"""Small, request-external disjoint LinUCB policies for the locked CSGB protocol.

One instance belongs to one context and one seed. ``choose(..., explore=False)``
uses predicted means only and is read-only, including the private RNG state.
Warm-up requires 32 *executed updates per arm*, not 32 updates in total.
No imports of model/GPU code or optional numerical dependencies are needed.
"""

import math
import random
import statistics
from dataclasses import dataclass


BUDGETS = (8, 16, 24, 32)
CONTEXTS = (4096, 8192)
TRAIN_DOCUMENTS = tuple(range(24))
SOURCE_SCHEMA = "csgb_online_disjoint_linucb_v1"
SCALE_SCHEMA = "csgb_fixed8_train24_population_scale_v1"


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"Invalid {name}")
    return value


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {name}")
    return value


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Unexpected {name} schema")


def _raw_features(raw_x, mode):
    size = 3 if mode == "history_native" else 4
    if not isinstance(raw_x, (list, tuple)) or len(raw_x) != size:
        raise ValueError(f"Expected {size} raw features")
    x = tuple(_number(v, "feature") for v in raw_x)
    if x[0] != 1.0 or any(not 0 <= v <= 1 for v in x[1:]):
        raise ValueError("Expected intercept 1 and bounded raw features")
    if size == 4 and not math.isclose(x[3], x[1] * x[2], rel_tol=0, abs_tol=1e-12):
        raise ValueError("CSGB interaction must be the raw C*D product")
    return x


@dataclass(frozen=True)
class FeatureScale:
    """Population statistics for raw C, D, and raw C*D, in that order."""

    context: int
    mean: tuple
    scale: tuple
    count: int
    documents: tuple = TRAIN_DOCUMENTS

    def __post_init__(self):
        if type(self.context) is not int or self.context not in CONTEXTS:
            raise ValueError("Invalid scale context")
        if not isinstance(self.documents, (list, tuple)) or tuple(self.documents) != TRAIN_DOCUMENTS or any(type(v) is not int for v in self.documents):
            raise ValueError("Scale must cover exactly the 24 training documents")
        _integer(self.count, "scale count", minimum=24)
        if not isinstance(self.mean, (list, tuple)) or not isinstance(self.scale, (list, tuple)) or len(self.mean) != 3 or len(self.scale) != 3:
            raise ValueError("Expected three scale statistics")
        object.__setattr__(self, "mean", tuple(_number(v, "mean") for v in self.mean))
        object.__setattr__(self, "scale", tuple(_number(v, "scale", positive=True) for v in self.scale))
        object.__setattr__(self, "documents", tuple(self.documents))

    def to_state(self):
        return {"schema": SCALE_SCHEMA, "context": self.context, "mean": list(self.mean),
                "scale": list(self.scale), "count": self.count, "documents": list(self.documents)}

    @classmethod
    def from_state(cls, state):
        _keys(state, ("schema", "context", "mean", "scale", "count", "documents"), "scale")
        if state["schema"] != SCALE_SCHEMA:
            raise ValueError("Unsupported scale schema")
        return cls(state["context"], state["mean"], state["scale"],
                   state["count"], state["documents"])


def fit_csgb_scales(rows):
    """Fit once from *all* valid Fixed8 training rounds; never use dev/test rows.

    Every row has exactly document_id/context/raw_x/policy/valid_round/split.
    Both contexts must contain every training document 0..23. Raw interaction
    is standardized separately; it is never recomputed from standardized C/D.
    Population std <= 1e-12 is replaced by exactly 1.
    """
    groups = {c: [] for c in CONTEXTS}
    docs = {c: set() for c in CONTEXTS}
    for row in rows:
        _keys(row, ("document_id", "context", "raw_x", "policy", "valid_round", "split"), "scale row")
        d, c = row["document_id"], row["context"]
        if type(d) is not int or d not in TRAIN_DOCUMENTS or type(c) is not int or c not in CONTEXTS:
            raise ValueError("Scaling accepts only train documents 0..23 and both locked contexts")
        if row["split"] != "train" or row["policy"] != "fixed8" or row["valid_round"] is not True:
            raise ValueError("Scaling requires valid Fixed8 training rounds only")
        groups[c].append(_raw_features(row["raw_x"], "csgb")[1:])
        docs[c].add(d)
    result = {}
    for c in CONTEXTS:
        if docs[c] != set(TRAIN_DOCUMENTS):
            raise ValueError(f"Missing full 24-document training coverage for context {c}")
        columns = list(zip(*groups[c]))
        means = tuple(statistics.fmean(v) for v in columns)
        scales = tuple(1.0 if statistics.pstdev(v) <= 1e-12 else statistics.pstdev(v) for v in columns)
        result[c] = FeatureScale(c, means, scales, len(groups[c]))
    return result


def _dot(a, b):
    return math.fsum(v * w for v, w in zip(a, b))


def _cholesky(matrix):
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j] - _dot(lower[i][:j], lower[j][:j])
            if i == j:
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Arm covariance must be positive definite")
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    return lower


def _solve(lower, vector):
    n = len(lower)
    forward, result = [0.0] * n, [0.0] * n
    for i in range(n):
        forward[i] = (vector[i] - _dot(lower[i][:i], forward[:i])) / lower[i][i]
    for i in reversed(range(n)):
        result[i] = (forward[i] - math.fsum(lower[j][i] * result[j] for j in range(i + 1, n))) / lower[i][i]
    return result


class OnlineCSGBPolicy:
    """Disjoint LinUCB, with executed-arm-only updates and explicit reward units.

    ``rho`` is fixed tokens/second; reward is useful_n - rho * elapsed_s.
    ``history_native`` inputs are already divided by 10 and remain unscaled.
    ``None`` denotes unavailable startup state: choose 8, update nothing.
    """

    def __init__(self, mode, *, context, seed, rho, scale=None):
        if mode not in ("csgb", "constant", "history_native"):
            raise ValueError("Unknown online policy mode")
        if type(context) is not int or context not in CONTEXTS:
            raise ValueError("Invalid context")
        _integer(seed, "seed")
        if seed >= 2**63:
            raise ValueError("Seed must be less than 2**63")
        self.mode, self.context, self.seed = mode, context, seed
        self.rho = _number(rho, "rho", positive=True)
        if scale is not None and (not isinstance(scale, FeatureScale) or scale.context != context):
            raise ValueError("Scale/context mismatch")
        if mode == "csgb" and scale is None:
            raise ValueError("CSGB requires fixed training-only scaling")
        if mode == "history_native" and scale is not None:
            raise ValueError("History policy must use identity scaling")
        self.scale = scale
        self.budgets = BUDGETS
        self.alpha, self.ridge, self.min_exploration, self.epsilon = 1.0, 1.0, 32, 0.05
        self.dimension = {"csgb": 4, "constant": 1, "history_native": 3}[mode]
        self._rng = random.Random(seed)
        self.counts = [0] * len(BUDGETS)
        self.rewards = [0.0] * len(BUDGETS)
        self._a = [[[float(i == j) for j in range(self.dimension)] for i in range(self.dimension)] for _ in BUDGETS]
        self._b = [[0.0] * self.dimension for _ in BUDGETS]

    def transform(self, raw_x):
        """Pure validated projection; safe for logging before frozen choice."""
        if raw_x is None:
            return None
        raw = _raw_features(raw_x, self.mode)
        if self.mode == "constant":
            return (1.0,)
        if self.mode == "history_native":
            return raw
        transformed = (1.0,) + tuple((v - m) / s for v, m, s in zip(raw[1:], self.scale.mean, self.scale.scale))
        if not all(math.isfinite(v) for v in transformed):
            raise ValueError("Nonfinite transformed feature")
        return transformed

    def choose(self, raw_x, explore=True):
        if type(explore) is not bool:
            raise ValueError("explore must be bool")
        x = self.transform(raw_x)
        if x is None:
            return BUDGETS[0]
        if explore:
            least = min(self.counts)
            if least < self.min_exploration:
                return self._rng.choice([b for b, count in zip(BUDGETS, self.counts) if count == least])
            if self._rng.random() < self.epsilon:
                return self._rng.choice(BUDGETS)
        scores = []
        for a, b in zip(self._a, self._b):
            lower = _cholesky(a)
            score = _dot(x, _solve(lower, b))
            if explore:
                score += self.alpha * math.sqrt(max(0.0, _dot(x, _solve(lower, x))))
            if not math.isfinite(score):
                raise ValueError("Nonfinite policy score")
            scores.append(score)
        # max returns the first index: exact score ties use the smaller budget.
        return BUDGETS[max(range(len(BUDGETS)), key=scores.__getitem__)]

    def update(self, raw_x, budget, useful_n, elapsed_s, rho=None):
        """Update only the arm actually executed; optional rho must equal frozen rho."""
        if type(budget) is not int or budget not in BUDGETS:
            raise ValueError("Invalid executed budget")
        _integer(useful_n, "useful_n")
        _number(useful_n, "useful_n", nonnegative=True)
        elapsed_s = _number(elapsed_s, "elapsed_s", positive=True)
        if rho is not None and _number(rho, "rho", positive=True) != self.rho:
            raise ValueError("Reward rho cannot change")
        x = self.transform(raw_x)
        if x is None:
            return None
        reward = useful_n - self.rho * elapsed_s
        if not math.isfinite(reward):
            raise ValueError("Nonfinite reward")
        arm = BUDGETS.index(budget)
        a = [[v + x[i] * x[j] for j, v in enumerate(row)] for i, row in enumerate(self._a[arm])]
        b = [v + reward * z for v, z in zip(self._b[arm], x)]
        total = self.rewards[arm] + reward
        if not all(math.isfinite(v) for row in a for v in row) or not all(math.isfinite(v) for v in b) or not math.isfinite(total):
            raise ValueError("Nonfinite updated parameters")
        _cholesky(a)
        self._a[arm], self._b[arm] = a, b
        self.rewards[arm] = total
        self.counts[arm] += 1
        return reward

    def to_state(self):
        version, state, spare = self._rng.getstate()
        return {"schema": 1, "source_schema": SOURCE_SCHEMA, "mode": self.mode,
                "context": self.context, "seed": self.seed, "rho": self.rho,
                "budgets": list(BUDGETS), "fallback": 8, "alpha": self.alpha,
                "ridge": self.ridge, "min_exploration": self.min_exploration,
                "exploration_unit": "executed_updates_per_arm", "epsilon": self.epsilon,
                "scale": None if self.scale is None else self.scale.to_state(),
                "counts": list(self.counts), "rewards": list(self.rewards),
                "a": [[list(row) for row in arm] for arm in self._a],
                "b": [list(row) for row in self._b],
                "rng": {"version": version, "state": list(state), "gauss_next": spare}}

    @classmethod
    def from_state(cls, state):
        expected = ("schema", "source_schema", "mode", "context", "seed", "rho", "budgets", "fallback",
                    "alpha", "ridge", "min_exploration", "exploration_unit", "epsilon", "scale",
                    "counts", "rewards", "a", "b", "rng")
        _keys(state, expected, "policy")
        locked = {"schema": 1, "source_schema": SOURCE_SCHEMA, "budgets": list(BUDGETS), "fallback": 8,
                  "alpha": 1.0, "ridge": 1.0, "min_exploration": 32,
                  "exploration_unit": "executed_updates_per_arm", "epsilon": 0.05}
        if any(state[k] != v or (isinstance(v, (int, float)) and isinstance(state[k], bool)) for k, v in locked.items()):
            raise ValueError("Policy protocol changed")
        if any(type(state[k]) is not int for k in ("schema", "fallback", "min_exploration")) or any(type(v) is not int for v in state["budgets"]):
            raise ValueError("Protocol integer fields must be integers")
        scale = None if state["scale"] is None else FeatureScale.from_state(state["scale"])
        result = cls(state["mode"], context=state["context"], seed=state["seed"], rho=state["rho"], scale=scale)
        dim = result.dimension
        if any(not isinstance(state[k], list) or len(state[k]) != 4 for k in ("counts", "rewards", "a", "b")):
            raise ValueError("Expected four disjoint arm states")
        result.counts = [_integer(v, "arm count") for v in state["counts"]]
        result.rewards = [_number(v, "reward total") for v in state["rewards"]]
        for arm in range(4):
            matrix, vector = state["a"][arm], state["b"][arm]
            if not isinstance(matrix, list) or len(matrix) != dim or any(not isinstance(row, list) or len(row) != dim for row in matrix):
                raise ValueError("Invalid covariance shape")
            if not isinstance(vector, list) or len(vector) != dim:
                raise ValueError("Invalid reward vector shape")
            matrix = [[_number(v, "covariance") for v in row] for row in matrix]
            vector = [_number(v, "reward vector") for v in vector]
            if any(not math.isclose(matrix[i][j], matrix[j][i], rel_tol=1e-12, abs_tol=1e-12) for i in range(dim) for j in range(dim)):
                raise ValueError("Nonsymmetric covariance")
            _cholesky(matrix)
            if not math.isclose(matrix[0][0], 1 + result.counts[arm], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Intercept/count mismatch")
            if not math.isclose(vector[0], result.rewards[arm], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Intercept/reward mismatch")
            if result.counts[arm] == 0 and (any(vector) or any(matrix[i][j] != float(i == j) for i in range(dim) for j in range(dim))):
                raise ValueError("Unexecuted arm was modified")
            result._a[arm], result._b[arm] = matrix, vector
        rng = state["rng"]
        _keys(rng, ("version", "state", "gauss_next"), "RNG")
        if type(rng["version"]) is not int or rng["version"] != 3 or rng["gauss_next"] is not None:
            raise ValueError("Invalid RNG format")
        if not isinstance(rng["state"], list) or len(rng["state"]) != 625:
            raise ValueError("Invalid RNG state length")
        if any(type(v) is not int or not 0 <= v < 2**32 for v in rng["state"][:-1]) or type(rng["state"][-1]) is not int or not 0 <= rng["state"][-1] <= 624:
            raise ValueError("Invalid RNG state values")
        result._rng.setstate((3, tuple(rng["state"]), None))
        return result
