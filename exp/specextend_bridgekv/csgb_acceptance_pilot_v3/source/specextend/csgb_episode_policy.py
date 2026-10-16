"""Immutable CPU softmax policy and whole-request REINFORCE arithmetic.

Sampling is identical during training/deployment; there is no greedy/explore
switch. RNG belongs to the caller, never parameters. The future runner must
prove canonical useful N, boundary-synchronized complete-request T, and an
action-independent baseline frozen before this episode. No fitting occurs here.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import re

BUDGETS = (8, 16, 24, 32)
PARAMETER_SCHEMA = "csgb_episode_softmax_v1"
DECISION_SCHEMA = "csgb_episode_decision_v1"


def _number(value, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError("Finite numeric scalar required, never bool")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("Unrepresentable scalar") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("Invalid finite scalar")
    return value


def _vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError("Wrong vector shape")
    return tuple(_number(v) for v in value)


def _matrix(value, width):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Exactly four action rows required")
    return tuple(_vector(row, width) for row in value)


def _version(value):
    if type(value) is not int or value < 0:
        raise ValueError("Nonnegative integer parameter version required")
    return value


def _hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid parameter SHA256")
    return value


def _raw(value, width):
    x = _vector(value, width)
    if (x[0] != 1. or any(not 0 <= v <= 1 for v in x[1:])
            or width == 4 and not math.isclose(x[3], x[1]*x[2], rel_tol=0., abs_tol=1e-12)):
        raise ValueError("Raw features require intercept1, bounded inputs and raw C*D")
    return x


def _sum(values):
    try:
        return _number(math.fsum(values))
    except OverflowError as exc:
        raise ValueError("Nonfinite arithmetic") from exc


def _action(u, probabilities):
    # Fixed three comparisons, no action-dependent early-exit sampling path.
    return BUDGETS[sum(u >= math.fsum(probabilities[:i]) for i in (1, 2, 3))]


@dataclass(frozen=True)
class Decision:
    parameter_sha256: str
    parameter_version: int
    raw_x: object
    transformed_x: object
    probabilities: tuple
    u: object
    budget: int

    def __post_init__(self):
        _hash(self.parameter_sha256); _version(self.parameter_version)
        p = _vector(self.probabilities, 4)
        if any(not 0 <= v <= 1 for v in p) or not math.isclose(math.fsum(p), 1., rel_tol=0., abs_tol=1e-12):
            raise ValueError("Invalid categorical probabilities")
        if type(self.budget) is not int or self.budget not in BUDGETS:
            raise ValueError("Unsupported action")
        if self.raw_x is None:
            if self.transformed_x is not None or self.u is not None or self.budget != 8 or p != (1.,0.,0.,0.):
                raise ValueError("Startup must force8 without RNG or score")
        else:
            if not isinstance(self.raw_x, (list, tuple)) or len(self.raw_x) not in (3, 4):
                raise ValueError("Wrong raw decision shape")
            raw = _raw(self.raw_x, len(self.raw_x))
            if not isinstance(self.transformed_x, (list, tuple)) or len(self.transformed_x) not in (1, 3, 4):
                raise ValueError("Wrong transformed decision shape")
            transformed = _vector(self.transformed_x, len(self.transformed_x))
            u = _number(self.u)
            if not 0 <= u < 1 or min(p) <= 0 or _action(u, p) != self.budget:
                raise ValueError("Invalid uniform/probability/action alignment")
            object.__setattr__(self, "raw_x", raw)
            object.__setattr__(self, "transformed_x", transformed)
            object.__setattr__(self, "u", u)
        object.__setattr__(self, "probabilities", p)

    def to_state(self):
        return {"schema": 1, "source_schema": DECISION_SCHEMA, "parameter_sha256": self.parameter_sha256,
                "parameter_version": self.parameter_version, "raw_x": None if self.raw_x is None else list(self.raw_x),
                "transformed_x": None if self.transformed_x is None else list(self.transformed_x),
                "probabilities": list(self.probabilities), "u": self.u, "budget": self.budget}

    @classmethod
    def from_state(cls, state):
        keys = {"schema","source_schema","parameter_sha256","parameter_version","raw_x","transformed_x","probabilities","u","budget"}
        if (not isinstance(state, dict) or set(state) != keys or type(state["schema"]) is not int
                or state["schema"] != 1 or state["source_schema"] != DECISION_SCHEMA):
            raise ValueError("Unsupported decision serialization")
        return cls(**{k:v for k,v in state.items() if k not in {"schema","source_schema"}})


@dataclass(frozen=True)
class EpisodeGradient:
    parameter_sha256: str
    parameter_version: int
    values: tuple

    def __post_init__(self):
        _hash(self.parameter_sha256); _version(self.parameter_version)
        if not isinstance(self.values, (list, tuple)) or not self.values:
            raise ValueError("Missing gradient rows")
        width = len(self.values[0]) if isinstance(self.values[0], (list, tuple)) else 0
        if width not in (1, 3, 4):
            raise ValueError("Wrong gradient dimension")
        object.__setattr__(self, "values", _matrix(self.values, width))


@dataclass(frozen=True)
class EpisodePolicy:
    mode: str
    weights: tuple
    temperature: float
    scale_mean: object = None
    scale_std: object = None
    parameter_version: int = 0
    _sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.mode not in ("csgb", "constant", "history_native"):
            raise ValueError("Unknown policy mode")
        _version(self.parameter_version)
        object.__setattr__(self, "temperature", _number(self.temperature, positive=True))
        object.__setattr__(self, "weights", _matrix(self.weights, self.dimension))
        if self.mode == "csgb":
            means, std = _vector(self.scale_mean, 3), _vector(self.scale_std, 3)
            if min(std) <= 0:
                raise ValueError("Fixed three-feature scale must be positive")
            object.__setattr__(self, "scale_mean", means)
            object.__setattr__(self, "scale_std", std)
        elif self.scale_mean is not None or self.scale_std is not None:
            raise ValueError("Constant/history may not hide a feature scale")
        payload = json.dumps(self.to_state(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        object.__setattr__(self, "_sha256", hashlib.sha256(payload).hexdigest())

    @property
    def dimension(self):
        return {"csgb": 4, "constant": 1, "history_native": 3}[self.mode]

    @property
    def parameter_sha256(self):
        return self._sha256

    def transform(self, raw_x):
        if raw_x is None:
            return None
        raw = _raw(raw_x, 3 if self.mode == "history_native" else 4)
        if self.mode == "constant":
            return (1.,)
        if self.mode == "history_native":
            return raw
        # The third normalized feature is raw C*D, never normalizedC*normalizedD.
        return _vector((1.,)+tuple((v-m)/s for v,m,s in zip(raw[1:], self.scale_mean, self.scale_std)), 4)

    def probabilities(self, raw_x):
        x = self.transform(raw_x)
        if x is None:
            return (1.,0.,0.,0.)
        logits = tuple(_number(_sum(w*v for w,v in zip(row,x))/self.temperature) for row in self.weights)
        top = max(logits)
        exponentials = tuple(math.exp(v-top) for v in logits)
        total = math.fsum(exponentials)
        p = tuple(v/total for v in exponentials)
        if min(p) <= 0 or any(not math.isfinite(v) for v in p):
            raise ValueError("Softmax probability underflow/nonfinite; never silently clip")
        return p

    def sample(self, raw_x, rng):
        if not isinstance(rng, random.Random):
            raise ValueError("The caller must supply its request-local Random")
        x = self.transform(raw_x); p = self.probabilities(raw_x)
        raw = None if raw_x is None else tuple(float(v) for v in raw_x)
        # All validation precedes RNG consumption; exactly one draw if valid.
        u = None if raw_x is None else _number(rng.random())
        return Decision(self.parameter_sha256, self.parameter_version, raw, x, p, u,
                        8 if u is None else _action(u,p))

    def replay(self, decision):
        if not isinstance(decision, Decision):
            raise ValueError("Replay requires an audited Decision")
        if (decision.parameter_sha256 != self.parameter_sha256 or decision.parameter_version != self.parameter_version
                or decision.transformed_x != self.transform(decision.raw_x)
                or decision.probabilities != self.probabilities(decision.raw_x)):
            raise ValueError("Decision parameter version/scale/probabilities differ")
        return decision

    def sgd(self, gradient, learning_rate):
        if (not isinstance(gradient, EpisodeGradient) or gradient.parameter_sha256 != self.parameter_sha256
                or gradient.parameter_version != self.parameter_version):
            raise ValueError("Gradient belongs to a different immutable parameter version")
        values = _matrix(gradient.values, self.dimension)
        eta = _number(learning_rate, positive=True)
        updated = tuple(tuple(_number(w+eta*g) for w,g in zip(row,step)) for row,step in zip(self.weights,values))
        return EpisodePolicy(self.mode, updated, self.temperature, self.scale_mean, self.scale_std, self.parameter_version+1)

    def to_state(self):
        return {"schema": 1, "source_schema": PARAMETER_SCHEMA, "native_v": True,
                "budgets": list(BUDGETS), "fallback": 8, "mode": self.mode, "parameter_version": self.parameter_version,
                "temperature": self.temperature, "weights": [list(row) for row in self.weights],
                "scale_mean": None if self.scale_mean is None else list(self.scale_mean),
                "scale_std": None if self.scale_std is None else list(self.scale_std)}

    @classmethod
    def from_state(cls, state):
        keys = {"schema","source_schema","native_v","budgets","fallback","mode","parameter_version","temperature","weights","scale_mean","scale_std"}
        if (not isinstance(state, dict) or set(state) != keys or type(state["schema"]) is not int
                or state["schema"] != 1 or state["source_schema"] != PARAMETER_SCHEMA or state["native_v"] is not True
                or type(state["fallback"]) is not int or state["fallback"] != 8
                or not isinstance(state["budgets"], list) or state["budgets"] != list(BUDGETS)
                or any(type(v) is not int for v in state["budgets"])):
            raise ValueError("Unsupported/mutated NativeV policy serialization")
        return cls(**{k:v for k,v in state.items() if k not in {"schema","source_schema","native_v","budgets","fallback"}})


def episode_gradient(policy, decisions, *, useful_n, elapsed_s, rho, baseline):
    """(N-rho*T-B) times the SUM of actual valid-decision log-policy scores.

    No division by random episode length. Baseline independence, canonical N,
    on-policy parameters fixed throughout this request, and complete-request
    timing are obligations of the future trainer/runner, not inferred here.
    """
    if not isinstance(policy, EpisodePolicy) or not isinstance(decisions, (list, tuple)):
        raise ValueError("Policy and complete request decision sequence required")
    if type(useful_n) is not int or useful_n < 1:
        raise ValueError("Useful returned-token count must be a positive integer")
    return_value = _number(_number(useful_n)-_number(rho, positive=True)*_number(elapsed_s, positive=True))
    advantage = _number(return_value-_number(baseline))
    valid = [policy.replay(d) for d in decisions if isinstance(d, Decision)]
    if len(valid) != len(decisions):
        raise ValueError("Unrecorded or malformed decision in episode")
    values = tuple(tuple(_number(advantage*_sum(
        ((1. if d.budget == b else 0.)-d.probabilities[k])*d.transformed_x[j]/policy.temperature
        for d in valid if d.raw_x is not None)) for j in range(policy.dimension)) for k,b in enumerate(BUDGETS))
    return EpisodeGradient(policy.parameter_sha256, policy.parameter_version, values)
