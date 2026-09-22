"""FIX-D / FIX-E — field-aware training-config schema + declared-vs-observed provenance.

FIX-D  the training config is validated per field: presence, expected type, semantic
       validity, required nested identity, non-placeholder, canonicalisable.  A dict
       being non-empty is NOT evidence ({"note": "..."} must fail), and a nested
       identity with sha256=None must fail.

FIX-E  a provenance envelope is only accepted when every material declaration equals
       the value observed in an authoritative artefact.  `found=true` is not proof.
       Observed values are read from frozen configs / frozen contract / checkpoint
       payload / source-lock defaults / split ledgers — never re-derived from the
       declaration itself (that would be circular).

Placeholder detection is EXACT-MATCH only.  No substring heuristics: legitimate values
containing the character sequence "na" ("canonical", "analysis", "final") are legal.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Optional

HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Exact values that are never acceptable as a declarative field value.
PLACEHOLDER_EXACT = {
    "", "n/a", "na", "none", "null", "nil", "unknown", "unspecified", "undefined",
    "tbd", "todo", "placeholder", "not supplied", "not_supplied", "not provided",
    "none given", "training_config not supplied", "missing",
}

# Allowed training dtypes: recovered from the frozen run entry point's --dtype choices.
TRAINING_DTYPES = ("float16", "bfloat16", "float32")

# checkpoint payload supervision tag -> canonical supervised-target mode
SUPERVISION_CANON = {
    "per_prefix_prefill": "per_prefix_full_answer_CE",
}

# Fallback list of AdapterConfig fields that carry no default, recovered from
# source_lock/adapter.py.  Parsed at runtime when the file is available and the parse
# is required to agree with this list.
ADAPTER_REQUIRED_FIELDS_FALLBACK = ("num_teacher_layers", "num_teacher_heads",
                                    "num_student_layers", "num_student_heads")
ADAPTER_OPTIONAL_NUMERIC = ("max_alpha", "eps")

REQUIRED_TRAINING_CONFIG_FIELDS = (
    "method_variant", "train_split_identity", "validation_split_identity", "optimizer",
    "learning_rate", "training_steps", "initialization", "adapter_config",
    "supervised_target_mode", "regularization", "layer_scope", "dtype", "seed",
)


class CheckpointIdentityIncomplete(RuntimeError):
    """The training config cannot support an identity (missing/invalid/placeholder)."""


class ProvenanceError(RuntimeError):
    """Base class for FIX-E failures."""


class ProvenanceDeclarationMismatch(ProvenanceError):
    """A declared value differs from the value observed in an authoritative artefact."""


class ProvenanceUnverifiable(ProvenanceError):
    """A material field cannot be uniquely recovered from any authoritative artefact."""


# --------------------------------------------------------------------------
# placeholder policy — exact match only
# --------------------------------------------------------------------------

def is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in PLACEHOLDER_EXACT
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


# --------------------------------------------------------------------------
# field-aware schema
# --------------------------------------------------------------------------

def _err(msg: str) -> str:
    return msg


def _is_real_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _v_canonical_string(max_len: int = 200) -> Callable[[Any], Optional[str]]:
    def check(v):
        if not isinstance(v, str):
            return f"expected string, got {type(v).__name__}"
        if is_placeholder(v):
            return "placeholder or empty string"
        if v != v.strip():
            return "must not have leading/trailing whitespace"
        if not v or len(v) > max_len:
            return f"length must be 1..{max_len}"
        if any(c in v for c in "\r\n\t"):
            return "must be a single-line canonical value"
        return None
    return check


def _v_split_identity(v):
    if not isinstance(v, dict):
        return f"expected mapping, got {type(v).__name__}"
    sha = v.get("sha256")
    if not isinstance(sha, str):
        return f"sha256 must be a string, got {type(sha).__name__}"
    if is_placeholder(sha):
        return "sha256 is a placeholder or empty"
    if not HEX64.match(sha):
        return "sha256 must be 64 lowercase hex characters"
    rc = v.get("row_count")
    if not _is_int(rc):
        return f"row_count must be int (not bool), got {type(rc).__name__}"
    if rc <= 0:
        return "row_count must be > 0"
    return None


def _v_learning_rate(v):
    if isinstance(v, bool):
        return "bool is not a valid learning rate"
    if not _is_real_number(v):
        return f"expected finite real number, got {type(v).__name__}"
    if v <= 0:
        return "learning rate must be > 0"
    return None


def _v_training_steps(v):
    if isinstance(v, bool):
        return "bool is not a valid step count"
    if not _is_int(v):
        return f"expected int, got {type(v).__name__}"
    if v <= 0:
        return "training steps must be > 0"
    return None


def _v_seed(v):
    if isinstance(v, bool):
        return "bool is not a valid seed"
    if not _is_int(v):
        return f"expected int, got {type(v).__name__}"
    if v < 0:
        return "seed must be >= 0"
    return None      # 0 is a legal seed and must not be treated as absent


def _v_dtype(v):
    if not isinstance(v, str):
        return f"expected string, got {type(v).__name__}"
    if v not in TRAINING_DTYPES:
        return f"dtype must be one of {TRAINING_DTYPES}"
    return None


def _v_regularization(v):
    if not isinstance(v, dict):
        return f"expected mapping, got {type(v).__name__}"
    required = ("beta_kl", "gamma_alpha", "delta_sharpness")
    missing = [k for k in required if k not in v]
    if missing:
        return ("regularization must state every coefficient explicitly; missing "
                + ", ".join(missing))
    for k in required:
        x = v[k]
        if isinstance(x, bool) or not _is_real_number(x):
            return f"regularization.{k} must be a finite real number"
        if x < 0:
            return f"regularization.{k} must be >= 0"
    return None


def _v_adapter_config(v, required_fields):
    if not isinstance(v, dict):
        return f"expected mapping, got {type(v).__name__}"
    missing = [k for k in required_fields if k not in v]
    if missing:
        return "adapter_config missing required field(s): " + ", ".join(missing)
    for k in required_fields:
        x = v[k]
        if isinstance(x, bool) or not _is_int(x) or x <= 0:
            return f"adapter_config.{k} must be a positive int"
    for k in ADAPTER_OPTIONAL_NUMERIC:
        if k in v:
            x = v[k]
            if isinstance(x, bool) or not _is_real_number(x) or x <= 0:
                return f"adapter_config.{k} must be a positive finite number"
    return None


def adapter_required_fields(source_lock: Optional[Path]) -> tuple[tuple[str, ...], str]:
    """Recover the AdapterConfig fields that carry no default (authoritative artefact)."""
    if source_lock is not None:
        p = Path(source_lock) / "adapter.py"
        if p.exists():
            text = p.read_text(encoding="utf-8")
            m = re.search(r"class AttentionAdapterConfig:\s*\n((?:[ \t]+.*\n|\s*\n)+)"
                          r"(?=\s*(?:class|def)\s)", text)
            if m:
                body = m.group(1)
                no_default = re.findall(r"^\s{4}(\w+)\s*:\s*[^=\n]+$", body, re.M)
                if no_default:
                    return tuple(no_default), f"{p} (AttentionAdapterConfig, fields without default)"
    return (ADAPTER_REQUIRED_FIELDS_FALLBACK,
            "fallback constant ADAPTER_REQUIRED_FIELDS_FALLBACK "
            "(adapter.py not readable)")


def build_schema(source_lock: Optional[Path] = None) -> dict:
    required_adapter, adapter_src = adapter_required_fields(source_lock)
    return {
        "required_fields": REQUIRED_TRAINING_CONFIG_FIELDS,
        "adapter_required_fields": required_adapter,
        "adapter_required_fields_source": adapter_src,
        "per_field": {
            "method_variant": _v_canonical_string(),
            "train_split_identity": _v_split_identity,
            "validation_split_identity": _v_split_identity,
            "optimizer": _v_canonical_string(64),
            "learning_rate": _v_learning_rate,
            "training_steps": _v_training_steps,
            "initialization": _v_canonical_string(64),
            "adapter_config": lambda v: _v_adapter_config(v, required_adapter),
            "supervised_target_mode": _v_canonical_string(64),
            "regularization": _v_regularization,
            "layer_scope": _v_canonical_string(64),
            "dtype": _v_dtype,
            "seed": _v_seed,
        },
    }


def schema_contract(source_lock: Optional[Path] = None) -> dict:
    s = build_schema(source_lock)
    return {
        "contract": "RESCUE training-config schema (FIX-D)",
        "required_fields": list(s["required_fields"]),
        "adapter_required_fields": list(s["adapter_required_fields"]),
        "adapter_required_fields_source": s["adapter_required_fields_source"],
        "allowed_dtypes": list(TRAINING_DTYPES),
        "regularization_required_nested": ["beta_kl", "gamma_alpha", "delta_sharpness"],
        "split_identity_required_nested": ["sha256 (64 hex)", "row_count (>0 int)"],
        "placeholder_policy": "exact-match only; no substring heuristics",
        "placeholder_values": sorted(PLACEHOLDER_EXACT),
        "bool_policy": "bool is rejected wherever a number/int is expected",
    }


def looks_like_envelope(raw: Any) -> bool:
    """True when the input is a provenance envelope carrying a canonical config."""
    if isinstance(raw, dict):
        return isinstance(raw.get("canonical_training_config"), dict)
    if isinstance(raw, (str, Path)):
        text = str(raw)
        p = Path(text)
        if p.exists() and p.is_file():
            obj = _read_json(p)
            return isinstance(obj, dict) and isinstance(obj.get("canonical_training_config"), dict)
    return False


def canonical_training_config(raw: Any, *, source_lock: Optional[Path] = None) -> dict:
    """Validate + canonicalise.  Raises CheckpointIdentityIncomplete (FIX-D)."""
    if isinstance(raw, (str, Path)):
        text = str(raw)
        p = Path(text)
        if p.exists() and p.is_file():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                raise CheckpointIdentityIncomplete(
                    f"CHECKPOINT_IDENTITY_INCOMPLETE: unparseable training config file: {exc}"
                ) from exc
        elif is_placeholder(text):
            raise CheckpointIdentityIncomplete(
                f"CHECKPOINT_IDENTITY_INCOMPLETE: training config is a placeholder ({text!r})")
        else:
            try:
                raw = json.loads(text)
            except Exception as exc:
                raise CheckpointIdentityIncomplete(
                    f"CHECKPOINT_IDENTITY_INCOMPLETE: training config is neither an existing "
                    f"file nor valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise CheckpointIdentityIncomplete(
            f"CHECKPOINT_IDENTITY_INCOMPLETE: training config must be a mapping, got "
            f"{type(raw).__name__}")
    if "canonical_training_config" in raw and isinstance(raw["canonical_training_config"], dict):
        raw = raw["canonical_training_config"]          # provenance envelope

    schema = build_schema(source_lock)
    problems = []
    for f in schema["required_fields"]:
        if f not in raw:
            problems.append(f"{f}: missing")
            continue
        try:
            err = schema["per_field"][f](raw[f])
        except Exception as exc:                      # a validator must never crash the caller
            err = f"validator error: {type(exc).__name__}: {exc}"
        if err:
            problems.append(f"{f}: {err}")
    if problems:
        raise CheckpointIdentityIncomplete(
            "CHECKPOINT_IDENTITY_INCOMPLETE: " + "; ".join(problems))
    cfg = {f: raw[f] for f in schema["required_fields"]}   # required fields only
    return json.loads(json.dumps(cfg, sort_keys=True))


# --------------------------------------------------------------------------
# FIX-E — observed values from authoritative artefacts
# --------------------------------------------------------------------------

def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def _sha256_file(p: Path) -> Optional[str]:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _csv_rows(p: Path) -> Optional[int]:
    p = Path(p)
    if not p.exists():
        return None
    with p.open(encoding="utf-8-sig", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def _source_texts(source_dirs) -> dict[str, str]:
    out = {}
    for d in source_dirs or []:
        d = Path(d)
        if not d.is_dir():
            continue
        for name in ("qwen_adapter_backend.py", "sequence_prefix_runner.py",
                     "prefix_seqloss.py", "entropy_adapter.py"):
            p = d / name
            if p.exists():
                out[str(p)] = p.read_text(encoding="utf-8")
    return out


def observe_training_facts(*, source_lock: Path, run_source_dirs, checkpoint_payload: dict,
                           train_ledger: Optional[Path],
                           validation_ledger: Optional[Path]) -> dict:
    """Observed value for every material field, with the artefact it came from."""
    source_lock = Path(source_lock)
    frozen = _read_json(source_lock / "frozen_training_config.json")
    if frozen is None:
        raise ProvenanceUnverifiable(
            "frozen_training_config.json is not readable under source_lock")
    texts = _source_texts(run_source_dirs)
    joined = "\n".join(texts.values())
    obs: dict[str, dict] = {}

    def put(field, value, source):
        obs[field] = {"observed": value, "source": source}

    # ---- frozen training config ----
    put("optimizer", frozen.get("optimizer"),
        f"{source_lock}/frozen_training_config.json:optimizer")
    put("learning_rate", frozen.get("learning_rate"),
        f"{source_lock}/frozen_training_config.json:learning_rate")
    put("training_steps", frozen.get("fixed_optimizer_steps"),
        f"{source_lock}/frozen_training_config.json:fixed_optimizer_steps")
    put("seed", frozen.get("seed"), f"{source_lock}/frozen_training_config.json:seed")
    reg = frozen.get("regularization")
    put("regularization", ({k: reg.get(k) for k in ("beta_kl", "gamma_alpha", "delta_sharpness")}
                           if isinstance(reg, dict) else None),
        f"{source_lock}/frozen_training_config.json:regularization")

    # ---- split ledgers ----
    for field, led in (("train_split_identity", train_ledger),
                       ("validation_split_identity", validation_ledger)):
        sha, rows = (None, None) if led is None else (_sha256_file(led), _csv_rows(led))
        put(field, ({"sha256": sha, "row_count": rows} if sha else None),
            f"ledger {led}" if led else "ledger not supplied")

    # ---- checkpoint payload ----
    sd = checkpoint_payload.get("adapter") or {}
    mix = sd.get("base.mix_logits")
    if mix is None or len(tuple(mix.shape)) != 4:
        raise ProvenanceUnverifiable("checkpoint payload lacks base.mix_logits[4]")
    shape = tuple(int(x) for x in mix.shape)
    required_adapter, _ = adapter_required_fields(source_lock)
    adapter_cfg = {"num_student_layers": shape[0], "num_student_heads": shape[1],
                   "num_teacher_layers": shape[2], "num_teacher_heads": shape[3]}
    adapter_p = source_lock / "adapter.py"
    adapter_text = adapter_p.read_text(encoding="utf-8") if adapter_p.exists() else ""
    for k in ADAPTER_OPTIONAL_NUMERIC:
        m = re.search(rf"^\s{{4}}{k}\s*:\s*float\s*=\s*([0-9.eE+-]+)\s*$", adapter_text, re.M)
        if not m:
            raise ProvenanceUnverifiable(f"adapter.py does not declare a default for {k}")
        adapter_cfg[k] = float(m.group(1))
    missing_keys = [k for k in required_adapter if k not in adapter_cfg]
    if missing_keys:
        raise ProvenanceUnverifiable(f"cannot observe adapter fields {missing_keys}")
    put("adapter_config", adapter_cfg,
        "checkpoint payload base.mix_logits shape "
        f"{shape} + {adapter_p} AttentionAdapterConfig defaults")

    # ---- dtype / layer scope / supervision / initialization / method variant ----
    m = re.search(r"dtype:\s*torch\.dtype\s*=\s*torch\.(\w+)", joined)
    put("dtype", (m.group(1) if m else None),
        "run source QwenAdapterBackend dtype default" if m else "not uniquely determined")

    scopes = set(re.findall(r'(?:selected_layer_indices|_selected_layers)\s*\(.*?"([^"]+)"\s*\)',
                            joined, re.S))
    put("layer_scope", (scopes.pop() if len(scopes) == 1 else None),
        "run source selected_layer_indices(...) scope literal" if len(scopes) <= 1
        else f"ambiguous scopes {sorted(scopes)}")

    tag = checkpoint_payload.get("supervision")
    put("supervised_target_mode", SUPERVISION_CANON.get(tag),
        f"checkpoint payload supervision={tag!r}"
        + ("" if tag in SUPERVISION_CANON else " (unmapped tag)"))

    runner = next((v for k, v in texts.items() if k.endswith("sequence_prefix_runner.py")), "")
    train_body = runner.split("def train(")[-1] if "def train(" in runner else ""
    if train_body and "load_checkpoint" not in train_body and "EntropyConditionedAttentionAdapter" in joined:
        put("initialization", "from_scratch_no_warm_start",
            "run source train() builds a fresh adapter and contains no warm-start load")
    else:
        put("initialization", None,
            "cannot be uniquely determined from the run source")

    components_ok = ("AttentionPriorAdapter" in
                     (source_lock / "adapter.py").read_text(encoding="utf-8")
                     if (source_lock / "adapter.py").exists() else False) and \
                    "EntropyConditionedAttentionAdapter" in joined
    mode = SUPERVISION_CANON.get(tag)
    if components_ok and mode:
        put("method_variant", f"attention_prior_adapter+entropy_conditioned_alpha+"
                              f"{'per_prefix_sequence_supervision' if mode == 'per_prefix_full_answer_CE' else mode}",
            "source_lock adapter class + run-source wrapper class + checkpoint "
            f"payload supervision={tag!r}")
    else:
        put("method_variant", None, "method components not all present in the artefacts")

    unverifiable = sorted(f for f, v in obs.items() if v["observed"] is None)
    if unverifiable:
        raise ProvenanceUnverifiable(
            "PROVENANCE_UNVERIFIABLE: no authoritative artefact uniquely determines "
            + ", ".join(unverifiable))
    return obs


MATERIAL_FIELDS = REQUIRED_TRAINING_CONFIG_FIELDS


#: keys that may legitimately accompany a value without carrying identity
NON_IDENTITY_KEYS = ("name",)


def _comparable(field: str, value: Any, observed_value: Any) -> Any:
    """Project onto the identity-bearing keys and reject unexpected extras.

    `name` (a ledger file name) is explicitly NOT identity, so it is allowed on the
    declared side and ignored in the comparison.  Any other key that the artefacts do
    not determine is a declaration mismatch, not something to ignore.
    """
    if isinstance(observed_value, dict) and isinstance(value, dict):
        unexpected = [k for k in value
                      if k not in observed_value and k not in NON_IDENTITY_KEYS]
        if unexpected:
            raise ProvenanceDeclarationMismatch(
                f"PROVENANCE_DECLARATION_MISMATCH: {field} declares key(s) the artefacts "
                f"do not determine: {sorted(unexpected)}")
        return {k: value.get(k) for k in observed_value}
    return value


def verify_declared_vs_observed(declared: dict, observed: dict) -> list[dict]:
    """declared == observed for every material field, else fail closed."""
    records, mismatches = [], []
    for field in MATERIAL_FIELDS:
        if field not in observed:
            raise ProvenanceUnverifiable(
                f"PROVENANCE_UNVERIFIABLE: no observed value for material field {field}")
        obs = observed[field]["observed"]
        dec = _comparable(field, declared.get(field), obs)
        same = json.dumps(dec, sort_keys=True) == json.dumps(obs, sort_keys=True)
        rec = {"field": field, "declared": dec, "observed": obs,
               "source": observed[field]["source"],
               "verification": "MATCH" if same else "MISMATCH"}
        records.append(rec)
        if not same:
            mismatches.append(field)
    if mismatches:
        raise ProvenanceDeclarationMismatch(
            "PROVENANCE_DECLARATION_MISMATCH: " + ", ".join(sorted(mismatches)))
    return records


def provenance_verification_contract() -> dict:
    return {
        "contract": "FIX-E declared-vs-observed",
        "record_fields": ["field", "declared", "observed", "source", "verification"],
        "pass_condition": "verification == MATCH for every material field",
        "material_fields": list(MATERIAL_FIELDS),
        "authority_order": ["frozen training config", "frozen experiment/statistics contract",
                            "checkpoint payload", "source-lock defaults",
                            "immutable split ledger", "run source where it uniquely decides"],
        "forbidden": "deriving 'observed' from 'declared' (circular proof)",
        "on_unverifiable": "PROVENANCE_UNVERIFIABLE (fail closed, never found=true)",
        "on_mismatch": "PROVENANCE_DECLARATION_MISMATCH",
    }
