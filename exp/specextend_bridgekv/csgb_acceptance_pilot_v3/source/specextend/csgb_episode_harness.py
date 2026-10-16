"""Request-only softmax adapter; no GPU imports, fitting, baseline or split logic.

Both train and deployment reuse clean OnlineHarness: no explicit round sync or
round update. A future runner supplies canonical AR and boundary-timed whole
request seconds, and may update a NEW immutable policy only after finish().
"""
import copy
import hashlib
import json
import math
import random

from csgb_episode_policy import BUDGETS, Decision, EpisodePolicy
from csgb_online_harness import FRESH_POLICIES, NATIVE_FIXED, POLICIES, OnlineHarness


def _finite(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("Finite complete-request measurement required")
    return value


def parameter_sha(policy):
    return hashlib.sha256(json.dumps(policy.to_state(), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def rng_state(rng):
    version, state, spare = rng.getstate()
    return {"version": version, "state": list(state), "gauss_next": spare}


def _restore_rng(state):
    if (not isinstance(state, dict) or set(state) != {"version", "state", "gauss_next"}
            or type(state["version"]) is not int or state["version"] != 3
            or not isinstance(state["state"], list) or len(state["state"]) != 625
            or any(type(v) is not int or not 0 <= v < 2**32 for v in state["state"][:-1])
            or type(state["state"][-1]) is not int or not 0 <= state["state"][-1] <= 624):
        raise ValueError("Invalid serialized request-local Random state")
    if state["gauss_next"] is not None:
        _finite(state["gauss_next"])
    rng = random.Random()
    rng.setstate((3, tuple(state["state"]), state["gauss_next"]))
    return rng


def audit_decisions(policy, decisions, rows, before, after):
    """Replay real sampling, including no-RNG fallback; parameters never update."""
    if (not isinstance(policy, EpisodePolicy) or parameter_sha(policy) != policy.parameter_sha256
            or len(decisions) != len(rows)):
        raise ValueError("Changed parameters or unverified/missing episode decision")
    rng, states = _restore_rng(before), []
    _restore_rng(after)
    for index, (value, row) in enumerate(zip(decisions, rows)):
        decision = Decision.from_state(value) if isinstance(value, dict) else value
        policy.replay(decision)
        expected = policy.sample(row["raw_x"], rng)
        x = None if row["learner_input"] is None else tuple(row["learner_input"])
        if (decision != expected or row["budget"] != decision.budget
                or x != decision.transformed_x or type(row.get("episode_decision_index")) is not int
                or row["episode_decision_index"] != index or row.get("policy") != policy.mode
                or policy.mode != "history_native" and row["raw_x"] != row["raw_features"]):
            raise ValueError("Decision/row/raw features/transform/action/RNG misalignment")
        states.append(decision.to_state())
    if rng_state(rng) != after:
        raise ValueError("Request RNG consumed unrecorded draws or changed outside sampling")
    return states


def validate_feature_chronology(rows, mode):
    """Audit the frozen refresh4/Root-suffix feature observation contract.

    This checks recorded chronology, not attention values reconstructed from
    full attention tensors. The caller must separately bind the collection
    source/geometry SHA and fixed refresh cadence to its experiment protocol.
    Native fixed arms keep selection readiness but never extract C/D;
    fixed8_fresh is the feature-matched calibration control, not a Native arm.
    """
    if not isinstance(mode, str) or mode not in POLICIES or not isinstance(rows, list) or not rows:
        raise ValueError("Known policy and nonempty verified rows required")
    required = {"proposal", "policy", "context", "prefix", "input_tokens",
                "root_processed_prefix_before", "committed_prefix_before",
                "committed_prefix_after", "useful_committed", "raw_x", "x",
                "raw_features", "coverage", "shift", "learner_input",
                "attention_observed_timestep", "anchor_timestep",
                "has_real_anchor", "eligible_for_calibration"}
    context = rows[0].get("context") if isinstance(rows[0], dict) else None
    if type(context) is not int or context < 1:
        raise ValueError("Actual prompt context required")
    previous_prefix, next_prefix, previous_accept, ema = 0, context+1, 0., 0.
    feature_count = 0
    for j, row in enumerate(rows):
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError("Missing feature chronology fields")
        integers = {"proposal": j, "context": context, "prefix": next_prefix,
                    "input_tokens": next_prefix,
                    "root_processed_prefix_before": previous_prefix,
                    "committed_prefix_before": next_prefix-1}
        ready, anchor = j >= 2, None if j < 2 else 4*((j-2)//4)
        if (any(type(row[k]) is not int or row[k] != v for k,v in integers.items())
                or row["policy"] != mode
                or type(row["committed_prefix_after"]) is not int
                or row["committed_prefix_after"] < next_prefix
                or type(row["useful_committed"]) is not int or row["useful_committed"] < 1
                or row["has_real_anchor"] is not ready
                or row["eligible_for_calibration"] is not ready
                or (anchor is None and row["anchor_timestep"] is not None)
                or (anchor is not None and (type(row["anchor_timestep"]) is not int
                                           or row["anchor_timestep"] != anchor))
                or row["raw_x"] != row["x"]):
            raise ValueError("Root prefix, actual-anchor readiness or feature chronology differs")
        raw = row["raw_x"]
        if mode in FRESH_POLICIES:
            timestamp = None if j == 0 else j-1
            if (row["raw_features"] != raw or (raw is not None) is not ready
                    or (timestamp is None and row["attention_observed_timestep"] is not None)
                    or (timestamp is not None and
                        (type(row["attention_observed_timestep"]) is not int
                         or row["attention_observed_timestep"] != timestamp))):
                raise ValueError("Fresh attention must precede this exact proposal")
            if raw is not None:
                if (not isinstance(raw, (list, tuple)) or len(raw) != 4
                        or any(not 0 <= _finite(v) <= 1 for v in raw) or raw[0] != 1.
                        or not math.isclose(raw[3], raw[1]*raw[2], rel_tol=0., abs_tol=1e-12)
                        or _finite(row["coverage"]) != raw[1] or _finite(row["shift"]) != raw[2]):
                    raise ValueError("Invalid raw Coverage/Shift/product or recorded C/D differs")
                feature_count += 1
                transformed = row["learner_input"]
                width = 1 if mode == "constant" else 4
                if (not isinstance(transformed, (list, tuple)) or len(transformed) != width
                        or any(not math.isfinite(_finite(v)) for v in transformed) or transformed[0] != 1.
                        or (mode == "fixed8_fresh" and transformed != raw)):
                    raise ValueError("Missing/invalid feature-matched learner input")
            elif any(row[k] is not None for k in ("coverage", "shift", "learner_input")):
                raise ValueError("Unanchored fresh proposal must not fabricate features")
        else:
            if any(row[k] is not None for k in ("raw_features", "coverage", "shift", "attention_observed_timestep")):
                raise ValueError("Native path must not extract fresh attention features")
            expected = [1., previous_accept/10., ema/10.] if mode == "history_native" and ready else None
            if raw != expected:
                raise ValueError("History/startup features use future or incorrect useful acceptance")
            if raw is not None:
                if (any(not 0 <= _finite(v) <= 1 for v in raw)
                        or row["learner_input"] != raw):
                    raise ValueError("Native-history input must use its fixed identity scale")
            elif row["learner_input"] is not None:
                raise ValueError("Native fixed/startup must not hide a learner input")
        previous_accept = float(max(0, row["useful_committed"]-1))
        ema = .75*ema+.25*previous_accept
        previous_prefix, next_prefix = row["prefix"], row["committed_prefix_after"]+1
    return {"verified_rounds": len(rows), "fresh_feature_rounds": feature_count,
            "full_attention_reconstructed": False}


def validate_request_accounting(result, reference, rows, *, context, seconds,
                                max_new_tokens=128, eos_token_id=None, mode=None):
    """Dataset/phase-free canonical Token and real verified-round arithmetic."""
    _finite(seconds, positive=True)
    if type(context) is not int or context < 1 or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("Invalid prompt length or output limit")
    if eos_token_id is not None and (type(eos_token_id) is not int or eos_token_id < 0):
        raise ValueError("Invalid EOS Token")
    reference = reference.get("output_token_ids") if isinstance(reference, dict) else reference
    output, scored, raw = (result.get(k) for k in ("output_token_ids", "accept_length_list", "raw_accept_length_list"))
    if (not isinstance(output, list) or not 0 < len(output) <= max_new_tokens
            or any(type(v) is not int or v < 0 for v in output) or output != reference
            or not isinstance(rows, list) or not rows or not isinstance(scored, list) or not isinstance(raw, list)
            or len(scored) != len(raw) or len(raw) != len(rows)
            or any(type(a) is not int or a < 0 for a in scored+raw)
            or any(type(result.get(k)) is not int for k in ("total_generated", "raw_total_generated", "generation_overshoot"))
            or result["total_generated"] != len(output) or sum(a+1 for a in scored) != len(output)
            or sum(a+1 for a in raw) != result["raw_total_generated"]
            or result["generation_overshoot"] != result["raw_total_generated"]-len(output)):
        raise ValueError("Canonical output or raw/scored Token accounting failed")
    has_eos = eos_token_id is not None and eos_token_id in output
    if ((has_eos and output.index(eos_token_id) != len(output)-1)
            or (len(output) < max_new_tokens and not has_eos)
            or (has_eos and result["raw_total_generated"] != len(output))):
        raise ValueError("EOS must stop inclusively; only output-limit clipping may overshoot")
    generated, stop = 0, len(output)
    for j, (a, s, row) in enumerate(zip(raw, scored, rows)):
        useful = min(generated+a+1, stop)-min(generated, stop)
        expected = {"proposal": j, "prefix": context+1+generated, "committed": a+1,
                    "raw_committed": a+1, "useful_committed": useful,
                    "committed_prefix_before": context+generated, "committed_prefix_after": context+generated+a+1,
                    "useful_generated_before": min(generated,stop), "useful_generated_after": min(generated+a+1,stop)}
        if (any(type(row.get(k)) is not int or row[k] != v for k,v in expected.items())
                or type(row.get("budget")) is not int or row["budget"] not in BUDGETS
                or type(row.get("actual_draft_nodes")) is not int or row["actual_draft_nodes"] != row["budget"]
                or a > min(10,row["budget"]) or useful < 1 or useful != s+1
                or (j < len(rows)-1 and generated+a+1 >= stop)
                or row.get("terminal_commit") is not (j == len(rows)-1)
                or row.get("seconds") is not None or row.get("reward") is not None or row.get("updated") is not False
                or _finite(row.get("update_cpu_seconds")) != 0
                or (mode is not None and row.get("policy") != mode)
                or (mode in NATIVE_FIXED and row["budget"] != NATIVE_FIXED[mode])
                or (mode == "fixed8_fresh" and row["budget"] != 8)):
            raise ValueError("Verified round/action/useful commitment or no-round-update contract failed")
        if _finite(row.get("decision_wall_seconds")) < 0 or row.get("decision_cpu_seconds") != row["decision_wall_seconds"]:
            raise ValueError("Invalid feature/decision WALL timing")
        generated += a+1
    if math.fsum(row["decision_wall_seconds"] for row in rows) > seconds+1e-9:
        raise ValueError("Feature/decision work exceeds the supplied complete-request time")
    return {"exact": True, "useful_n": len(output), "seconds": seconds,
            "request_tps": len(output)/seconds, "scored_acceptance": math.fsum(scored)/len(scored)}


class _Bridge:
    def __init__(self, policy, rng):
        self.policy, self.rng, self.decisions = policy, rng, []

    def choose(self, raw_x, explore=False):
        if explore is not False:
            raise ValueError("Episode sampling never uses the old explore switch")
        decision = self.policy.sample(raw_x,self.rng)
        self.decisions.append(decision)
        return decision.budget

    def transform(self, raw_x):
        return self.policy.transform(raw_x)


class EpisodeHarness(OnlineHarness):
    def __init__(self, model, **dependencies):
        self._episode_finalized = True
        super().__init__(model, **dependencies)

    def reset(self, mode, episode_policy=None, rng=None, *, phase, context,
              max_new_tokens=128, eos_token_id=None, provenance=None):
        if not self._episode_finalized:
            raise RuntimeError("Finish the previous real request; never discard a partial episode")
        if (not isinstance(phase,str) or not phase.strip() or not isinstance(mode,str)
                or provenance is not None and (not isinstance(provenance,dict) or provenance.get("phase",phase) != phase)):
            raise ValueError("Explicit phase must be preserved, never relabeled by clean evaluation")
        learned = mode in {"csgb","constant","history_native"}
        if (learned and (not isinstance(episode_policy,EpisodePolicy) or episode_policy.mode != mode or not isinstance(rng,random.Random))
                or not learned and (episode_policy is not None or rng is not None)
                or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128):
            raise ValueError("Mode/policy/request-local RNG/output-limit mismatch")
        self.episode_policy, self.episode_rng, self.episode_phase = episode_policy,rng,phase
        self.bridge = _Bridge(episode_policy,rng) if learned else None
        self._parameters_before = None if not learned else copy.deepcopy(episode_policy.to_state())
        self._rng_before = None if not learned else rng_state(rng)
        super().reset(mode,self.bridge,explore=False,clean=True,context=context,max_new_tokens=max_new_tokens,
                      eos_token_id=eos_token_id,provenance={**(provenance or {}),"phase":phase})
        self._episode_finalized = False

    def draft(self, model, *args, **kwargs):
        before = 0 if self.bridge is None else len(self.bridge.decisions)
        result = super().draft(model,*args,**kwargs)
        if self.bridge is not None:
            if len(self.bridge.decisions) == before and self.pending["raw_x"] is None:
                self.bridge.decisions.append(self.episode_policy.sample(None,self.episode_rng))
            if len(self.bridge.decisions) != before+1:
                raise RuntimeError("Exactly one recorded Decision per real proposal required")
            self.pending["episode_decision_index"] = before
        return result

    def finish(self, result, reference, elapsed_s):
        if self._episode_finalized or not self.finished or self.pending is not None:
            raise RuntimeError("Only one genuinely terminal, completely verified episode may finish")
        checked = validate_request_accounting(result,reference,self.rows,context=self.context,seconds=elapsed_s,
            max_new_tokens=self.max_new_tokens,eos_token_id=self.eos_token_id,mode=self.policy)
        validate_feature_chronology(self.rows,self.policy)
        decisions, after = [], None
        if self.bridge is not None:
            if self.episode_policy.to_state() != self._parameters_before or self.bridge.policy is not self.episode_policy or self.bridge.rng is not self.episode_rng:
                raise ValueError("Episode parameters or RNG identity changed inside request")
            after = rng_state(self.episode_rng)
            decisions = audit_decisions(self.episode_policy,self.bridge.decisions,self.rows,self._rng_before,after)
        if any(row["provenance"].get("phase") != self.episode_phase for row in self.rows):
            raise ValueError("Round provenance phase was relabeled")
        self._episode_finalized = True
        return {"phase":self.episode_phase,"policy":self.policy,**checked,"rows":copy.deepcopy(self.rows),
                "result":copy.deepcopy(result),
                "reference_output_token_ids":copy.deepcopy(reference.get("output_token_ids") if isinstance(reference,dict) else reference),
                "decisions":decisions,"rng_before":copy.deepcopy(self._rng_before),"rng_after":after,
                "parameter_sha256":None if self.episode_policy is None else self.episode_policy.parameter_sha256,
                "parameter_version":None if self.episode_policy is None else self.episode_policy.parameter_version,
                "parameters_unchanged":True,"explicit_round_sync":False,"round_updates":False,
                "whole_request_timing_supplied_by_caller":True,"baseline_or_sgd_performed":False}
