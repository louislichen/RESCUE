"""Request-local instrumentation for a NEW all-round online CSGB protocol.

This module does not import torch/numpy until a default runtime dependency is
used.  It changes neither the existing Harness nor a frozen experiment.  The
engine must support the opt-in terminal-commit callback: no unverified terminal
proposal is constructed.  Complete-request timing remains the runner's job.

Training round time includes feature/decision/Draft/verification/commit and a
closing GPU synchronization.  Learner updates are timed separately, and finish
BEFORE the next round begins.  Clean evaluation has no explicit round sync,
reward labels, or updates.  Its per-round wall times must not be inferred from
the decision timings retained here.  The compatibility key
``decision_cpu_seconds`` is feature/decision WALL time (including a feature
copy's device wait), not an isolated CPU or D2H stage measurement.
"""

import copy
import math
import time
import types

from csgb_cache_geometry import root_readable_positions


BUDGETS = (8, 16, 24, 32)
FRESH_POLICIES = frozenset(("csgb", "constant", "fixed8_fresh"))
NATIVE_FIXED = {f"fixed{b}_native": b for b in BUDGETS}
POLICIES = FRESH_POLICIES | frozenset(NATIVE_FIXED) | {"history_native"}


def _synchronize():
    import torch
    torch.cuda.synchronize()


def _attention_to_host(attention):
    return attention.detach().float().cpu().numpy().copy()


def _features(attention, readable, anchor):
    from csgb import features
    return features(attention, readable, anchor)


def _generated_tokens(ids, context):
    return ids[0, context:].tolist()


def _vector(value, *, size=None):
    if value is None:
        return None
    result = [float(v) for v in value]
    if (size is not None and len(result) != size) or not result or not all(math.isfinite(v) for v in result):
        raise ValueError("Invalid feature vector")
    return result


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {name}")
    return value


class OnlineHarness:
    """Bind one model once; reset its request-local state before each request.

    ``learner.choose(raw_x, explore=...)`` and
    ``learner.update(raw_x, budget, useful_n, seconds, rho)`` use raw features.
    Optional ``learner.transform(raw_x)`` is pure and only exposes the scaled
    learner input for the audit.  All injected dependencies are CPU-testable.
    Fixed8-fresh is a feature-matched calibration control, not a learner.
    """

    def __init__(self, model, *, clock=time.perf_counter, synchronize=None,
                 attention_to_host=None, feature_fn=None,
                 coverage_positions_fn=root_readable_positions,
                 generated_tokens_fn=None):
        if getattr(model, "_csgb_online_harness", None) is not None:
            raise ValueError("An online Harness is already attached")
        if getattr(model, "csgb_terminal_commit_callback", None) is not None:
            raise ValueError("Refusing to replace an existing terminal callback")
        self.model = model
        self.clock = clock
        self.synchronize = synchronize or _synchronize
        self.attention_to_host = attention_to_host or _attention_to_host
        self.feature_fn = feature_fn or _features
        self.coverage_positions_fn = coverage_positions_fn
        self.generated_tokens_fn = generated_tokens_fn or _generated_tokens
        self.original_draft = model.draft
        self.original_refresh = model.update_working_cache_retrieval
        self.request_index = -1
        self.pending = None
        self.active = False
        model.draft = types.MethodType(self.draft, model)
        model.update_working_cache_retrieval = types.MethodType(self.refresh, model)
        model.csgb_terminal_commit_callback = self.finalize_terminal
        model.csgb_skip_terminal_draft = True
        model.csgb_root_coverage = True
        model._csgb_online_harness = self

    def reset(self, policy, learner=None, explore=False, rho=0., clean=False, *,
              max_new_tokens=128, eos_token_id=None, context,
              calibration=False, provenance=None):
        if self.pending is not None:
            raise RuntimeError("Cannot discard an unfinished proposal")
        if policy not in POLICIES:
            raise ValueError("Unknown online Harness policy")
        if any(type(v) is not bool for v in (explore, clean, calibration)):
            raise ValueError("Mode flags must be bool")
        if clean and explore:
            raise ValueError("Clean evaluation cannot update a learner")
        if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho < 0:
            raise ValueError("rho must be finite nonnegative tokens/second")
        if policy in {"csgb", "constant", "history_native"} and learner is None:
            raise ValueError("A learned policy requires a learner")
        if explore and (learner is None or policy in NATIVE_FIXED or policy == "fixed8_fresh"):
            raise ValueError("Only a learned policy can explore/update")
        _integer(context, "prompt context", 1)
        _integer(max_new_tokens, "maximum output tokens", 1)
        if eos_token_id is not None:
            _integer(eos_token_id, "EOS token ID")
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("Provenance must be a dictionary")
        self.policy, self.learner = policy, learner
        self.explore, self.rho, self.clean = explore, float(rho), clean
        self.max_new_tokens, self.eos_token_id, self.context = max_new_tokens, eos_token_id, context
        self.calibration, self.provenance = calibration, copy.deepcopy(provenance or {})
        self.request_index += 1
        self.active, self.finished = True, False
        self.rows, self.round_index = [], 0
        self.anchor, self.anchor_timestep = None, None
        self.has_real_anchor = False
        self.latest_attention, self.latest_attention_timestep = None, None
        self.latest_attention_source = None
        self.last_accept, self.ema = 0., 0.
        self.model.csgb_attention_every_round = policy in FRESH_POLICIES
        self.model.csgb_skip_terminal_draft = True
        self.model.csgb_root_coverage = True

    def refresh(self, model, *args, **kwargs):
        """Anchor only an actual noninitial selection, AFTER Root used old KV.

        Native/history track only readiness/timestamp (no feature D2H).  Fresh
        paths reuse the immutable host snapshot already copied for this Root's
        decision, rather than perform another copy at the same refresh.
        """
        if not self.active or self.pending is None:
            raise RuntimeError("Cache refresh outside an active proposal")
        initial = getattr(model, "selected_chunks", None) is None
        requested = kwargs.get("do_retrieval", args[1] if len(args) > 1 else False)
        attention = getattr(model, "attn_scores_final", None)
        real = not initial and bool(requested) and attention is not None
        anchor, timestamp = None, None
        if real:
            timestamp = getattr(model, "csgb_attention_observed_timestep", None)
            if type(timestamp) is not int or timestamp != self.pending["proposal"] - 1:
                raise RuntimeError("Selection anchor has a stale attention timestamp")
            if self.policy in FRESH_POLICIES:
                if (self.latest_attention_source is not attention or
                        self.latest_attention_timestep != timestamp):
                    raise RuntimeError("Selection did not use the decision attention snapshot")
                anchor = self.latest_attention
        result = self.original_refresh(*args, **kwargs)
        if real:
            self.has_real_anchor = True
            self.anchor, self.anchor_timestep = anchor, timestamp
        return result

    def _close(self, committed_length, *, terminal=False, eos_index=None):
        """Close/update exactly one executed round, before any next-round start."""
        if self.pending is None:
            raise RuntimeError("Verified pending round is missing/already closed")
        row = self.pending
        raw_n = committed_length - (row["prefix"] - 1)
        generated_after = committed_length - self.context
        generated_before = row["prefix"] - 1 - self.context
        if raw_n < 1 or generated_before < 0:
            raise RuntimeError("A verified proposal must commit a positive prefix")
        useful_after = min(generated_after, self.max_new_tokens)
        if eos_index is not None:
            useful_after = min(useful_after, eos_index + 1)
        useful_before = min(generated_before, self.max_new_tokens)
        useful_n = useful_after - useful_before
        if useful_n < 1:
            raise RuntimeError("No useful tokens: an unused terminal proposal was executed")
        if not terminal and generated_after >= self.max_new_tokens:
            raise RuntimeError("Engine attempted a proposal after the output limit")
        if not self.clean:
            self.synchronize()
        end = self.clock()
        elapsed = end - row.pop("start")
        if not self.clean and (not math.isfinite(elapsed) or elapsed <= 0):
            raise RuntimeError("Round time must be finite positive seconds")
        row.update(committed=raw_n, raw_committed=raw_n, useful_committed=useful_n,
                   committed_prefix_after=committed_length,
                   useful_generated_before=useful_before, useful_generated_after=useful_after,
                   seconds=None if self.clean else elapsed,
                   reward=None if self.clean else useful_n - self.rho * elapsed,
                   terminal_commit=terminal, updated=False, update_cpu_seconds=0.)
        # Clear before invoking the learner: duplicate finalization cannot update.
        self.pending = None
        self.rows.append(row)
        self.last_accept = max(0, useful_n - 1)
        self.ema = .75 * self.ema + .25 * self.last_accept
        if self.explore and row["raw_x"] is not None:
            update_start = self.clock()
            self.learner.update(list(row["raw_x"]), row["budget"], useful_n, elapsed, self.rho)
            row["update_cpu_seconds"] = self.clock() - update_start
            if not math.isfinite(row["update_cpu_seconds"]) or row["update_cpu_seconds"] < 0:
                raise RuntimeError("Invalid learner-update wall time")
            row["updated"] = True

    def finalize_terminal(self, committed_ids):
        """The engine supplies committed-only IDs; never fabricate a new Root."""
        if not self.active or self.finished or self.pending is None:
            raise RuntimeError("Terminal callback is missing a live proposal or repeated")
        length = int(committed_ids.shape[-1])
        eos_index = None
        if self.eos_token_id is not None:
            tokens = list(self.generated_tokens_fn(committed_ids, self.context))
            if len(tokens) != length - self.context:
                raise RuntimeError("Terminal token view has an inconsistent length")
            if self.eos_token_id in tokens:
                eos_index = tokens.index(self.eos_token_id)
                if eos_index != len(tokens) - 1:
                    raise RuntimeError("Committed tokens after first EOS: terminal engine is not corrected")
        if length - self.context < self.max_new_tokens and eos_index is None:
            raise RuntimeError("Terminal callback without EOS or output limit")
        self._close(length, terminal=True, eos_index=eos_index)
        self.finished = True

    def _raw_features(self, model, root_length):
        j = self.round_index
        if self.policy not in FRESH_POLICIES or j == 0:
            return None, None
        attention = getattr(model, "attn_scores_final", None)
        timestamp = getattr(model, "csgb_attention_observed_timestep", None)
        if attention is None or type(timestamp) is not int or timestamp != j - 1:
            raise RuntimeError("Fresh controller attention must come from preceding verification")
        self.latest_attention = self.attention_to_host(attention)
        self.latest_attention_source, self.latest_attention_timestep = attention, timestamp
        if len(self.latest_attention) != root_length - 1:
            raise RuntimeError("Fresh attention must cover exactly the committed prefix")
        selected = getattr(model, "selected_chunks", None)
        if selected is None:
            raise RuntimeError("A noninitial Root is missing its stable-cache selection")
        # Crucially includes the not-yet-processed committed suffix, not only
        # selected chunks, and excludes the next uncommitted Root.
        readable = self.coverage_positions_fn(selected, int(model.total_seq_len),
                                              len(self.latest_attention), root_length)
        x = _vector(self.feature_fn(self.latest_attention, readable, self.anchor), size=4)
        if x is not None:
            if (not self.has_real_anchor or self.anchor is None or x[0] != 1. or
                    any(not 0 <= v <= 1 for v in x[1:]) or
                    not math.isclose(x[3], x[1]*x[2], rel_tol=0., abs_tol=1e-12)):
                raise RuntimeError("Invalid Coverage/Shift or fabricated selection anchor")
        return x, timestamp

    def draft(self, model, input_ids, nodes, threshold, max_depth):
        if not self.active or self.finished:
            raise RuntimeError("Draft called outside an unfinished request")
        root_length = int(input_ids.shape[-1])
        if root_length < self.context + 1:
            raise RuntimeError("Root input is shorter than prompt plus uncommitted Root")
        had_pending = self.pending is not None
        if had_pending:
            self._close(root_length - 1)
        elif self.round_index != 0:
            raise RuntimeError("A proposal disappeared without verification")
        elif root_length != self.context + 1:
            raise RuntimeError("Initial Root contains unaccounted generated tokens")
        if not had_pending and not self.clean:
            self.synchronize()  # Separate prefill from the first measured round.
        # All previous-round learning is now complete, and is not charged here.
        start = self.clock()
        raw_features, timestamp = self._raw_features(model, root_length)
        raw_x = raw_features
        if self.policy == "history_native":
            raw_x = ([1., self.last_accept / 10., self.ema / 10.]
                     if self.has_real_anchor else None)
        if self.policy in NATIVE_FIXED:
            budget = NATIVE_FIXED[self.policy]
        elif self.policy == "fixed8_fresh" or raw_x is None:
            budget = 8
        else:
            budget = self.learner.choose(list(raw_x), explore=self.explore)
        if type(budget) is not int or budget not in BUDGETS:
            raise RuntimeError("Learner selected an unsupported budget")
        learner_input = None
        if raw_x is not None:
            transform = getattr(self.learner, "transform", None)
            learner_input = _vector(transform(list(raw_x)) if callable(transform) else raw_x)
        decision_seconds = self.clock() - start
        if not math.isfinite(decision_seconds) or decision_seconds < 0:
            raise RuntimeError("Invalid feature/decision wall time")
        self.pending = {
            "request_index": self.request_index, "provenance": copy.deepcopy(self.provenance),
            "policy": self.policy, "calibration": self.calibration,
            "proposal": self.round_index, "prefix": root_length, "input_tokens": root_length,
            "context": self.context, "committed_prefix_before": root_length - 1,
            "root_processed_prefix_before": int(getattr(model, "total_seq_len", 0)),
            "budget": budget, "raw_features": copy.deepcopy(raw_features),
            "raw_x": copy.deepcopy(raw_x), "x": copy.deepcopy(raw_x),
            "learner_input": learner_input,
            "coverage": None if raw_features is None else raw_features[1],
            "shift": None if raw_features is None else raw_features[2],
            "attention_observed_timestep": timestamp,
            "anchor_timestep": self.anchor_timestep, "has_real_anchor": self.has_real_anchor,
            "eligible_for_calibration": self.has_real_anchor,
            "decision_cpu_seconds": decision_seconds,
            "decision_wall_seconds": decision_seconds, "start": start,
        }
        self.round_index += 1
        result = self.original_draft(input_ids, budget, threshold, max_depth)
        actual_nodes = int(result[0].shape[-1])
        if actual_nodes != budget:
            raise RuntimeError("Executed tree size differs from the selected arm")
        self.pending["actual_draft_nodes"] = actual_nodes
        return result
