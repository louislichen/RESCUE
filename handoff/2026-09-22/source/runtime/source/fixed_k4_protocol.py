from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


class DraftModel(Protocol):
    def prefill(self, prompt_ids: Sequence[int]) -> tuple[int, Any]: ...
    def step(self, token: int, cache: Any) -> tuple[int, Any]: ...
    def crop_cache(self, cache: Any, length: int) -> Any: ...


class TargetModel(Protocol):
    def prefill(self, prompt_ids: Sequence[int]) -> tuple[int, Any]: ...
    def verify(self, proposal_ids: Sequence[int], cache: Any) -> tuple[list[int], Any]: ...
    def step(self, token: int, cache: Any) -> tuple[int, Any]: ...
    def clone_cache(self, cache: Any) -> Any: ...
    def crop_cache(self, cache: Any, length: int) -> Any: ...


@dataclass
class ProtocolCounters:
    proposed_draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    correction_tokens: int = 0
    bonus_tokens: int = 0
    target_prefill_calls: int = 0
    target_verify_calls: int = 0
    target_step_calls: int = 0
    draft_prefill_calls: int = 0
    draft_step_calls: int = 0
    blocks: int = 0
    full_accept_blocks: int = 0
    reject_blocks: int = 0
    short_proposals: int = 0
    eos_terminated: bool = False
    rollback_events: list[dict[str, int]] = field(default_factory=list)


@dataclass
class FixedK4Result:
    token_ids: list[int]
    counters: ProtocolCounters
    k: int
    max_new_tokens: int


def _draft_propose(draft: DraftModel, prompt_ids: Sequence[int], cache: Any, next_id: int,
                   eos_id: int | None, limit: int, counters: ProtocolCounters) -> tuple[list[int], Any, bool]:
    proposals: list[int] = []
    draft_eos = False
    current_cache = cache
    for _ in range(limit):
        if eos_id is not None and next_id == eos_id:
            draft_eos = True
            break
        proposals.append(int(next_id))
        counters.proposed_draft_tokens += 1
        next_id, current_cache = draft.step(int(next_id), current_cache)
        counters.draft_step_calls += 1
    return proposals, current_cache, draft_eos


def fixed_k4_decode(prompt_ids: Sequence[int], draft: DraftModel, target: TargetModel,
                    eos_id: int | None, *, k: int = 4, max_new_tokens: int = 16) -> FixedK4Result:
    if k != 4:
        raise ValueError("this task freezes K=4")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    counters = ProtocolCounters()
    draft_next, draft_cache = draft.prefill(prompt_ids)
    target_next, target_cache = target.prefill(prompt_ids)
    counters.draft_prefill_calls += 1
    counters.target_prefill_calls += 1
    output: list[int] = []
    prompt_len = len(prompt_ids)

    while len(output) < max_new_tokens:
        counters.blocks += 1
        prefix_len = prompt_len + len(output)
        limit = min(k, max_new_tokens - len(output))
        proposals, draft_after, draft_eos = _draft_propose(
            draft, prompt_ids, draft_cache, int(draft_next), eos_id, limit, counters
        )
        if len(proposals) < limit:
            counters.short_proposals += 1
        if not proposals:
            correction = int(target_next)
            counters.correction_tokens += 1
            if eos_id is not None and correction == eos_id:
                counters.eos_terminated = True
                break
            output.append(correction)
            target_next, target_cache = target.step(correction, target_cache)
            draft_next, draft_cache = draft.step(correction, draft_cache)
            counters.target_step_calls += 1
            counters.draft_step_calls += 1
            continue

        target_predictions, target_after = target.verify(proposals, target_cache)
        counters.target_verify_calls += 1
        if len(target_predictions) != len(proposals) + 1:
            raise RuntimeError("target verify must return one prediction per proposal plus next-token prediction")
        mismatch = next((i for i, (draft_id, target_id) in enumerate(zip(proposals, target_predictions))
                         if int(draft_id) != int(target_id)), None)
        if mismatch is not None:
            counters.reject_blocks += 1
            counters.accepted_draft_tokens += mismatch
            output.extend(int(x) for x in proposals[:mismatch])
            correction = int(target_predictions[mismatch])
            counters.correction_tokens += 1
            counters.rollback_events.append({"prefix_len": prefix_len, "proposal_len": len(proposals), "accepted_before_reject": mismatch})
            if eos_id is not None and correction == eos_id:
                counters.eos_terminated = True
                break
            output.append(correction)
            target_valid = target.clone_cache(target_after)
            target_valid = target.crop_cache(target_valid, prefix_len + mismatch)
            target_next, target_cache = target.step(correction, target_valid)
            draft_valid = draft.crop_cache(draft_after, prefix_len + mismatch)
            draft_next, draft_cache = draft.step(correction, draft_valid)
            counters.target_step_calls += 1
            counters.draft_step_calls += 1
            continue

        counters.full_accept_blocks += 1
        counters.accepted_draft_tokens += len(proposals)
        output.extend(int(x) for x in proposals)
        if len(output) >= max_new_tokens:
            break
        bonus = int(target_predictions[len(proposals)])
        if eos_id is not None and bonus == eos_id:
            counters.eos_terminated = True
            break
        counters.bonus_tokens += 1
        output.append(bonus)
        target_next, target_cache = target.step(bonus, target_after)
        draft_next, draft_cache = draft.step(bonus, draft_after)
        counters.target_step_calls += 1
        counters.draft_step_calls += 1

    return FixedK4Result(output[:max_new_tokens], counters, k, max_new_tokens)


def greedy_reference(prompt_ids: Sequence[int], target: TargetModel, eos_id: int | None,
                     max_new_tokens: int = 16) -> list[int]:
    next_id, cache = target.prefill(prompt_ids)
    out: list[int] = []
    for _ in range(max_new_tokens):
        if eos_id is not None and int(next_id) == eos_id:
            break
        out.append(int(next_id))
        next_id, cache = target.step(int(next_id), cache)
    return out
