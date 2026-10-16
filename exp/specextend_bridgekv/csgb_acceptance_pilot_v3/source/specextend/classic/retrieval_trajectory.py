"""Lossless-enough traces for all-round, retrieval-aware Mapper training.

The original Bridge pair collector samples aligned Target/Draft cache rows.  It
does not retain the speculative round in which a row was used, the working
cache that produced the Draft root distribution, or the verifier decision.
Those omissions make it impossible to train against later-round acceptance.

``RetrievalTrajectoryCollector`` records that missing causal structure.  Large
cache tensors are stored once per request in shared banks; individual rounds
refer to absolute rows in those banks.  Full Target and Draft *root* logits are
kept because they are inexpensive and support the exact soft-overlap loss.
For the remaining tree nodes only Target top-k logits are retained.

Collection deliberately copies tensors to CPU while generation is running.
It is a training-data job, not a latency benchmark; payload metadata states
that its wall-clock measurements must not be used as TPS results.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


SCHEMA_NAME = "specextend_retrieval_trajectory"
SCHEMA_VERSION = 1


def _cpu_tensor(
    value: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected Tensor, got {type(value).__name__}")
    result = value.detach()
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result.to(device="cpu").contiguous()


def _flatten_cache_rows(value: torch.Tensor) -> torch.Tensor:
    """Convert ``[1, heads, rows, head_dim]`` cache data to ``[rows, dim]``."""

    if value.ndim != 4 or int(value.shape[0]) != 1:
        raise ValueError(f"Expected [1,H,L,D] cache tensor, got {tuple(value.shape)}")
    return value.permute(0, 2, 1, 3).reshape(int(value.shape[2]), -1)


def _strict_selected_indices(
    chunks: Optional[Iterable[Sequence[int]]],
) -> torch.Tensor:
    positions: List[int] = []
    for chunk in chunks or ():
        if len(chunk) != 3:
            raise ValueError(f"Invalid selected chunk: {chunk!r}")
        _, start, stop = map(int, chunk)
        if start < 0 or stop <= start:
            raise ValueError(f"Invalid selected chunk range [{start}, {stop})")
        positions.extend(range(start, stop))
    if not positions:
        return torch.empty(0, dtype=torch.long)
    indices = torch.tensor(sorted(set(positions)), dtype=torch.long)
    if indices.numel() != len(positions):
        raise RuntimeError("Selected chunks overlap; cache-row identity is ambiguous")
    return indices


def _past_position_ids(model) -> Optional[torch.Tensor]:
    draft_core = getattr(getattr(model, "draft_model", None), "model", None)
    value = getattr(draft_core, "past_key_position_ids", None)
    if value is None:
        return None
    return _cpu_tensor(value, dtype=torch.long)


def _logical_cache_length(cache) -> int:
    try:
        shape = tuple(int(value) for value in cache.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("Cache object has no valid logical shape") from error
    if len(shape) != 4:
        raise RuntimeError(f"Expected logical rank-4 cache, got {shape}")
    return shape[2]


def _working_cache_length(model) -> int:
    cache = getattr(model, "draft_stable_kv", None)
    if cache is None:
        return 0
    if not isinstance(cache, (list, tuple)) or not cache:
        raise RuntimeError("Draft working cache is empty or malformed")
    return int(cache[0][0].shape[2])


def _sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RetrievalTrajectoryCollector:
    """Record one request at a time with strict round/cache causality.

    The engine invokes methods in this order::

        begin_request
        begin_proposal -> finish_proposal -> record_target_tree
                       -> record_verification
        ...
        begin_proposal -> finish_proposal  # eager, unverified trailing draft
        finalize_request

    A proposal has two working-cache identities.  ``root_cache`` is present at
    entry and is consumed by the root forward.  ``tree_cache`` is materialized
    only after the new root-input rows have been appended; it is consumed by
    tree expansion and becomes the next proposal's root cache.  Keeping these
    separate prevents an off-by-one causal claim about attention guidance.
    """

    def __init__(self, target_tree_topk: int = 32):
        if int(target_tree_topk) <= 0:
            raise ValueError("target_tree_topk must be positive")
        self.target_tree_topk = int(target_tree_topk)
        self.sample_metadata: Dict[str, Any] = {}
        self._request_active = False
        self._pending: Optional[Dict[str, Any]] = None
        self._rounds: List[Dict[str, Any]] = []
        self._cache_refreshes: List[Dict[str, Any]] = []
        self._active_cache_refresh_id: Optional[int] = None
        self._selection_lineage_source_round: Optional[int] = None
        self._draft_token_bank = torch.empty(0, dtype=torch.long)
        self._request: Dict[str, Any] = {}
        self._payload: Optional[Dict[str, Any]] = None

    def set_sample_metadata(self, **metadata: Any) -> None:
        if self._request_active:
            raise RuntimeError("Cannot change sample metadata during a request")
        self.sample_metadata = dict(metadata)

    def begin_request(
        self,
        model,
        input_ids: torch.Tensor,
        generation_config: Dict[str, Any],
    ) -> None:
        if self._request_active:
            raise RuntimeError("Previous trajectory request was not finalized")
        if not bool(generation_config.get("use_specextend")):
            raise ValueError("Retrieval trajectories require use_specextend=True")
        if str(generation_config.get("retrieval_policy")) != "attention":
            raise ValueError(
                "Formal Mapper trajectories require retrieval_policy='attention'"
            )
        if float(generation_config.get("temperature", 0.0)) != 0.0:
            raise ValueError(
                "Formal retrieval trajectories require deterministic temperature=0"
            )
        if str(generation_config.get("dynamic_cache_controller", "none")) != "none":
            raise ValueError(
                "Mapper trajectories require a fixed cache budget; dynamic cache "
                "control is a separate ablation"
            )
        if generation_config.get("return_output_ids") is not True:
            raise ValueError(
                "Formal retrieval trajectories require return_output_ids=True"
            )
        if getattr(model, "attn_bridge_mapper", None) is not None:
            raise RuntimeError(
                "Formal attention_native trajectories require no active Mapper"
            )
        if getattr(model, "bridge_collector", None) is not None:
            raise RuntimeError(
                "Pair collection and retrieval trajectory collection are exclusive"
            )
        if getattr(model, "attn_bridge_config", None):
            raise RuntimeError(
                "Formal attention_native trajectories require an empty bridge config"
            )
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("Trajectory collection requires batch size one")
        self._request_active = True
        self._pending = None
        self._rounds = []
        self._cache_refreshes = []
        self._active_cache_refresh_id = None
        self._selection_lineage_source_round = None
        self._draft_token_bank = torch.empty(0, dtype=torch.long)
        self._payload = None
        self._request = {
            "input_token_ids": _cpu_tensor(input_ids, dtype=torch.long).squeeze(0),
            "input_tokens": int(input_ids.shape[1]),
            "generation_config": dict(generation_config),
            "sample_metadata": dict(self.sample_metadata),
        }

    def begin_proposal(self, model, input_ids: torch.Tensor) -> Dict[str, Any]:
        if not self._request_active:
            raise RuntimeError("begin_request must precede begin_proposal")
        if self._pending is not None:
            raise RuntimeError("Previous proposal has not been verified")
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("Proposal input must have batch size one")

        proposal_round = len(self._rounds)
        full_length_before = int(getattr(model, "total_seq_len", 0))
        root_indices = _strict_selected_indices(
            getattr(model, "selected_chunks", None)
        )
        working_length = _working_cache_length(model)
        if root_indices.numel() != working_length:
            if not (working_length == 0 and root_indices.numel() == 0):
                raise RuntimeError(
                    "Root selected indices disagree with the working cache: "
                    f"indices={root_indices.numel()}, cache={working_length}"
                )
        if root_indices.numel() and int(root_indices[-1]) >= full_length_before:
            raise RuntimeError(
                "Root working cache references a future Draft-bank row"
            )
        if self._active_cache_refresh_id is not None:
            previous = self._cache_refreshes[self._active_cache_refresh_id]
            if not torch.equal(root_indices, previous["selected_indices"]):
                raise RuntimeError(
                    "Next root cache indices differ from the preceding tree cache"
                )
        if full_length_before < 0 or full_length_before > int(input_ids.shape[1]):
            raise RuntimeError(
                "Draft full-cache length is incompatible with proposal input: "
                f"cache={full_length_before}, input={input_ids.shape[1]}"
            )
        root_input = input_ids[:, full_length_before:]
        if root_input.numel() == 0:
            raise RuntimeError("Every proposal must append at least one Draft root row")
        if int(self._draft_token_bank.numel()) != full_length_before:
            raise RuntimeError(
                "Draft token bank/full-cache length mismatch before proposal: "
                f"tokens={self._draft_token_bank.numel()}, cache={full_length_before}"
            )
        self._draft_token_bank = torch.cat(
            [
                self._draft_token_bank,
                _cpu_tensor(root_input, dtype=torch.long).reshape(-1),
            ],
            dim=0,
        )
        target_cache = getattr(model, "past_key_values", None)
        if not isinstance(target_cache, (list, tuple)) or not target_cache:
            raise RuntimeError("Target cache is unavailable before Draft proposal")
        target_source_limit = _logical_cache_length(target_cache[-1][1])
        if target_source_limit != int(input_ids.shape[1]) - 1:
            raise RuntimeError(
                "Target source limit/proposal prefix mismatch: "
                f"Target={target_source_limit}, prefix={int(input_ids.shape[1]) - 1}"
            )

        pending = {
            "round": proposal_round,
            "prefix_tokens_before_root": int(input_ids.shape[1]) - 1,
            "target_source_limit": target_source_limit,
            "full_draft_cache_tokens_before": full_length_before,
            "root_input_token_ids": _cpu_tensor(root_input, dtype=torch.long).squeeze(0),
            "root_input_draft_position_ids": torch.arange(
                working_length,
                working_length + int(root_input.shape[1]),
                dtype=torch.long,
            ),
            "root_cache_indices": root_indices,
            "root_cache_tokens": working_length,
            # The root path passes position_ids=None, so Draft attention ignores
            # its stale tree state and rebuilds compressed positions 0..W-1.
            "root_cache_position_ids": torch.arange(
                working_length, dtype=torch.long
            ).unsqueeze(0),
            "root_cache_refresh_id": self._active_cache_refresh_id,
        }
        self._pending = pending
        return {
            "round": proposal_round,
            "full_draft_cache_tokens_before": full_length_before,
        }

    def finish_proposal(
        self,
        model,
        *,
        tree_token_ids: torch.Tensor,
        tree_position_ids: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        tree_parent: torch.Tensor,
        tree_cache_position_ids: Optional[torch.Tensor],
        draft_tree_position_ids: torch.Tensor,
        retrieval_requested: bool,
        selection_attention: Optional[torch.Tensor],
        selection_chunk_means: Optional[torch.Tensor],
        selection_ranked_chunk_ids: Optional[torch.Tensor],
        mandatory_frontier_chunk_id: Optional[int],
        effective_chunk_budget: Optional[int],
        attention_query_node_ids: Optional[torch.Tensor],
    ) -> None:
        pending = self._require_pending("finish_proposal")
        if tree_token_ids.ndim != 2 or int(tree_token_ids.shape[0]) != 1:
            raise ValueError("tree_token_ids must have shape [1, nodes]")
        node_count = int(tree_token_ids.shape[1])
        if tree_position_ids.ndim != 1 or int(tree_position_ids.numel()) != node_count:
            raise ValueError("tree_position_ids must align with tree_token_ids")
        if tuple(tree_attention_mask.shape) != (node_count, node_count):
            raise ValueError("tree_attention_mask must be square over tree nodes")

        root_token = pending["root_input_token_ids"][-1:].reshape(1, 1)
        full_tree_tokens = torch.cat(
            [root_token, _cpu_tensor(tree_token_ids, dtype=torch.long)], dim=1
        ).squeeze(0)
        positions = _cpu_tensor(tree_position_ids, dtype=torch.long)
        draft_positions = _cpu_tensor(draft_tree_position_ids, dtype=torch.long)
        if draft_positions.shape != positions.shape:
            raise ValueError(
                "Draft-compressed and Target-absolute proposal positions disagree"
            )
        full_positions = torch.cat([positions[:1] - 1, positions], dim=0)
        full_draft_positions = torch.cat(
            [draft_positions[:1] - 1, draft_positions], dim=0
        )
        mask = _cpu_tensor(tree_attention_mask)
        full_mask = torch.cat(
            [torch.zeros(1, node_count, dtype=mask.dtype), mask], dim=0
        )
        full_mask = torch.cat(
            [torch.ones(node_count + 1, 1, dtype=mask.dtype), full_mask], dim=1
        )

        tree_indices = _strict_selected_indices(
            getattr(model, "selected_chunks", None)
        )
        working_length = _working_cache_length(model)
        if tree_indices.numel() != working_length:
            raise RuntimeError(
                "Tree selected indices disagree with the working cache: "
                f"indices={tree_indices.numel()}, cache={working_length}"
            )
        total_seq_len = int(getattr(model, "total_seq_len", 0))
        if total_seq_len != int(self._draft_token_bank.numel()):
            raise RuntimeError(
                "Proposal append did not match the Draft token bank: "
                f"cache={total_seq_len}, tokens={self._draft_token_bank.numel()}"
            )
        if total_seq_len != int(pending["target_source_limit"]) + 1:
            raise RuntimeError(
                "Draft full cache must contain exactly one unverified root beyond "
                f"Target support: Draft={total_seq_len}, "
                f"Target={pending['target_source_limit']}"
            )
        if tree_indices.numel() and int(tree_indices[-1]) >= total_seq_len:
            raise RuntimeError("Tree cache selects a row beyond the Draft full cache")

        selection_source_round = (
            int(getattr(model, "timestep", -1)) if retrieval_requested else None
        )
        if retrieval_requested and selection_source_round != int(pending["round"]) - 1:
            raise RuntimeError(
                "Attention/cache causality changed: a retrieval used by proposal "
                f"{pending['round']} reports source round {selection_source_round}"
            )
        if selection_source_round is not None:
            self._selection_lineage_source_round = selection_source_round
        attention_cpu = None
        if selection_attention is not None:
            attention_cpu = _cpu_tensor(
                selection_attention.reshape(-1), dtype=torch.float16
            )
        policy = str(getattr(model, "retrieval_policy", ""))
        if retrieval_requested and policy == "attention" and attention_cpu is None:
            raise RuntimeError("Attention retrieval has no recorded selection scores")

        chunks = [
            [int(chunk_id), int(start), int(stop)]
            for chunk_id, start, stop in (getattr(model, "selected_chunks", None) or ())
        ]
        chunk_means = None
        all_chunks = [
            [int(chunk_id), int(start), int(stop)]
            for chunk_id, start, stop in (getattr(model, "chunks", None) or ())
        ]
        attention_valid_tokens = None
        attention_missing_draft_suffix_tokens = None
        target_source_tokens = None
        if attention_cpu is not None:
            attention_valid_tokens = int(attention_cpu.numel())
            target_cache = getattr(model, "past_key_values", None)
            if not isinstance(target_cache, (list, tuple)) or not target_cache:
                raise RuntimeError("Target cache is unavailable during retrieval")
            target_source_tokens = _logical_cache_length(target_cache[-1][1])
            attention_missing_draft_suffix_tokens = (
                total_seq_len - attention_valid_tokens
            )
            # The verifier attention covers the newly committed Target prefix.
            # draft() has already appended the *next*, still-unverified root to
            # the Draft full cache, so that cache is exactly one row longer.
            if attention_valid_tokens != target_source_tokens:
                raise RuntimeError(
                    "Selection attention/Target-cache length mismatch: "
                    f"attention={attention_valid_tokens}, Target={target_source_tokens}"
                )
            if attention_missing_draft_suffix_tokens != 1:
                raise RuntimeError(
                    "Expected exactly one unverified Draft root beyond verifier "
                    f"attention, got Draft={total_seq_len}, "
                    f"attention={attention_valid_tokens}"
                )
            if not isinstance(selection_chunk_means, torch.Tensor):
                raise RuntimeError(
                    "Attention retrieval did not expose engine-native chunk means"
                )
            means_cpu = _cpu_tensor(selection_chunk_means.reshape(-1))
            if int(means_cpu.numel()) != len(all_chunks):
                raise RuntimeError(
                    "Engine chunk-mean count does not match chunk metadata"
                )
            # Preserve the values used by the GPU top-k itself.  Recomputing
            # from a CPU/FP32 copy can change fp16 cumsum ties and select a
            # different chunk at the budget boundary.
            chunk_means = means_cpu
            if not isinstance(selection_ranked_chunk_ids, torch.Tensor):
                raise RuntimeError(
                    "Attention retrieval did not expose ranked non-frontier chunks"
                )
            ranked_chunk_ids = _cpu_tensor(
                selection_ranked_chunk_ids.reshape(-1), dtype=torch.long
            )
            if mandatory_frontier_chunk_id is None:
                raise RuntimeError("Attention retrieval has no mandatory frontier chunk")
            mandatory_frontier_chunk_id = int(mandatory_frontier_chunk_id)
            if effective_chunk_budget is None:
                raise RuntimeError("Attention retrieval has no effective chunk budget")
            effective_chunk_budget = int(effective_chunk_budget)
            if effective_chunk_budget != len(chunks):
                raise RuntimeError(
                    "Effective retrieval budget differs from selected chunks"
                )
            if ranked_chunk_ids.numel() != max(0, effective_chunk_budget - 1):
                raise RuntimeError(
                    "Ranked non-frontier chunk count differs from retrieval budget"
                )
            if not isinstance(attention_query_node_ids, torch.Tensor):
                raise RuntimeError("Attention retrieval has no query-node identity")
            query_node_ids = _cpu_tensor(
                attention_query_node_ids.reshape(-1), dtype=torch.long
            )
            if selection_source_round >= len(self._rounds):
                raise RuntimeError(
                    "Attention query IDs reference an unavailable verification round"
                )
            source_round = self._rounds[selection_source_round]
            accepted_ids = source_round.get("accepted_node_ids")
            if not isinstance(accepted_ids, torch.Tensor):
                raise RuntimeError(
                    "Attention source round has no recorded accepted path"
                )
            accepted_ids = accepted_ids.detach().cpu().reshape(-1).to(torch.long)
            requested_queries = int(
                self._request["generation_config"]["retrieval_num_queries"]
            )
            expected_query_ids = (
                accepted_ids
                if requested_queries == 0
                else accepted_ids[-min(requested_queries, accepted_ids.numel()) :]
            )
            if not torch.equal(query_node_ids, expected_query_ids):
                raise RuntimeError(
                    "Attention query node IDs differ from the accepted-path suffix "
                    f"of verification round {selection_source_round}"
                )
        elif selection_chunk_means is not None:
            raise RuntimeError("Chunk means were exposed without attention scores")
        else:
            ranked_chunk_ids = None
            query_node_ids = None

        refresh_id = len(self._cache_refreshes)
        refresh_kind = (
            f"{policy}_selection" if retrieval_requested
            else ("initial_recent_fill" if refresh_id == 0 else "carry_or_frontier_update")
        )
        refresh = {
            "refresh_id": refresh_id,
            "kind": refresh_kind,
            "attention_source_verification_round": selection_source_round,
            "selection_lineage_attention_source_round": (
                self._selection_lineage_source_round
            ),
            "applies_to_tree_proposal_round": int(pending["round"]),
            "applies_to_next_root_proposal_round": int(pending["round"]) + 1,
            # Mapper eligibility is frozen when this working cache is
            # materialized.  The next root may see a longer Target prefix, but
            # rows that were not mapped here must not gain a counterfactual
            # residual during offline replay.
            "materialization_target_source_limit": int(
                pending["target_source_limit"]
            ),
            "full_draft_cache_tokens": total_seq_len,
            "selected_indices": tree_indices,
            "selected_chunks": chunks,
            "all_chunks": all_chunks,
            "attention_scores": attention_cpu,
            "attention_query_node_ids": query_node_ids,
            "attention_valid_tokens": attention_valid_tokens,
            "target_source_tokens": target_source_tokens,
            "attention_missing_draft_suffix_tokens": (
                attention_missing_draft_suffix_tokens
            ),
            "chunk_mean_attention": chunk_means,
            "requested_chunk_budget": int(
                getattr(model, "retrieve_top_k", 0)
            ),
            "effective_chunk_budget": effective_chunk_budget,
            "ranked_non_frontier_chunk_ids": ranked_chunk_ids,
            "mandatory_frontier_chunk_id": mandatory_frontier_chunk_id,
        }
        self._cache_refreshes.append(refresh)
        self._active_cache_refresh_id = refresh_id

        draft_logits = getattr(model, "last_draft_root_logits", None)
        if not isinstance(draft_logits, torch.Tensor):
            raise RuntimeError("Draft root logits were not exposed by the engine")
        draft_logits = draft_logits.reshape(-1, int(draft_logits.shape[-1]))
        if int(draft_logits.shape[0]) != 1:
            raise RuntimeError("Expected one Draft root distribution")

        pending.update({
            "tree_cache_refresh_id": refresh_id,
            "tree_cache_indices": tree_indices,
            "tree_cache_tokens": working_length,
            "tree_cache_position_ids": (
                None if tree_cache_position_ids is None
                else _cpu_tensor(tree_cache_position_ids, dtype=torch.long)
            ),
            "full_draft_cache_tokens_after": total_seq_len,
            "proposal_token_ids": full_tree_tokens,
            "proposal_target_position_ids": full_positions,
            "proposal_draft_position_ids": full_draft_positions,
            "proposal_tree_attention_mask": full_mask,
            "proposal_parent_raw": _cpu_tensor(tree_parent, dtype=torch.long),
            "native_draft_root_logits": _cpu_tensor(
                draft_logits[0], dtype=torch.float32
            ),
        })

    def record_target_tree(
        self,
        tree_logits: torch.Tensor,
        *,
        native_root_target_draft_kl: Optional[float],
    ) -> None:
        pending = self._require_pending("record_target_tree")
        if "proposal_token_ids" not in pending:
            raise RuntimeError("finish_proposal must precede target verification")
        if tree_logits.ndim != 3 or int(tree_logits.shape[0]) != 1:
            raise ValueError(
                f"Expected Target tree logits [1,N,V], got {tuple(tree_logits.shape)}"
            )
        node_count = int(pending["proposal_token_ids"].numel())
        if int(tree_logits.shape[1]) != node_count:
            raise RuntimeError(
                "Target logits/tree node mismatch: "
                f"logits={tree_logits.shape[1]}, proposal={node_count}"
            )
        logits = tree_logits[0].float()
        topk = min(self.target_tree_topk, int(logits.shape[-1]))
        top_values, top_indices = torch.topk(logits, k=topk, dim=-1)
        pending.update({
            "target_root_logits": _cpu_tensor(logits[0], dtype=torch.float32),
            "target_next_token_ids": _cpu_tensor(
                logits.argmax(dim=-1), dtype=torch.long
            ),
            "target_tree_topk_token_ids": _cpu_tensor(
                top_indices, dtype=torch.int32
            ),
            "target_tree_topk_logits": _cpu_tensor(
                top_values, dtype=torch.float16
            ),
            "target_tree_logsumexp": _cpu_tensor(
                torch.logsumexp(logits, dim=-1), dtype=torch.float32
            ),
            "native_root_target_draft_kl": (
                None
                if native_root_target_draft_kl is None
                else float(native_root_target_draft_kl)
            ),
        })

    def record_verification(
        self,
        *,
        input_ids_before: torch.Tensor,
        normalized_parent: torch.Tensor,
        correctness: torch.Tensor,
        verifier_next_token_ids: torch.Tensor,
        best_candidate_ids: Sequence[int | torch.Tensor],
        best_candidate_tokens: Sequence[int],
        accept_length: int,
        resampled_token_id: int,
    ) -> None:
        pending = self._require_pending("record_verification")
        if "target_root_logits" not in pending:
            raise RuntimeError("Target logits must be recorded before verification")
        candidate_ids = torch.tensor(
            [int(value.item()) if hasattr(value, "item") else int(value)
             for value in best_candidate_ids],
            dtype=torch.long,
        )
        candidate_tokens = torch.tensor(
            [int(value) for value in best_candidate_tokens], dtype=torch.long
        )
        accept_length = int(accept_length)
        if candidate_ids.numel() != accept_length + 1:
            raise RuntimeError(
                "Accepted path must contain the Target-provided root plus "
                f"accept_length Draft tokens, got path={candidate_ids.numel()}, "
                f"accept={accept_length}"
            )
        if candidate_tokens.numel() != candidate_ids.numel():
            raise RuntimeError("Accepted node/token path lengths disagree")
        node_count = int(pending["proposal_token_ids"].numel())
        if candidate_ids.numel() and (
            int(candidate_ids.min()) < 0 or int(candidate_ids.max()) >= node_count
        ):
            raise IndexError("Accepted path references a nonexistent proposal node")
        parent = _cpu_tensor(normalized_parent, dtype=torch.long)
        correct = _cpu_tensor(correctness, dtype=torch.bool)
        verifier_next = _cpu_tensor(
            verifier_next_token_ids, dtype=torch.long
        ).reshape(-1)
        if int(parent.numel()) != node_count or int(correct.numel()) != node_count:
            raise RuntimeError("Verification vectors do not align with proposal nodes")
        if int(verifier_next.numel()) != node_count:
            raise RuntimeError("Verifier next-token vector does not align with proposal")
        if not torch.equal(verifier_next, pending["target_next_token_ids"]):
            raise RuntimeError(
                "Temperature-zero verifier tokens differ from Target argmax"
            )
        if int(candidate_ids[0]) != 0:
            raise RuntimeError("Accepted path must begin at the Target-provided root")
        if not torch.equal(
            candidate_tokens,
            pending["proposal_token_ids"].index_select(0, candidate_ids),
        ):
            raise RuntimeError("Accepted token IDs do not match proposal nodes")

        expected_correct = torch.zeros(node_count, dtype=torch.bool)
        expected_correct[0] = True
        for node in range(1, node_count):
            ancestor = int(parent[node])
            if not 0 <= ancestor < node:
                raise RuntimeError(
                    f"Proposal parent {ancestor} is invalid for node {node}"
                )
            expected_correct[node] = bool(expected_correct[ancestor]) and (
                int(pending["proposal_token_ids"][node])
                == int(verifier_next[ancestor])
            )
        if not torch.equal(correct, expected_correct):
            raise RuntimeError("Saved correct mask does not reproduce verifier logic")

        target_positions = pending["proposal_target_position_ids"]
        deepest_correct = int(
            torch.argmax(correct.to(torch.long) * target_positions).item()
        )
        if int(candidate_ids[-1]) != deepest_correct:
            raise RuntimeError("Accepted path does not end at the deepest correct node")
        reconstructed_path = []
        cursor = deepest_correct
        while cursor != -1:
            reconstructed_path.append(cursor)
            cursor = int(parent[cursor])
        reconstructed_path.reverse()
        if reconstructed_path != candidate_ids.tolist():
            raise RuntimeError("Accepted node IDs are not the verifier parent chain")

        ancestry_mask = pending["proposal_tree_attention_mask"].to(torch.bool)
        if tuple(ancestry_mask.shape) != (node_count, node_count):
            raise RuntimeError("Proposal attention mask shape changed before verification")
        for node in range(node_count):
            expected_ancestors = {node}
            cursor = int(parent[node])
            while cursor != -1:
                expected_ancestors.add(cursor)
                cursor = int(parent[cursor])
            observed_ancestors = set(
                torch.nonzero(ancestry_mask[node], as_tuple=False).reshape(-1).tolist()
            )
            if observed_ancestors != expected_ancestors:
                raise RuntimeError(
                    f"Proposal attention mask is not the parent closure at node {node}"
                )

        last_accepted = int(candidate_ids[-1])
        if int(resampled_token_id) != int(verifier_next[last_accepted]):
            raise RuntimeError(
                "Resampled token does not equal the verifier token after the "
                "deepest accepted proposal node"
            )
        rejection_children = torch.nonzero(
            parent == last_accepted, as_tuple=False
        ).reshape(-1)
        matching_children = rejection_children[
            pending["proposal_token_ids"].index_select(0, rejection_children)
            == verifier_next[last_accepted]
        ] if rejection_children.numel() else rejection_children
        if matching_children.numel():
            raise RuntimeError(
                "Accepted path stopped even though the verifier token exists as a child"
            )
        termination_kind = (
            "tree_leaf_or_budget_boundary"
            if rejection_children.numel() == 0
            else "canonical_token_absent_from_children"
        )

        depths = torch.zeros(node_count, dtype=torch.int16)
        for node in range(1, node_count):
            ancestor = int(parent[node])
            depths[node] = depths[ancestor] + 1

        pending.update({
            "prefix_tokens_before_verification": int(input_ids_before.shape[1]),
            "normalized_parent": parent,
            "proposal_depth": depths,
            "correct_node_mask": correct,
            "verifier_next_token_ids": verifier_next,
            "accepted_node_ids": candidate_ids,
            "accepted_token_ids": candidate_tokens,
            "accept_length": accept_length,
            "first_rejection_position": accept_length + 1,
            "rejection_parent_node_id": last_accepted,
            "rejection_target_token_id": int(verifier_next[last_accepted]),
            "rejection_child_node_ids": rejection_children,
            "termination_kind": termination_kind,
            "resampled_token_id": int(resampled_token_id),
        })
        self._rounds.append(pending)
        self._pending = None

    def finalize_request(
        self,
        model,
        final_input_ids: torch.Tensor,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self._request_active:
            raise RuntimeError("No active trajectory request")
        trailing = self._pending
        if trailing is not None:
            if "proposal_token_ids" not in trailing:
                raise RuntimeError("Trailing proposal was only partially constructed")
            if "target_root_logits" in trailing:
                raise RuntimeError("Trailing proposal was verified but not committed")
            trailing = dict(trailing)
            trailing["status"] = "eager_unverified_after_last_committed_round"

        target_cache = getattr(model, "past_key_values", None)
        if not isinstance(target_cache, (list, tuple)) or not target_cache:
            raise RuntimeError("Target cache is unavailable at trajectory finalization")
        target_value_cache = target_cache[-1][1]
        target_data = getattr(target_value_cache, "data", None)
        if not isinstance(target_data, torch.Tensor):
            raise RuntimeError("Target cache does not expose tensor data")
        target_length = _logical_cache_length(target_value_cache)
        if target_length > int(target_data.shape[2]):
            raise RuntimeError("Target logical cache length exceeds allocation")
        target_values = _flatten_cache_rows(
            target_data[:, :, :target_length, :]
        )
        final_sequence = _cpu_tensor(final_input_ids, dtype=torch.long).reshape(-1)
        if int(final_sequence.numel()) != target_length:
            raise RuntimeError(
                "Committed token sequence/Target-cache length mismatch: "
                f"tokens={final_sequence.numel()}, Target={target_length}"
            )

        full_draft = getattr(model, "full_draft_kv", None)
        draft_length = int(getattr(model, "total_seq_len", -1))
        if not isinstance(full_draft, (list, tuple)) or not full_draft:
            raise RuntimeError("Draft full cache is unavailable at finalization")
        if draft_length <= 0:
            raise RuntimeError("Draft full cache has no populated rows")
        if int(self._draft_token_bank.numel()) != draft_length:
            raise RuntimeError(
                "Draft token bank/full-cache length mismatch at finalization: "
                f"tokens={self._draft_token_bank.numel()}, Draft={draft_length}"
            )
        if draft_length != target_length + 1:
            raise RuntimeError(
                "Final Draft cache must contain exactly one eager unverified root "
                f"beyond Target support: Draft={draft_length}, Target={target_length}"
            )
        if not torch.equal(self._draft_token_bank[:target_length], final_sequence):
            raise RuntimeError(
                "Draft token bank verified prefix differs from the committed "
                "Target sequence"
            )
        draft_keys = []
        draft_values = []
        for layer_index, pair in enumerate(full_draft):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise RuntimeError(f"Draft layer {layer_index} is not a K/V pair")
            key, value = pair
            if draft_length > int(key.shape[2]) or key.shape != value.shape:
                raise RuntimeError(f"Invalid Draft cache shape at layer {layer_index}")
            draft_keys.append(_flatten_cache_rows(key[:, :, :draft_length, :]))
            draft_values.append(_flatten_cache_rows(value[:, :, :draft_length, :]))

        raw_accept = [int(value) for value in result.get("raw_accept_length_list", ())]
        if raw_accept != [int(round_["accept_length"]) for round_ in self._rounds]:
            raise RuntimeError(
                "Trajectory accept lengths disagree with generation result: "
                f"trace={[round_['accept_length'] for round_ in self._rounds]}, "
                f"result={raw_accept}"
            )

        payload = {
            "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
            "collection_semantics": {
                "benchmark_eligible": False,
                "reason": "synchronous CPU trace copies perturb end-to-end latency",
                "target_cache_read_only": True,
                "root_cache_and_tree_cache_recorded_separately": True,
                "retrieval_policy": "attention",
                "mapper_active": False,
                "bridge_config_empty": True,
                "pair_collector_active": False,
                "attention_causality": (
                    "verification round t selection is materialized after the root "
                    "forward of proposal t+1; it affects that proposal's tree cache "
                    "and becomes proposal t+2's root cache"
                ),
            },
            "request": self._request,
            "shared_banks": {
                "target_last_value": _cpu_tensor(
                    target_values, dtype=torch.float16
                ),
                "target_token_ids": final_sequence,
                "draft_key": _cpu_tensor(
                    torch.stack(draft_keys, dim=0), dtype=torch.float16
                ),
                "draft_value": _cpu_tensor(
                    torch.stack(draft_values, dim=0), dtype=torch.float16
                ),
                "draft_token_ids": self._draft_token_bank.clone(),
                "target_cache_tokens": target_length,
                "draft_cache_tokens": draft_length,
            },
            "rounds": self._rounds,
            "cache_refreshes": self._cache_refreshes,
            "trailing_unverified_proposal": trailing,
            "final_output": {
                "full_sequence_token_ids": _cpu_tensor(
                    final_input_ids, dtype=torch.long
                ).squeeze(0),
                "generated_token_ids": _cpu_tensor(
                    final_input_ids[:, int(self._request["input_tokens"]):],
                    dtype=torch.long,
                ).squeeze(0),
                "total_generated": int(result["total_generated"]),
                "raw_total_generated": int(result["raw_total_generated"]),
                "raw_accept_length_list": raw_accept,
            },
        }
        self._payload = payload
        self._request_active = False
        self._pending = None
        return payload

    def payload(self) -> Dict[str, Any]:
        if self._payload is None:
            raise RuntimeError("Trajectory request has not been finalized")
        return self._payload

    def save_atomic(self, path: os.PathLike[str] | str) -> Dict[str, Any]:
        payload = self.payload()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return {
            "path": str(path),
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
            "rounds": len(payload["rounds"]),
        }

    def _require_pending(self, operation: str) -> Dict[str, Any]:
        if not self._request_active or self._pending is None:
            raise RuntimeError(f"{operation} requires an active proposal")
        return self._pending


__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "RetrievalTrajectoryCollector",
]
