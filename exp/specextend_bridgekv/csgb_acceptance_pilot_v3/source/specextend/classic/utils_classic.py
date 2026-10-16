from typing import List
import time
import torch
from .attention_observation import wants_attention
from .terminal_commit import candidate_prefix_through_eos, finish_terminal_commit

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from termcolor import colored

def timer(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f'{func.__name__} took {elapsed} seconds')
        return result

    return wrapper


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values

def tree_decoding(
        model,
        draft_input_ids,
        past_key_values,
        draft_position_ids,
        tree_attention_mask,
):
    if model.use_retrieval_cache:
        retrieval_condition = model.timestep % model.retrieve_every_n_steps == 0
        model.retrieval_condition = retrieval_condition
    else:
        retrieval_condition = False
    
    need_attention_scores = wants_attention(model, retrieval_condition)
    outputs, tree_logits, hidden_state = model(
        draft_input_ids,
        tree_attention_mask=tree_attention_mask,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=draft_position_ids,
        init=False,
        retrieve_attn_scores = need_attention_scores
    )

    draft_logits = getattr(model, "last_draft_root_logits", None)
    if draft_logits is not None:
        target_root = tree_logits[0, 0].float()
        draft_root = draft_logits.reshape(-1, draft_logits.shape[-1])[0].float()
        root_kl = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(draft_root, dim=-1),
            torch.nn.functional.softmax(target_root, dim=-1),
            reduction="sum",
        )
        model.last_root_target_draft_kl = float(root_kl.item())

    trajectory_collector = getattr(model, "trajectory_collector", None)
    if trajectory_collector is not None:
        trajectory_collector.record_target_tree(
            tree_logits,
            native_root_target_draft_kl=getattr(
                model, "last_root_target_draft_kl", None
            ),
        )

    return tree_logits, hidden_state, outputs

def verify(input_ids, logits, draft_input_ids, draft_position_ids, tree_attention_mask, 
           past_key_values_data, current_length_data, parent, model, nodes, threshold, max_depth, logits_processor,
           stop_after_commit=None, terminal_eos_token_id=None):
    """Commit a verified path, optionally stopping before the next Draft.

    ``stop_after_commit`` sees only committed IDs, never ``next_token``.  The
    default path is unchanged.  ``terminal_eos_token_id`` is supplied only by
    the opt-in generation path to trim a committed path at its first EOS before
    compacting Target KV and recording acceptance/attention bookkeeping.
    """
    if terminal_eos_token_id is not None and stop_after_commit is None:
        raise ValueError("EOS commit trimming requires stop_after_commit")
    if logits_processor is None:
        next = torch.argmax(logits, dim=-1)
    else:
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        next = torch.multinomial(probabilities, 1).view(1, -1)
    next=next.to(draft_input_ids.device)

    parent = torch.where(parent == torch.arange(parent.size(0),device=parent.device), -1, parent)
    parent = torch.cat([torch.tensor([0],device=parent.device), parent + 1], dim=-1).to(draft_input_ids.device)

    correct = torch.where(draft_input_ids[0] != next[0][parent], 0, torch.ones(draft_input_ids.size(1), device=draft_input_ids.device))
    correct[0] = 1
    last_sum = torch.sum(correct)
    while True:
        correct = torch.where(correct[parent] == 0, 0, correct)
        if torch.sum(correct) == last_sum:
            break
        else:
            last_sum = torch.sum(correct)

    id = torch.argmax(correct * draft_position_ids)
    best_candidate = []
    best_candidate_id = []
    max_id = id
    parent[0] = -1
    while id != -1:
        best_candidate.append(draft_input_ids[0][id].item())
        best_candidate_id.append(id)
        id = parent[id].item()

    best_candidate.reverse()
    best_candidate_id.reverse()
    if stop_after_commit is not None:
        kept = candidate_prefix_through_eos(best_candidate, terminal_eos_token_id)
        best_candidate = best_candidate[:kept]
        best_candidate_id = best_candidate_id[:kept]
        max_id = best_candidate_id[-1]
    next_token = next[0][max_id].unsqueeze(0).unsqueeze(0)
    accept_length=len(best_candidate)-1

    trajectory_collector = getattr(model, "trajectory_collector", None)
    if trajectory_collector is not None:
        trajectory_collector.record_verification(
            input_ids_before=input_ids,
            normalized_parent=parent,
            correctness=correct,
            verifier_next_token_ids=next,
            best_candidate_ids=best_candidate_id,
            best_candidate_tokens=best_candidate,
            accept_length=accept_length,
            resampled_token_id=int(next_token.item()),
        )

    # Compute final attention scores with newly accepted tokens, to be used for retrieval
    if model.use_retrieval_cache:
        if wants_attention(model, model.retrieval_condition):
            prev_input_len = input_ids.shape[1]
            last_query_index = best_candidate_id[-1]
            requested = int(getattr(model, "retrieval_num_queries", 1))
            num_queries = len(best_candidate_id) if requested == 0 else min(requested, len(best_candidate_id))
            query_indices = best_candidate_id[-num_queries:]
            if getattr(model, "trajectory_collector", None) is not None:
                model._trajectory_pending_attention_query_node_ids = torch.tensor(
                    [
                        int(value.item()) if hasattr(value, "item") else int(value)
                        for value in query_indices
                    ],
                    dtype=torch.long,
                    device=model.attn_scores.device,
                )
            branch_attn = model.attn_scores[:, :, query_indices, :].mean(dim=1).squeeze(0)
            prefix_rows = branch_attn[:, :prev_input_len]
            aggregation = getattr(model, "retrieval_query_aggregation", "mean")
            if aggregation == "max":
                prefix_attn_scores = prefix_rows.max(dim=0).values
            elif aggregation == "decay":
                decay = float(getattr(model, "retrieval_decay_lambda", 0.7))
                recency = torch.arange(num_queries - 1, -1, -1, device=prefix_rows.device, dtype=prefix_rows.dtype)
                weights = torch.softmax(-decay * recency, dim=0)
                prefix_attn_scores = (prefix_rows * weights[:, None]).sum(dim=0)
            else:
                prefix_attn_scores = prefix_rows.mean(dim=0)
            last_attn_scores = model.attn_scores[:, :, last_query_index, :].mean(dim=1).squeeze(0)
            
            best_candidate_id_abs = torch.tensor(best_candidate_id, device=last_attn_scores.device) + prev_input_len
            model.attn_scores_final = torch.cat(
                (prefix_attn_scores,
                last_attn_scores[best_candidate_id_abs]),
                dim=0
            )
            model.csgb_attention_observed_timestep = int(model.timestep)

    start=current_length_data[0].item()-draft_input_ids.size(1)
    select_indices=torch.tensor(best_candidate_id)+start

    for data in past_key_values_data:
        # select_indices=tensor([29, 30, 34, 44], device='cuda:0')
        tgt = data[..., select_indices.to(data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = data[..., start: start + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(start + tgt.shape[-2])

    input_ids=torch.cat([input_ids,torch.tensor(best_candidate,device=input_ids.device).unsqueeze(0)],dim=-1)

    model.update_dynamic_cache_budget(
        accept_length,
        getattr(model, "last_root_target_draft_kl", None),
    )

    # Every accepted-round counter and Target KV update above remains valid,
    # but a terminal request has no next proposal.  In particular, do not call
    # a controller's draft wrapper just to finalize the preceding round: that
    # would also select an unused action and compute an unused tree.
    if finish_terminal_commit(model, input_ids, stop_after_commit):
        return input_ids, best_candidate, accept_length, None, None, None, None

    next_draft, next_position_ids, next_tree_attention_mask, parent = model.draft(torch.cat((input_ids, next_token.to(input_ids.device)), dim=1),nodes,threshold,max_depth)

    next_draft=torch.cat([next_token, next_draft], dim=-1)
    next_position_ids = torch.cat([torch.tensor([next_position_ids[0] - 1],device=next_position_ids.device), next_position_ids], dim=-1)
    next_tree_attention_mask = torch.cat(
        [torch.zeros(1, next_tree_attention_mask.size(1), dtype=next_tree_attention_mask.dtype,device=next_tree_attention_mask.device), next_tree_attention_mask],
        dim=0)
    next_tree_attention_mask = torch.cat(
        [torch.ones(next_tree_attention_mask.size(0), 1, dtype=next_tree_attention_mask.dtype,device=next_tree_attention_mask.device), next_tree_attention_mask],
        dim=1)

    return input_ids, best_candidate, accept_length, next_draft, next_position_ids, next_tree_attention_mask,parent


def print_newly_accepted_tokens(
    old_len,
    input_ids_after,
    tokenizer,
    accepted_color='green',
    resampled_color='blue',
    verbose=False,
):
    """
    Print tokens appended to 'input_ids_after' from index 'old_len' onward.
    
    According to the new logic:
      - The *first* of these newly appended tokens is actually "resampled".
      - Any remaining tokens are "accepted".
    
    We no longer have "rejected" tokens in 'input_ids_after'.
    """
    # Slice the newly appended tokens
    new_tokens = input_ids_after[0, old_len:]
    if new_tokens.numel() == 0:
        return []

    typed_tokens = []

    # 1) The first newly appended token is "resampled"
    resampled_token = new_tokens[0]
    resampled_str = tokenizer.decode(
        resampled_token.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=True
    ).replace("<0x0A>", "\n")
    typed_tokens.append((resampled_str, "resampled"))

    # 2) All remaining newly appended tokens are "accepted"
    if new_tokens.numel() > 1:
        accepted_tokens = new_tokens[1:]
        for t in accepted_tokens:
            decoded = tokenizer.decode(
                t.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).replace("<0x0A>", "\n")
            typed_tokens.append((decoded, "accepted"))

    # 3) Print them with color if verbose
    if verbose:
        for (token_str, token_type) in typed_tokens:
            if token_type == "accepted":
                clr = accepted_color
            else:  # token_type == "resampled"
                clr = resampled_color
            print(colored(token_str, clr), flush=True, end=" ")

    return typed_tokens

def compute_branch_length(tensor_1d):
    """
    Returns the number of elements in `tensor_1d` up to (but not including)
    the first -1, or the entire length if -1 is absent.
    """
    # Find all indices where tensor_1d == -1
    neg1_indices = (tensor_1d == -1).nonzero(as_tuple=True)[0]
    if len(neg1_indices) == 0:
        # No -1 found => entire length
        return tensor_1d.size(0) - 1
    else:
        # The first -1 is at neg1_indices[0]
        return max(neg1_indices[0].item() - 1, 0)


def print_colored_draft_tree(
    candidates: torch.Tensor,
    best_candidate: int,
    accept_length: int,
    tokenizer
) -> None:
    """
    Print each row (branch) in 'candidates' (shape [num_branches, max_depth]),
    color-coded as follows:
      - The first token in every branch is blue (the resampled token).
      - For the accepted branch (i == best_candidate):
           tokens after the first up to 'accept_length' are green,
           and anything after that is red.
      - For non-accepted branches: tokens after the first are red.
    """
    num_branches = candidates.size(0)

    for i in range(num_branches):
        branch_tokens = candidates[i].tolist()
        branch_tokens = [t for t in branch_tokens if t >= 0]  # remove any -1 padding

        colored_text_parts = []
        for idx, token_id in enumerate(branch_tokens):
            token_str = tokenizer.decode([token_id], skip_special_tokens=True)

            if idx == 0:
                # First token always blue
                clr = "blue"
            else:
                # For the accepted branch
                if i == best_candidate:
                    if idx <= accept_length:
                        clr = "green"  # accepted portion
                    else:
                        clr = "red"    # beyond accepted length
                else:
                    # Non-accepted branches: everything after the first is red
                    clr = "red"

            colored_text_parts.append(colored(token_str, clr))

        # Join them into a single string for the branch
        branch_text = " ".join(colored_text_parts)
        print(f"Branch {i:2d}: {branch_text}")
