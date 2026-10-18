from typing import List
import time
import torch

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
    if model.ea_layer.use_retrieval_cache:
        retrieval_condition = model.ea_layer.timestep % model.ea_layer.retrieve_every_n_steps == 0
        model.ea_layer.retrieval_condition = retrieval_condition
    else:
        retrieval_condition = False
    
    need_attention_scores = retrieval_condition and model.ea_layer.retrieval_policy == "attention"
    outputs, tree_logits, hidden_state = model(
        draft_input_ids,
        tree_attention_mask=tree_attention_mask,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=draft_position_ids,
        init=False,
        retrieve_attn_scores = need_attention_scores
    )

    return tree_logits, hidden_state, outputs

def verify(input_ids,logits,draft,position_ids,hidden_states,tree_attention_mask,past_key_values_data,current_length_data,parent,model,nodes,threshold,max_depth,logits_processor):
    if logits_processor is None:
        next=torch.argmax(logits,dim=-1)
    else:
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        next = torch.multinomial(probabilities, 1).view(1,-1)

    next=next.to(draft.device)

    parent = torch.where(parent == torch.arange(parent.size(0),device=parent.device), -1, parent)
    parent = torch.cat([torch.tensor([0],device=parent.device), parent + 1], dim=-1).to(draft.device)

    correct = torch.where(draft[0] != next[0][parent], 0, torch.ones(draft.size(1), device=draft.device))
    correct[0] = 1

    last_sum = torch.sum(correct)
    while True:
        correct=torch.where(correct[parent] == 0, 0, correct)
        if torch.sum(correct)==last_sum:
            break
        else:
            last_sum=torch.sum(correct)

    id = torch.argmax(correct * position_ids)

    best_candidate = []
    best_candidate_id = []
    max_id=id

    parent[0]=-1
    while id != -1:
        best_candidate.append(draft[0][id].item())
        best_candidate_id.append(id)
        id = parent[id].item()

    best_candidate.reverse()
    best_candidate_id.reverse()
    next_token = next[0][max_id].unsqueeze(0).unsqueeze(0)
    accept_length=len(best_candidate)-1

    start=current_length_data[0].item()-draft.size(1)
    select_indices=torch.tensor(best_candidate_id)+start

    # For retrieval
    if model.ea_layer.use_retrieval_cache:
        if model.ea_layer.retrieval_condition and model.ea_layer.retrieval_policy == "attention":
            prev_input_len = input_ids.shape[1]
            last_query_index = best_candidate_id[-1]
            requested = int(getattr(model.ea_layer, "retrieval_num_queries", 1))
            num_queries = len(best_candidate_id) if requested == 0 else min(requested, len(best_candidate_id))
            query_indices = best_candidate_id[-num_queries:]
            branch_attn = model.ea_layer.attn_scores[:, :, query_indices, :].mean(dim=1)
            # Aggregate several accepted-branch queries for the old context.
            # Earlier queries cannot see later branch nodes, so retain the final
            # query when assigning scores to newly accepted branch tokens.
            prefix_rows = branch_attn.squeeze(0)[:, :prev_input_len]
            aggregation = getattr(model.ea_layer, "retrieval_query_aggregation", "mean")
            if aggregation == "max":
                prefix_attn_scores = prefix_rows.max(dim=0).values
            elif aggregation == "decay":
                decay = float(getattr(model.ea_layer, "retrieval_decay_lambda", 0.7))
                recency = torch.arange(num_queries - 1, -1, -1, device=prefix_rows.device, dtype=prefix_rows.dtype)
                weights = torch.softmax(-decay * recency, dim=0)
                prefix_attn_scores = (prefix_rows * weights[:, None]).sum(dim=0)
            else:
                prefix_attn_scores = prefix_rows.mean(dim=0)
            last_attn_scores = model.ea_layer.attn_scores[:, :, last_query_index, :].mean(dim=1).squeeze(0)
            best_candidate_id_abs = torch.tensor(best_candidate_id, device=last_attn_scores.device) + prev_input_len
            model.ea_layer.attn_scores_final = torch.cat(
                (prefix_attn_scores,
                last_attn_scores[best_candidate_id_abs]),
                dim=0
            )

    # update target model kv cache
    for data in past_key_values_data:
        tgt = data[..., select_indices.to(data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = data[..., start: start + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(start + tgt.shape[-2])

    # Append accepted branch
    input_ids=torch.cat([input_ids,torch.tensor(best_candidate,device=input_ids.device).unsqueeze(0)],dim=-1)
    
    # hidden states of the newly appended branch (EXCLUDING root node, includes PREVIOUS ROOT NODE)
    new_accept_hidden=hidden_states[:,torch.tensor(best_candidate_id,device=hidden_states.device),:]
    # print(colored(f'new_accept_hidden.shape (= newly appended tokens (EXCLUDING ROOT NODE)): {new_accept_hidden.shape}','red'))

    # next_token: additionally sampled token (used as next root node)
    next_draft, next_position_ids, next_tree_attention_mask,next_parent = model.ea_layer.topK_genrate(new_accept_hidden,
                                              input_ids=torch.cat((input_ids, next_token.to(input_ids.device)), dim=1),
                                              head=model.base_model.lm_head, nodes=nodes, threshold=threshold, max_depth=max_depth)

    # newly drafted tokens (+ root node) and its corresponding position_ids
    next_draft=torch.cat([next_token, next_draft], dim=-1)
    next_position_ids = torch.cat([torch.tensor([next_position_ids[0] - 1],device=next_position_ids.device), next_position_ids], dim=-1)
    next_tree_attention_mask = torch.cat(
        [torch.zeros(1, next_tree_attention_mask.size(1), dtype=next_tree_attention_mask.dtype,device=next_tree_attention_mask.device), next_tree_attention_mask],
        dim=0)
    next_tree_attention_mask = torch.cat(
        [torch.ones(next_tree_attention_mask.size(0), 1, dtype=next_tree_attention_mask.dtype,device=next_tree_attention_mask.device), next_tree_attention_mask],
        dim=1)

    return input_ids,best_candidate,accept_length,next_draft, next_position_ids, next_tree_attention_mask,next_parent


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

def print_draft_branches(draft_input_ids: torch.Tensor,
                         draft_position_ids: torch.Tensor,
                         parent: torch.Tensor,
                         best_candidate_id: list,
                         accept_length: int,
                         tokenizer) -> None:
    """
    Prints all branches in the drafted tree.

    Args:
      draft_input_ids: Tensor of shape [1, N] (N includes the root token at index 0 and N-1 drafted tokens).
      draft_position_ids: Tensor of shape [N] (position ids for all tokens).
      parent: Tensor of shape [N-1] for the drafted tokens (excluding the root).
      best_candidate_id: List of indices representing the accepted branch (computed in verify()).
      accept_length: Number of tokens (after the root) that are accepted (i.e. len(best_candidate)-1).
      tokenizer: A tokenizer with a decode() method.
    """
    print('\n')
    device = draft_input_ids.device
    N = draft_input_ids.size(1)
    
    # Prepend -1 to parent so that its length becomes N.
    if parent.size(0) == N - 1:
        parent = torch.cat([torch.tensor([-1], device=device), parent + 1], dim=0)
    
    # Reconstruct branches: a branch is a path from the root (index 0) to a leaf.
    all_indices = set(range(N))
    used_as_parent = set(parent[1:].tolist())  # ignoring the root's -1
    leaf_indices = sorted(list(all_indices - used_as_parent))
    
    branches = []
    for leaf in leaf_indices:
        branch = []
        cur = leaf
        while cur != -1:
            branch.append(cur)
            cur = parent[cur].item() if cur >= 0 and cur < N else -1
        branch.reverse()  # Now branch goes from root to leaf.
        branches.append(branch)
    
    # Print each branch.
    for i, branch in enumerate(branches):
        branch_len = len(branch)
        if branch == best_candidate_id:
            header = colored(f"Branch {i} ", "green")
            is_accepted = True
        else:
            header = colored(f"Branch {i} ", "white")
            is_accepted = False
        
        print(header, end=' ')
        
        token_strs = []
        for j, idx in enumerate(branch):
            token_id = draft_input_ids[0, idx].item()
            token_text = tokenizer.decode([token_id], skip_special_tokens=True).strip()
            if j == 0:
                # Root token always printed in blue.
                token_strs.append(colored(token_text, "blue"))
            else:
                if is_accepted:
                    # For the accepted branch, print tokens up to accept_length in green,
                    # and any tokens after that in cyan.
                    if j <= accept_length:
                        token_strs.append(colored(token_text, "green"))
                    else:
                        token_strs.append(colored(token_text, "cyan"))
                else:
                    token_strs.append(colored(token_text, "cyan"))
        print(" ".join(token_strs))
