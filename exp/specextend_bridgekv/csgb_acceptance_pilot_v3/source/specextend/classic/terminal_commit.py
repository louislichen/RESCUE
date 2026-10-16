"""Pure stopping helpers for opt-in terminal Draft elision.

The verifier's committed prefix does not include its sampled *next* root.
Only that committed prefix may decide whether a request has finished.
"""


def candidate_prefix_through_eos(candidate_tokens, eos_token_id):
    """Number of path tokens to commit, retaining the first EOS itself."""
    if eos_token_id is not None:
        for index, token in enumerate(candidate_tokens):
            if int(token) == int(eos_token_id):
                return index + 1
    return len(candidate_tokens)


def committed_generation_should_stop(generated_token_ids, max_new_tokens, eos_token_id):
    """Stop on a completed requested prefix or an EOS already committed."""
    if int(max_new_tokens) < 1:
        raise ValueError("terminal Draft elision requires max_new_tokens >= 1")
    return bool(
        len(generated_token_ids) >= int(max_new_tokens)
        or (
            eos_token_id is not None
            and int(eos_token_id) in generated_token_ids
        )
    )


def finish_terminal_commit(model, committed_input_ids, stop_after_commit):
    """Finalize a stopped round without selecting another Draft action.

    A request-local ``model.csgb_terminal_commit_callback`` can close external
    timing/reward accounting.  It receives committed-only IDs and is called
    exactly once when this predicate is true.  It must not generate a proposal
    or make a next-round controller decision.  Missing predicates preserve the
    original verifier path, including not consulting this optional callback.
    """
    if stop_after_commit is None or not stop_after_commit(committed_input_ids):
        return False
    callback = getattr(model, "csgb_terminal_commit_callback", None)
    if callback is not None:
        callback(committed_input_ids)
    return True
