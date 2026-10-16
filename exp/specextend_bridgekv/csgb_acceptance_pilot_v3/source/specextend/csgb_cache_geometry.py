"""Absolute positions available to a Root query, before Root forward.

The stable cache covers selected historical intervals. The same Root forward
also processes the suffix starting at total_seq_len, including accepted tokens
and the next uncommitted Root. Only committed history has attention mass here;
the final Root itself is excluded from the coverage universe.
"""


def root_readable_positions(selected_chunks, processed_prefix_length,
                            attention_length, root_input_length):
    for value in (processed_prefix_length, attention_length, root_input_length):
        if type(value) is not int or value < 0:
            raise ValueError('Lengths must be nonnegative integers')
    if processed_prefix_length > root_input_length:
        raise ValueError('Processed prefix exceeds current Root input')
    # Attention can be older/shorter, but must not include uncommitted Root.
    if attention_length > max(0, root_input_length-1):
        raise ValueError('Attention extends beyond committed history')
    positions=set()
    for _,lo,hi in selected_chunks:
        if lo < 0 or hi < lo or hi > processed_prefix_length:
            raise ValueError('Selected interval is not in the stable processed prefix')
        positions.update(range(lo,min(hi,attention_length)))
    positions.update(range(processed_prefix_length,attention_length))
    return sorted(positions)
