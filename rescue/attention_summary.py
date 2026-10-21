"""Read-only attention summaries shared within one committed-prefix round."""


def mean_attention(state):
    """Use the eagerly prepared mean, or support an uncached legacy state.

    Runtime read states are immutable, request-local round snapshots. Their
    summary is computed before publishing them to the planner's CUDA stream.
    Do not lazily populate a legacy state here: its first consumers can run on
    different streams, and a cached tensor would then need another ready event.
    """
    cached = getattr(state, 'mean_attention', None)
    return state.a.mean(0) if cached is None else cached
