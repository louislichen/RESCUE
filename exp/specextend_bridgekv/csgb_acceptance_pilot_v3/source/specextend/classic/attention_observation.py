"""Observation cadence is independent of working-cache refresh cadence."""


def wants_attention(model, retrieval_condition):
    return bool(getattr(model, 'use_retrieval_cache', False)
                and getattr(model, 'retrieval_policy', None) == 'attention'
                and (retrieval_condition or getattr(model, 'csgb_attention_every_round', False)))
