from typing import List, Optional, Dict

import torch
from transformers import AutoModelForCausalLM


def _find_decoder_layers_list(model: AutoModelForCausalLM):
    candidates = [
        (lambda m: m.model.layers),
        (lambda m: m.model.model.layers),
        (lambda m: m.transformer.h),
        (lambda m: m.gpt_neox.layers),
        (lambda m: m.model.decoder.layers),
    ]
    for getter in candidates:
        try:
            layers = getter(model)
            if isinstance(layers, (list, torch.nn.ModuleList)) and len(layers) > 0:
                return layers
        except Exception:
            continue
    raise RuntimeError("Could not locate decoder layers list in this model.")


@torch.no_grad()
def sampled_attention_rows_last_layer(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,                       
    attention_mask: Optional[torch.Tensor],        
    sample_Ls: List[int],                          
    attention_precision: int,
) -> Dict[int, Optional[List[float]]]:
    layers = _find_decoder_layers_list(model)
    last_layer = layers[-1]
    if not hasattr(last_layer, "self_attn"):
        return {L: None for L in sample_Ls}

    Lmax = int(input_ids.shape[-1])
    sample_Ls = [int(L) for L in sample_Ls if 1 < int(L) <= Lmax]
    sample_Ls = sorted(list(dict.fromkeys(sample_Ls)))

    row_indices = [L - 1 for L in sample_Ls]
    row_to_L = {r: (r + 1) for r in row_indices}

    cache: Dict[int, Optional[List[float]]] = {L: None for L in sample_Ls}

    def hook_fn(module, inputs, output):
        attn_weights = None
        if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[1]):
            attn_weights = output[1]
        else:
            return

        if attn_weights.dim() != 4:
            return

        for r in row_indices:
            L = row_to_L[r]
            v = attn_weights[0, :, r, :L].float().mean(dim=0).detach().cpu()
            if not torch.isfinite(v).all():
                cache[L] = None
                continue
            cache[L] = [round(float(x), attention_precision) for x in v.tolist()]

    h = last_layer.self_attn.register_forward_hook(hook_fn)
    try:
        _ = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
        )
    finally:
        h.remove()

    return cache