"""Full archived SpecExtend tree/retrieval/hybrid engine on Qwen2 weights.

Architecture port only: QKV bias and RoPE base are taken from Qwen2. FP32
prefix attention uses PyTorch's efficient SDPA and its LSE, since upstream
FlashAttention does not implement FP32. This is explicitly a Qwen2 engine
port, not the published Vicuna/FP16 benchmark or an unmodified reproduction.
"""
import sys
from pathlib import Path
from types import MethodType
import time

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, AutoTokenizer
from transformers.cache_utils import Cache

from .loading import ROOT

SOURCE = ROOT/'exp/specextend_bridgekv/specextend'


class PrefixView(Cache):
    """Expose upstream preallocated buffers through the shared prefill API."""
    def __init__(self, pairs):
        self.pairs = pairs

    def get_seq_length(self, layer_idx=0):
        return self.pairs[layer_idx][0].shape[2]

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def get_max_length(self):
        return self.pairs[0][0].data.shape[2]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        keys, values = self.pairs[layer_idx]
        return keys.cat(key_states, dim=2), values.cat(value_states, dim=2)


def _fp32_flash(query, keys, values, causal=False, **kwargs):
    q, k, v = (a.transpose(1, 2) for a in (query, keys, values))
    mask = None
    if causal and q.shape[-2] != k.shape[-2]:
        prefix = k.shape[-2]-q.shape[-2]
        mask = torch.arange(k.shape[-2], device=q.device)[None, :] <= (
            prefix+torch.arange(q.shape[-2], device=q.device)[:, None])
        causal = False
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal).transpose(1, 2)


def _fp32_prefix(query, keys, values, cache_seqlens, return_softmax_lse=False, **kwargs):
    n = int(cache_seqlens)
    q, k, v = query.transpose(1, 2), keys[:, :n].transpose(1, 2), values[:, :n].transpose(1, 2)
    # The upstream attention expands GQA heads before this call.
    out, lse, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, None, True, 0., False)
    return (out.transpose(1, 2), lse[:, :, :q.shape[-2]]) if return_softmax_lse else out.transpose(1, 2)


def _patch(module):
    if getattr(module, '_rescue_qwen_port', False):
        return
    module._rescue_qwen_port = True
    original = module.LlamaAttention.__init__
    def init(attention, config):
        original(attention, config)
        for name in ('q_proj', 'k_proj', 'v_proj'):
            linear = getattr(attention, name)
            linear.bias = torch.nn.Parameter(linear.weight.new_zeros(linear.out_features))
    def rope(attention):
        if attention.config.rope_scaling is not None:
            raise ValueError('This Qwen2 port does not implement rope scaling')
        attention.rotary_emb = module.LlamaRotaryEmbedding(attention.head_dim,
            max_position_embeddings=attention.max_position_embeddings,
            base=attention.config.rope_theta)
    module.LlamaAttention.__init__ = init
    module.LlamaAttention._init_rope = rope
    flash = module.flash_attn_func
    def dispatch(q, k, v, **kwargs):
        k, v = k.to(q.dtype), v.to(q.dtype)
        return _fp32_flash(q, k, v, **kwargs) if q.dtype == torch.float32 else flash(q, k, v, **kwargs)
    module.flash_attn_func = dispatch
    if hasattr(module, 'flash_attn_with_kvcache'):
        prefix = module.flash_attn_with_kvcache
        def dispatch_prefix(q, k, v, **kwargs):
            return (_fp32_prefix(q, k, v, **kwargs) if q.dtype == torch.float32
                    else prefix(q, k, v, **kwargs))
        module.flash_attn_with_kvcache = dispatch_prefix


def build(teacher, student, tokenizer_path):
    sys.path.insert(0, str(SOURCE))
    import shared.modeling_llama_kv_target as target_module
    import classic.modeling_llama_kv_draft as draft_module
    from classic.model_classic import SPModel
    wrappers = []
    vocab = len(AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True))
    for model, module in ((teacher, target_module), (student, draft_module)):
        _patch(module)
        config = LlamaConfig(**model.config.to_dict())
        config.pretraining_tp = 1
        with torch.device('meta'):
            wrapper = module.LlamaForCausalLM(config)
        missing, unexpected = wrapper.load_state_dict(model.state_dict(), strict=False, assign=True)
        if unexpected or any(not key.endswith('.rotary_emb.inv_freq') for key in missing):
            raise ValueError(f'Architecture port lost weights: {missing}, {unexpected}')
        for layer in wrapper.model.layers:
            layer.self_attn._init_rope()
            layer.self_attn.rotary_emb.to(model.device)
        head = torch.nn.Linear(config.hidden_size, vocab, bias=False, device='meta')
        head.weight = torch.nn.Parameter(wrapper.lm_head.weight[:vocab], requires_grad=False)
        wrapper.lm_head = head
        wrapper.eval().requires_grad_(False)
        wrappers.append(wrapper)
    engine = SPModel(wrappers[0], str(tokenizer_path), wrappers[1]).eval()
    def init_caches(engine):
        c = engine.draft_model.config
        shape = (1, c.num_key_value_heads, engine.full_cache_budget,
                 c.hidden_size//c.num_attention_heads)
        engine.full_draft_kv = [(torch.zeros(shape, device=student.device, dtype=student.dtype),
                                 torch.zeros(shape, device=student.device, dtype=student.dtype))
                                for _ in range(c.num_hidden_layers)]
        engine.total_seq_len = engine.seq_len_total_old = engine.evicted = 0
        engine.draft_stable_kv = engine.chunks = engine.cached_chunks = engine.selected_chunks = None
        engine.draft_model.model.past_key_position_ids = None
        engine.recent_start = engine.recent_end = 0
    engine.init_caches = MethodType(init_caches, engine)
    original_forward = engine.base_model.model.forward
    engine.prefill_precision = 'native'
    def target_forward(model, input_ids=None, **kwargs):
        if kwargs.get('init', True) and input_ids.shape[1] > 1:
            if kwargs.get('attention_mask') is not None or kwargs.get('position_ids') is not None:
                raise ValueError('Shared prefill requires unpadded implicit positions')
            cache = PrefixView(kwargs['past_key_values'])
            mixed = engine.prefill_precision == 'bfloat16'
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
                teacher.model(input_ids=input_ids[:, :-1], past_key_values=cache, use_cache=True,
                    output_attentions=False, return_dict=True)
            # The final prompt query is FP32 in both the reference and verifier.
            return teacher.model(input_ids=input_ids[:, -1:], past_key_values=cache, use_cache=True,
                output_attentions=False, return_dict=True)
        return original_forward(input_ids=input_ids, **kwargs)
    engine.base_model.model.forward = MethodType(target_forward, engine.base_model.model)
    return engine


@torch.no_grad()
def generate(model, ids, max_new_tokens=256, nodes=32, top_chunks=32, max_depth=10):
    device = model.base_model.device
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = model.spgenerate(torch.tensor([ids], device=device), max_new_tokens=max_new_tokens,
        nodes=nodes, max_depth=max_depth, temperature=0., use_specextend=True,
        retrieval_chunk_size=32, retrieve_top_k=top_chunks, retrieve_every_n_steps=4,
        force_flash_prefill=True, force_hybrid_tree_attn=True, return_output_ids=True)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter()-start
    generated = result['output_token_ids']
    if model.tokenizer.eos_token_id in generated:
        generated = generated[:generated.index(model.tokenizer.eos_token_id)+1]
    return dict(generated=generated, tokens=len(generated), total_seconds=seconds,
                total_tps=len(generated)/seconds, upstream=result, nodes=nodes,
                top_chunks=top_chunks, max_depth=max_depth)
