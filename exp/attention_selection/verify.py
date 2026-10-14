"""Integration audits for equality, budgets, visibility, and cache rollback."""
from __future__ import annotations

import json
from types import MethodType

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from attention import AttentionController
from selection import mandatory, retrieve, token_features


@torch.no_grad()
def dense_mask_reference(model, controller, ids, cache, selected, core):
    originals = []
    controller.enabled = False
    for layer in model.model.layers:
        module = layer.self_attn
        original = module.forward
        originals.append((module, original))

        def forward(module, hidden_states, attention_mask=None, position_ids=None,
                    _original=original, **kwargs):
            length = attention_mask.shape[-1]
            visible = torch.zeros(length, device=hidden_states.device, dtype=torch.bool)
            visible[selected[selected < length]] = True
            visible[position_ids[0]] = True
            visible[selected[-1]+1:] = True
            mask = attention_mask.clone()
            mask[..., ~visible] = torch.finfo(mask.dtype).min
            return _original(hidden_states, attention_mask=mask, position_ids=position_ids, **kwargs)

        module.forward = MethodType(forward,module)
    try:
        return core(model, ids, cache)
    finally:
        for module, original in originals:
            module.forward = original


@torch.no_grad()
def verify(e):
    from run import core_forward, device_cache, legacy_cache, tensor_ids, trim, write

    # A small FP32 model isolates mathematical gathering/position correctness
    # from BF16 matrix-shape rounding differences in the real model.
    torch.manual_seed(42)
    toy = Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
          num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
          attention_dropout=0.0,_attn_implementation='eager')).eval()
    c = AttentionController(toy).install()
    inputs=torch.tensor([[1,4,8,12,15,19,23,27,30,31,39,48,57,61,66,71]])
    prefix=legacy_cache(core_forward(toy,inputs[:,:-1]).past_key_values)
    selected=torch.tensor([0,2,5,9,13,14,15])
    c.configure(selected)
    sparse=core_forward(toy,inputs[:,-1:],prefix).last_hidden_state
    dense=dense_mask_reference(toy,c,inputs[:,-1:],prefix,selected,core_forward).last_hidden_state
    toy_diff=float((sparse-dense).abs().max())
    assert toy_diff<2e-5, toy_diff

    items=[r for group in e.data.values() for r in group]
    all_titles=[]
    for item in items:
        all_titles.extend(item['titles'])
    assert len(set(all_titles)) == len(all_titles), 'New splits share document titles'
    old=json.loads((e.out/'config.json').read_text()) if (e.out/'config.json').exists() else None
    item=e.data['train'][0]
    ids=item['ids']
    e.sc.enabled=False
    e.sc.capture=False
    prefix=legacy_cache(core_forward(e.student,tensor_ids(ids[:-1],e.a.device)).past_key_values)
    baseline=e.logit(e.student,core_forward(e.student,tensor_ids(ids[-1:],e.a.device),prefix))
    e.sc.configure(None)
    wrapped=e.logit(e.student,core_forward(e.student,tensor_ids(ids[-1:],e.a.device),prefix))
    equality=float((baseline-wrapped).abs().max())
    assert torch.equal(baseline,wrapped), equality
    e.sc.configure(torch.arange(len(ids),device=e.a.device))
    gathered=e.logit(e.student,core_forward(e.student,tensor_ids(ids[-1:],e.a.device),prefix))
    assert torch.equal(baseline,gathered), float((baseline-gathered).abs().max())
    e.sc.enabled=False
    e.sc.capture=True
    sout=core_forward(e.student,tensor_ids(ids[-1:],e.a.device),prefix)
    s,visible,q=e.sc.snapshot()
    t=s.flip(0).clone()
    tested=0
    for b in e.a.budgets:
        for method in ('recent','random','student_chunk','teacher_chunk','student_token','teacher_token','union','difference','difference_core'):
            for ratio in (0.25,0.5,0.75):
                selection=retrieve(method,t,s,b,e.a.recent,e.a.chunk,ratio)
                assert selection.numel()==b
                assert torch.isin(mandatory(len(ids),e.a.recent,t.device),selection).all()
                assert torch.unique(selection).numel()==b
                tested+=1
    features=token_features(t,s,q)
    assert features.shape==(len(ids),29) and torch.isfinite(features).all()
    selection=retrieve('teacher_token',t,s,e.a.budgets[0],e.a.recent)
    e.sc.configure(selection)
    out=core_forward(e.student,tensor_ids(ids[-1:],e.a.device),prefix)
    first=e.logit(e.student,out)
    cache=legacy_cache(out.past_key_values)
    assert all(k.shape[-2]==len(ids) for k,v in cache)
    core_forward(e.student,tensor_ids([123],e.a.device),cache)
    _,visible,_=e.sc.snapshot()
    assert visible[:,-1].all() and visible[:,-2].all(), 'Draft suffix lost between retrievals'
    rolled=trim(cache,len(ids)-1)
    again=e.logit(e.student,core_forward(e.student,tensor_ids(ids[-1:],e.a.device),rolled))
    assert torch.equal(first,again), 'Rollback changed a repeated query'
    result=dict(fp32_gather_vs_dense_mask_max_error=toy_diff,real_full_native_max_error=equality,
                full_gather_native_equal=True,exact_budget_cases=tested,features_finite=True,
                draft_suffix_preserved=True,cache_rollback_equal=True,title_disjoint=True,
                counts={k:len(v) for k,v in e.data.items()})
    write(e.out/'verification.json',result)
    e.sc.capture=False
    e.sc.enabled=False
    print(json.dumps(result,indent=2),flush=True)
