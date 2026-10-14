#!/usr/bin/env python3
"""E3 equal-budget source ablations, and E9 utility/attention correlations."""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from run import Experiment, append, device_cache, rows, write
from selection import mandatory, normalize, retrieve


@torch.no_grad()
def main():
    # Reuse the exact frozen primary protocol; never tune on these test ablations.
    from argparse import Namespace
    config=json.loads(open('exp/attention_selection/out_stage2/config.json').read())
    allowed=['data','output','cache','teacher','student','device','budgets','query_steps','recent','chunk',
             'max_new_tokens','draft_k','utility_candidates','utility_joint','seed']
    a=Namespace(**{k:config[k] for k in allowed})
    torch.set_num_threads(4)
    e=Experiment(a)
    chosen=json.loads(e.selected_path.read_text())
    path=e.out/'union_ablation_test.jsonl'
    done={(r['source_index'],r['step'],r['budget'],r['method']) for r in rows(path)}
    diagnostics=[]
    correlations=[]
    utility={(r['source_index'],r['step'],r['budget']):r for r in rows(e.out/'utility_test.jsonl')}
    for index,item in enumerate(e.data['test']):
        bundle=torch.load(e.doc_path(item),weights_only=False)
        base=device_cache(bundle['initial_student'],a.device)
        for state in bundle['states']:
            t,s=(state[k].to(a.device) for k in ('trow','srow'))
            cache=e.state_cache(base,state)
            n=t.shape[-1]
            fixed=mandatory(n,a.recent,t.device)
            candidates=torch.arange(n,device=t.device)
            candidates=candidates[~torch.isin(candidates,fixed)]
            tr=candidates[torch.argsort(t[-1,candidates],descending=True,stable=True)]
            sr=candidates[torch.argsort(s[-1,candidates],descending=True,stable=True)]
            for b in a.budgets:
                slots=min(b,n)-fixed.numel()
                ratio=chosen['ratios'][f'union_{b}']
                pt=tr[:round(slots*ratio)]
                ps=sr[:slots-round(slots*ratio)]
                intersection=pt[torch.isin(pt,ps)]
                te=pt[~torch.isin(pt,ps)]
                se=ps[~torch.isin(ps,pt)]
                union=retrieve('union',t,s,b,a.recent,ratio=ratio)
                diagnostics.append(dict(source_index=item['source_index'],step=state['step'],budget=b,
                    peak_jaccard=len(intersection)/max(1,len(pt)+len(ps)-len(intersection)),
                    teacher_exclusive_fraction=len(te)/max(1,slots),student_exclusive_fraction=len(se)/max(1,slots),
                    overlap_count=len(intersection),teacher_peak_count=len(pt),student_peak_count=len(ps)))
                for method,removed,fallback in [('drop_student_exclusive',se,tr),('drop_teacher_exclusive',te,sr)]:
                    key=(item['source_index'],state['step'],b,method)
                    if key in done:
                        continue
                    kept=union[~torch.isin(union,removed)]
                    replacements=fallback[~torch.isin(fallback,torch.cat((kept,removed)))]
                    chosen_positions=torch.cat((kept,replacements[:b-len(kept)])).sort().values
                    assert chosen_positions.numel()==b and not torch.isin(chosen_positions,removed).any()
                    result=e.prediction(cache,state,chosen_positions)
                    append(path,dict(source_index=item['source_index'],step=state['step'],budget=b,method=method,
                                     removed_count=len(removed),retained=len(chosen_positions),**result))
                label=utility.get((item['source_index'],state['step'],b))
                if label:
                    pool=label['candidates']
                    gain=np.asarray(label['gains'])
                    tt=normalize(t[-1]);ss=normalize(s[-1])
                    for name,values in [('teacher',tt[pool]),('student',ss[pool]),
                                        ('difference',(tt*(tt-ss).clamp_min(0))[pool])]:
                        x=values.cpu().numpy()
                        corr=float(np.corrcoef(x,gain)[0,1]) if np.std(x)>0 and np.std(gain)>0 else 0.0
                        correlations.append(dict(source_index=item['source_index'],step=state['step'],budget=b,
                                                 feature=name,utility_correlation=corr))
        print(f'supplement {index+1}/{len(e.data["test"])}',flush=True)
    write(e.out/'union_diagnostics_test.json',diagnostics)
    write(e.out/'utility_attention_correlations.json',correlations)


if __name__=='__main__':
    main()
