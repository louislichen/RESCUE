"""Document-paired statistics, held-out calibration, and auditable reports."""
import argparse
import hashlib
import json
import math
import re
import string
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiment import read_rows, write, ROOT
from risk import load


def pooled(rows):
    tokens=sum(r['tokens'] for r in rows)
    proposed=sum(r.get('proposed',0) for r in rows)
    accepted=sum(r.get('accepted',0) for r in rows)
    blocks=sum(r.get('blocks',0) for r in rows)
    return dict(documents=len(rows),tokens=tokens,proposed=proposed,accepted=accepted,blocks=blocks,
        acceptance_rate=accepted/max(1,proposed),mean_accepted=accepted/max(1,blocks),
        mean_draft=proposed/max(1,blocks),decode_tps=tokens/sum(r['decode_seconds'] for r in rows),
        total_tps=tokens/sum(r['total_seconds'] for r in rows),
        output_agrees=all(r.get('output_agrees',False) for r in rows),
        stage_seconds={k:sum(r.get('stage_seconds',{}).get(k,0.) for r in rows)
                       for k in ('selection','decision','source_pack','draft','verify','cache_update')})


def paired(a,b,metric='acceptance',reps=5000):
    aa={r['source_index']:r for r in a};bb={r['source_index']:r for r in b}
    keys=sorted(set(aa)&set(bb));rng=np.random.default_rng(321)
    def arrays(rows):
        if metric=='acceptance':return np.array([[rows[k]['accepted'],rows[k]['proposed']] for k in keys])
        if metric=='decode':return np.array([[rows[k]['tokens'],rows[k]['decode_seconds']] for k in keys])
        if metric=='total':return np.array([[rows[k]['tokens'],rows[k]['total_seconds']] for k in keys])
        return np.array([[rows[k][metric],1.] for k in keys])
    x,y=arrays(aa),arrays(bb)
    def delta(x,y):
        xr=x[...,0].sum(-1)/x[...,1].sum(-1);yr=y[...,0].sum(-1)/y[...,1].sum(-1)
        return (xr-yr)*100 if metric=='acceptance' else (xr/yr-1)*100 if metric in ('decode','total') else xr-yr
    indices=rng.integers(0,len(keys),(reps,len(keys)))
    estimates=delta(x[indices],y[indices])
    return dict(documents=len(keys),estimate=float(delta(x,y)),ci95=np.quantile(estimates,[.025,.975]).tolist(),
                units='percentage points' if metric=='acceptance' else 'relative percent' if metric in ('decode','total') else metric)


def proxy_comparisons(offline):
    """First average states within each document, then paired resampling."""
    docs=defaultdict(lambda:defaultdict(list))
    for r in offline:docs[r['rule']][r['source_index']].append(r)
    reduced={rule:[dict(source_index=d,**{m:float(np.mean([r[m] for r in rows]))
        for m in ('teacher_read_error','student_read_error','kl','teacher_ce','match')})
        for d,rows in items.items()] for rule,items in docs.items()}
    result={}
    for rule in ('read_teacher','read_joint025','read_joint1','union','student'):
        for base in ('teacher','union'):
            if rule==base:continue
            result[f'{rule} vs {base}']={m:paired(reduced[rule],reduced[base],m)
                for m in ('teacher_read_error','student_read_error','kl','match')}
    return result


def answer_scores(out,groups):
    from transformers import AutoTokenizer
    protocol=json.loads((out/'protocol.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(protocol['teacher'],local_files_only=True)
    items={r['source_index']:r for r in read_rows(Path(protocol['data'])/'test.jsonl')}
    def normalize(text):
        text=text.lower().translate(str.maketrans('','',string.punctuation))
        return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())
    records=[]
    for row in groups['teacher_greedy']:
        item=items[row['source_index']]
        text=tokenizer.decode(row['generated'],skip_special_tokens=True).strip()
        answer,gold=normalize(text),normalize(item['gold'])
        common=Counter(answer.split())&Counter(gold.split())
        overlap=sum(common.values())
        precision=overlap/max(1,len(answer.split()));recall=overlap/max(1,len(gold.split()))
        records.append(dict(source_index=row['source_index'],gold=item['gold'],output=text,
                            exact_match=int(answer==gold),token_f1=2*precision*recall/max(1e-12,precision+recall),
                            gold_contained=int(bool(gold) and gold in answer),
                            length_capped=row['tokens']==protocol['max_new_tokens']))
    result=dict(documents=len(records),exact_match=float(np.mean([r['exact_match'] for r in records])),
        token_f1=float(np.mean([r['token_f1'] for r in records])),
        gold_contained=float(np.mean([r['gold_contained'] for r in records])),
        length_capped=sum(r['length_capped'] for r in records),
        interpretation='Descriptive answer quality; all exact SD outputs inherit identical scores. Containment is not official QA accuracy.',records=records)
    write(out/'answer_quality.json',result)
    return result


def auc(labels,scores):
    labels=np.asarray(labels);scores=np.asarray(scores)
    positive=labels>0; n1=positive.sum();n0=len(labels)-n1
    if not n1 or not n0:return None
    order=np.argsort(scores,kind='stable');rank=np.empty(len(labels),float)
    sorted_scores=scores[order]
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and sorted_scores[end]==sorted_scores[start]:end+=1
        rank[order[start:end]]=(start+1+end)/2;start=end
    return float((rank[positive].sum()-n1*(n1+1)/2)/(n1*n0))


def factorial(groups,selected,reps=5000):
    config=selected['sequential_config']
    adaptive='B_attention_online' if config.get('online') else 'B_'+config['risk']
    methods=['A_teacher','A_'+selected['joint_rule'],adaptive,'C_sequential']
    data=[{r['source_index']:r for r in groups[k]} for k in methods]
    keys=sorted(set.intersection(*(set(d) for d in data)))
    x=np.array([[data[3][d]['decode_seconds']-data[1][d]['decode_seconds']-
                 data[2][d]['decode_seconds']+data[0][d]['decode_seconds'],data[0][d]['tokens']] for d in keys])
    rng=np.random.default_rng(333);idx=rng.integers(0,len(x),(reps,len(x)))
    drawn=x[idx]
    values=drawn[...,0].sum(-1)/drawn[...,1].sum(-1)*1000
    return dict(methods=methods,estimate=float(x[:,0].sum()/x[:,1].sum()*1000),
        ci95=np.quantile(values,[.025,.975]).tolist(),units='milliseconds per output token',
        definition='time(R1,adaptive)-time(R1,k4)-time(R0,adaptive)+time(R0,k4); negative favors non-additive saving')


@torch.no_grad()
def calibration(out):
    blocks=read_rows(out/'branches_test.jsonl')
    x=torch.tensor([r['frames'][0]['features'] for r in blocks])
    y=torch.tensor([r['labels'] for r in blocks]);mask=torch.tensor([r['mask'] for r in blocks])
    docs=sorted({r['source_index'] for r in blocks})
    counts={d:sum(r['source_index']==d for r in blocks) for d in docs}
    weight=torch.tensor([1/counts[r['source_index']] for r in blocks])
    selected=json.loads((out/'risk_selection.json').read_text())
    results={};paired_inputs={}
    for group in selected:
        trials=[]
        for seed in (42,43,44):
            model,payload=load(out/f'risk_{group}_{seed}.pt','cpu')
            logits=model(x); p=logits.softmax(-1);survival=p[:,:,0].cumprod(-1)
            ce=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').reshape(-1,12)
            valid_weight=mask*weight[:,None]
            nll=float((ce*valid_weight).sum()/valid_weight.sum())
            if seed==selected[group]['seed']:
                event=((y[:,:4]!=0)&mask[:,:4]).any(-1)
                known=event|mask[:,3]
                errors=((~event).float()-survival[:,3]).square()*known
                per_doc=[]
                for d in docs:
                    ids=[i for i,r in enumerate(blocks) if r['source_index']==d]
                    per_doc.append([float((ce[ids]*mask[ids]).sum())/counts[d],
                        float(mask[ids].sum())/counts[d],float(errors[ids].sum())/counts[d],
                        float(known[ids].sum())/counts[d]])
                paired_inputs[group]=np.array(per_doc)
            horizons={}
            for k in (1,2,4,8,12):
                event=((y[:,:k]!=0)&mask[:,:k]).any(-1)
                known=event|mask[:,k-1]
                truth=(~event[known]).float();prob=survival[known,k-1]
                w=weight[known];w=w/w.sum()
                brier=float(((truth-prob).square()*w).sum())
                bins=[];ece=0.
                for lower in np.arange(0,1,.1):
                    member=(prob>=lower)&(prob<lower+.1+1e-7)
                    if not member.any():continue
                    mass=float(w[member].sum());pred=float((w[member]*prob[member]).sum()/w[member].sum())
                    actual=float((w[member]*truth[member]).sum()/w[member].sum())
                    ece+=mass*abs(pred-actual);bins.append(dict(lower=float(lower),predicted=pred,observed=actual,mass=mass))
                horizons[k]=dict(n=int(known.sum()),brier=brier,ece=ece,
                    auc_failure=auc((1-truth).numpy(),(1-prob).numpy()),bins=bins)
            trials.append(dict(seed=seed,nll=nll,horizons=horizons,selected=seed==selected[group]['seed']))
        results[group]=trials
    write(out/'calibration_test.json',results)
    rng=np.random.default_rng(91);indices=rng.integers(0,len(docs),(5000,len(docs)))
    contrasts={}
    for a,b in [('attention','structural'),('attention','confidence'),('attention_read','structural'),
                ('attention_read','confidence'),('combined','confidence')]:
        row={}
        for metric,numerator,denominator in [('nll',0,1),('k4_brier',2,3)]:
            x,z=paired_inputs[a],paired_inputs[b]
            xx,zz=x[indices],z[indices]
            samples=xx[...,numerator].sum(-1)/xx[...,denominator].sum(-1)-zz[...,numerator].sum(-1)/zz[...,denominator].sum(-1)
            delta=x[:,numerator].sum()/x[:,denominator].sum()-z[:,numerator].sum()/z[:,denominator].sum()
            row[metric]=dict(estimate=float(delta),ci95=np.quantile(samples,[.025,.975]).tolist(),lower_is_better=True)
        contrasts[f'{a} vs {b}']=row
    write(out/'calibration_comparisons.json',contrasts)
    return results


def completeness(out,groups):
    protocol=json.loads((out/'protocol.json').read_text())
    manifest=protocol['manifest']
    counts=manifest['counts'];methods=json.loads((out/'test_methods.json').read_text())
    runtime=read_rows(out/'runtime_test.jsonl')
    keys=[(r['source_index'],r['method']) for r in runtime]
    assert len(keys)==len(set(keys))
    for name in ['teacher_greedy']+list(methods):assert len(groups[name])==counts['test'],name
    assert all(r['output_agrees'] for r in runtime)
    all_titles=set();all_questions=set();data=Path(protocol['data'])
    old_titles=set();old_questions=set()
    for path in manifest['previous_hashes']:
        for item in read_rows(ROOT/path):
            old_questions.add(item.get('question',''))
            title_list=item.get('titles',[])
            if isinstance(title_list,str):
                try:title_list=json.loads(title_list)
                except json.JSONDecodeError:title_list=[]
            old_titles.update(title_list)
            old_titles.update(re.findall(r'\[Doc\s+\d+\]\s*\n([^\n]+)',item.get('prompt','')))
    for split in counts:
        path=data/f'{split}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['hashes'][split]
        items=read_rows(path);assert len(items)==counts[split]
        for item in items:
            titles=set(item['titles']);question=item['question']
            assert not titles&all_titles and not titles&old_titles
            assert question not in all_questions and question not in old_questions
            assert 600<=len(item['ids'])<=2048
            all_titles.update(titles);all_questions.add(question)
    for a,b in [('O_teacher_k4','A_teacher'),('O_native_full','A_native_full')]:
        aa={r['source_index']:r for r in groups[a]};bb={r['source_index']:r for r in groups[b]}
        for key in aa:
            assert aa[key]['generated']==bb[key]['generated']
            assert [(r['proposed'],r['accepted']) for r in aa[key]['trace']]==[(r['proposed'],r['accepted']) for r in bb[key]['trace']]
    for split in ('train','validation','test'):
        items=read_rows(out/f'offline_{split}.jsonl')
        assert len({r['source_index'] for r in items})==counts[split]
        branches=read_rows(out/f'branches_{split}.jsonl')
        assert len({r['source_index'] for r in branches})==counts[split]
        for r in branches:
            assert len(r['labels'])==len(r['mask'])==12
            assert 0<=r['accepted']<=r['proposed']<=12
            for f in r['frames']:
                assert len(f['features'])==31 and np.isfinite(f['features']).all()
    phase_output_counts={}
    for path in out.glob('*validation.jsonl'):
        if path.name.startswith(('offline_','branches_')):continue
        rows=read_rows(path)
        assert all(r['output_agrees'] for r in rows),path
        phase_output_counts[path.stem]=len(rows)
    for split in ('train','validation','test'):
        rows=read_rows(out/f'collection_{split}.jsonl')
        assert all(r['output_agrees'] for r in rows)
        phase_output_counts[f'collection_{split}']=len(rows)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'exp/attention_horizon').glob('*.py')}
    for path in ('exp/attention_selection/attention.py','exp/attn_adapter_transfer/run_attention_oracle.py',
                 'exp/attn_adapter_transfer/common.py','exp/selective_kv/bridge.py'):
        p=ROOT/path
        if p.exists():hashes[path]=hashlib.sha256(p.read_bytes()).hexdigest()
    from checks import check
    result=dict(runtime_records=len(runtime),sd_runs=len(runtime)-counts['test'],
                output_matches=sum(r['output_agrees'] for r in runtime if r['method']!='teacher_greedy'),
                test_methods=len(methods),fresh_counts=counts,duplicate_runtime_keys=False,
                current_source_sha256=hashes,
                terminal_and_budget_checks=check(),lean_probe_decisions_identical=True,
                environment=dict(torch=torch.__version__,cuda=torch.version.cuda),
                split_hashes_match=True,questions_and_titles_disjoint=True,input_length_range=[600,2048],
                phase_output_counts=phase_output_counts,
                offline_records={s:len(read_rows(out/f'offline_{s}.jsonl')) for s in counts},
                branch_records={s:len(read_rows(out/f'branches_{s}.jsonl')) for s in counts})
    write(out/'completeness.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='exp/attention_horizon/out')
    p.add_argument('--update-progress',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    out=Path(args.output);runtime=read_rows(out/'runtime_test.jsonl')
    groups=defaultdict(list)
    for r in runtime:groups[r['method']].append(r)
    aggregate={k:pooled(v) for k,v in groups.items()}
    comparisons={}
    pairs=[('A_read_teacher','A_teacher'),('A_read_joint025','A_teacher'),('A_read_joint1','A_teacher'),
           ('A_union','A_teacher'),('B_attention','B_best_static'),('B_attention','B_confidence'),
           ('A_read_joint025','A_union'),('A_read_joint1','A_union'),
           ('A_read_joint025','A_read_teacher'),('A_read_joint1','A_read_teacher'),
           ('B_attention_read','B_best_static'),('B_combined','B_best_static'),
           ('B_confidence','B_best_static'),('B_structural','B_best_static'),
           ('B_attention_read','B_attention'),('B_combined','B_confidence'),('B_attention_online','B_attention'),
           ('C_sequential','C_jointworkset_static'),('C_joint','C_sequential'),
           ('C_sequential_V','C_sequential'),('C_joint_V','C_joint'),
           ('B_attention','O_teacher_static'),('B_attention_read','O_teacher_static'),
           ('B_combined','O_teacher_static'),('O_teacher_k4','A_teacher'),
           ('B_attention','O_history'),('B_combined','O_history'),
           ('B_attention','O_confidence'),('B_combined','O_confidence')]
    for a,b in pairs:
        comparisons[f'{a} vs {b}']={m:paired(groups[a],groups[b],m) for m in ('acceptance','decode','total')}
    offline=defaultdict(list)
    for r in read_rows(out/'offline_test.jsonl'):offline[r['rule']].append(r)
    offline_summary={k:{metric:float(np.mean([r[metric] for r in rows])) for metric in
        ('teacher_read_error','student_read_error','kl','teacher_ce','match','planning_seconds')} for k,rows in offline.items()}
    cal=calibration(out);complete=completeness(out,groups)
    cal_comparisons=json.loads((out/'calibration_comparisons.json').read_text())
    answers=answer_scores(out,groups)
    selected=json.loads((out/'selected.json').read_text())
    interaction=factorial(groups,selected)
    proxy=proxy_comparisons(read_rows(out/'offline_test.jsonl'))
    decision_counts={method:dict(worksets=dict(Counter(
        b['rule'] for r in rows for b in r.get('trace',[]))),
        draft_lengths=dict(Counter(b['proposed'] for r in rows for b in r.get('trace',[]))))
        for method,rows in groups.items() if method!='teacher_greedy'}
    summary=dict(runtime=aggregate,comparisons=comparisons,offline=offline_summary,selected=selected,
                 factorial_interaction=interaction,offline_comparisons=proxy,decision_counts=decision_counts,
                 completeness=complete,calibration=cal,calibration_comparisons=cal_comparisons,answer_quality=answers)
    write(out/'summary.json',summary)
    lines=['# 读取保持选择、注意力长度与联合决策：完整实验报告','',
        '新采集 48/16/30 篇训练/验证/测试文档，标题互斥并排除旧实验文档。Qwen2-7B → 1.5B；FP32 teacher / BF16 student；H100；预算 512；最多输出 64 token。所有测试选择只用验证集，teacher 保持完整上下文。使用贪心精确推测验证，接受连续 teacher argmax 一致的草稿；不是随机采样 min(1,p/q) 的接受概率实验。','',
        f'验证选中的共同选择：`{selected["joint_rule"]}`；teacher 最佳固定长度：{selected["best_static"]["teacher"]}；主线注意力策略：`{selected["attention_policy"]}`。','',
        '## A：等预算、固定 k=4','',
        '| 方法 | T 读取误差 | S 读取误差 | 输出 KL | 接受率 | Decode TPS | 总 TPS |','|---|---:|---:|---:|---:|---:|---:|']
    for rule in ('teacher','student','union','read_teacher','read_joint025','read_joint1','native_full'):
        r=aggregate['A_'+rule];o=offline_summary[rule]
        lines.append(f'| {rule} | {o["teacher_read_error"]:.4f} | {o["student_read_error"]:.4f} | {o["kl"]:.4f} | {100*r["acceptance_rate"]:.2f}% | {r["decode_tps"]:.2f} | {r["total_tps"]:.2f} |')
    lines+=['','读取误差是每个模型自身的 A·V 读出经筛选后重归一化的相对平方误差，再按 query head 平均；不直接比较两个模型的 V。选择使用上一个已提交 query，离线输出测量当前 query，因此代理改善并不保证输出改善。候选采用有限四次交换，不是全局最优子集。','',
        '## B/C：真实解码','',
        '| 方法 | 接受率 | 每轮接受 | 每轮草稿 | 验证次数 | Decode TPS | 总 TPS |','|---|---:|---:|---:|---:|---:|---:|']
    order=['teacher_greedy','B_best_static','B_structural','B_confidence','B_attention',
        'B_attention_read','B_combined','B_attention_online','C_jointworkset_static',
        'C_sequential','C_joint','C_sequential_V','C_joint_V','O_teacher_k4',
        'O_teacher_static','O_history','O_confidence','O_native_full']
    for k in order:
        r=aggregate[k]
        if k=='teacher_greedy':
            lines.append(f'| {k} | — | — | — | — | {r["decode_tps"]:.2f} | {r["total_tps"]:.2f} |')
            continue
        lines.append(f'| {k} | {r["acceptance_rate"]*100:.2f}% | {r["mean_accepted"]:.3f} | {r["mean_draft"]:.3f} | {r["blocks"]} | {r["decode_tps"]:.2f} | {r["total_tps"]:.2f} |')
    lines+=['',f'四因素对照的共同基准固定 k=4，对应方法为 `{interaction["methods"]}`。每输出 token 时间的交互项 Δ11−Δ10−Δ01+Δ00 为 {interaction["estimate"]:+.3f} ms，95% CI [{interaction["ci95"][0]:+.3f}, {interaction["ci95"][1]:+.3f}]；负值表示超出简单相加的时间节省，区间跨零则不能证明非加性收益。联合策略相对顺序策略是另一项独立对照，见下表。']
    lines+=['','## 配对 95% 区间','',
        '按文档配对 bootstrap 5,000 次；接受率单位为百分点，TPS 为相对百分比。未作多重比较校正。','',
        '| 比较 | 接受率增量 [95% CI] | Decode TPS 变化 [95% CI] |','|---|---:|---:|']
    for name,c in comparisons.items():
        ac,sp=c['acceptance'],c['decode']
        lines.append(f'| {name} | {ac["estimate"]:+.2f} [{ac["ci95"][0]:+.2f}, {ac["ci95"][1]:+.2f}] | {sp["estimate"]:+.2f}% [{sp["ci95"][0]:+.2f}%, {sp["ci95"][1]:+.2f}%] |')
    lines+=['','## 长度预测与成本','',
        '| 特征 | 选中种子 | 测试边界 NLL | k=4 Brier | k=4 拒绝/终止 AUC |','|---|---:|---:|---:|---:|']
    for group,trials in cal.items():
        r=next(t for t in trials if t['selected']);h=r['horizons'][4]
        av='NA' if h['auc_failure'] is None else f'{h["auc_failure"]:.3f}'
        lines.append(f'| {group} | {r["seed"]} | {r["nll"]:.4f} | {h["brier"]:.4f} | {av} |')
    lines+=['','按文档配对的校准差异（负值为改善，95% CI；未作多重比较校正）：','',
        '| 对照 | NLL 差 | k=4 Brier 差 |','|---|---:|---:|']
    for name,row in cal_comparisons.items():
        n,b=row['nll'],row['k4_brier']
        lines.append(f'| {name} | {n["estimate"]:+.4f} [{n["ci95"][0]:+.4f}, {n["ci95"][1]:+.4f}] | {b["estimate"]:+.4f} [{b["ci95"][0]:+.4f}, {b["ci95"][1]:+.4f}] |')
    lines+=['','每组预测器训练三个种子，在验证集选择 checkpoint、温度和种子。测试校准使用独立最大长度轨迹，只统计块边界，按文档加权；AUC 是非加权描述统计。各位置分继续接受、接受 EOS、拒绝三类；EOS 与拒绝分别标注，预测可继续性时都结束本轮。未观察到的尾部右删失，不标成拒绝。','',
        '结构特征包括长度、预算比例和参考年龄；confidence 在结构基础上加 student 熵、最大概率、top-2 margin；attention 加注意力覆盖、分歧、熵和分布位置；attention_read 另加本模型读取保持误差；combined 合并注意力、读取与置信度。所有组用相同数据、训练程序及三个种子。','',
        '代价模型仅拟合训练测量，预期每轮提交数为 1 + Σ(j=1…min(k,剩余预算−1)) P(前 j 个均接受且非 EOS)。截断的是生存概率的求和范围；预算用完时不再计入 bonus 或补算 student。联合决策在 teacher/read-teacher/所选共同集合与有限长度菜单中评分，付出所有集合构造和评分成本；这是受限候选决策，不能宣称全局最优。在线注意力控制只使用已生成 student 观测与陈旧 teacher 参考，不能把它解释成未调用 teacher 得到的当前 query 真值。','',
        '`O_*` 是额外的实际成本对照：关闭不需要的 student full-key shadow 和多层注意力捕获。固定长度及历史接受概率更新率在验证集选择；历史策略仅用训练先验及当前文档已经完成的 teacher 验证，不取当前未验证真值。它们仍是本仓库 eager 实现，不能代表专用推测解码引擎的极限速度。','',
        '## 完整性与限制','',
        f'{complete["sd_runs"]}/{complete["sd_runs"]} 次实际测试 SD 精确匹配各自 FP32 teacher 参考；离线、最大分支、计时记录见 completeness.json。原生完整读取加诊断探针 logits 完全一致，GQA 手算、交换误差重算、缓存不被探针修改及稀疏/完整回滚通过。','',
        '这是单模型对、约 2K 上下文的机制预实验。基础模型和 V mapper 的训练种子未扩展，预测器三个种子不等于整个系统三次独立训练。完整 backing KV 仍保留，因此不是物理 KV 存储压缩；full-key shadow 读取和 Python eager 执行均计入实际时延。冻结 V 是跨新数据集的集成检查，未与新选择/长度策略联合训练。接受率受长度与 EOS 影响，判断长度策略以吞吐、验证次数和精确输出为主。','',
        '若新策略未超过对应基线，结果只支持发现瓶颈，不证明论文假说。长上下文、不同模型层数、原版 SpecExtend 和优化部署尚未在本轮执行；依照事先协议，只有机制与实测成本出现可用信号才进入这些扩展。','']
    main_policy='B_attention_online' if selected['sequential_config'].get('online') else 'B_'+selected['sequential_config']['risk']
    static=aggregate['B_best_static'];dynamic=aggregate[main_policy]
    seq,joint=aggregate['C_sequential'],aggregate['C_joint']
    def milliseconds(row,names):
        return sum(row['stage_seconds'][name] for name in names)/row['tokens']*1000
    lines+=['## 结果解释','',
        f'所选动态路线 `{main_policy}` 的验证次数为 {dynamic["blocks"]}，最佳固定长度为 {static["blocks"]}，减少 {(1-dynamic["blocks"]/static["blocks"])*100:.2f}%；但每输出 token 的 draft 时间由 {milliseconds(static,["draft"]):.2f} 增至 {milliseconds(dynamic,["draft"]):.2f} ms。少验证不等于少总计算，必须与额外草稿共同判断。','',
        f'联合与顺序策略的选择＋长度决策开销分别为 {milliseconds(joint,["selection","decision"]):.2f}/{milliseconds(seq,["selection","decision"]):.2f} ms/输出 token，验证次数分别为 {joint["blocks"]}/{seq["blocks"]}。联合策略确实会切换候选集合；决策生效不能替代净吞吐检验。','',
        '读取保持是局部目标：这里保持上一个 query 的最后一层读出，而实际下一 token 会经过全部层并改变 query。小幅降低这个目标，不能保证更接近 teacher 的输出分布。若本轮实际接受率未提升，应保留该负结果，不能把读出误差的下降当成主张已经成立。','',
        '低开销 O 对照在主测试之后单独执行，内部仍按文档随机顺序计时。跨轮次的微小时差可能受系统漂移影响；相同诊断配置的 B 对照用于隔离信息价值，O 对照用于补充实际成本判断。','',
        f'参考输出的描述性 QA 分数：归一化 EM {answers["exact_match"]*100:.2f}%，token F1 {answers["token_f1"]*100:.2f}%；{answers["length_capped"]}/{answers["documents"]} 条达到输出上限。所有精确 SD 继承相同分数，这项实验改变提议效率，不提升最终 teacher 答案质量。完整输出见 answer_quality.json。','']
    (out/'report.md').write_text('\n'.join(lines))
    def change(name,metric):
        c=comparisons[name][metric]
        unit='个百分点' if metric=='acceptance' else '%'
        return f'{c["estimate"]:+.2f}{unit}，95% CI [{c["ci95"][0]:+.2f}, {c["ci95"][1]:+.2f}]'
    joint=selected['joint_rule']
    primary=paired(groups['A_'+joint],groups['A_teacher'],'acceptance')
    a_verdict='接受率有正向证据' if primary['ci95'][0]>0 else '接受率下降' if primary['ci95'][1]<0 else '未证明接受率提升'
    main_length='B_attention_online' if selected['sequential_config'].get('online') else 'B_'+selected['sequential_config']['risk']
    main_comparison={m:paired(groups[main_length],groups['O_teacher_static'],m) for m in ('acceptance','decode','total')}
    write(out/'primary_horizon_comparison.json',dict(method=main_length,comparison=main_comparison))
    b=main_comparison['decode']
    b_verdict='注意力长度策略相对低开销固定长度有吞吐收益' if b['ci95'][0]>0 else '注意力长度策略未超过低开销固定长度对照'
    c=comparisons['C_joint vs C_sequential']['decode']
    c_verdict='联合决策相对顺序决策有吞吐收益' if c['ci95'][0]>0 else '联合决策未证明优于顺序决策'
    best_sd=max((k for k in aggregate if k!='teacher_greedy'),key=lambda k:aggregate[k]['decode_tps'])
    concise=['# 三组实验结论','',
        f'1. **读取保持共同选择：{a_verdict}。** 验证选择 `{joint}`；相对 teacher-token 接受率 {primary["estimate"]:+.2f} 个百分点，95% CI [{primary["ci95"][0]:+.2f}, {primary["ci95"][1]:+.2f}]。读出误差和输出质量需分别判断。',
        f'2. **动态长度：{b_verdict}。** 所选主线 `{main_length}` 相对低开销固定长度 decode TPS {b["estimate"]:+.2f}%，95% CI [{b["ci95"][0]:+.2f}, {b["ci95"][1]:+.2f}]。纯注意力相对置信度 {change("B_attention vs B_confidence","decode")}。额外报告读取误差、置信度联合和逐 token 停止控制，见主报告。',
        f'3. **选择与长度联合：{c_verdict}。** 相对顺序决策 decode TPS {change("C_joint vs C_sequential","decode")}。冻结 V 集成相对顺序策略 {change("C_sequential_V vs C_sequential","decode")}，不将旧 mapper 的收益自动归因于新模块。','',
        f'当前 eager 实现：teacher 单独生成 {aggregate["teacher_greedy"]["decode_tps"]:.2f} decode TPS，最快 SD `{best_sd}` 为 {aggregate[best_sd]["decode_tps"]:.2f}。模块间比较不自动等于相对 teacher 的加速。','',
        f'全部 {complete["sd_runs"]}/{complete["sd_runs"]} 次实际测试 SD 匹配 FP32 teacher 参考。完整表格、逐文档记录、三个预测种子和配对区间均已保留。','',
        '本轮为单模型对、约 2K 的机制验证。三个预测器种子不代表基础模型或 V mapper 的多种子复现；长文本及部署效率仍需后续独立验证。','']
    (out/'conclusions.md').write_text('\n'.join(concise))
    if args.update_progress:
        path=ROOT/'实验进度.md'
        start='<!-- attention-horizon-results:start -->';end='<!-- attention-horizon-results:end -->'
        block='\n'.join([start,'##### 10. 读取保持共同选择、注意力动态长度与联合决策（2026-10-14）','',
            '三组核心机制实验及低开销对照已完成。48/16/30 篇全新文档，排除旧标题并跨划分互斥；Qwen2-7B → 1.5B、FP32 teacher / BF16 student、H100，预算 512、最大输出 64。固定长度菜单 1/2/4/8/12；预测器每组训练三个种子，验证选择，基础模型和旧 V mapper 冻结。','',
            *concise[2:5],'',
            f'主测试 {complete["sd_runs"]}/{complete["sd_runs"]} 次 SD 与大模型参考一致；另有独立最大分支校准。输入哈希、标题互斥、GQA、EOS/删失、预算末尾 bonus 与缓存回滚核验通过。全部配置、分阶段耗时、配对区间和源码哈希均保存。','',
            '单模型对、约 2K 的机制预实验；尚未扩展到 8K/16K 或不同层数模型。完整 backing KV 保留；full-key 学生探针的成本已计入。低开销 O 对照单独计时，细小时差需考虑运行批次影响。三组预测种子不代表整个系统的多种子复现。','',
            '实现与复现：[README](exp/attention_horizon/README.md)；[结论](exp/attention_horizon/out/conclusions.md)；[完整报告](exp/attention_horizon/out/report.md)；[逐项汇总](exp/attention_horizon/out/summary.json)；[完整性](exp/attention_horizon/out/completeness.json)。',end])
        text=path.read_text()
        if start in text:
            text=text[:text.index(start)]+block+text[text.index(end)+len(end):]
        else:text+='\n\n'+block+'\n'
        path.write_text(text)
    print(json.dumps(complete,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
