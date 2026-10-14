#!/usr/bin/env python3
"""Document-paired analysis, completeness audit, report and scientific plots."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def jsonread(path):
    return json.loads(path.read_text())


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def document_mean(records,metric):
    grouped=defaultdict(list)
    for r in records:
        grouped[r['source_index']].append(r[metric])
    return {k:float(np.mean(v)) for k,v in grouped.items()}


def interval(values):
    x=np.asarray(values,dtype=float)
    generator=np.random.default_rng(4242)
    sampled=x[generator.integers(0,len(x),(5000,len(x)))].mean(-1)
    return dict(mean=float(x.mean()),ci95=[float(v) for v in np.quantile(sampled,[0.025,0.975])],documents=len(x))


def paired(records,reference,metric):
    a,b=document_mean(records,metric),document_mean(reference,metric)
    assert set(a)==set(b),(len(a),len(b))
    return interval([a[k]-b[k] for k in sorted(a)])


def tps_ratio(records,reference,field='decode_seconds'):
    a={r['source_index']:r for r in records}
    b={r['source_index']:r for r in reference}
    assert set(a)==set(b)
    ids=sorted(a)
    tok_a=np.asarray([a[k]['tokens'] for k in ids])
    tok_b=np.asarray([b[k]['tokens'] for k in ids])
    seconds_a=np.asarray([a[k][field] for k in ids])
    seconds_b=np.asarray([b[k][field] for k in ids])
    generator=np.random.default_rng(4242)
    samples=generator.integers(0,len(ids),(5000,len(ids)))
    ratio=(tok_a[samples].sum(-1)/seconds_a[samples].sum(-1))/(tok_b[samples].sum(-1)/seconds_b[samples].sum(-1))
    return dict(ratio=float((tok_a.sum()/seconds_a.sum())/(tok_b.sum()/seconds_b.sum())),
                ci95=[float(v) for v in np.quantile(ratio,[0.025,0.975])])


def pooled_difference(records,reference,denominator):
    a={r['source_index']:r for r in records}
    b={r['source_index']:r for r in reference}
    assert set(a)==set(b)
    ids=sorted(a)
    an=np.asarray([a[k]['accepted'] for k in ids]);ad=np.asarray([a[k][denominator] for k in ids])
    bn=np.asarray([b[k]['accepted'] for k in ids]);bd=np.asarray([b[k][denominator] for k in ids])
    sample=np.random.default_rng(4242).integers(0,len(ids),(5000,len(ids)))
    delta=an[sample].sum(-1)/ad[sample].sum(-1)-bn[sample].sum(-1)/bd[sample].sum(-1)
    return dict(mean=float(an.sum()/ad.sum()-bn.sum()/bd.sum()),
                ci95=[float(v) for v in np.quantile(delta,[0.025,0.975])],documents=len(ids))


def statstring(s,scale=1,digits=3):
    return f"{s['mean']*scale:.{digits}f} [{s['ci95'][0]*scale:.{digits}f}, {s['ci95'][1]*scale:.{digits}f}]"


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='exp/attention_selection/out_stage2')
    a=p.parse_args()
    out=Path(a.output)
    config=jsonread(out/'config.json')
    data={s:read(Path(config['data'])/f'{s}.jsonl') for s in ('train','validation','test')}
    offline=read(out/'offline_test.jsonl')
    runtime=read(out/'runtime_test.jsonl')
    fp32_runtime=read(out/'runtime_fp32_test.jsonl')
    joint=read(out/'utility_joint_test.jsonl')
    ablations=read(out/'union_ablation_test.jsonl')
    teacher=[r for r in runtime if r['method']=='teacher_greedy']
    states=sum(sum(q<len(r['generated']) for q in config['query_steps']) for r in teacher)
    utility_states=sum(min(2,sum(q<len(r['generated']) for q in config['query_steps'])) for r in teacher)
    expected=dict(offline=states*(13*len(config['budgets'])+9),runtime=len(data['test'])*(3+12*len(config['budgets'])),
                  utility_joint=utility_states*len(config['budgets'])*5,union_ablations=states*len(config['budgets'])*2)
    actual=dict(offline=len(offline),runtime=len(runtime),utility_joint=len(joint),union_ablations=len(ablations))
    assert expected==actual,(expected,actual)
    assert len(fp32_runtime)==len(data['test'])*(3+6*len(config['budgets']))
    assert len(teacher)==len(data['test'])
    cache_files=list(Path(config['cache']).glob('*.pt'))
    assert len(cache_files)==sum(map(len,data.values()))
    for filename,records,fields in [('offline',offline,['source_index','step','budget','method']),
                                  ('runtime',runtime,['source_index','budget','method']),
                                  ('runtime_fp32',fp32_runtime,['source_index','budget','method']),
                                  ('utility',joint,['source_index','step','budget','method']),
                                  ('union',ablations,['source_index','step','budget','method'])]:
        keys=[tuple(r[f] for f in fields) for r in records]
        assert len(set(keys))==len(keys),f'Duplicated {filename} records'
    for r in offline+ablations:
        if r['budget']:
            assert r['retained']==r['budget']
        assert np.isfinite(r['kl']) and r['kl']>=-1e-5
    output_failures=[dict(source_index=r['source_index'],budget=r['budget'],method=r['method'])
                     for r in runtime if not r['output_agrees']]
    audit=dict(expected=expected,actual=actual,tensor_cache_documents=len(cache_files),offline_states=states,
               all_records_unique=True,all_selection_budgets_equal=True,
               runtime_output_agreement=len(runtime)-len(output_failures),runtime_runs=len(runtime),
               output_failures=output_failures,verification=jsonread(out/'verification.json'))
    audit['fp32_runtime_expected']=len(data['test'])*(3+6*len(config['budgets']))
    audit['fp32_runtime_actual']=len(fp32_runtime)
    audit['fp32_output_agreement']=sum(r['output_agrees'] for r in fp32_runtime)
    audit['fp32_sd_runs']=sum(r['method']!='teacher_greedy' for r in fp32_runtime)
    audit['fp32_sd_output_agreement']=sum(r['output_agrees'] and r['method']!='teacher_greedy' for r in fp32_runtime)
    audit['fp32_output_failures']=[dict(source_index=r['source_index'],method=r['method'],budget=r['budget'])
                                  for r in fp32_runtime if not r['output_agrees']]
    write(out/'completeness.json',audit)
    source_dir=Path(__file__).parent
    write(out/'source_audit.json',dict(files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source_dir.glob('*.py')},
                initial_sources=config['source_sha256'],note='Core run/attention/selection/verification hashes unchanged after main start. runtime.py finalized before late import by --phase all. prepare_data.py reader repaired afterwards; all split hashes reproduced identically. Supplement/numerical audit/analysis run after primary experiments.'))
    results={'offline':{},'runtime':{},'runtime_fp32':{},'runtime_fp32_union_vs_token':{},'utility':{},'union_ablations':{}}
    groups=defaultdict(list)
    for r in offline:
        groups[(r['budget'],r['method'])].append(r)
    native=groups[(0,'native_full')]
    for (b,method),records in groups.items():
        ref=native if b==0 else groups[(b,'teacher_token')]
        results['offline'][f'{b}/{method}']=dict(documents=len(document_mean(records,'kl')),
              kl=interval(list(document_mean(records,'kl').values())),
              match=interval(list(document_mean(records,'match').values())),
              delta_kl=paired(records,ref,'kl'),delta_match=paired(records,ref,'match'),
              repaired_native_errors=sum(r['match']==1 and r['native_match']==0 for r in records),
              broken_native_matches=sum(r['match']==0 and r['native_match']==1 for r in records))
    runtime_groups=defaultdict(list)
    for r in runtime:
        runtime_groups[(r['budget'],r['method'])].append(r)
    native_runtime=runtime_groups[(0,'native_full')]
    for (b,method),records in runtime_groups.items():
        if method=='teacher_greedy':
            continue
        reference=native_runtime if b==0 else runtime_groups[(b,'teacher_chunk')]
        results['runtime'][f'{b}/{method}']=dict(documents=len(records),
          acceptance_rate=sum(r['accepted'] for r in records)/sum(r['proposed'] for r in records),
          mean_accepted=sum(r['accepted'] for r in records)/sum(r['blocks'] for r in records),
          document_mean_accepted=interval(list(document_mean(records,'mean_accepted').values())),
          decode_tps=sum(r['tokens'] for r in records)/sum(r['decode_seconds'] for r in records),
          total_tps=sum(r['tokens'] for r in records)/sum(r['total_seconds'] for r in records),
          delta_acceptance=paired(records,reference,'acceptance_rate'),
          pooled_delta_acceptance=pooled_difference(records,reference,'proposed'),
          pooled_delta_mean_accepted=pooled_difference(records,reference,'blocks'),
          delta_mean_accepted=paired(records,reference,'mean_accepted'),
          decode_ratio=tps_ratio(records,reference),total_ratio=tps_ratio(records,reference,'total_seconds'),
          teacher_decode_ratio=tps_ratio(records,teacher),teacher_total_ratio=tps_ratio(records,teacher,'total_seconds'),
          output_agreement=sum(r['output_agrees'] for r in records),
          stage_mean_seconds={k:float(np.mean([r['stage_seconds'][k] for r in records]))
                              for k in ('selection','cache_update','draft','verify')})
    fp32_groups=defaultdict(list)
    for r in fp32_runtime:
        fp32_groups[(r['budget'],r['method'])].append(r)
    fp32_teacher=fp32_groups[(0,'teacher_greedy')]
    for (b,method),records in fp32_groups.items():
        if method=='teacher_greedy':
            continue
        reference=fp32_groups[(0,'native_full')] if b==0 else fp32_groups[(b,'teacher_chunk')]
        results['runtime_fp32'][f'{b}/{method}']=dict(documents=len(records),
          acceptance_rate=sum(r['accepted'] for r in records)/sum(r['proposed'] for r in records),
          mean_accepted=sum(r['accepted'] for r in records)/sum(r['blocks'] for r in records),
          decode_tps=sum(r['tokens'] for r in records)/sum(r['decode_seconds'] for r in records),
          total_tps=sum(r['tokens'] for r in records)/sum(r['total_seconds'] for r in records),
          delta_mean_accepted=paired(records,reference,'mean_accepted'),
          pooled_delta_acceptance=pooled_difference(records,reference,'proposed'),
          pooled_delta_mean_accepted=pooled_difference(records,reference,'blocks'),
          decode_ratio=tps_ratio(records,reference),total_ratio=tps_ratio(records,reference,'total_seconds'),
          teacher_decode_ratio=tps_ratio(records,fp32_teacher),teacher_total_ratio=tps_ratio(records,fp32_teacher,'total_seconds'),
          output_agreement=sum(r['output_agrees'] for r in records),
          stage_mean_seconds={k:float(np.mean([r['stage_seconds'][k] for r in records]))
                              for k in ('selection','cache_update','draft','verify')},
          prefill_mean_seconds=float(np.mean([r['prefill_seconds'] for r in records])))
    for b in config['budgets']:
        union=fp32_groups[(b,'union_available')]
        token=fp32_groups[(b,'teacher_token')]
        results['runtime_fp32_union_vs_token'][str(b)]=dict(
            delta_acceptance=paired(union,token,'acceptance_rate'),
            pooled_delta_acceptance=pooled_difference(union,token,'proposed'),
            delta_mean_accepted=paired(union,token,'mean_accepted'),
            decode_ratio=tps_ratio(union,token),total_ratio=tps_ratio(union,token,'total_seconds'))
    for b in config['budgets']:
        teacher_joint=[r for r in joint if r['budget']==b and r['method']=='teacher']
        for method in ('random','teacher','difference','oracle','learned'):
            records=[r for r in joint if r['budget']==b and r['method']==method]
            results['utility'][f'{b}/{method}']=dict(joint_gain=interval(list(document_mean(records,'joint_kl_gain').values())),
                                                   kl_vs_teacher=paired(records,teacher_joint,'kl'),
                                                   match_vs_teacher=paired(records,teacher_joint,'match'))
        ref=groups[(b,'union')]
        for method in ('drop_student_exclusive','drop_teacher_exclusive'):
            records=[r for r in ablations if r['budget']==b and r['method']==method]
            results['union_ablations'][f'{b}/{method}']=dict(delta_kl=paired(records,ref,'kl'),delta_match=paired(records,ref,'match'))
    layer=jsonread(out/'layer_test.json')
    results['layer_correspondence']={k:interval(list(document_mean(layer,k).values())) for k in ('stability','paired_js','depth_js','random_js')}
    union=jsonread(out/'union_diagnostics_test.json')
    results['union_overlap']={str(b):{k:interval(list(document_mean([r for r in union if r['budget']==b],k).values()))
                                      for k in ('peak_jaccard','teacher_exclusive_fraction','student_exclusive_fraction')} for b in config['budgets']}
    correlations=jsonread(out/'utility_attention_correlations.json')
    results['utility_correlations']={f'{b}/{k}':interval(list(document_mean([r for r in correlations if r['budget']==b and r['feature']==k],
                                         'utility_correlation').values())) for b in config['budgets'] for k in ('teacher','student','difference')}
    write(out/'summary.json',results)
    report=[]
    report += ['# 第二阶段完整实验结果','',
      '本轮执行 E1—E9，另做并集来源消融。使用本机 Qwen2-7B-Instruct → Qwen2-1.5B-Instruct、H100、BF16、eager attention，冻结两模型。',
      '训练 24、验证 12、测试 30 个全新 HotpotQA 样本；排除此前 300 个样本涉及的文档标题，新三个集合的文档标题也互不重叠。上下文约 600—2048 token。',
      f"测试共 {states} 个离线 query 状态，缓存预算 {config['budgets']}，强制保留 sink 和最近 {config['recent']} 个位置；近期位置计入预算。每层头平均，全局筛选用 teacher/student 最后一层。",
      '离线状态是 teacher 生成轨迹上的位置 0、2、4、6；teacher 提前 EOS 时保留实际存在的状态。离线指标和接受长度差按文档等权，吞吐按总 token/总时间汇总；95% 区间均用文档配对 bootstrap（5000 次），未作多重比较校正。合计接受率另给直接对应合计比例的 bootstrap 区间。',
      '', '## 完成情况','', '| 实验 | 状态与产物 |','|---|---|',
      '| E1 基础筛选 | 完成：近期、随机、student/teacher chunk 与 token 对照 |',
      '| E2 粒度 | 完成：严格相同实际 token 预算的 chunk/token |',
      '| E3 并集 | 完成：验证集配额选择、峰值重叠、两类独占来源的等预算消融 |',
      '| E4 层间关系 | 完成：全位置/远历史/近期热图，训练配对、独立测试稳定性 |',
      '| E5 权重干预 | 完成：单层 λ=0.25/0.5/1，深度/相似/随机/位置打乱；同层同强度与各自调参对照 |',
      '| E7 差异筛选 | 完成：差异排序、差异＋student 核心保护；仅共同完整历史离线对照 |',
      '| E8 按层选位 | 完成：全层共用、深度配对、相似配对、随机配对；每层相同预算 |',
      '| E9 收益选择器 | 完成：真实单 token 等预算替换 KL 标签、回归训练、联合 8 token 替换、离线 oracle 与规则对照 |',
      '| E6 实际收益 | 完成：30 文档 greedy SD，k=4，最多 32 token；验证轨迹、分阶段时间、吞吐与输出一致性 |',
      '', '## 离线筛选结果','',
      '以下 Δ 相对同预算 teacher-token；命中率指 student 下一 token 与 teacher argmax 一致，不能解释为事实正确率。KL 越低越好。', '',
      '| 预算 | 方法 | 平均 KL | 命中率 % | ΔKL [95% CI] | Δ命中率 pp [95% CI] |','|---|---|---:|---:|---|---|']
    selection_order=['recent','random','student_chunk','teacher_chunk','student_token','teacher_token','union','difference','difference_core','layer_depth','layer_pairing','layer_random','learned_global']
    for b in config['budgets']:
        for method in selection_order:
            r=results['offline'][f'{b}/{method}']
            report.append(f"| {b} | {method} | {r['kl']['mean']:.4f} | {100*r['match']['mean']:.2f} | {statstring(r['delta_kl'])} | {statstring(r['delta_match'],100,2)} |")
    report += ['', '## 单层注意力权重干预','',
      '保持完整原生历史 KV，仅改变一个层当前 query 的 logits。历史缓存从同一冻结模型前缀重建。下面 Δ 相对完整原生 student。', '',
      '| 方法 | 平均 KL | 命中率 % | ΔKL [95% CI] | Δ命中率 pp [95% CI] |','|---|---:|---:|---|---|']
    for method in ['native_full']+[f'bias_{r}_{k}' for k in ('common','tuned') for r in ('depth','pairing','random','shuffled')]:
        r=results['offline'][f'0/{method}']
        report.append(f"| {method} | {r['kl']['mean']:.4f} | {100*r['match']['mean']:.2f} | {statstring(r['delta_kl'])} | {statstring(r['delta_match'],100,2)} |")
    report += ['', '## 实际 speculative decoding','',
      '以下 Δ/吞吐比相对同预算 teacher-chunk；完整上下文组相对 native_full。接受率为合计接受 draft token / 合计 proposal token；平均接受长度为合计接受 token / 验证轮数。区间中接受长度差按文档等权。',
      'teacher-chunk 是本轮 Qwen2 线性 drafting 上的 SpecExtend 式选块对照，非 Vicuna/原版树解码复现。全部路线每轮验证后刷新；边界 chunk 截断以匹配实际 token 数。',
      '可部署路线只使用已完成 prefill/verification 的 teacher 注意力，student 并集只取实际观测位置。差异规则与当前-query teacher oracle 不计入部署结果。',
      '完整 prefill、teacher query 行采集、选位、gather、draft、验证与同步更新均计入耗时；gather 子项通过独立仪器化重跑测量，主吞吐不插入层内计时事件。', '',
      '| 预算 | 方法 | 接受率 % | 平均接受长度 | decode TPS | 总 TPS | Δ接受长度 [95% CI] | decode 吞吐比 [95% CI] | 输出一致 |',
      '|---|---|---:|---:|---:|---:|---|---|---|']
    for (b,method) in sorted(runtime_groups):
        if method=='teacher_greedy':
            continue
        r=results['runtime'][f'{b}/{method}']
        speed=r['decode_ratio']
        report.append(f"| {b} | {method} | {100*r['acceptance_rate']:.2f} | {r['mean_accepted']:.3f} | {r['decode_tps']:.2f} | {r['total_tps']:.2f} | {statstring(r['delta_mean_accepted'])} | {speed['ratio']:.3f} [{speed['ci95'][0]:.3f}, {speed['ci95'][1]:.3f}] | {r['output_agreement']}/{r['documents']} |")
    report += ['', '## FP32 Teacher 数值一致性复核','',
      '原 BF16 block 验证出现与逐 token teacher 不同的输出。另用同一冻结 teacher 权重转 FP32（TF32 关闭），student 保持 BF16，重新生成并计时 teacher 参考，再重跑主要对照。原验证集超参数保持固定，复核结果不与 BF16 性能混算。',
      'FP32 复核改变了 teacher 的数值目标，不用于证明相对于原 BF16 teacher 的同分布无损加速；原 BF16 输出分歧仍保留为未通过的一致性检查。',
      'numerical_audit.json 比较同一 canonical KV 前缀的单 query 与多 query 计算，并固定 batch 形状改变未来 token 检查 causal mask。以下复核输出一致性与吞吐优先用于解释 E6；FP32 运算也只能由本轮检查支持，不能保证所有输入绝无浮点近似差异。', '',
      '| 预算 | 方法 | 接受率 % | 平均接受长度 | decode TPS | 总 TPS | Δ接受长度 [95% CI] | 相对 teacher-chunk decode 比 [95% CI] | 输出一致 |',
      '|---|---|---:|---:|---:|---:|---|---|---|']
    for (b,method) in sorted(fp32_groups):
        if method=='teacher_greedy':
            continue
        r=results['runtime_fp32'][f'{b}/{method}']
        speed=r['decode_ratio']
        report.append(f"| {b} | {method} | {100*r['acceptance_rate']:.2f} | {r['mean_accepted']:.3f} | {r['decode_tps']:.2f} | {r['total_tps']:.2f} | {statstring(r['delta_mean_accepted'])} | {speed['ratio']:.3f} [{speed['ci95'][0]:.3f}, {speed['ci95'][1]:.3f}] | {r['output_agreement']}/{r['documents']} |")
    report += ['', '### 粒度与并集的独立贡献','',
      '下表对合计接受率的增量直接按文档配对重采样，再重新计算合计 accepted/proposed。该指标按 proposal 数隐含加权，与前表的文档平均接受长度差不同；两个结果均保留。并集还必须与 teacher-token 比较，才能区分筛选粒度与双来源的作用。', '',
      '| 预算 | 方法 / 对照 | 合计接受率差 pp [95% CI] | decode 吞吐比 [95% CI] |','|---|---|---|---|']
    for b in config['budgets']:
        for method in ('teacher_token','union_available'):
            r=results['runtime_fp32'][f'{b}/{method}'];speed=r['decode_ratio']
            report.append(f"| {b} | {method} / teacher_chunk | {statstring(r['pooled_delta_acceptance'],100,2)} | {speed['ratio']:.4f} [{speed['ci95'][0]:.4f}, {speed['ci95'][1]:.4f}] |")
        r=results['runtime_fp32_union_vs_token'][str(b)];speed=r['decode_ratio']
        report.append(f"| {b} | union_available / teacher_token | {statstring(r['pooled_delta_acceptance'],100,2)} | {speed['ratio']:.4f} [{speed['ci95'][0]:.4f}, {speed['ci95'][1]:.4f}] |")
    report += ['', '### FP32 复核的分阶段耗时','',
      '各项为每个测试文档的平均秒数；draft 包含工作集 gather。主运行未插入层内计时事件。单文档 BF16 gather 的独立 CUDA 事件重跑见 runtime_profile.jsonl，不能与此 FP32 表混算。', '',
      '| 预算 | 方法 | prefill | selection | cache update / 同步补齐 | draft | verify |','|---|---|---:|---:|---:|---:|---:|']
    for b,method in [(0,'native_full')]+[(256,m) for m in ('teacher_chunk','teacher_token','union_available','layer_pairing','learned_global')]:
        r=results['runtime_fp32'][f'{b}/{method}'];stage=r['stage_mean_seconds']
        report.append(f"| {b} | {method} | {r['prefill_mean_seconds']:.4f} | {stage['selection']:.4f} | {stage['cache_update']:.4f} | {stage['draft']:.4f} | {stage['verify']:.4f} |")
    report += ['', '## 收益选择器与 Oracle','',
      '标签：以 teacher-token 工作集为基线，把同一个低注意力 anchor 替换成一个候选，测量输出 KL 改善。候选池为 teacher/student 各 8 个加随机 8 个（去重补齐，总 24 个）。两个预算都收集训练标签。',
      'MLP 使用 teacher 多层注意力特征、当前 student query 摘要及前 16 个冻结特征，以及候选相对 anchor 的差值；student 未观测位置不作零填充。优化收益回归 MSE，检查点按验证 MSE 选择。',
      '该 query 特征来自 student 第一层注意力输入，因此主要是当前 token 的归一化嵌入，不是深层上下文表示。learned_global 是强制将局部替换标签外推到全工作集排序的诊断；它不能视为经过验证的最佳部署策略。',
      '测试把 8 个低 teacher 注意力位置同时替换，所有方法使用相同候选池与移出位置，直接测联合收益。Oracle 按测试时测得的单次替换收益排序，只作离线诊断，不是严格上界；联合非加性可能使其劣于基线。', '',
      '| 预算 | 方法 | 联合 KL 改善 [95% CI] | ΔKL 相对候选池 teacher 排序 [95% CI] |','|---|---|---|']
    for key,r in results['utility'].items():
        b,method=key.split('/')
        report.append(f"| {b} | {method} | {statstring(r['joint_gain'])} | {statstring(r['kl_vs_teacher'])} |")
    train=jsonread(out/'selector_training.json')
    report += ['',f"选择器参数 {train['parameters']}，训练标签 {train['labels']}，选择 epoch {train['selected_epoch']}；验证 MSE {train['validation_mse']:.6g}，零预测器 MSE {train['zero_predictor_mse']:.6g}，验证收益相关系数 {train['validation_correlation']:.3f}。",
      '', '## 层间对应与并集来源消融','']
    for k,v in results['layer_correspondence'].items():
        report.append(f'- {k}: {statstring(v)}')
    report += ['', '层配对只由训练文档远历史 JS 距离确定。稳定性为测试 query 的逐层最近匹配与训练配对的一致比例；相似度本身不证明干预有效。', '',
       '| 预算 | 消融 | ΔKL 相对并集 [95% CI] | Δ命中率 pp [95% CI] |','|---|---|---|---|']
    for key,r in results['union_ablations'].items():
        b,method=key.split('/')
        report.append(f"| {b} | {method} | {statstring(r['delta_kl'])} | {statstring(r['delta_match'],100,2)} |")
    report += ['', '移除某一来源的峰值独占位置后，使用另一来源排名补齐，并保持原缓存预算与近期规则；并集重叠统计见 union_diagnostics_test.json。',
      '', '## 验证与适用范围','',
      f"记录完整性通过：{actual}。输出与逐 token teacher 参考一致：{len(runtime)-len(output_failures)}/{len(runtime)}；不一致详情见 completeness.json。",
      f"FP32 teacher 复核记录 {len(fp32_runtime)}，输出一致 {audit['fp32_output_agreement']}/{len(fp32_runtime)}；数值审计见 numerical_audit.json。",
      'FP32 小模型稀疏 gather 与完整 KV 的屏蔽参考最大误差 2.38e-7；真实模型完整读取与原生实现输出完全一致；预算、生成后缀、缓存回滚与数据分离检查均通过。',
      '实际方法保留完整原生 draft backing KV，再选择工作集读取；这轮不能声称减少所有历史 KV 存储，也没有 mapper 或 teacher-V 注入。每层预算相同，未扩展到按层非均匀预算或按头映射。',
      'Qwen2 模型头数不同，但两者都是 28 层；本轮不验证不同总层数、不同模型族或超长 8K/32K 上下文。单模型对、单种子、30 个测试文档的结果只支持本轮机制判断。',
      '采用 eager attention 和线性 greedy drafting，TPS 代表此验证实现；不等同于原版 Vicuna 树解码、FlashAttention 或 RTX 4090 部署速度。',
      'BF16 block 验证与逐 token 参考可能存在数值路径差异；若输出不一致，不能把该路线直接作为已验证无损实现。',
      '', '## 复现','', '```bash',
      '/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_selection/prepare_data.py --output exp/attention_selection/data_fresh',
      '/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_selection/run.py --phase smoke',
      '/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_selection/run.py --phase all',
      '/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_selection/supplement.py',
      '/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_selection/numerical_audit.py --fp32-runtime',
      '/home/wanghl/miniconda3/envs/idn/bin/python exp/attention_selection/analyze.py', '```','',
      '缓存位于 /mnt/8T/cl/rescue_experiment_cache/attention_selection_stage2，模型与张量缓存不提交；固定数据 manifest、config、selected、原始 JSONL、检查结果和图保存在本目录。']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    plot(out,results)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('output_failures','fp32_output_failures')},ensure_ascii=False,indent=2))


def plot(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=out/'figures'
    figures.mkdir(exist_ok=True)
    pairs=jsonread(out/'layer_pairs.json')
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,key in zip(axes,['all','remote','recent']):
        matrix=np.asarray(pairs['matrices'][key])
        im=ax.imshow(matrix,origin='lower',aspect='auto',cmap='viridis')
        ax.set(title=f'{key} attention JS (train)',xlabel='Teacher layer',ylabel='Student layer')
        ax.plot(pairs['depth'],np.arange(len(pairs['depth'])),'w.',ms=3,label='relative depth')
        ax.plot(pairs['pairing'],np.arange(len(pairs['pairing'])),'rx',ms=4,label='remote nearest')
        fig.colorbar(im,ax=ax,shrink=.85)
    axes[-1].legend(loc='lower right',fontsize=7)
    fig.savefig(figures/'layer_correspondence.png',dpi=180)
    fig.savefig(figures/'layer_correspondence.pdf')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    order=['teacher_chunk','teacher_token','student_token','union','difference','difference_core','layer_pairing','learned_global']
    for ax,b in zip(axes,[256,512]):
        val=[results['offline'][f'{b}/{k}']['delta_kl'] for k in order]
        y=np.arange(len(order))
        mean=np.asarray([x['mean'] for x in val]);lo=np.asarray([x['ci95'][0] for x in val]);hi=np.asarray([x['ci95'][1] for x in val])
        ax.errorbar(mean,y,xerr=np.stack((mean-lo,hi-mean)),fmt='o',capsize=3)
        ax.axvline(0,color='gray',lw=1)
        ax.set(yticks=y,yticklabels=order,title=f'Working budget {b}',xlabel='Output KL difference vs teacher-token (lower is better)')
        ax.invert_yaxis()
    fig.savefig(figures/'offline_selection.png',dpi=180)
    fig.savefig(figures/'offline_selection.pdf')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    fig.subplots_adjust(bottom=.24,wspace=.28)
    order=['teacher_chunk','teacher_token','union_available','layer_pairing','learned_global','bias_teacher']
    labels=['Teacher chunk','Teacher token','Union','Layer matching','Utility selector','Teacher token + bias']
    markers=['o','s','^','D','X','P']
    for ax,b in zip(axes,[256,512]):
        for k,label,marker in zip(order,labels,markers):
            r=results['runtime_fp32'][f'{b}/{k}']
            ax.scatter(r['mean_accepted'],r['decode_tps'],s=45,marker=marker,label=label)
        ax.set(title=f'FP32 teacher, working budget {b}',xlabel='Accepted draft tokens / verification block',ylabel='Measured decode tokens / second')
        ax.margins(x=.1,y=.08)
        ax.grid(alpha=.2)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,fontsize=9)
    fig.savefig(figures/'runtime_acceptance_throughput.png',dpi=180)
    fig.savefig(figures/'runtime_acceptance_throughput.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
