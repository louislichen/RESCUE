"""Development refinement with upstream-size tree controls, then held-out test.

Initial and refinement development timings are normalized against a matched
retrieval control in each batch. No test timing chooses configurations.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from .baselines import BaselineEngine
from .compare_baselines import append, write
from .config import SystemConfig
from .engine import RescueEngine
from .loading import ROOT, DEFAULT_TEACHER, load_models, load_modules
from .long_benchmark import summary
from .long_data import context
from .packing import share_packed_projections
from .policy import RiskModel
from .specextend_qwen import SOURCE, build, generate as tree_generate


def config(value):
    value = dict(value)
    if value['injection_layers'] is not None:
        value['injection_layers'] = tuple(value['injection_layers'])
    return SystemConfig(**value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--test-documents', type=int, default=8)
    p.add_argument('--repeats', type=int, default=2)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory')
    initial = json.loads((args.initial/'protocol.json').read_text())
    choices = json.loads((args.initial/'selection.json').read_text())['chosen']
    old = [json.loads(l) for l in (args.initial/'development.jsonl').read_text().splitlines()]
    assert len(old) == 114 and all(r['output_agrees'] for r in old)
    assert not (args.initial/'runtime.jsonl').exists(), 'Independent test must remain unopened for timing'
    args.output.mkdir(parents=True)
    configs = {name: config(value) for name, value in initial['configs'].items()}
    retrieval = {name: tuple(value) for name, value in initial['retrieval'].items()}
    full = {name: tuple(value) for name, value in initial['full_tree'].items()}
    positive = configs[choices['rescue_positive']]
    configs.update(refine_k2=replace(positive, length_policy='fixed', fixed_k=2),
        refine_teacher025=replace(positive, student_weight=.25),
        refine_teacher025_k2=replace(positive, student_weight=.25, length_policy='fixed', fixed_k=2),
        refine_strength1=replace(positive, injection_strength=1.),
        refine_matched_noinj=replace(positive, inject_tokens=0),
        refine_matched_noinj_k2=replace(positive, inject_tokens=0, length_policy='fixed', fixed_k=2))
    full.update({f'full{chunks}_n{nodes}': (chunks, nodes) for chunks in (16, 32) for nodes in (50, 64)})
    names = list(dict.fromkeys([*choices.values(), 'refine_k2', 'refine_teacher025',
        'refine_teacher025_k2', 'refine_strength1', 'refine_matched_noinj',
        'refine_matched_noinj_k2', *[n for n in full if full[n][1] in (50, 64)]]))
    sources = [*sorted((ROOT/'rescue').glob('*.py')), *sorted((ROOT/'tests').glob('*.py')),
        *sorted(SOURCE.rglob('*.py')), ROOT/'exp/attention_horizon/selection.py', ROOT/'exp/selective_kv/bridge.py']
    hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = args.output/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    data = ROOT/'exp/long_system/data'
    train = [json.loads(l) for l in (data/'train.jsonl').read_text().splitlines()]
    development = [json.loads(l) for l in (data/'development.jsonl').read_text().splitlines()][:3]
    write(args.output/'protocol.json', dict(scope='Development refinement before any independent test timing',
        args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        initial_protocol_sha256=hashlib.sha256((args.initial/'protocol.json').read_bytes()).hexdigest(),
        initial_development_sha256=hashlib.sha256((args.initial/'development.jsonl').read_bytes()).hexdigest(),
        source_sha256=hashes, data_sha256=initial['data_sha256'], checkpoint_sha256=initial['checkpoint_sha256'],
        configs={n:asdict(v) for n,v in configs.items()}, retrieval=retrieval, full_tree=full,
        refinement_candidates=names, reference_control=choices['retrieval'],
        development_indices=[r['source_index'] for r in development], lengths=[2048,4096,8192,16384],
        tuning_lengths=[8192,16384], max_new_tokens=256,
        selection='Geometric mean per-document/length TPS ratio vs same-batch retrieval reference; initial records retained, repeats replace their initial records',
        precision='Same initial FP32 Target weights/decode, BF16 prefix and Draft, TF32 off; shared immutable packing',
        success_gate='Nonzero RESCUE and dynamic separately: >=10% paired mean with CI lower>0 against both controls at8K/16K'))
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    teacher, student, tokenizer = load_models(efficient=True)
    training = Path(initial['args']['training'])
    mapper, _, _ = load_modules(teacher.device, mapper=training/'mapper_joint.pt')
    risk = RiskModel.load(training/'risk_injected.pt', teacher.device)
    cost = json.loads((args.initial/'cost.json').read_text())
    tree = build(teacher, student, DEFAULT_TEACHER)
    tree.prefill_precision = 'bfloat16'
    torch.cuda.synchronize(); tick = time.perf_counter(); memory = torch.cuda.memory_allocated()
    share_packed_projections(teacher, tree.base_model)
    share_packed_projections(student, tree.draft_model)
    torch.cuda.synchronize()
    write(args.output/'environment.json', dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        packing_seconds=time.perf_counter()-tick, packing_allocated_delta_bytes=torch.cuda.memory_allocated()-memory))
    references = json.loads((args.initial/'references.json').read_text())
    base = configs[choices['rescue_positive']]
    def reference(item, length):
        key = f"{item['split']}:{item['source_index']}:{length}"
        if key not in references:
            with BaselineEngine(teacher, student, replace(base, length_policy='fixed'), vocab_size=len(tokenizer)) as engine:
                references[key] = engine.reference(context(item,length), tokenizer.eos_token_id)['generated']
            write(args.output/'references.json', references)
        return references[key]
    write(args.output/'references.json', references)
    def run(item, length, candidate, limit=256):
        ids = context(item, length)
        if candidate in configs:
            cfg = replace(configs[candidate], max_new_tokens=limit)
            with RescueEngine(teacher, student, cfg, mapper, risk, cost, vocab_size=len(tokenizer)) as engine:
                result = engine.generate(ids, tokenizer.eos_token_id)
        elif candidate in retrieval:
            chunks,k = retrieval[candidate]
            cfg = replace(base, length_policy='fixed', fixed_k=k, max_new_tokens=limit)
            with BaselineEngine(teacher, student, cfg, top_chunks=chunks, vocab_size=len(tokenizer)) as engine:
                result = engine.generate(ids, tokenizer.eos_token_id)
        else:
            chunks,nodes = full[candidate]
            result = tree_generate(tree, ids, max_new_tokens=limit, nodes=nodes, top_chunks=chunks)
        result.update(source_index=item['source_index'], split=item['split'], length=length, candidate=candidate,
            output_agrees=result['generated']==reference(item,length)[:limit])
        if not result['output_agrees']:
            append(args.output/'mismatches.jsonl',result)
            raise AssertionError(f'Output mismatch: {item["split"]}/{item["source_index"]}/{length}/{candidate}')
        return result
    for length in (8192,16384):
        for candidate in names:
            run(train[0],length,candidate,limit=8)
        print('refinement warmup',length,flush=True)
    records=[]
    for item in development:
        for length in (8192,16384):
            order=list(names);random.Random(913+item['source_index']*1000+length).shuffle(order)
            for candidate in order:
                row=run(item,length,candidate)
                records.append(row);append(args.output/'development.jsonl',row)
            print('refinement complete',item['source_index'],length,flush=True)
    def normalized(rows):
        control={(r['source_index'],r['length']):r['total_tps'] for r in rows if r['candidate']==choices['retrieval']}
        return {name:math.exp(sum(math.log(r['total_tps']/control[(r['source_index'],r['length'])])
            for r in rows if r['candidate']==name)/sum(r['candidate']==name for r in rows))
            for name in {r['candidate'] for r in rows}}
    scores=normalized(old);scores.update(normalized(records))
    chosen=dict(rescue_positive=max((n for n,c in configs.items() if c.inject_tokens),key=scores.get),
        rescue_guarded=max(configs,key=scores.get),rescue_dynamic='dynamic',
        retrieval=max(retrieval,key=scores.get),full_specextend=max(full,key=scores.get))
    write(args.output/'selection.json',dict(chosen=chosen,normalized_development_scores=scores))
    print('refinement locked',chosen,scores,flush=True)
    test=[json.loads(l) for l in (data/'test.jsonl').read_text().splitlines()][:args.test_documents]
    write(args.output/'test_matrix.json',dict(source_indices=[r['source_index'] for r in test],
        lengths=[2048,4096,8192,16384],repeats=args.repeats,modes=chosen))
    measured=[]
    for item in test:
        for length in (2048,4096,8192,16384):
            for repeat in range(args.repeats):
                order=list(chosen);random.Random(4242+item['source_index']*1000+length+repeat).shuffle(order)
                for mode in order:
                    row=run(item,length,chosen[mode]);row.update(mode=mode,repeat=repeat)
                    measured.append(row);append(args.output/'runtime.jsonl',row)
                print('test complete',item['source_index'],length,repeat,flush=True)
    write(args.output/'summary.json',summary(measured))


if __name__=='__main__':
    main()
