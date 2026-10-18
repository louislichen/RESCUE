"""Training-only long-context execution/selection pilot against retrieval ports."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random

import torch

from .baselines import BaselineEngine
from .compare_baselines import append, write
from .config import SystemConfig
from .engine import RescueEngine
from .loading import ROOT, load_models, load_modules
from .long_data import context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--documents', type=int, default=2)
    parser.add_argument('--lengths', type=int, nargs='+', default=[2048, 4096, 8192, 16384])
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new run directory')
    args.output.mkdir(parents=True)
    sources = [*sorted((ROOT/'rescue').glob('*.py')), *sorted((ROOT/'tests').glob('*.py'))]
    hashes = {}
    for source in sources:
        rel = source.relative_to(ROOT)
        hashes[str(rel)] = hashlib.sha256(source.read_bytes()).hexdigest()
        dest = args.output/'source'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    data = ROOT/'exp/long_system/data/train.jsonl'
    rows = [json.loads(l) for l in data.read_text().splitlines()][:args.documents]
    base = SystemConfig(length_policy='fixed', fixed_k=4, max_new_tokens=args.max_new_tokens,
        prefill_chunk_size=0, selector='hierarchical', injection_layers=(27,), inject_tokens=64)
    configs = dict(hierarchical=replace(base, inject_tokens=0), injected=base,
        adaptive=replace(base, refresh_policy='adaptive', refresh_max=16, attention_drift=.15, coverage_floor=.65))
    modes = ['specextend16', 'specextend32', *configs]
    write(args.output/'protocol.json', dict(scope='Training-only pilot, retrieval ports; not full upstream SpecExtend',
        target_dtype='bfloat16', draft_dtype='bfloat16', backend='SDPA with compact attention observations',
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(), sources=hashes,
        configs={k: asdict(v) for k, v in configs.items()}, modes=modes,
        all_contexts='Natural nested GovReport report prefixes, all from new training split'))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher, student, tokenizer = load_models(target_dtype=torch.bfloat16, efficient=True)
    mapper, risk, cost = load_modules(teacher.device)
    write(args.output/'environment.json', dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        teacher_dtype=str(teacher.dtype), student_dtype=str(student.dtype)))
    for item in rows:
        for length in args.lengths:
            ids = context(item, length)
            with BaselineEngine(teacher, student, base, top_chunks=16, vocab_size=len(tokenizer)) as engine:
                reference = engine.reference(ids, tokenizer.eos_token_id)
            order = list(modes)
            random.Random(42+length+item['source_index']).shuffle(order)
            for mode in order:
                torch.cuda.reset_peak_memory_stats()
                if mode in configs:
                    engine = RescueEngine(teacher, student, configs[mode], mapper, risk, cost, vocab_size=len(tokenizer))
                else:
                    engine = BaselineEngine(teacher, student, base, top_chunks=int(mode.removeprefix('specextend')), vocab_size=len(tokenizer))
                with engine:
                    result = engine.generate(ids, tokenizer.eos_token_id)
                result.update(mode=mode, length=length, split='train', source_index=item['source_index'],
                    output_agrees=result['generated'] == reference['generated'],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
                append(args.output/'runtime.jsonl', result)
                print(item['source_index'], length, mode, 'total', result['total_tps'],
                      'decode', result['decode_tps'], 'exact', result['output_agrees'], flush=True)
                if not result['output_agrees']:
                    write(args.output/'mismatch.json', dict(reference=reference, result=result))
                    raise AssertionError('Pilot output differs from same-backend Target reference')


if __name__ == '__main__':
    main()
