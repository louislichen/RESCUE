"""Functional full-engine Qwen2 port check, using training documents only."""
import argparse
import json
from pathlib import Path
import torch
from .loading import load_models, DEFAULT_TEACHER
from .specextend_qwen import build, generate
from .long_data import context
from .config import SystemConfig
from .baselines import BaselineEngine
from .packing import pack_projections


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', nargs='+', type=int, default=[2048, 8192])
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--prefill-precision', choices=['native', 'bfloat16'], default='native')
    p.add_argument('--pack-projections', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    t, s, tok = load_models(efficient=True)
    full = build(t, s, DEFAULT_TEACHER)
    full.prefill_precision = a.prefill_precision
    if a.pack_projections:
        for model in (t, s, full.base_model, full.draft_model):
            pack_projections(model)
    item = json.loads(Path('exp/long_system/data/train.jsonl').read_text().splitlines()[0])
    records = []
    for length in a.lengths:
        ids = context(item, length)
        cfg = SystemConfig(length_policy='fixed', max_new_tokens=a.max_new_tokens,
                           prefill_precision=a.prefill_precision)
        with BaselineEngine(t, s, cfg, vocab_size=len(tok)) as engine:
            reference = engine.reference(ids, tok.eos_token_id)['generated']
        for nodes in (8, 32):
            result = generate(full, ids, max_new_tokens=a.max_new_tokens, nodes=nodes, top_chunks=32)
            result.update(length=length, output_agrees=result['generated'] == reference, reference=reference)
            records.append(result)
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(records, indent=2)+'\n')
            print(length, nodes, result['output_agrees'], result['total_tps'], flush=True)
            assert result['output_agrees']


if __name__ == '__main__':
    main()
