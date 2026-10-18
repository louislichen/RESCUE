"""Training-only actual-engine timing calibration, separate from evaluation."""
from dataclasses import replace
import statistics

import numpy as np

from .engine import RescueEngine


def calibrate(teacher, student, mapper, risk, config, documents, eos, vocab):
    records = []
    for k in (1, 4, 8):
        cfg = replace(config, length_policy='fixed', fixed_k=k, schedule='serial',
                      length_endpoint=None, profile=True)
        with RescueEngine(teacher, student, cfg, mapper, risk, vocab_size=vocab) as engine:
            for item in documents:
                reference = engine.reference(item['ids'], eos)
                result = engine.generate(item['ids'], eos)
                if result['generated'] != reference['generated']:
                    raise AssertionError('Calibration output does not match Target')
                records.append(dict(source_index=item['source_index'], k=k,
                                    generated=result['generated'], cost_samples=result['cost_samples'],
                                    total_seconds=result['total_seconds']))
    samples = [v for r in records for v in r['cost_samples']['verify']]
    x = np.array([[1., r['n']/2048, r['k'], r['k']*r['n']/2048] for r in samples])
    y = np.array([r['seconds'] for r in samples])
    beta = np.zeros(4)
    for _ in range(400):
        for j in range(4):
            beta[j] = max(0., np.dot(x[:, j], y-x@beta+x[:, j]*beta[j])/max(1e-12, np.dot(x[:, j], x[:, j])))
    cost = dict(verify_beta=beta.tolist(),
                student_step=statistics.median(v for r in records for v in r['cost_samples']['student_step']),
                fixed_round=statistics.median(v for r in records for v in r['cost_samples']['overhead']),
                cache_update=0., calibration_documents=[d['source_index'] for d in documents],
                verify_mae=float(np.abs(x@beta-y).mean()), verify_samples=len(samples),
                scope='Actual engine, fixed horizons, training documents only; additive cost approximation')
    return cost, records


def main():
    import argparse
    import json
    from pathlib import Path
    import torch
    from .config import SystemConfig
    from .loading import ROOT, load_models, load_modules
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--documents', type=int, default=2)
    args = parser.parse_args()
    if args.documents < 1 or args.output.exists():
        parser.error('Use a new output file and a positive document count')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher, student, tokenizer = load_models()
    mapper, risk, _ = load_modules(teacher.device)
    records = [json.loads(line) for line in (ROOT/'exp/attention_horizon/data/train.jsonl').read_text().splitlines()][:args.documents]
    if len(records) != args.documents:
        parser.error('Not enough training documents')
    cost, timings = calibrate(teacher, student, mapper, risk, SystemConfig(), records,
                              tokenizer.eos_token_id, len(tokenizer))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cost, indent=2)+'\n')
    args.output.with_suffix('.samples.json').write_text(json.dumps(timings, indent=2)+'\n')


if __name__ == '__main__':
    main()
