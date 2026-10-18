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
