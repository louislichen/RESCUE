"""Isolated, development-only request test of corrected CSGB features.

Preparation is not an experimental result. Frozen parameters come only from
the complete 24-document training audit. Native fixed budgets do not pay for
fresh controller attention. Every policy shares terminal-Draft elision.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time


POLICIES = ['fixed8_native', 'fixed16_native', 'fixed24_native', 'fixed32_native',
            'constant_gain', 'csgb']
FEATURE_COLUMNS = {'constant': [0], 'coverage': [0, 1], 'shift': [0, 2],
                   'csgb': [0, 1, 2, 3]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_models(state):
    expected = {'schema': 1, 'train_documents': list(range(24)),
                'corrected_root_coverage': True, 'attention_observation_every_round': True,
                'contexts': [4096, 8192], 'budgets': [8, 16, 24, 32],
                'intervention_rounds': [5, 6, 7, 8], 'repeats': 2, 'ridge': 1.0}
    if any(state.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong frozen training protocol or feature version')
    if set(state.get('models', {})) != {'4096', '8192'}:
        raise ValueError('Both actual context lengths require frozen models')
    for models in state['models'].values():
        if set(models) != set(FEATURE_COLUMNS):
            raise ValueError('Missing or unexpected frozen ablation model')
        for name, cols in FEATURE_COLUMNS.items():
            model = models[name]
            if model.get('cols') != cols or model.get('budgets') != expected['budgets']:
                raise ValueError('Frozen model columns/budgets mismatch')
            mean, scale, theta = (model.get(k) for k in ('mean', 'scale', 'theta'))
            if (not isinstance(mean, list) or not isinstance(scale, list)
                    or len(mean) != len(cols)-1 or len(scale) != len(mean)
                    or not isinstance(theta, list) or len(theta) != len(cols)
                    or any(not isinstance(row, list) or len(row) != 4 for row in theta)):
                raise ValueError('Invalid frozen parameter shape')
            numbers = mean + scale + [v for row in theta for v in row]
            if (any(type(v) not in (float, int) or not math.isfinite(v) for v in numbers)
                    or any(v <= 0 for v in scale)):
                raise ValueError('Invalid frozen parameter values')


def validate_documents(documents, smoke=False):
    allowed = {0} if smoke else set(range(24, 32))
    if not documents or len(set(documents)) != len(documents) or not set(documents) <= allowed:
        raise ValueError('Only training doc0 smoke or locked development docs24..31 allowed')


def validate_model_evidence(state):
    """Require the fitter's complete immutable training inputs, not metadata alone."""
    fixed = {'cases': 1536, 'states': 192, 'feature_meaning': 'corrected_root_coverage',
             'cache_refresh_every_rounds': 4, 'target_dtype': 'float32', 'draft_dtype': 'float16',
             'no_hyperparameter_search': True, 'development_read': False, 'confirmation_read': False}
    if any(state.get(k) != v for k, v in fixed.items()):
        raise ValueError('Unproven frozen model provenance')
    inputs = state.get('input_sha256')
    provenance = state.get('provenance', {})
    sources = provenance.get('fitter_source_sha256', {})
    if (not isinstance(inputs, dict) or len(inputs) < 1536 or not sources
            or not any(Path(p).name == 'fit_csgb_train24.py' and d == state.get('fitter_sha256')
                       for p, d in sources.items())
            or inputs.get(provenance.get('sidecar_path')) != provenance.get('sidecar_sha256')):
        raise ValueError('Missing immutable training inputs/fitter/sidecar')
    for path, expected in {**inputs, **sources}.items():
        if inputs.get(path) != expected or digest(path) != expected:
            raise ValueError('Frozen model training evidence changed: '+path)


def current_gpu_uuid():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                             '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=10)
    uuids = {parts[0].strip() for line in result.stdout.splitlines()
             if len(parts := line.split(',')) == 2 and parts[1].strip() == str(os.getpid())}
    if len(uuids) != 1:
        raise RuntimeError('Cannot prove one physical GPU for this worker')
    return uuids.pop()


def assert_gpu_uncontended(uuid):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                             '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=10)
    pids = {int(parts[1].strip()) for line in result.stdout.splitlines()
            if len(parts := line.split(',')) == 2 and parts[0].strip() == uuid}
    if pids != {os.getpid()}:
        raise RuntimeError('Physical GPU contention: stop without touching any other process')


def validate_request(result, reference, rows, pending, policy, context, seconds, learner=None):
    """Independent accounting and frozen-action replay, outside the timer."""
    output = result['output_token_ids']
    raw, scored = result['raw_accept_length_list'], result['accept_length_list']
    if (output != reference['output_token_ids'] or not output or len(output) > 128
            or result['total_generated'] != len(output)):
        raise ValueError('Canonical output or scored denominator mismatch')
    if (not math.isfinite(seconds) or seconds <= 0 or not rows or pending is not None
            or len(rows) != len(raw) or len(scored) != len(raw)
            or any(type(a) is not int or a < 0 for a in raw + scored)):
        raise ValueError('Incomplete verified rounds or invalid request time')
    if (sum(a+1 for a in raw) != result['raw_total_generated']
            or sum(a+1 for a in scored) != len(output)
            or result['generation_overshoot'] != result['raw_total_generated']-len(output)
            or scored[:-1] != raw[:-1] or scored[-1] > raw[-1]
            or result['avg_accept_length'] != round(sum(scored)/len(scored), 3)):
        raise ValueError('Raw/scored accepted-token accounting mismatch')
    prefix = context+1
    fixed = int(policy[5:-7]) if policy.startswith('fixed') else None
    for j, (row, accepted) in enumerate(zip(rows, raw)):
        budget = row['budget']; x = row['x']
        if (row['proposal'] != j or row['prefix'] != prefix or row['committed'] != accepted+1
                or row['actual_draft_nodes'] != budget or budget not in (8, 16, 24, 32)
                or type(row.get('terminal_commit', False)) is not bool
                or row.get('terminal_commit', False) != (j == len(rows)-1)
                or row['seconds'] is not None or row['reward'] is not None
                or not math.isfinite(row['decision_cpu_seconds']) or row['decision_cpu_seconds'] < 0):
            raise ValueError('Round/action/terminal alignment mismatch')
        if fixed is not None:
            if budget != fixed or x is not None:
                raise ValueError('Native fixed policy changed action or extracted controller features')
        else:
            if learner is None or learner.choose(x, explore=False) != budget:
                raise ValueError('Frozen action does not reproduce saved decision')
            if x is not None:
                if (len(x) != 4 or x[0] != 1 or not all(math.isfinite(v) for v in x)
                        or not 0 <= x[1] <= 1+1e-12 or not 0 <= x[2] <= 1
                        or abs(x[3]-x[1]*x[2]) > 1e-12
                        or row['coverage'] != x[1] or row['shift'] != x[2]
                        or row['attention_observed_timestep'] != j-1):
                    raise ValueError('Invalid/stale controller feature')
            elif budget != 8:
                raise ValueError('Missing-feature fallback is not budget eight')
        prefix += accepted+1


def main():
    ap = argparse.ArgumentParser()
    for name in ('base', 'draft', 'source', 'models', 'out'):
        ap.add_argument('--'+name, required=True)
    ap.add_argument('--documents', type=int, nargs='+', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    validate_documents(args.documents, args.smoke)
    frozen_bytes = Path(args.models).read_bytes()
    state = json.loads(frozen_bytes); validate_models(state); validate_model_evidence(state)
    out = Path(args.out)
    if out.exists():
        raise ValueError('Output already exists; never reuse a partial run')
    # Heavy dependencies are delayed so accounting/preflight tests need no GPU.
    import numpy as np
    import torch
    from classic.model_classic import SPModel
    from baseline_profiler import exact_input, load_texts
    from csgb_same_state import release_request_caches
    from csgb_frozen_gain import FrozenGain
    from run_csgb_controller import Harness, atomic

    class AuditedHarness(Harness):
        def draft(self, model, input_ids, nodes, threshold, max_depth):
            result = super().draft(model, input_ids, nodes, threshold, max_depth)
            self.pending['attention_observed_timestep'] = getattr(model, 'csgb_attention_observed_timestep', None)
            self.pending['processed_prefix_length'] = int(getattr(model, 'total_seq_len', 0))
            self.pending['selected_chunk_count'] = len(getattr(model, 'selected_chunks', None) or [])
            return result

    if torch.cuda.device_count() != 1:
        raise RuntimeError('One visible GPU per worker is required; parallelize documents, not pairs')
    torch.manual_seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model = SPModel.from_pretrained(base_model_path=args.base, draft_model_path=args.draft,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={'': 0}).eval()
    if {str(p.device) for p in model.parameters()} != {'cuda:0'}:
        raise RuntimeError('All Target/Draft components must use the worker single visible GPU')
    model.base_model.float(); model.csgb_root_coverage = True; model.csgb_skip_terminal_draft = True
    h = AuditedHarness(model); model.csgb_terminal_commit_callback = h.finalize_terminal
    learners = {int(c): {k: FrozenGain(v) for k, v in models.items()} for c, models in state['models'].items()}
    # load_texts stops at this limit, so dev never loads documents32..47.
    data_digest = digest(args.source)
    texts = load_texts(args.source, max(args.documents)+1)
    if digest(args.source) != data_digest:
        raise RuntimeError('Input dataset changed while loading')
    sources = {str(p.resolve()): digest(p) for p in Path(__file__).parent.rglob('*.py')}
    gpu_uuid = current_gpu_uuid()
    repeats = 1 if args.smoke else 3
    out.mkdir(exist_ok=False)
    atomic(out/'protocol.json', {'schema': 1, 'phase': 'training_smoke' if args.smoke else 'reused_development_only',
        'documents': args.documents, 'contexts': [4096, 8192], 'policies': POLICIES, 'repeats': repeats,
        'warmup_repeats': 1, 'tokens': 128, 'max_depth': 10, 'threshold': .7,
        'models_sha256': hashlib.sha256(frozen_bytes).hexdigest(), 'frozen_models': str(Path(args.models).resolve()),
        'source': str(Path(args.source).resolve()), 'source_sha256': data_digest, 'source_manifest': sources,
        'recorded_source_root': str(Path(__file__).parent.resolve()),
        'runner_sha256': digest(__file__), 'target_dtype': 'float32', 'draft_dtype': 'float16',
        'base_model_path': str(Path(args.base).resolve()), 'draft_model_path': str(Path(args.draft).resolve()),
        'seed': 42, 'corrected_root_coverage': True, 'skip_terminal_draft': True,
        'attention_every_round': ['constant_gain', 'csgb'], 'cache_refresh_every': 4,
        'retrieval_chunk_size': 32, 'retrieve_top_k': 32,
        'gpu_name': torch.cuda.get_device_name(0), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_uuid': gpu_uuid,
        'gpu_total_memory': torch.cuda.get_device_properties(0).total_memory,
        'torch_version': torch.__version__, 'numpy_version': np.__version__, 'cuda_version': torch.version.cuda,
        'tf32_enabled': False,
        'timing': 'complete request wall time; boundary sync only; eager FP32, NOT production speed',
        'request_cleanups_outside_timer': True, 'gpu_contention_boundary_checks': True,
        'confirmation_evaluated': False})
    count = 0
    for doc in args.documents:
        for context in (4096, 8192):
            release_request_caches(model)
            ids = exact_input(model.tokenizer, texts[doc], context, torch, torch.device('cuda:0'))
            if int(ids.shape[-1]) != context:
                raise RuntimeError('Actual input length differs from context label')
            with torch.inference_mode():
                ref = model.ar_generate(ids, max_new_tokens=128, verbose=False, return_metrics=True)
            atomic(out/f'reference_{context}_{doc}.json', ref)
            for repeat in range(-1, repeats):
                order = POLICIES.copy(); random.Random(918+doc*100+context+repeat).shuffle(order)
                for policy in order:
                    release_request_caches(model)
                    fresh = policy in ('constant_gain', 'csgb')
                    model.csgb_attention_every_round = fresh
                    learner = learners[context]['constant' if policy == 'constant_gain' else 'csgb'] if fresh else None
                    h.reset('csgb' if fresh else policy[:-7], learner, False, 0., clean=True)
                    nodes = 8 if fresh else int(policy[5:-7])
                    assert_gpu_uncontended(gpu_uuid)
                    torch.cuda.synchronize(); start = time.perf_counter()
                    with torch.inference_mode():
                        result = model.spgenerate(ids, nodes=nodes, max_depth=10, threshold=.7,
                            max_new_tokens=128, temperature=0, use_specextend=True,
                            retrieval_chunk_size=32, retrieve_top_k=32, retrieve_every_n_steps=4,
                            retrieval_num_queries=1, exact_target_attention=True, return_output_ids=True,
                            verbose=False, output_result_line=False)
                    torch.cuda.synchronize(); seconds = time.perf_counter()-start
                    assert_gpu_uncontended(gpu_uuid)
                    validate_request(result, ref, h.rows, h.pending, policy, context, seconds, learner)
                    row = {'document': doc, 'context': context, 'actual_input_tokens': int(ids.shape[-1]),
                           'policy': policy, 'repeat': repeat, 'seconds': seconds,
                           'tps': result['total_generated']/seconds, 'exact': True,
                           'gpu_uuid': gpu_uuid, 'gpu_uncontended_at_request_boundaries': True,
                           'result': result, 'rounds': h.rows, 'pending_is_none': h.pending is None}
                    atomic(out/f'{"warmup" if repeat < 0 else "case"}_{context}_{doc}_{policy}_{repeat}.json', row)
                    count += repeat >= 0
                    print(json.dumps({'document': doc, 'context': context, 'policy': policy,
                                      'repeat': repeat, 'tps': row['tps'], 'exact': True}), flush=True)
    if (Path(args.models).read_bytes() != frozen_bytes or digest(args.source) != data_digest
            or any(digest(p) != v for p, v in sources.items())):
        raise RuntimeError('Frozen parameters or source changed during evaluation')
    validate_model_evidence(state)
    if count != len(args.documents)*2*len(POLICIES)*repeats:
        raise RuntimeError('Incomplete evaluation matrix')
    atomic(out/'DONE.json', {'passed': True, 'cases': count, 'failures': [], 'confirmation_evaluated': False})


if __name__ == '__main__':
    main()
