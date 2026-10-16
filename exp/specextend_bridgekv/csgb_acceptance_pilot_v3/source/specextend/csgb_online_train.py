"""Training-only actual-feedback CSGB; isolated from all previous run roots.

No development/confirmation arguments exist. Parameters are learned only from
executed actions, using useful output commitments and round-complete timing.
Request timings here include online exploration and updates, NOT deployment TPS.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

from csgb_online_policy import CONTEXTS, BUDGETS, FeatureScale, OnlineCSGBPolicy


MODES = ('csgb', 'constant', 'history_native')
SEEDS = (101, 202, 303)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive_json(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('Refusing to replace '+str(path))
    pending = path.with_name(path.name+'.pending')
    with pending.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    # Hard-link publication is atomic AND exclusive; rename could overwrite.
    # A failed partial write remains .pending for inspection, never reused.
    os.link(pending, path)
    pending.unlink()


def private_seed(seed, context, mode):
    if type(seed) is not int or seed not in SEEDS or context not in CONTEXTS or mode not in MODES:
        raise ValueError('Unregistered seed/context/mode')
    return seed * 100000 + context + MODES.index(mode) * 10000


def validate_raw_x(x, mode):
    if x is None:
        return
    width = 3 if mode == 'history_native' else 4
    if (not isinstance(x, list) or len(x) != width or x[0] != 1.
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in x)
            or any(not 0 <= v <= 1 for v in x[1:])
            or (width == 4 and not math.isclose(x[3], x[1]*x[2], rel_tol=0., abs_tol=1e-12))):
        raise ValueError('Malformed raw policy features')


def validate_training_source(path):
    """Reject even accidentally staged dev/test data before loading a model."""
    lines = Path(path).read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    if len(rows) != 24 or any(not isinstance(r, dict) or not isinstance(r.get('text'), str)
                              or not r['text'] for r in rows):
        raise ValueError('Exactly24 training texts required; never load full development/test source')
    return [row['text'] for row in rows]


def validate_scaler(path):
    state = json.loads(Path(path).read_text())
    expected = {'schema': 1, 'purpose': 'train-only fixed8 full-round scaling',
                'train_documents': list(range(24)), 'contexts': [4096, 8192],
                'attention_observation_every_round': True, 'corrected_root_coverage': True,
                'verified_feature_observations': 2280}
    if any(state.get(key) != value or type(state.get(key)) is not type(value)
           for key, value in expected.items()):
        raise ValueError('Wrong scaler protocol or incomplete training evidence')
    if set(state.get('scalers', {})) != {'4096', '8192'}:
        raise ValueError('Both context scalers required')
    scalers = {int(c): FeatureScale.from_state(s) for c, s in state['scalers'].items()}
    if sum(s.count for s in scalers.values()) != 2280:
        raise ValueError('Incomplete full-round training scaling')
    mappings = [state.get(name) for name in ('input_sha256', 'collection_source_sha256', 'source_sha256')]
    if (any(not isinstance(mapping, dict) or not mapping for mapping in mappings)
            or len(mappings[0]) != 2522):
        raise ValueError('Missing immutable scaler raw/source evidence')
    evidence = {}
    for mapping in mappings:
        for source, expected_hash in mapping.items():
            if source in evidence and evidence[source] != expected_hash:
                raise ValueError('Contradictory scaler evidence')
            if not Path(source).is_absolute() or digest(source) != expected_hash:
                raise ValueError('Changed scaler evidence: '+source)
            evidence[source] = expected_hash
    return state, scalers, evidence


def validate_rounds(result, reference, rows, pending, context, seconds, *,
                    mode, learner_before=None, learner_after=None, rho=0.):
    """Independent accounting plus chronological RNG/action/update replay."""
    output = result.get('output_token_ids')
    if (not isinstance(output, list) or not output or len(output) > 128
            or output != reference.get('output_token_ids')
            or result.get('total_generated') != len(output)
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('Canonical equality/output/time validation failed')
    raw = result.get('raw_accept_length_list', [])
    scored = result.get('accept_length_list', [])
    if (not rows or pending is not None or len(rows) != len(raw) or len(raw) != len(scored)
            or any(type(a) is not int or a < 0 for a in raw + scored)
            or sum(a+1 for a in raw) != result.get('raw_total_generated')
            or sum(a+1 for a in scored) != len(output)
            or result.get('generation_overshoot') != result['raw_total_generated']-len(output)
            or raw[:-1] != scored[:-1] or scored[-1] > raw[-1]):
        raise ValueError('Raw/scored round completeness failed')
    replay = None if learner_before is None else OnlineCSGBPolicy.from_state(learner_before)
    if (mode in MODES) != (replay is not None):
        raise ValueError('Missing/unexpected actual-feedback model')
    if replay is not None and (replay.mode != mode or replay.context != context
                               or replay.rho != rho or not isinstance(learner_after, dict)):
        raise ValueError('Model/request context, mode or calibration mismatch')
    prefix, generated = context+1, 0
    history_last, history_ema = 0., 0.
    for j, (row, a_raw, a_scored) in enumerate(zip(rows, raw, scored)):
        # Harness schema intentionally checked here rather than trusted via DONE.
        x = row['x']
        validate_raw_x(x, mode)
        useful = min(generated+a_raw+1, 128)-min(generated, 128)
        if (row['proposal'] != j or row['prefix'] != prefix or row['budget'] not in BUDGETS
                or row['actual_draft_nodes'] != row['budget']
                or row['committed'] != a_raw+1 or row.get('raw_committed') != a_raw+1
                or row['useful_committed'] != useful
                or useful != a_scored+1
                or type(row.get('terminal_commit')) is not bool
                or row['terminal_commit'] != (j == len(rows)-1)
                or not isinstance(row['seconds'], (int, float))
                or not math.isfinite(row['seconds']) or row['seconds'] <= 0
                or not math.isfinite(row['decision_cpu_seconds']) or row['decision_cpu_seconds'] < 0
                or row['decision_cpu_seconds'] > row['seconds']+1e-9
                or not math.isfinite(row['update_cpu_seconds']) or row['update_cpu_seconds'] < 0):
            raise ValueError('Executed action, terminal or useful commitment mismatch')
        if (row.get('policy') != mode or row.get('context') != context or row.get('raw_x') != x
                or type(row.get('has_real_anchor')) is not bool
                or row['has_real_anchor'] != (x is not None)
                or row.get('eligible_for_calibration') is not row['has_real_anchor']
                or row.get('committed_prefix_before') != prefix-1
                or row.get('committed_prefix_after') != prefix+a_raw):
            raise ValueError('Raw feature/readiness or committed-prefix metadata mismatch')
        anchor = row.get('anchor_timestep')
        if x is not None:
            if type(anchor) is not int or j < 2 or anchor != 4*((j-2)//4):
                raise ValueError('Anchor is not a prior actual attention selection')
        elif anchor is not None:
            raise ValueError('Startup has a fabricated anchor timestamp')
        if row['reward'] != useful-rho*row['seconds']:
            raise ValueError('Reward not based on actual useful tokens and elapsed time')
        if mode in ('fixed8_fresh', 'csgb', 'constant'):
            if x is not None:
                if row.get('attention_observed_timestep') != j-1:
                    raise ValueError('Stale/future Target attention')
                if row.get('raw_features') != x:
                    raise ValueError('Policy does not receive recorded full raw features')
            if mode == 'fixed8_fresh' and row['budget'] != 8:
                raise ValueError('Calibration action is not fixed8')
        elif mode == 'history_native':
            if row.get('raw_features') is not None:
                raise ValueError('Native history paid for unneeded CSGB features')
            if x is not None and x != [1., history_last/10., history_ema/10.]:
                raise ValueError('History features contain future or incorrectly counted acceptance')
        else:
            raise ValueError('Unexpected training policy')
        if replay is not None:
            if replay.choose(x, explore=True) != row['budget']:
                raise ValueError('Executed action does not reproduce chronological RNG/parameters')
            if replay.transform(x) != (None if row.get('learner_input') is None else tuple(row['learner_input'])):
                raise ValueError('Saved standardized learner input differs')
            reward = replay.update(x, row['budget'], useful, row['seconds'], rho)
            if (reward is not None) != (x is not None):
                raise ValueError('Startup/no-anchor update disagreement')
        if row.get('updated') is not (replay is not None and x is not None):
            raise ValueError('Unexpected/missing training update')
        if x is None and row['budget'] != 8:
            raise ValueError('Missing-feature fallback must be8')
        history_last = float(max(0, useful-1))
        history_ema = .75*history_ema + .25*history_last
        prefix += a_raw+1; generated += a_raw+1
    if sum(r['useful_committed'] for r in rows) != len(output):
        raise ValueError('Useful tokens do not equal returned output tokens')
    if sum(r['seconds']+r['update_cpu_seconds'] for r in rows) > seconds+1e-9:
        raise ValueError('Round/update intervals exceed complete-request wall time')
    if replay is not None and replay.to_state() != learner_after:
        raise ValueError('Learner state diverges from independent saved-round replay')


def calibrate_rho(rows):
    """Use only post-prefill rounds with real module1 selection and features."""
    grouped = {c: [] for c in CONTEXTS}
    docs = {c: set() for c in CONTEXTS}
    for request in rows:
        d, c = request['document'], request['context']
        if (type(d) is not int or not 0 <= d < 24 or c not in CONTEXTS
                or request['phase'] != 'calibration' or request['policy'] != 'fixed8_fresh'):
            raise ValueError('Calibration may only use locked training Fixed8 requests')
        if d in docs[c]:
            raise ValueError('Duplicate calibration request')
        docs[c].add(d)
        for row in request['rounds']:
            validate_raw_x(row['x'], 'fixed8_fresh')
            if row['x'] is not None:
                if (row['proposal'] < 2 or row['attention_observed_timestep'] != row['proposal']-1
                        or row.get('has_real_anchor') is not True
                        or row.get('eligible_for_calibration') is not True
                        or row.get('calibration') is not True
                        or row.get('budget') != 8 or row.get('actual_draft_nodes') != 8
                        or type(row.get('anchor_timestep')) is not int
                        or row['anchor_timestep'] != 4*((row['proposal']-2)//4)):
                    raise ValueError('Invalid calibration feature/round alignment')
                if (type(row['useful_committed']) is not int or row['useful_committed'] < 1
                        or not math.isfinite(row['seconds']) or row['seconds'] <= 0):
                    raise ValueError('Invalid calibration useful count or round seconds')
                grouped[c].append((row['useful_committed'], row['seconds']))
    if any(docs[c] != set(range(24)) or not grouped[c] for c in CONTEXTS):
        raise ValueError('Full24-document calibration at both contexts is required')
    return {str(c): {'rho': sum(n for n, _ in grouped[c])/sum(t for _, t in grouped[c]),
                     'useful_tokens': sum(n for n, _ in grouped[c]),
                     'round_seconds': sum(t for _, t in grouped[c]),
                     'eligible_rounds': len(grouped[c]), 'documents': list(range(24))}
            for c in CONTEXTS}


def main():
    ap = argparse.ArgumentParser()
    for key in ('base', 'draft', 'source', 'scaler', 'protocol', 'out'):
        ap.add_argument('--'+key, required=True)
    ap.add_argument('--seed', type=int, choices=SEEDS, required=True)
    ap.add_argument('--expected-gpu-uuid', required=True)
    ap.add_argument('--physical-gpu-index', type=int, choices=(0, 1, 2), required=True)
    ap.add_argument('--smoke', action='store_true', help='doc0 only; no final training artifacts')
    args = ap.parse_args()
    if (not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', args.expected_gpu_uuid)
            or args.physical_gpu_index != SEEDS.index(args.seed)
            or os.environ.get('CUDA_VISIBLE_DEVICES') != args.expected_gpu_uuid):
        raise ValueError('CUDA must be bound to the exact physical UUID checked idle')
    texts = validate_training_source(args.source)
    scale_state, scales, scaler_evidence = validate_scaler(args.scaler)
    training_source_hashes = [sha for p, sha in scale_state['collection_source_sha256'].items()
                              if Path(p).name == 'train24.jsonl']
    if training_source_hashes != [digest(args.source)]:
        raise ValueError('Training data do not match the immutable scaler collection inputs')
    protocol = json.loads(Path(args.protocol).read_text())
    if (protocol.get('schema') != 1 or protocol.get('method', {}).get('seeds') != list(SEEDS)
            or protocol.get('training_documents') != list(range(24))
            or protocol.get('no_development_or_confirmation_generation_in_this_launcher') is not True):
        raise ValueError('Wrong training-only experiment protocol')
    out = Path(args.out)
    if out.exists():
        raise ValueError('Output exists; never restart or reuse a partial run')
    import numpy as np
    import torch
    from classic.model_classic import SPModel
    from baseline_profiler import exact_input
    from csgb_same_state import release_request_caches
    from csgb_corrected_request import current_gpu_uuid, assert_gpu_uncontended
    from csgb_online_harness import OnlineHarness

    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible GPU per independent worker')
    # Establish only a tiny context, prove physical identity and exclusivity
    # BEFORE expensive model allocation. UUID binding avoids CUDA-index order.
    probe = torch.empty(1, device='cuda:0')
    gpu = current_gpu_uuid()
    if gpu != args.expected_gpu_uuid:
        raise RuntimeError('Actual CUDA device differs from the checked physical GPU')
    assert_gpu_uncontended(gpu)
    del probe
    torch.manual_seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model = SPModel.from_pretrained(base_model_path=args.base, draft_model_path=args.draft,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={'': 0}).eval()
    model.base_model.float()
    if {str(p.device) for p in model.parameters()} != {'cuda:0'}:
        raise RuntimeError('All Target/Draft components must be on worker cuda:0')
    h = OnlineHarness(model)
    if current_gpu_uuid() != gpu:
        raise RuntimeError('Physical GPU identity changed while loading models')
    assert_gpu_uncontended(gpu)
    sources = {str(p.resolve()): digest(p) for p in Path(__file__).parent.rglob('*.py')}
    inputs = {str(Path(p).resolve()): digest(p) for p in (args.source, args.scaler, args.protocol)}
    out.mkdir(exist_ok=False)
    exclusive_json(out/'protocol.json', {'schema': 1, 'phase': 'training_smoke' if args.smoke else 'online_train_only',
        'seed': args.seed, 'documents': [0] if args.smoke else list(range(24)),
        'contexts': list(CONTEXTS), 'policies': list(MODES), 'budgets': list(BUDGETS),
        'training_passes': 1, 'model_seed': 42, 'target_dtype': 'float32', 'draft_dtype': 'float16',
        'max_new_tokens': 128, 'max_depth': 10, 'threshold': .7, 'cache_refresh_every': 4,
        'chunk_size': 32, 'retrieve_top_k': 32, 'corrected_root_coverage': True,
        'terminal_draft_elision': True, 'tf32': False,
        'private_seeds': {str(c): {m: private_seed(args.seed, c, m) for m in MODES} for c in CONTEXTS},
        'gpu_uuid': gpu, 'gpu_name': torch.cuda.get_device_name(0),
        'physical_gpu_index': args.physical_gpu_index, 'expected_gpu_uuid': args.expected_gpu_uuid,
        'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'source_manifest': sources, 'input_manifest': inputs,
        'scaler_evidence': scaler_evidence, 'runner_sha256': digest(__file__),
        'performance_claim': False, 'timing': 'round-synchronized online training; NOT frozen deployment TPS',
        'development_opened': False, 'confirmation_opened': False})
    references = {}
    def get_input_reference(doc, context):
        ids = exact_input(model.tokenizer, texts[doc], context, torch, torch.device('cuda:0'))
        if int(ids.shape[-1]) != context:
            raise RuntimeError('Actual context length mismatch')
        if (doc, context) not in references:
            release_request_caches(model)
            assert_gpu_uncontended(gpu)
            with torch.inference_mode():
                reference = model.ar_generate(ids, max_new_tokens=128, verbose=False, return_metrics=True)
            assert_gpu_uncontended(gpu)
            references[doc, context] = reference
            exclusive_json(out/f'reference_{context}_{doc}.json', reference)
        return ids, references[doc, context]

    def run(doc, context, mode, phase, learner=None, rho=0.):
        ids, ref = get_input_reference(doc, context)
        release_request_caches(model)
        explore = phase in ('train', 'smoke_train')
        h.reset(mode, learner=learner, explore=explore, rho=rho, clean=False,
                max_new_tokens=128, eos_token_id=model.tokenizer.eos_token_id, context=context,
                calibration=phase in ('calibration', 'smoke_calibration'),
                provenance={'document': doc, 'context': context, 'phase': phase, 'seed': args.seed})
        before = None if learner is None else copy.deepcopy(learner.to_state())
        assert_gpu_uncontended(gpu)
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode():
            result = model.spgenerate(ids, nodes=8, max_depth=10, threshold=.7,
                max_new_tokens=128, temperature=0, use_specextend=True,
                retrieval_chunk_size=32, retrieve_top_k=32, retrieve_every_n_steps=4,
                retrieval_num_queries=1, exact_target_attention=True, return_output_ids=True,
                verbose=False, output_result_line=False)
        torch.cuda.synchronize(); seconds = time.perf_counter()-start
        assert_gpu_uncontended(gpu)
        after = None if learner is None else copy.deepcopy(learner.to_state())
        validate_rounds(result, ref, h.rows, h.pending, context, seconds, mode=mode,
                        learner_before=before, learner_after=after, rho=rho)
        row = {'document': doc, 'context': context, 'policy': mode, 'phase': phase,
               'seed': args.seed, 'gpu_uuid': gpu, 'seconds': seconds,
               'instrumented_training_request_tps': result['total_generated']/seconds,
               'performance_claim': False, 'exact': True, 'rho': rho,
               'result': result, 'rounds': h.rows, 'pending_is_none': h.pending is None,
               'learner_before': before, 'learner_after': after}
        exclusive_json(out/f'{phase}_{context}_{doc}_{mode}.json', row)
        print(json.dumps({'phase': phase, 'document': doc, 'context': context, 'policy': mode,
                          'seed': args.seed, 'rounds': len(h.rows), 'exact': True,
                          'instrumented_training_request_tps': row['instrumented_training_request_tps']}), flush=True)
        return row

    # Disposable smoke models never leak into formal training or calibration.
    if args.smoke:
        for c in CONTEXTS:
            run(0, c, 'fixed8_fresh', 'smoke_calibration')
            for mode in MODES:
                learner = OnlineCSGBPolicy(mode, context=c, seed=private_seed(args.seed,c,mode),
                    rho=20., scale=scales[c] if mode == 'csgb' else None)
                run(0, c, mode, 'smoke_train', learner, 20.)
        exclusive_json(out/'DONE.json', {'passed': True, 'smoke': True, 'requests': 8,
            'no_final_models': True, 'performance_claim': False, 'development_opened': False,
            'confirmation_opened': False})
        return

    for c in CONTEXTS:
        run(0, c, 'fixed8_fresh', 'warmup')
    calibration = [run(d, c, 'fixed8_fresh', 'calibration') for d in range(24) for c in CONTEXTS]
    rho_state = calibrate_rho(calibration)
    exclusive_json(out/'rho.json', {'schema': 1, 'gpu_uuid': gpu, 'by_context': rho_state,
        'source': 'all24 training docs; Fixed8 fresh anchored generation rounds, prefill excluded',
        'calibration_sha256': {str(p.resolve()): digest(p) for p in sorted(out.glob('calibration_*.json'))}})
    learners = {(c,m): OnlineCSGBPolicy(m, context=c, seed=private_seed(args.seed,c,m),
                    rho=rho_state[str(c)]['rho'], scale=scales[c] if m == 'csgb' else None)
                for c in CONTEXTS for m in MODES}
    doc_order = list(range(24)); random.Random(args.seed).shuffle(doc_order)
    exclusive_json(out/'training_order.json', {'documents': doc_order,
        'policy_order_rule': 'Random(seed*1000000+document*100+context).shuffle(csgb,constant,history_native)',
        'contexts': list(CONTEXTS), 'passes': 1})
    for doc in doc_order:
        for c in CONTEXTS:
            order = list(MODES); random.Random(args.seed*1000000+doc*100+c).shuffle(order)
            for mode in order:
                run(doc, c, mode, 'train', learners[c,mode], rho_state[str(c)]['rho'])
    if any(min(learner.counts) < 32 for learner in learners.values()):
        raise RuntimeError('Insufficient executed exploration per arm; preserve partial run, no extra passes')
    evidence = {str(p.resolve()): digest(p) for p in sorted(out.glob('*.json'))}
    frozen = {'schema': 1, 'purpose': 'all-round actual-feedback online training; not held-out benefit',
              'seed': args.seed, 'gpu_uuid': gpu, 'train_documents': list(range(24)),
              'contexts': list(CONTEXTS), 'models': {str(c): {m: learners[c,m].to_state() for m in MODES} for c in CONTEXTS},
              'input_manifest': inputs, 'source_manifest': sources, 'training_evidence': evidence,
              'scaler_sha256': digest(args.scaler), 'rho_sha256': digest(out/'rho.json'),
              'development_opened': False, 'confirmation_opened': False}
    # Full immutable inputs/sources rechecked before writing any completed freeze.
    for path, expected in {**scaler_evidence, **inputs, **sources, **evidence}.items():
        if digest(path) != expected:
            raise RuntimeError('Evidence/source changed during online training: '+path)
    exclusive_json(out/'frozen_models.json', frozen)
    exclusive_json(out/'DONE.json', {'passed': True, 'smoke': False, 'seed': args.seed,
        'calibration_requests': 48, 'training_requests': 144, 'warmup_requests': 2,
        'frozen_models_sha256': digest(out/'frozen_models.json'), 'performance_claim': False,
        'goal_achieved': False, 'development_opened': False, 'confirmation_opened': False})
    print('TRAINING_COMPLETE_NO_HELD_OUT_PERFORMANCE_CLAIM', flush=True)


if __name__ == '__main__':
    main()
