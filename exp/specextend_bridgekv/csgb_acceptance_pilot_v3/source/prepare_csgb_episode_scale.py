"""Reproduce a fit0..15-only scale from immutable, already collected features.

No current-policy fitting, timing reuse, monitor outcomes, development, CUDA or
model imports. The legacy synchronized seconds are audited for integrity only;
the episode pilot MUST calibrate its own complete-request rho and baseline.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


TOP = Path(__file__).resolve().parent
ORIGINAL_ROOT = Path('/storage/dzj/spec_repro/CSGB_online_allround_20261016_v1')
ORIGINAL_ANALYSIS = Path('/storage/dzj/spec_repro/CSGB_online_migrated202_20261016_v1/training_analysis/manifest.json')
PINS = {
    'protocol.json': 'd3b0a3ab6ad4809025e532a3f808a3a3c8e608c6c8f4807ee42337b6ee959031',
    'frozen_models.json': '796e064b72a2540480c4daa0e4b81c50b6254ec203bd4a1a538f4c915f9517e6',
    'analysis_manifest': '8741949008bbf517cfe35020a9533976c0032e9de373f8dac7918a4d62eb5eba',
    'stage.sha256': 'a288d1a934d9c3540bca94aca5e02a451f90c4d73f1947598144f17663b19fff',
}
FIT_DOCS = tuple(range(16))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def population_scale(observations):
    """All valid verified rounds, never standardize monitor documents."""
    grouped = {d: [] for d in FIT_DOCS}
    for d, x in observations:
        if (type(d) is not int or d not in FIT_DOCS or not isinstance(x, list) or len(x) != 4
                or x[0] != 1. or any(type(v) not in (int, float) or not math.isfinite(v) for v in x)
                or any(not 0 <= v <= 1 for v in x[1:])
                or not math.isclose(x[3], x[1]*x[2], rel_tol=0, abs_tol=1e-12)):
            raise ValueError('Only valid fit-only raw C,D,C*D may define this scale')
        grouped[d].append(x[1:])
    if any(not v for v in grouped.values()):
        raise ValueError('Every fit document0..15 needs verified feature observations')
    columns = list(zip(*(x for values in grouped.values() for x in values)))
    means = [math.fsum(col)/len(col) for col in columns]
    scales = [math.sqrt(math.fsum((v-m)**2 for v in col)/len(col)) for col,m in zip(columns, means)]
    if any(v <= 1e-12 for v in scales):
        raise ValueError('Degenerate fit-only scale; do not silently replace it')
    return {'mean': means, 'scale': scales, 'count': len(columns[0]),
            'per_document_count': {str(d): len(v) for d,v in grouped.items()}}


def build(remote_root):
    remote_root = Path(remote_root)
    if (not remote_root.is_absolute() or remote_root.parent != Path('/storage/dzj/spec_repro')
            or not remote_root.name.startswith('CSGB_episode_pilot_')):
        raise ValueError('Explicit isolated episode pilot root required')
    collected = TOP/'results/csgb_online_allround_20261016_v1'
    raw = collected/'train_completed/seed_101'
    capsule = TOP/'results/csgb_online_migrated_development_20261016_v1/frozen_source'
    legacy_capsule = collected/'frozen_source'
    analysis = TOP/'results/csgb_online_migrated202_20261016_v1/training_analysis/manifest.json'
    checks = [(raw/name, digest) for name,digest in PINS.items() if name not in ('analysis_manifest','stage.sha256')]
    checks += [(analysis,PINS['analysis_manifest']), (collected/'stage.sha256',PINS['stage.sha256'])]
    for path, wanted in checks:
        if sha(path) != wanted:
            raise ValueError('Changed independently audited feature provenance: '+str(path))
    protocol, frozen, manifest = load(raw/'protocol.json'), load(raw/'frozen_models.json'), load(analysis)
    source = legacy_capsule/'train24.jsonl'
    records = [json.loads(line) for line in source.read_text().splitlines()]
    if len(records) != 24 or any(not isinstance(r.get('text'),str) or not r['text'] for r in records):
        raise ValueError('Stage exactly the historical24 training texts, never full48 corpus')
    corpus_sha = sha(source)
    original_corpus = str(ORIGINAL_ROOT/'train24.jsonl')
    stage_entries = {line.split(maxsplit=1)[1].strip():line.split(maxsplit=1)[0]
                     for line in (collected/'stage.sha256').read_text().splitlines()}
    if (protocol['input_manifest'].get(original_corpus) != corpus_sha
            or manifest['input_and_source_sha256'].get(original_corpus) != corpus_sha
            or stage_entries.get('train24.jsonl') != corpus_sha):
        raise ValueError('Training corpus differs from original protocol/stage/completed audit')
    sources = protocol['source_manifest']
    for original, wanted in sources.items():
        relative = Path(original).relative_to(ORIGINAL_ROOT)
        candidates = (legacy_capsule/relative, capsule/relative)
        path = next((p for p in candidates if p.is_file() and sha(p) == wanted), None)
        if path is None:
            raise ValueError('Cannot reproduce original collector source '+original)
    # Prove the *executed* auditor and import dependency, not another copy with
    # the same filename. Bind sys.modules temporarily so callers cannot inject
    # a previously imported working-tree policy into this legacy auditor.
    old_py = capsule/'specextend'
    for name in ('csgb_online_train.py','csgb_online_policy.py'):
        if sha(old_py/name) != sources[str(ORIGINAL_ROOT/'specextend'/name)]:
            raise ValueError('Actually executed frozen integrity module was not source-verified')
    policy_spec = importlib.util.spec_from_file_location('csgb_online_policy',old_py/'csgb_online_policy.py')
    policy_module = importlib.util.module_from_spec(policy_spec)
    previous = sys.modules.get('csgb_online_policy')
    try:
        sys.modules['csgb_online_policy'] = policy_module
        policy_spec.loader.exec_module(policy_module)
        spec = importlib.util.spec_from_file_location('_episode_legacy_integrity',old_py/'csgb_online_train.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    finally:
        if previous is None: sys.modules.pop('csgb_online_policy',None)
        else: sys.modules['csgb_online_policy'] = previous
    observations, inputs, terminal_count, total_rounds = [], {}, 0, 0
    for d in FIT_DOCS:
        filename = f'calibration_4096_{d}_fixed8_fresh.json'
        path = raw/filename
        original = str(ORIGINAL_ROOT/'seed_101'/filename)
        wanted = sha(path)
        if (frozen['training_evidence'].get(original) != wanted
                or manifest['input_and_source_sha256'].get(original) != wanted):
            raise ValueError('Fit feature row not bound to both completed immutable audits')
        row = load(path); reference_path = raw/f'reference_4096_{d}.json'; reference = load(reference_path)
        original_reference = str(ORIGINAL_ROOT/'seed_101'/reference_path.name)
        if (frozen['training_evidence'].get(original_reference) != sha(reference_path)
                or manifest['input_and_source_sha256'].get(original_reference) != sha(reference_path)):
            raise ValueError('Canonical fit reference differs from both completed audits')
        if ((row.get('document'),row.get('context'),row.get('policy'),row.get('phase'),row.get('seed'))
                != (d,4096,'fixed8_fresh','calibration',101)
                or row.get('exact') is not True or row.get('pending_is_none') is not True
                or row.get('learner_before') is not None or row.get('learner_after') is not None
                or len(reference['output_token_ids']) != 128):
            raise ValueError('Wrong original fit-only calibration identity/exactness')
        module.validate_rounds(row['result'],reference,row['rounds'],None,4096,row['seconds'],
                               mode='fixed8_fresh',rho=0.)
        total_rounds += len(row['rounds'])
        for r in row['rounds']:
            if r['raw_x'] is not None:
                observations.append((d,r['raw_x']))
                terminal_count += int(r['terminal_commit'])
        inputs[original] = wanted
        inputs[original_reference] = sha(reference_path)
    stats = population_scale(observations)
    if (stats['count'],total_rounds,terminal_count) != (699,731,16):
        raise ValueError('Original fit-only full verified feature set changed')
    inputs.update({str(ORIGINAL_ROOT/'seed_101'/name):PINS[name] for name in ('protocol.json','frozen_models.json')})
    inputs[str(ORIGINAL_ROOT/'stage.sha256')] = PINS['stage.sha256']
    inputs[str(ORIGINAL_ANALYSIS)] = PINS['analysis_manifest']
    source_manifest = dict(sources)
    source_manifest[str(remote_root/Path(__file__).name)] = sha(__file__)
    return {'schema':1,'purpose':'episode fit16 fixed population scale','context':4096,
            'fit_documents':list(FIT_DOCS),'monitor_documents_used':False,
            'development_opened':False,'confirmation_opened':False,'source_sha256':corpus_sha,**stats,
            'input_sha256':inputs,'source_manifest':source_manifest,
            'raw_total_verified_rounds':731,'verified_startup_missing_features':32,
            'final_verified_rounds_included':16,'original_timing_or_rho_reused':False,
            'scaling':'population mean/std of raw C,D,C*D, all699 fit-only valid verified rounds',
            'historical_source_scope':'fit0..15 subset of prior training24, not globally unseen data'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--remote-root',required=True)
    ap.add_argument('--out',required=True,type=Path)
    args = ap.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError('Never overwrite an existing feature scale')
    artifact = build(args.remote_root)
    with args.out.open('x') as handle:
        json.dump(artifact,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
    print(json.dumps({'count':artifact['count'],'mean':artifact['mean'],'scale':artifact['scale'],
                      'sha256':sha(args.out),'monitor_used':False,'timing_reused':False},sort_keys=True))


if __name__ == '__main__':
    main()
