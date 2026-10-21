"""CPU-only audit of the matched32-warm development horizon supplement.

The original six-warm selection remains immutable. This study keeps the union
of its choices and the independently scored32-warm choices; it is NOT a fresh
confirmation or a RESCUE speed-success experiment. Saved-data analysis imports
no torch and never reads model, original corpus, credential or server paths.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

if __package__:
    from . import baseline_selection_report as support
else:
    import analysis_support as support

LENGTHS = (4096, 8192)
GROUPS = ('retrieval/eager', 'retrieval/resident', 'retrieval/graph', 'full_tree')
SEED = 20261025
DRAWS = 10000
EXPECTED = dict(reference_requests=32, prime_requests=76, timing_requests=2432, owners=76)
TOTAL = 2540
DOCUMENTS = 16
REPEATS = 2
DOCS = (211, 213, 220, 226, 233, 236, 246, 248, 253, 271, 272, 288, 303, 306, 327, 604)
TRAIN = 'exp/long_system/data/train.jsonl'
OPT = 'exp/module_opt/data_v1_20261018/confirmation.jsonl'
ARCH = 'exp/module_arch/data_v1_20261018/confirmation.jsonl'
TUNE = 'exp/module_opt/data_v1_20261018/tune.jsonl'
DATA_SHA = {
    TRAIN: 'ae1e3f58a663dd505ddc9d1f1fdb87af55cad1d967c6b771f696bcbbadce7083',
    OPT: '03eb1b126b424cc99b3586fb42594430ed74706f3ade587dc7102abfa5df90bb',
    ARCH: '007fd9247bf8d77dc53eef51469c84c8a66d29bb2d9577815349c4601ecb29f9',
    TUNE: '85b2eb1a81d1617b2d158c069c420c452459a8bdb90da128f257220cc27ded09',
}
ORIGINS = {
    TRAIN: dict(original_corpus_split='train', original_shard='train-00000-of-00002.parquet', preparation_source='rescue/long_data.py'),
    TUNE: dict(original_corpus_split='train', original_shard='train-00000-of-00002.parquet', preparation_source='rescue/module_opt_data.py -> rescue/long_data.py'),
    OPT: dict(original_corpus_split='validation', original_shard='validation-00000-of-00001.parquet', preparation_source='rescue/module_opt_data.py'),
    ARCH: dict(original_corpus_split='validation', original_shard='validation-00000-of-00001.parquet', preparation_source='rescue/module_arch_data.py'),
}
positive, nonnegative, near = support.positive, support.nonnegative, support.near


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group(arm):
    return 'full_tree' if arm['family'] == 'full_tree' else 'retrieval/' + ('graph' if arm['backend'] == 'cuda_graph' else arm['backend'])


def pool_document(rows, *, complete=True):
    """The primary requires both repeats; a prefix diagnostic uses what ran."""
    if not rows or len({(r.get('mode'), r.get('length'), r.get('source_index')) for r in rows}) != 1:
        raise ValueError('One document/context/configuration required')
    repeats = [r.get('repeat') for r in rows]
    if len(set(repeats)) != len(repeats) or any(r not in (0, 1) for r in repeats) or (complete and set(repeats) != {0, 1}):
        raise ValueError('Both unique repeats required; no imputation')
    tokens, seconds = sum(r['tokens'] for r in rows), sum(r['outer_seconds'] for r in rows)
    if not positive(tokens) or not positive(seconds):
        raise ValueError('Positive actual client token/time pairs required')
    return dict(tokens=tokens, seconds=seconds, tps=tokens/seconds,
        accepted=sum(r['accepted'] for r in rows), blocks=sum(r['blocks'] for r in rows),
        proposed=None if any(r.get('proposed') is None for r in rows) else sum(r['proposed'] for r in rows))


def paired_interval(cells, other=None, *, seed=SEED, draws=DRAWS, index=None):
    """Whole-document resampling of ratio-of-sums, never mean request TPS."""
    import numpy as np
    if seed != SEED or draws != DRAWS or not cells:
        raise ValueError('Frozen seed/draws and nonempty document cells required')
    if any(not positive(c.get('tokens')) or not positive(c.get('seconds')) for c in cells):
        raise ValueError('Invalid document token/time pair')
    n = len(cells)
    token = np.array([c['tokens'] for c in cells], float)
    seconds = np.array([c['seconds'] for c in cells], float)
    point = float(token.sum()/seconds.sum())
    if other is not None:
        if len(other) != n or any(not positive(c.get('tokens')) or not positive(c.get('seconds')) for c in other):
            raise ValueError('Aligned document pairs required')
        ot = np.array([c['tokens'] for c in other], float)
        os = np.array([c['seconds'] for c in other], float)
        point = 100.*(point/(ot.sum()/os.sum())-1.)
    result = dict(documents=n, point=point, ci95=None)
    if n > 1:
        if index is None:
            index = np.random.default_rng(seed).integers(0, n, size=(draws, n))
        if index.shape != (draws, n):
            raise ValueError('Shared resample matrix has wrong document count')
        sample = token[index].sum(1)/seconds[index].sum(1)
        if other is not None:
            sample = 100.*(sample/(ot[index].sum(1)/os[index].sum(1))-1.)
        result['ci95'] = [float(v) for v in np.percentile(sample, [2.5, 97.5])]
    return result


def quality_interval(cells, denominator, index=None):
    import numpy as np
    if not cells or any(not nonnegative(c.get('accepted')) or not positive(c.get(denominator)) for c in cells):
        return dict(point=None, ci95=None)
    num = np.array([c['accepted'] for c in cells], float)
    den = np.array([c[denominator] for c in cells], float)
    result = dict(point=float(num.sum()/den.sum()), ci95=None)
    if len(cells) > 1:
        if index is None:
            index = np.random.default_rng(SEED).integers(0, len(cells), size=(DRAWS, len(cells)))
        sample = num[index].sum(1)/den[index].sum(1)
        result['ci95'] = [float(v) for v in np.percentile(sample, [2.5, 97.5])]
    return result


def score_rows(rows):
    if not rows or any(not positive(r.get('tokens')) or not positive(r.get('outer_seconds')) for r in rows):
        raise ValueError('Only actual finite warm intervals may be scored')
    return sum(r['tokens'] for r in rows)/sum(r['outer_seconds'] for r in rows)


def select_union(warm, documents, stage1, arms=None):
    """Exact32-warm score; preserve stage1 receipts before deduplication."""
    arms = support.menu() if arms is None else arms
    result = {}
    for length in LENGTHS:
        candidates = []
        for arm in arms:
            rows = [r for r in warm if r.get('length') == length and r.get('mode') == arm['mode']]
            if len(rows) != 32 or {(r.get('source_index'), r.get('repeat')) for r in rows} != {(d, r) for d in documents for r in (0, 1)}:
                raise ValueError('All32 warm requests required before authoritative selection')
            def prefix(values):
                return dict(tokens=sum(r['tokens'] for r in values), seconds=sum(r['outer_seconds'] for r in values),
                    tps=score_rows(values), captures=sum(r.get('draft_runtime', {}).get('request', {}).get('captures', 0) for r in values),
                    capture_seconds=sum(r.get('draft_runtime', {}).get('request', {}).get('capture_host_seconds', 0.) for r in values))
            candidates.append(dict(mode=arm['mode'], arm=arm, warm_requests=32,
                warm_tokens=sum(r['tokens'] for r in rows), warm_seconds=sum(r['outer_seconds'] for r in rows),
                warm_tps=score_rows(rows), request_ids=[r['request_id'] for r in rows], prefix6=prefix(rows[:6]), prefix32=prefix(rows)))
        best = lambda values: min(values, key=lambda r: (-r['warm_tps'], r['mode']))
        groups, union = {}, {}
        old = stage1['per_context'][str(length)]['groups']
        for name in GROUPS:
            groups[name] = best([r for r in candidates if group(r['arm']) == name])
            union[name] = [old[name]] + ([] if old[name]['mode'] == groups[name]['mode'] else [groups[name]])
        modes = list(dict.fromkeys(r['mode'] for name in GROUPS for r in union[name]))
        result[str(length)] = dict(groups=groups, stage1_groups=old, union_by_group=union, union_modes=modes,
            ranking=sorted(candidates, key=lambda r: (-r['warm_tps'], r['mode'])),
            overall=best(candidates),
            overall_family_winner={family: best([r for r in candidates if r['arm']['family'] == family]) for family in ('retrieval', 'full_tree')})
    return result


class Evidence:
    def __init__(self, root):
        self.root, self.problems = Path(root), []

    def problem(self, value):
        if value not in self.problems:
            self.problems.append(value)

    def path(self,name):
        if not isinstance(name,str) or Path(name).is_absolute():
            raise ValueError('Evidence path must be relative')
        path=(self.root/name).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError('Evidence path escapes the saved root')
        return path

    def json(self, name, default=None):
        default = {} if default is None else default
        try:
            value = json.loads(self.path(name).read_text())
            if not isinstance(value, type(default)):
                raise ValueError('Wrong JSON container')
            return value
        except (OSError, TypeError, ValueError):
            self.problem('Missing/malformed '+str(name))
            return default

    def rows(self, name, *, optional=False):
        try:
            path=self.path(name)
            if optional and not path.is_file():return []
            lines = path.read_text().splitlines()
        except (OSError,TypeError,ValueError):
            self.problem('Missing '+str(name))
            return []
        rows = []
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Not an object')
                rows.append(row)
            except ValueError:
                self.problem(f'Malformed {name}:{i}')
        return rows

    def pin(self, name, digest):
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            self.problem('Invalid local SHA pin: '+str(name))
            return False
        path = (self.root/name).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file() or sha(path) != digest:
            self.problem('Missing/changed/unsafe pinned evidence: '+name)
            return False
        return True


def stage1_chain(evidence, protocol):
    """Independent hashes/choice preservation, using the copied portable tree."""
    receipt = protocol.get('stage1_evidence', {})
    if receipt.get('root') != 'stage1_evidence':
        evidence.problem('Portable full stage1 evidence root missing')
        return {}, {}
    base = Evidence(evidence.root/'stage1_evidence')
    paths = {}
    for key in ('baseline_results', 'analysis_dir', 'verified_complete', 'final_manifest'):
        name = receipt.get(key)
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            base.problem('Unsafe/missing copied stage1 path: '+key)
            name = '__missing__'
        paths[key] = name
    final = base.json(paths['final_manifest'])
    inventory = final.get('file_sha256', {})
    if not inventory:
        base.problem('Empty stage1 final artifact manifest')
    for name, digest in inventory.items():
        base.pin(name, digest)
    evidence.pin(receipt.get('pin_index'), receipt.get('pin_index_sha256'))
    pins = evidence.json(receipt.get('pin_index')).get('original_to_saved', {})
    if not pins:
        base.problem('Stage1 original-to-portable pin index missing')
    for pin in pins.values():
        evidence.pin(pin.get('saved_path'), pin.get('sha256'))
    prefix, analysis = paths['baseline_results'], paths['analysis_dir']
    complete = base.json(prefix+'/COMPLETE.json')
    frozen = base.json(prefix+'/FROZEN_SELECTION.json')
    old_protocol = base.json(prefix+'/protocol.json')
    summary = base.json(analysis+'/summary.json')
    winners = base.json(analysis+'/VERIFIED_WINNERS.json')
    analyzed = base.json(analysis+'/ANALYSIS_COMPLETE.json')
    verified = base.json(paths['verified_complete'])
    pipeline = base.json('PIPELINE_EXIT.json')
    proof = protocol.get('stage1_freeze', {})
    if (type(pipeline.get('exit_code')) is not int or pipeline.get('exit_code') != 0 or
        'PIPELINE_EXIT.json' not in inventory or proof.get('status') != 'BASELINE_VERIFIED'):
        base.problem('Stage1 did not reach independently verified pipeline exit0')
    counts = dict(reference_requests=6, prime_requests=76, timing_requests=456, owners=76)
    if (complete.get('passed') is not True or complete.get('counts') != counts or complete.get('logical_requests') != 538 or
        complete.get('owner_closed') != 76 or complete.get('goal_complete') is not False or any(complete.get(k) is not True for k in
            ('sources_unchanged', 'assets_unchanged', 'dependencies_unchanged', 'all_canonical', 'backend_trajectories_equal'))):
        base.problem('Stage1 execution/exactness/immutability proof invalid')
    if (summary.get('status') != 'COMPLETE' or summary.get('problems') != [] or summary.get('winner_status') != 'VERIFIED_FROZEN' or
        summary.get('observed_counts') != counts or winners.get('status') != 'VERIFIED_FROZEN' or
        winners.get('authoritative') is not True or analyzed.get('status') != 'COMPLETE'):
        base.problem('Stage1 authoritative rendered winner analysis incomplete')
    for name, digest in analyzed.get('artifacts_sha256', {}).items():
        base.pin(analysis+'/'+name, digest)
    flags = ('passed', 'raw_independent_audit_passed', 'rendered_figures_inspected', 'strict_analysis_complete',
        'all538_requests_verified', 'all76_owners_verified', 'eight_frozen_choices_verified')
    if verified.get('status') != 'VERIFIED_COMPLETE' or any(verified.get(k) is not True for k in flags):
        base.problem('Stage1 independent raw/render inspection receipt invalid')
    file_links = dict(selection_freeze_sha256=prefix+'/FROZEN_SELECTION.json', verified_winners_sha256=analysis+'/VERIFIED_WINNERS.json',
        baseline_complete_sha256=prefix+'/COMPLETE.json', analysis_summary_sha256=analysis+'/summary.json',
        verified_complete_sha256=paths['verified_complete'], final_manifest_sha256=paths['final_manifest'],
        baseline_protocol_sha256=prefix+'/protocol.json', pipeline_exit_sha256='PIPELINE_EXIT.json')
    for field, name in file_links.items():
        base.pin(name, proof.get(field))
    links = dict(baseline_complete_sha256=proof.get('baseline_complete_sha256'), frozen_selection_sha256=proof.get('selection_freeze_sha256'),
        analysis_summary_sha256=proof.get('analysis_summary_sha256'), verified_winners_sha256=proof.get('verified_winners_sha256'),
        final_artifacts_manifest_sha256=proof.get('final_manifest_sha256'))
    if any(verified.get(k) != v for k, v in links.items()):
        base.problem('Stage1 independent verification links differ')
    if (frozen.get('complete_request_count') != 538 or frozen.get('group_candidates') != 8 or frozen.get('no_new_confirmation_opened') is not True or
        frozen.get('protocol_sha256') != proof.get('baseline_protocol_sha256') or complete.get('frozen_selection_sha256') != proof.get('selection_freeze_sha256')):
        base.problem('Stage1 choices no longer bound to the original complete six-warm study')
    for length in LENGTHS:
        for name in GROUPS:
            row = frozen.get('per_context', {}).get(str(length), {}).get('groups', {}).get(name, {})
            if (row.get('arm') != proof.get('choices', {}).get(str(length), {}).get(name) or
                row.get('mode') != winners.get('winners', {}).get('selected', {}).get(str(length), {}).get(name)):
                base.problem('One of the original eight choices was modified/replaced')
    if protocol.get('flash_attention') != old_protocol.get('flash_attention'):
        base.problem('Official Flash dependency changed between stages')
    current_assets = protocol.get('assets_sha256', {})
    if not old_protocol.get('assets_sha256') or any(current_assets.get(p) != d for p, d in old_protocol['assets_sha256'].items()):
        base.problem('Shared model/tokenizer/old271 asset receipts differ between stages')
    for name, digest in old_protocol.get('source_sha256', {}).items():
        if (name.startswith('rescue/') or name in ('exp/attention_horizon/selection.py','exp/selective_kv/bridge.py')) and protocol.get('source_sha256',{}).get(name) != digest:
            base.problem('Shared algorithm/runtime changed after stage1: '+name)
    tree_prefix='exp/specextend_bridgekv/specextend/'
    if {k:v for k,v in old_protocol.get('source_sha256',{}).items() if k.startswith(tree_prefix)} != {k:v for k,v in protocol.get('source_sha256',{}).items() if k.startswith(tree_prefix)}:
        base.problem('Archived full-tree source set changed between stages')
    if protocol.get('configs') != old_protocol.get('configs'):
        base.problem('One of the38 actual algorithm configurations differs from stage1')
    if any(protocol.get('environment',{}).get(k) != old_protocol.get('environment',{}).get(k) for k in ('nvidia_smi_gpu','torch','cuda','target_dtype','draft_dtype','prefix_dtype','tf32')):
        base.problem('Physical GPU/runtime/precision differs from stage1')
    evidence.problems.extend('Stage1: '+p for p in base.problems)
    return frozen, old_protocol


def owner_errors(owner, rows):
    errors = []
    if len(rows) != 33 or not rows or rows[0].get('phase') != 'cold_prime' or any(r.get('phase') != 'warm' for r in rows[1:]):
        return ['One charged cold prime plus all32 warm requests required']
    first, warm = rows[0], rows[1:]
    warm_seconds = sum(r['outer_seconds'] for r in warm)
    warm_tokens = sum(r['tokens'] for r in warm)
    constructor, close = owner.get('constructor_and_detach_seconds'), owner.get('close_seconds')
    if not nonnegative(constructor) or not nonnegative(close):
        return ['Constructor/close measured intervals missing']
    active = constructor+first['outer_seconds']+warm_seconds+close
    tokens = first['tokens']+warm_tokens
    values = dict(prime_request_id=first['request_id'], prime_tokens=first['tokens'], warm_tokens=warm_tokens,
        warm_request_ids=[r['request_id'] for r in warm])
    if any(owner.get(k) != v for k, v in values.items()) or any(not near(owner.get(k), v) for k, v in dict(
        cold_prime_outer_seconds=first['outer_seconds'], warm_outer_seconds_sum=warm_seconds,
        accounted_service_seconds=active, accounted_service_tokens=tokens, accounted_service_tps=tokens/active).items()):
        errors.append('Owner token/time/request ledger differs')
    if (owner.get('maximum_live_owners') != 1 or not owner.get('owner_signature') or
        any(r.get('owner_signature') != owner.get('owner_signature') for r in rows) or
        owner.get('engine_closed') is not True or owner.get('hook_restored') is not True or
        owner.get('closed_runtime_stats', {}).get('entries') != 0 or owner.get('final_runtime_stats') != warm[-1].get('runtime_stats')):
        errors.append('Private owner identity/closure/Graph release not proved')
    final = owner.get('final_runtime_stats', {})
    if final.get('mode') in ('resident', 'cuda_graph'):
        for name in ('captures', 'replays', 'resident_steps', 'capture_host_seconds'):
            if not near(final.get(name), sum(r.get('draft_runtime', {}).get('request', {}).get(name, 0) for r in rows)):
                errors.append('Lifetime/request counter mismatch: '+name)
    return errors


def audit_export(evidence):
    root = Evidence(evidence.root.parent)
    inventory = root.json('EVIDENCE_MANIFEST.json').get('file_sha256', {})
    if not inventory:
        root.problem('Exported source/raw evidence inventory is empty')
    for name, digest in inventory.items():
        root.pin(name, digest)
    evidence.problems.extend('Export: '+p for p in root.problems)
    pipeline=root.json('PIPELINE_EXIT.json')
    if type(pipeline.get('exit_code')) is not int or pipeline.get('exit_code') != 0:
        evidence.problem('Current scientific pipeline has not exited successfully')


def audit_telemetry(evidence, protocol, owners):
    rows=evidence.rows('gpu_diagnostics.jsonl');result=[]
    if len(rows)!=152:evidence.problem('All152 owner start/end GPU diagnostics required')
    fields=['uuid','temperature.gpu','clocks.current.sm','clocks.current.memory','power.draw','power.limit','utilization.gpu','memory.used','memory.total']
    gpu=protocol.get('environment',{}).get('nvidia_smi_gpu','').split(',')
    expected_uuid=gpu[1].strip() if len(gpu)>1 else None
    seen=set()
    for row in rows:
        owner=next((o for o in owners if o.get('owner_id')==row.get('owner_id')),None)
        telemetry=row.get('telemetry',{});values=telemetry.get('values',{});key=row.get('owner_id'),row.get('phase')
        if (owner is None or key in seen or row.get('phase') not in ('owner_start','owner_end') or
            row.get('mode')!=owner.get('mode') or row.get('length')!=owner.get('length') or
            telemetry.get('read_only') is not True or telemetry.get('primary_clock_included') is not False or
            telemetry.get('query_fields')!=fields or set(values)!=set(fields) or values.get('uuid')!=expected_uuid or
            not positive(telemetry.get('wall_time_unix'))):
            evidence.problem('Read-only GPU drift diagnostics identity/scope differs')
        seen.add(key)
        record=dict(mode=row.get('mode'),length=row.get('length'),phase=row.get('phase'),execution_index=row.get('execution_index'))
        for field in fields[1:]:
            try:value=float(values[field])
            except (TypeError,ValueError,KeyError):value=None
            record[field]=value if nonnegative(value) else None
        result.append(record)
    return result


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def input_sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def audit_data(evidence, protocol):
    """Rebuild selected old tokenized rows; never reopen original path strings."""
    if (protocol.get('data_manifest') != 'data/DATASET_MANIFEST.json' or
        protocol.get('data_source_paths') != {n: 'data/'+n for n in DATA_SHA} or
        protocol.get('selected_documents') != 'data/documents.jsonl' or protocol.get('prime_document') != 'data/prime.json'):
        evidence.problem('Portable old-data manifest/source layout differs')
    evidence.pin('data/DATASET_MANIFEST.json', protocol.get('data_manifest_sha256'))
    manifest = evidence.json('data/DATASET_MANIFEST.json')
    if (manifest.get('status') != 'FROZEN_OLD_DATA_ONLY_BEFORE_GENERATION' or
        manifest.get('receipt_sha256') != json_sha({k: v for k, v in manifest.items() if k != 'receipt_sha256'}) or
        manifest.get('source_indices') != list(DOCS) or manifest.get('expected_train_indices') != list(DOCS[:12]) or
        manifest.get('source_sha256') != DATA_SHA or manifest.get('document_count') != 16 or
        manifest.get('prime_source_index') != 271 or manifest.get('prime_in_warm_documents') is not True or
        manifest.get('tokenizer_inventory_bound') is not True or not manifest.get('tokenizer_sha256')):
        evidence.problem('Frozen old-data-only cohort/receipt/tokenizer contract differs')
    if manifest.get('original_corpus_provenance') != ORIGINS:
        evidence.problem('Historical train/validation shard provenance differs')
    scope = manifest.get('scope', {})
    if any(scope.get(k) is not False for k in ('retokenized_raw_text', 'fresh_confirmation_opened', 'runtime_or_paused_outcomes_read', 'model_generation_performed', 'training_performed')):
        evidence.problem('Data preparation accessed new/raw/outcome data or ran a model')
    rule = manifest.get('selection_rule', {})
    if rule.get('replacement_allowed') is not False or rule.get('selection_uses_outcomes_or_eos') is not False or rule.get('contexts') != list(LENGTHS) or rule.get('minimum_natural_tokens') != 8192:
        evidence.problem('Predefined old-data selection/replacement rule differs')
    expected_order = {TRAIN: DOCS[:12], OPT: (303, 306, 327), ARCH: (604, 613, 614), TUNE: (271, 272, 288)}
    all_rows = {}
    forbidden = {'generated', 'output_token_ids', 'request_id', 'outer_seconds', 'total_seconds', 'accepted', 'proposed', 'trace', 'performance_eligible', 'runtime_stats', 'output_agrees', 'execution_index', 'owner_id'}
    def no_outcomes(value):
        if isinstance(value, dict):
            return not forbidden.intersection(value) and all(no_outcomes(v) for v in value.values())
        if isinstance(value, list):
            return all(no_outcomes(v) for v in value if isinstance(v, (dict, list)))
        return True
    for name, digest in DATA_SHA.items():
        evidence.pin('data/'+name, digest)
        rows = evidence.rows('data/'+name)
        if tuple(r.get('source_index') for r in rows) != expected_order[name]:
            evidence.problem('Old source cohort count/order changed: '+name)
        split = 'train' if name in (TRAIN, TUNE) else 'confirmation'
        metadata = manifest.get('metadata_sources', {}).get(name, {})
        if (metadata.get('sha256') != digest or metadata.get('source_indices') != list(expected_order[name]) or
            metadata.get('rows') != len(expected_order[name]) or metadata.get('split') != split or
            not str(metadata.get('path', '')).endswith('/'+name)):
            evidence.problem('Old source identity receipt differs: '+name)
        for row in rows:
            arrays = [row.get(k) for k in ('prefix_ids', 'document_ids', 'suffix_ids')]
            if (row.get('split') != split or not no_outcomes(row) or
                type(row.get('original_tokens')) is not int or row['original_tokens'] < 8192 or
                not all(isinstance(v, list) and all(type(t) is int and t >= 0 for t in v) for v in arrays) or
                len(arrays[1]) < 8192 or len(arrays[1]) > row['original_tokens']):
                evidence.problem('Invalid natural old token arrays/split/outcome exclusion')
        all_rows[name] = rows
    selected = all_rows.get(TRAIN, [])+all_rows.get(OPT, [])+all_rows.get(ARCH, [])[:1]
    prime = all_rows.get(TUNE, [{}])[0] if all_rows.get(TUNE) else {}
    saved = evidence.rows('data/documents.jsonl')
    saved_prime = evidence.json('data/prime.json')
    if selected != saved or prime != saved_prime or [r.get('source_index') for r in selected] != list(DOCS):
        evidence.problem('Executed old selected documents/prime were substituted')
    if next((r for r in selected if r.get('source_index') == 271), None) != prime:
        evidence.problem('Paid cold271 is not identical to its disclosed warm document271')
    template = dict(prefix_ids=prime.get('prefix_ids'), suffix_ids=prime.get('suffix_ids'))
    if manifest.get('prompt_template') != template or manifest.get('template_sha256') != json_sha(template):
        evidence.problem('Common old271 prompt template changed')
    details, inputs, bodies, doc_hashes = [], {}, set(), set()
    ordinals = [(TRAIN, i) for i in range(12)]+[(OPT, i) for i in range(3)]+[(ARCH, 0)]
    for row, (cohort, ordinal) in zip(selected, ordinals):
        doc = row['source_index']
        body_sha = input_sha(row['document_ids'])
        if body_sha in bodies or row.get('document_sha256') in doc_hashes:
            evidence.problem('Duplicate old document body/text identity')
        bodies.add(body_sha)
        doc_hashes.add(row.get('document_sha256'))
        if dict(prefix_ids=row['prefix_ids'], suffix_ids=row['suffix_ids']) != template:
            evidence.problem('Old documents do not share the fixed prompt template')
        hashes = {}
        for length in LENGTHS:
            take = length-len(row['prefix_ids'])-len(row['suffix_ids'])
            if not 0 < take <= len(row['document_ids']):
                evidence.problem('Old natural tokens cannot supply exact4K/8K input')
                continue
            hashes[str(length)] = input_sha(row['prefix_ids']+row['document_ids'][:take]+row['suffix_ids'])
            inputs[doc, length] = hashes[str(length)]
        details.append(dict(source_identity=dict(cohort=cohort, row_ordinal=ordinal, source_index=doc, split=row['split'], **ORIGINS[cohort]),
            document_sha256=row['document_sha256'], original_tokens=row['original_tokens'], stored_document_tokens=len(row['document_ids']),
            stored_document_ids_sha256=body_sha, input_sha256=hashes))
    expected_inputs = {f'{d}:{n}': v for (d, n), v in inputs.items()}
    if (manifest.get('selected_documents') != details or manifest.get('input_sha256') != expected_inputs or
        manifest.get('prime_input_sha256') != {str(n): inputs.get((271, n)) for n in LENGTHS}):
        evidence.problem('Selected old input/body fingerprints differ from preserved metadata')
    prime_detail=next((dict(d,source_identity=dict(cohort=TUNE,row_ordinal=0,source_index=271,split='train',**ORIGINS[TUNE])) for d in details if d['source_identity']['source_index']==271),None)
    if manifest.get('prime')!=prime_detail:evidence.problem('Frozen prime source/body receipt differs')
    evidence.pin('data/documents.jsonl',protocol.get('selected_documents_sha256'))
    evidence.pin('data/prime.json',protocol.get('prime_document_sha256'))
    if protocol.get('data_receipt_sha256')!=manifest.get('receipt_sha256') or protocol.get('tokenizer_sha256')!=manifest.get('tokenizer_sha256'):
        evidence.problem('Executed data receipt/tokenizer pin differs')
    if any(len({inputs.get((d, n)) for d in DOCS}) != 16 for n in LENGTHS):
        evidence.problem('Duplicate exact context among old development inputs')
    return [r['source_index'] for r in selected], inputs


def summarize(warm, owners, bad, stage1, stage2, arms, sequence=None):
    import itertools
    import numpy as np
    metrics, pairs = {}, {}
    common_index = np.random.default_rng(SEED).integers(0, DOCUMENTS, size=(DRAWS, DOCUMENTS))
    cell_map = {}
    owner_map = {(o.get('mode'), o.get('length')): o for o in owners}
    for length in LENGTHS:
        values = metrics[str(length)] = {}
        for arm in arms:
            mode = arm['mode']
            rows = [r for r in warm if r.get('mode')==mode and r.get('length')==length]
            bydoc = {d: [r for r in rows if r.get('source_index')==d] for d in DOCS}
            docs = [d for d in DOCS if (mode, length, d) not in bad and len(bydoc[d])==2 and {r.get('repeat') for r in bydoc[d]}=={0,1}]
            if not docs:
                continue
            cells = [pool_document(bydoc[d]) for d in docs]
            cell_map[mode, length] = dict(zip(docs, cells))
            index = common_index if len(docs)==16 else None
            interval = paired_interval(cells, index=index)
            valid_rows = [r for r in rows if r['source_index'] in docs]
            metric = dict(documents=len(docs), paired_document_indices=docs, warm_requests=len(valid_rows),
                warm_tokens=sum(r['tokens'] for r in valid_rows), warm_seconds=sum(r['outer_seconds'] for r in valid_rows),
                pooled_tps=score_rows(valid_rows), tps_ci95=interval['ci95'],
                acceptance_length=quality_interval(cells, 'blocks', index=index),
                acceptance_rate=quality_interval(cells, 'proposed', index=index),
                warm_captures=sum(r.get('draft_runtime', {}).get('request', {}).get('captures', 0) for r in rows),
                warm_capture_seconds=sum(r.get('draft_runtime', {}).get('request', {}).get('capture_host_seconds', 0.) for r in rows),
                actual_raw_warm_requests=len(rows), cumulative=[], prefix6=None, prefix32=None)
            clean_prefix = []
            for position, row in enumerate(rows):
                expected = None if sequence is None or position >= len(sequence) else sequence[position]
                if (mode, length, row.get('source_index')) in bad or (expected is not None and any(row.get(k)!=v for k,v in expected.items())):
                    break
                clean_prefix.append(row)
                metric['cumulative'].append(dict(requests=len(clean_prefix), tps=score_rows(clean_prefix),
                    captures=sum(r.get('draft_runtime', {}).get('request', {}).get('captures', 0) for r in clean_prefix)))
                if len(clean_prefix) in (6, 32):
                    prefix_docs = list(dict.fromkeys(r['source_index'] for r in clean_prefix))
                    pcells = [pool_document([r for r in clean_prefix if r['source_index']==d], complete=False) for d in prefix_docs]
                    stat = paired_interval(pcells)
                    metric['prefix'+str(len(clean_prefix))] = dict(requests=len(clean_prefix), documents=len(prefix_docs),
                        point=score_rows(clean_prefix), ci95=stat['ci95'], document_indices=prefix_docs,
                        repeat_counts=[len([r for r in clean_prefix if r['source_index']==d]) for d in prefix_docs],
                        capture_seconds=sum(r.get('draft_runtime', {}).get('request', {}).get('capture_host_seconds', 0.) for r in clean_prefix))
            owner = owner_map.get((mode, length), {})
            metric.update(accounted_service_tps=owner.get('accounted_service_tps'),
                constructor_seconds=owner.get('constructor_and_detach_seconds'), close_seconds=owner.get('close_seconds'))
            cold = owner.get('cold_prime_outer_seconds')
            if positive(cold) and nonnegative(owner.get('constructor_and_detach_seconds')):
                metric['cold_complete_tps'] = owner.get('prime_tokens', 0)/(cold+owner['constructor_and_detach_seconds'])
            else:
                metric['cold_complete_tps'] = None
            for stage in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds'):
                if rows and all(nonnegative(r.get(stage)) for r in rows):
                    metric[stage] = sum(r[stage] for r in rows)/len(rows)
            values[mode] = metric
        retained = stage2.get(str(length), {}).get('union_modes', []) if stage2 else list(dict.fromkeys(
            r['mode'] for r in stage1.get('per_context', {}).get(str(length), {}).get('groups', {}).values()))
        comparisons = []
        for left, right in itertools.combinations(retained, 2):
            ac, bc = cell_map.get((left, length), {}), cell_map.get((right, length), {})
            docs = [d for d in DOCS if d in ac and d in bc]
            if not docs:
                continue
            stat = paired_interval([ac[d] for d in docs], [bc[d] for d in docs], index=common_index if len(docs)==16 else None)
            if len(docs)==16:
                stat['point'] = 100.*(values[left]['pooled_tps']/values[right]['pooled_tps']-1.)
            comparisons.append(dict(left=left, right=right, documents=len(docs), document_indices=docs,
                point_percent=stat['point'], ci95=stat['ci95'], scope='Exploratory document-paired pointwise interval, post-selection; no final success gate'))
        pairs[str(length)] = comparisons
    return metrics, pairs


def analyze(directory):
    try:
        return _analyze(directory)
    except (KeyError, ValueError, TypeError, AttributeError, IndexError, OverflowError) as error:
        return dict(schema=1, status='PARTIAL', authoritative_union=False, goal_complete=False,
            problems=['Malformed saved evidence: '+type(error).__name__+': '+str(error)],
            metrics={}, pairwise_retained={}, stage2=None, provisional_stage2=None, observed_counts={})


def _analyze(directory):
    import numpy as np
    evidence = Evidence(directory)
    protocol = evidence.json('protocol.json')
    arms = support.menu()
    amap = {a['mode']: a for a in arms}
    declared = protocol.get('arms', [])
    if (not isinstance(declared, list) or len(declared) != 38 or {a.get('mode'): a for a in declared if isinstance(a, dict)} != amap or
        protocol.get('source_indices') != list(DOCS) or protocol.get('lengths') != list(LENGTHS) or protocol.get('repeats') != 2 or
        protocol.get('max_new_tokens') != 128 or protocol.get('seed') != SEED or protocol.get('expected_counts') != EXPECTED or
        protocol.get('expected_requests') != TOTAL):
        evidence.problem('Frozen full38/configuration16doc/32warm/2540 plan differs')
    if (protocol.get('no_training') is not True or protocol.get('mapper_executed') is not False or protocol.get('controller_executed') is not False or
        protocol.get('max_live_owners') != 1 or protocol.get('template_prefill_precision') != 'bfloat16' or protocol.get('graph_max_entries') != 128):
        evidence.problem('Baseline-only shared execution/freeze contract differs')
    environment = protocol.get('environment', {})
    if ('H100' not in environment.get('gpu_name', '') or environment.get('total_memory', 0) < 80_000_000_000 or
        environment.get('target_dtype') != 'torch.float32' or environment.get('draft_dtype') != 'torch.bfloat16' or
        environment.get('prefix_dtype') != 'torch.bfloat16' or environment.get('tf32') is not False):
        evidence.problem('H10080GB/FP32Target/BF16Draft+prefix/TF32off environment differs')
    eos, vocab = protocol.get('tokenizer_eos_token_id'), protocol.get('vocab_size')
    if type(eos) is not int or type(vocab) is not int or not 0 <= eos < vocab:
        evidence.problem('Frozen tokenizer EOS/vocabulary metadata absent')
    sources = protocol.get('source_sha256', {})
    if not sources:
        evidence.problem('Immutable source inventory missing')
    for name, digest in sources.items():
        evidence.pin('source/'+name, digest)
    if not protocol.get('source_bindings') or not protocol.get('tree_source_bindings'):
        evidence.problem('Actual import bindings missing')
    for name, value in protocol.get('source_bindings', {}).items():
        relative = name.replace('.', '/')+'.py'
        if not str(value.get('path', '')).endswith('/'+relative) or sources.get(relative) != value.get('sha256'):
            evidence.problem('Actual runtime/source import mismatch: '+name)
    for value in protocol.get('tree_source_bindings', {}).values():
        if 'sha256' in value and sources.get(value.get('path')) != value['sha256']:
            evidence.problem('Archived full-tree import/source mismatch')
    flash = protocol.get('flash_attention', {})
    if flash.get('version') != '2.8.3' or not str(flash.get('files', {}).get('flash_attn_2_cuda', {}).get('path', '')).endswith('.so'):
        evidence.problem('Real official Flash2.8.3 binary receipt missing')
    frozen1, stage1_protocol = stage1_chain(evidence, protocol)
    # Dataset/source receipt integration is completed against the actual data
    # helper schema before this analyzer is eligible for a scientific freeze.
    documents, input_sha = audit_data(evidence, protocol)
    if documents != list(DOCS):
        evidence.problem('Exactly the sixteen predefined reused development documents required')
    gate=evidence.json('GATE_PASSED.json')
    if (gate.get('passed') is not True or gate.get('stage1_freeze')!=protocol.get('stage1_freeze') or
        gate.get('source_sha256')!=sources or gate.get('data_manifest_sha256')!=protocol.get('data_manifest_sha256') or
        gate.get('old_metadata_sources')!=DATA_SHA or gate.get('expected_counts')!=EXPECTED or gate.get('expected_requests')!=2540 or
        gate.get('no_new_confirmation_opened') is not True):
        evidence.problem('Stage1/source/data gate did not bind the study before generation')
    sequence = protocol.get('shared_warm_sequence', [])
    if (not isinstance(sequence, list) or len(sequence) != 32 or
        {(r.get('source_index'), r.get('repeat')) for r in sequence} != {(d, r) for d in DOCS for r in (0, 1)}):
        evidence.problem('One shared32-request document/repeat sequence is missing/duplicated')
    owner_plan = protocol.get('owner_order', [])
    expected_owners = {(a['mode'], n) for a in arms for n in LENGTHS}
    if len(owner_plan) != 76 or {(r.get('mode'), r.get('length')) for r in owner_plan} != expected_owners:
        evidence.problem('Exactly76 private owner blocks required')
    flattened = [dict(mode=o['mode'], length=o['length'], **r) for o in owner_plan for r in sequence]
    if protocol.get('warm_order') != flattened:
        evidence.problem('All methods/contexts must run the same frozen warm document order')
    canonical = evidence.rows('reference.jsonl')
    prime = evidence.rows('prime.jsonl')
    warm = evidence.rows('runtime.jsonl')
    owners = evidence.rows('owners.jsonl')
    counts = dict(reference_requests=len(canonical), prime_requests=len(prime), timing_requests=len(warm), owners=len(owners))
    if counts != EXPECTED:
        evidence.problem('Complete2540-request/76owner matrix missing')
    refs = evidence.json('references.json')
    ref_map = {}
    for row in canonical:
        key = row.get('source_index'), row.get('length')
        generated = row.get('generated')
        if (key not in {(d, n) for d in DOCS for n in LENGTHS} or key in ref_map or not isinstance(generated, list) or
            not 0 < len(generated) <= 128 or not all(type(t) is int and t >= 0 for t in generated) or refs.get(f'{key[0]}:{key[1]}') != generated):
            evidence.problem('Canonical32 references missing/duplicated/inconsistent')
        elif (len(generated) < 128 and generated[-1] != eos) or (eos in generated and (generated[-1] != eos or eos in generated[:-1])):
            evidence.problem('Canonical short/EOS first occurrence boundary differs')
        ref_map[key] = generated
    if len(ref_map) != 32:
        evidence.problem('All32 independent Target references required')
    if protocol.get('input_sha256') != {f'{d}:{n}': value for (d, n), value in input_sha.items()}:
        evidence.problem('Executed exact4K/8K inputs differ from the frozen old data')
    all_requests = canonical+prime+warm
    ids, indices = [r.get('request_id') for r in all_requests], [r.get('execution_index') for r in all_requests]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        evidence.problem('Scientific request UUID missing/reused')
    valid_indices = all(type(i) is int for i in indices) and len(set(indices)) == len(indices)
    if not valid_indices or sorted(indices) != list(range(TOTAL)):
        evidence.problem('Scientific execution ledger is not0..2539')
    pmap, wmap, bad = {}, {}, set()
    signatures = {}
    for phase, rows in (('cold_prime', prime), ('warm', warm)):
        for row in rows:
            mode, length, doc = row.get('mode'), row.get('length'), row.get('source_index')
            arm = amap.get(mode)
            if arm is None:
                evidence.problem('Unknown baseline configuration in raw timing')
                continue
            errors = support.row_errors(row, ref_map.get((doc, length)), arm)
            if (row.get('arm') != arm or length not in LENGTHS or row.get('phase') != phase or
                any(row.get(k) is not False for k in ('observer_enabled', 'profile_enabled', 'mapper_executed', 'controller_executed'))):
                errors.append('Unfrozen/instrumented/nonbaseline primary request')
            if arm['family'] == 'retrieval':
                if arm['backend'] != 'eager' and row.get('runtime_stats', {}).get('max_graph_entries') != 128:
                    errors.append('Actual private Graph entry limit differs from128')
                trace = row.get('trace', [])
                if trace and row.get('tokens') not in (trace[-1]['prefix']-(length-1)+trace[-1]['accepted'], trace[-1]['prefix']-(length-1)+trace[-1]['accepted']+1):
                    errors.append('Terminal commits cannot explain client output count')
            key = mode, length
            if phase == 'cold_prime':
                if doc != 271 or key in pmap:
                    errors.append('Cold prime must be one fully paid old271 request')
                pmap[key] = row
                if arm['backend'] == 'cuda_graph':
                    delta, lifetime = row.get('draft_runtime', {}).get('request', {}), row.get('runtime_stats', {})
                    if not positive(delta.get('captures')) or any(not near(delta.get(k), lifetime.get(k)) for k in ('captures', 'replays', 'resident_steps', 'capture_host_seconds')):
                        errors.append('Cold owner borrowed pre-existing Graph captures')
            else:
                key = mode, length, doc, row.get('repeat')
                if key in wmap or doc not in DOCS or row.get('repeat') not in (0, 1):
                    errors.append('Unregistered/duplicate warm matrix cell')
                wmap[key] = row
            logical = (support.linear_signature(row), [row.get(k) for k in ('accepted', 'proposed', 'blocks', 'draft_forwards',
                'working_gather_layers', 'working_full_gather_layers', 'working_gathered_position_layers', 'working_incremental_binds',
                'target_probe_rounds', 'draft_shadow_probes', 'projected_positions', 'rpc_calls')]) if arm['family']=='retrieval' else (
                    row.get('upstream', {}).get('accept_length_list'), row.get('raw_total_generated'), row.get('generation_overshoot'), row.get('scored_generated_before_eos'))
            strategy = arm['family'], arm['top_chunks'], arm.get('fixed_k', arm.get('nodes')), doc, length
            if strategy in signatures and signatures[strategy] != logical:
                errors.append('Same strategy repeat/backend changed its logical trajectory')
            signatures[strategy] = logical
            for error in errors:
                evidence.problem(f'{phase} {mode}/{doc}/{length}: '+error)
            if errors:
                bad.add((mode, length, doc))
    if set(pmap) != expected_owners or set(wmap) != {(m, n, d, r) for m, n in expected_owners for d in DOCS for r in (0, 1)}:
        evidence.problem('76prime/2432warm complete matrix missing')
    if [{k: r.get(k) for k in ('mode', 'length')} for r in prime] != owner_plan or [{k: r.get(k) for k in ('mode', 'length', 'source_index', 'repeat')} for r in warm] != flattened:
        evidence.problem('Observed request order differs from shared frozen plan')
    owner_map = {}
    for i, owner in enumerate(owners):
        key = owner.get('mode'), owner.get('length')
        rows = [r for r in prime+warm if r.get('owner_id') == owner.get('owner_id')]
        rows.sort(key=lambda r: r.get('execution_index', -1))
        errors = owner_errors(owner, rows)
        if key in owner_map or key not in expected_owners:
            errors.append('Duplicate/unknown owner')
        owner_map[key] = owner
        if valid_indices and [r['execution_index'] for r in rows] != list(range(32+i*33, 32+(i+1)*33)):
            errors.append('Private owner cold+32warm block is not contiguous in registered order')
        for error in errors:
            evidence.problem('Owner '+str(key)+': '+error)
    if set(owner_map) != expected_owners:
        evidence.problem('All76 private owners must be closed')
    telemetry=audit_telemetry(evidence,protocol,owners)
    tree_raw = evidence.rows('tree_raw.jsonl')
    tree_rows = [r for r in prime+warm if r.get('arm', {}).get('family') == 'full_tree']
    raw_map = {r.get('request_id'): r for r in tree_raw}
    if len(tree_raw) != 528 or len(raw_map) != len(tree_raw) or set(raw_map) != {r['request_id'] for r in tree_rows}:
        evidence.problem('All528 raw pre-check full-tree results required')
    for row in tree_rows:
        raw = raw_map.get(row['request_id'], {})
        original = raw.get('result', {})
        if (raw.get('execution_index') != row.get('execution_index') or original.get('generated') != row.get('wrapper_generated') or
            original.get('tokens') != row.get('wrapper_tokens') or original.get('upstream') != row.get('upstream')):
            evidence.problem('Normalized tree output differs from saved original wrapper result')
    complete = evidence.json('COMPLETE.json')
    if (complete.get('passed') is not True or complete.get('counts') != EXPECTED or complete.get('expected_counts')!=EXPECTED or complete.get('logical_requests') != TOTAL or
        complete.get('owner_closed') != 76 or complete.get('goal_complete') is not False or complete.get('analysis_required_locally') is not True or any(complete.get(k) is not True for k in
            ('sources_unchanged', 'assets_unchanged', 'dependencies_unchanged', 'all_canonical', 'backend_trajectories_equal'))):
        evidence.problem('Terminal complete/source/asset/exactness receipt invalid')
    if complete.get('data_unchanged') is not True or complete.get('stage1_verified_unchanged') is not True:
        evidence.problem('Old data/stage1 verification changed during stage2')
    frozen2 = evidence.json('FROZEN_SELECTION.json')
    if (frozen2.get('complete_request_count')!=2540 or frozen2.get('no_new_confirmation_opened') is not True or
        frozen2.get('stage1_selection_sha256')!=protocol.get('stage1_freeze',{}).get('selection_freeze_sha256') or
        frozen2.get('protocol_sha256')!=sha(Path(directory)/'protocol.json') or
        complete.get('frozen_selection_sha256')!=sha(Path(directory)/'FROZEN_SELECTION.json') or
        frozen2.get('source_sha256')!=sources or frozen2.get('assets_sha256')!=protocol.get('assets_sha256') or
        complete.get('data_manifest_sha256')!=protocol.get('data_manifest_sha256') or
        complete.get('stage1_selection_sha256')!=protocol.get('stage1_freeze',{}).get('selection_freeze_sha256')):
        evidence.problem('Stage2 union is not bound to complete2540/stage1/protocol hashes')
    computed = None
    if len(warm) == 2432 and len(wmap) == 2432 and frozen1.get('per_context'):
        computed = select_union(warm, DOCS, frozen1, arms)
        if frozen2.get('per_context') != computed:
            evidence.problem('Stage2 score/ranking/original-choice union differs from exact32warm rule')
    else:
        evidence.problem('Cannot freeze winners from incomplete warm data')
    audit_export(evidence)
    if (Path(directory)/'FAILURE.json').exists():
        evidence.problem('Failure marker exists; preserve partial root')
    status = 'COMPLETE' if not evidence.problems else 'PARTIAL'
    metrics, pairs = summarize(warm, owners, bad, frozen1, computed, arms, sequence)
    return dict(schema=1, status=status, goal_complete=False, authoritative_union=status=='COMPLETE',
        problems=evidence.problems, observed_counts=counts, expected_counts=EXPECTED,
        observed_requests=len(all_requests), expected_requests=TOTAL, documents=list(DOCS),
        metrics=metrics, pairwise_retained=pairs, stage2=computed if status=='COMPLETE' else None,
        provisional_stage2=computed if status!='COMPLETE' else None,
        gpu_telemetry=telemetry,
        stage1_freeze_sha256=protocol.get('stage1_freeze', {}).get('selection_freeze_sha256'),
        shared_warm_sequence=sequence, scope='Matched-horizon development on16 reused documents, not held-out or RESCUE success')


def retained_modes(summary, length):
    current=summary.get('stage2') or summary.get('provisional_stage2') or {}
    return current.get(str(length),{}).get('union_modes',[])


def label(mode):
    return mode.replace('retrieval','T').replace('tree','Tree T').replace('_k','/k').replace('_n','/n').replace('_graph',' Graph').replace('_resident',' resident').replace('_eager',' eager')


def render(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metrics=summary.get('metrics',{});status=summary.get('status','PARTIAL');figures=[]
    caption=(f'{status} | H10080GB,Qwen2 FP32 Target/BF16 Draft+prefix,TF32off; exact4096/8192,max128/EOS. '
        f'16 reused docs ×2warm repeats; {summary.get("observed_counts",{}).get("timing_requests",0)}/2432warm. '
        '76paid old271primes; all captures/overshoot charged. Pointwise document CIs; not held-out/production/Vicuna.')
    colors={'eager':'#64748b','resident':'#f59e0b','graph':'#2563eb'}
    def finish(fig,name):
        fig.text(.01,.015,caption,va='bottom',fontsize=8,wrap=True)
        fig.tight_layout(rect=(0,.10,1,.94));fig.savefig(output/(name+'.png'),dpi=160);fig.savefig(output/(name+'.svg'));plt.close(fig);figures.append(name)
    def interval(ax,x,point,ci,color):
        if point is None:return
        ax.scatter([x],[point],color=color,s=21,zorder=4)
        if ci is not None:ax.vlines(x,ci[0],ci[1],color=color,linewidth=1)
    def tree(ax,n,key):
        for top,color in ((16,'#10b981'),(32,'#9333ea')):
            points=[]
            for nodes in (8,32,50,64):
                row=metrics.get(str(n),{}).get(f'tree{top}_n{nodes}',{});v=row.get(key)
                p=v.get('point') if isinstance(v,dict) else v;ci=v.get('ci95') if isinstance(v,dict) else row.get('tps_ci95')
                if p is not None:points.append((nodes,p));interval(ax,nodes,p,ci,color)
            if points:ax.plot(*zip(*points),color=color,label=f'Top{top}')
        ax.set_xticks((8,32,50,64));ax.set_xlabel('Tree nodes; maximum depth10');ax.grid(axis='y',alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    fig,axes=plt.subplots(3,2,figsize=(12,9))
    for col,n in enumerate(LENGTHS):
        for ri,key in enumerate(('acceptance_length','acceptance_rate')):
            ax=axes[ri,col]
            for backend,color in colors.items():
                for top in (16,32):
                    points=[]
                    for k in (1,2,4,8,12):
                        v=metrics.get(str(n),{}).get(f'retrieval{top}_k{k}_{backend}',{}).get(key,{})
                        if v.get('point') is not None:points.append((k,v['point']));interval(ax,k,v['point'],v.get('ci95'),color)
                    if points:ax.plot(*zip(*points),color=color,linestyle='--' if top==16 else '-',label=f'Top{top} {backend}')
            ax.set_title(f'{n}: linear quality');ax.set_xticks((1,2,4,8,12));ax.set_xlabel('Fixed draft length k')
            ax.set_ylabel('Accepted Draft\ntokens / round' if ri==0 else 'Accepted / proposed\nDraft tokens');ax.grid(axis='y',alpha=.2)
            if ax.lines:ax.legend(fontsize=6,ncol=2)
        tree(axes[2,col],n,'acceptance_length');axes[2,col].set_title(f'{n}: raw tree quality, includes overshoot')
        axes[2,col].set_ylabel('Raw tree accepted tokens\n/ raw round')
    fig.suptitle('Quality only — whole-document bootstrap; tree proposal rate/tail IDs are not invented');finish(fig,'quality')
    def heatmaps(key,title,unit,name):
        fig,axes=plt.subplots(3,2,figsize=(12,8))
        for col,n in enumerate(LENGTHS):
            for ri,backend in enumerate(colors):
                ax=axes[ri,col];grid=np.full((2,5),np.nan)
                for y,top in enumerate((16,32)):
                    for x,k in enumerate((1,2,4,8,12)):
                        v=metrics.get(str(n),{}).get(f'retrieval{top}_k{k}_{backend}',{}).get(key)
                        if nonnegative(v):grid[y,x]=v
                if np.isfinite(grid).any():
                    scale=dict(vmin=0,vmax=max(1.,float(np.nanmax(grid)))) if key in ('warm_captures','warm_capture_seconds') else {}
                    im=ax.imshow(np.ma.masked_invalid(grid),aspect='auto',cmap='viridis',**scale);fig.colorbar(im,ax=ax,label=unit)
                    for y,top in enumerate((16,32)):
                        for x,k in enumerate((1,2,4,8,12)):
                            if np.isfinite(grid[y,x]):
                                text=str(int(grid[y,x])) if key=='warm_captures' else f'{grid[y,x]:.2f}'
                                if status=='PARTIAL':text+='\nn='+str(metrics[str(n)][f'retrieval{top}_k{k}_{backend}']['documents'])
                                ax.text(x,y,text,ha='center',va='center',fontsize=8,color='white')
                else:ax.text(.5,.5,'No complete repeat-pair cells',ha='center',transform=ax.transAxes)
                ax.set_xticks(range(5),(1,2,4,8,12));ax.set_yticks(range(2),('Top16','Top32'));ax.set_xlabel('Fixed draft length k');ax.set_title(f'{n}: {backend}')
        fig.suptitle(title);finish(fig,name)
    heatmaps('pooled_tps','Primary32-warm TPS points — full95% intervals in REPORT; no cross-stage timing splice','Client tokens / outer second','tps')
    heatmaps('warm_captures','All actual new captures across32 warm requests — never filter slow requests','New captures','captures')
    heatmaps('warm_capture_seconds','Actual charged warm capture-host interval — inside full-generate outer','Capture-host seconds','capture_time')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,n in enumerate(LENGTHS):
        modes=retained_modes(summary,n);ax=axes[0,col]
        for i,m in enumerate(modes):
            v=metrics.get(str(n),{}).get(m,{});interval(ax,i,v.get('pooled_tps'),v.get('tps_ci95'),'#2563eb')
        ax.set_xticks(range(len(modes)),[label(m) for m in modes],rotation=25,ha='right',fontsize=7);ax.set_ylabel('Pooled32-warm client TPS');ax.grid(axis='y',alpha=.2)
        ax.set_title(f'{n}: original stage1 + new stage2 union')
        tree(axes[1,col],n,'pooled_tps');axes[1,col].set_ylabel('Full-tree client TPS');axes[1,col].set_title(f'{n}: full-tree grid with document95% intervals')
    fig.suptitle('Retained contenders and all full-tree TPS — pointwise/post-selection intervals, not a final success gate');finish(fig,'retained_tps')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,n in enumerate(LENGTHS):
        modes=retained_modes(summary,n)
        for ri,key in enumerate(('cold_complete_tps','accounted_service_tps')):
            ax=axes[ri,col]
            for i,m in enumerate(modes):
                value=metrics.get(str(n),{}).get(m,{}).get(key)
                if positive(value):ax.bar(i,value,color='#2563eb')
            ax.set_xticks(range(len(modes)),[label(m) for m in modes],rotation=25,ha='right',fontsize=7)
            ax.set_ylabel('Client tokens / charged second');ax.set_title(f'{n}: '+('constructor + paid old271 cold' if ri==0 else 'constructor + cold +32warm + close'))
    fig.suptitle('Cold/service diagnostics — one private owner per configuration/context; descriptive, no document CI');finish(fig,'cold_service')
    fig,axes=plt.subplots(1,2,figsize=(12,5));palette=plt.get_cmap('tab10')
    for col,n in enumerate(LENGTHS):
        ax=axes[col]
        for i,m in enumerate(retained_modes(summary,n)):
            v=metrics.get(str(n),{}).get(m,{});curve=v.get('cumulative',[])
            if curve:ax.plot([r['requests'] for r in curve],[r['tps'] for r in curve],color=palette(i),label=label(m))
            for key in ('prefix6','prefix32'):
                p=v.get(key)
                if p:interval(ax,p['requests'],p['point'],p['ci95'],palette(i))
        ax.set_title(f'{n}: shared order;6 requests=6 docs ×1,32 requests=16 docs ×2');ax.set_xticks((1,6,16,32));ax.set_xlabel('Warm requests after separately paid cold prime');ax.set_ylabel('Cumulative client TPS');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=6,ncol=2)
    fig.suptitle('Prefix6 is this stage’s first6 requests, NOT the old stage1 three-doc ×2 cohort; endpoint intervals use actual document counts');finish(fig,'cumulative')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for col,n in enumerate(LENGTHS):
        modes=retained_modes(summary,n);ax=axes[col]
        for j,(key,color) in enumerate((('prefill_seconds','#60a5fa'),('decode_seconds','#fbbf24'),('cleanup_seconds','#c4b5fd'))):
            for i,m in enumerate(modes):
                v=metrics.get(str(n),{}).get(m,{}).get(key)
                if nonnegative(v):ax.bar(i+(j-1)*.23,v,width=.23,color=color,label=key.replace('_seconds','') if i==0 else None)
        ax.set_xticks(range(len(modes)),[label(m) for m in modes],rotation=25,ha='right',fontsize=7);ax.set_ylabel('Saved stage seconds / warm request');ax.set_title(f'{n}: available inner stages; not causal kernel attribution')
        if ax.patches:ax.legend(fontsize=7)
    fig.suptitle('Collected latency diagnostics — no primary profiler; unavailable stages omitted, not zero');finish(fig,'latency')
    fig,axes=plt.subplots(2,1,figsize=(12,6))
    for n,color in ((4096,'#2563eb'),(8192,'#f59e0b')):
        records=[r for r in summary.get('gpu_telemetry',[]) if r.get('length')==n]
        for ax,key in zip(axes,('temperature.gpu','clocks.current.sm')):
            for phase,marker in (('owner_start','o'),('owner_end','x')):
                points=[(r['execution_index'],r.get(key)) for r in records if r.get('phase')==phase and nonnegative(r.get(key))]
                if points:ax.scatter(*zip(*points),s=18,color=color,marker=marker,label=f'{n} {phase}')
            ax.grid(alpha=.2);ax.set_xlabel('Scientific execution index at owner boundary')
            if ax.collections:ax.legend(fontsize=7,ncol=2)
    axes[0].set_ylabel('Observed GPU temperature (C)');axes[1].set_ylabel('Observed SM clock (MHz)')
    fig.suptitle('Read-only owner-boundary telemetry — outside primary clocks; no filtering, clock adjustment or causal correction')
    finish(fig,'gpu_telemetry')
    return figures


def write_report(summary, directory, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    figures=render(summary,output)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    shutil.copyfile(Path(__file__),output/'analysis_source.py');shutil.copyfile(Path(support.__file__),output/'analysis_support.py')
    lines=[f"# {summary['status']}: matched32-warm baseline development",'',
        'This is a separate matched-horizon development supplement, not fresh confirmation, a RESCUE comparison, or goal success. '
        'Original six-warm stage1 choices remain unchanged. Stage2 maximizes SUM32warm client tokens / SUM32warm full-generate outer seconds; '
        'exact ties use lexical mode order. The deduplicated union keeps both stage1 and stage2 choices in every backend/family group, up to8/context. '
        'A weaker opponent never silently replaces an earlier retained opponent.', '',
        'Fixed scope:38 configurations,16 already-used documents,exact4096/8192 inputs,max128/EOS,two warm repeats; '
        '32AR+76cold primes+2432warm=2540 scientific requests. H10080GB,Qwen2 Target FP32,Draft/prefix BF16,TF32off. '
        'Every configuration/context has one private owner and the same frozen32-request document order. '
        'Owners execute blocked, not interleaved; clock drift between blocks remains a limitation. '
        'Cold prime271 intentionally overlaps a warm training document and is separately charged to all configurations. '
        'All valid lazy captures and tree overshoot work are included; common model preparation is excluded identically.', '',
        'Statistics pool actual token/time repeats within whole documents, then bootstrap the ratio over documents using10000 shared '
        'resamples (seed20261025). These are pointwise exploratory intervals after development selection, not adjusted final-success tests. '
        'Prefix6 is the actual first6 requests of this stage’s shared sequence, not the old stage1 three-doc/two-repeat cohort. '
        'Its actual document and repeat counts are saved in summary.json; no cross-run timing is spliced in. '
        'No capture-free filtering is allowed. Active-service throughput charges constructor+cold+32warm+close and is not production wall throughput.', '',
        'The input audit rebuilds exact Token prefixes from four saved, pinned old metadata files. Full-text document SHA values are inherited '
        'metadata; no original text is re-tokenized. The paused71/320 test and new confirmation cohort remain untouched. '
        'Owner-boundary GPU temperature/clocks/power records are observational, outside primary clocks and never used to filter or adjust performance.', '',
        '## Audit problems','']
    lines += ['- '+p for p in summary.get('problems',[])] or ['None.']
    stage2=summary.get('stage2') or summary.get('provisional_stage2') or {}
    lines += ['','## Preserved and supplemental choices','', '| Input | Group | Original stage1 | Stage2 | Retained union |','|---|---|---|---|---|']
    for n in LENGTHS:
        entry=stage2.get(str(n),{})
        for name in GROUPS:
            old=entry.get('stage1_groups',{}).get(name,{});new=entry.get('groups',{}).get(name,{})
            modes=', '.join(r['mode'] for r in entry.get('union_by_group',{}).get(name,[]))
            lines.append(f"| {n} | {name} | {old.get('mode','unavailable')} | {new.get('mode','unavailable')} | {modes} |")
    lines += ['','## Full32-warm ranking','', '| Input | Rank | Mode | TPS | Pointwise95% CI | Paired documents | Warm captures |','|---|---:|---|---:|---|---:|---:|']
    for n in LENGTHS:
        values=summary.get('metrics',{}).get(str(n),{})
        for rank,mode in enumerate(sorted(values,key=lambda m:(-values[m]['pooled_tps'],m)),1):
            v=values[mode];ci=v.get('tps_ci95');ci_text='unavailable' if ci is None else f'[{ci[0]:.6f},{ci[1]:.6f}]'
            lines.append(f"| {n} | {rank} | {mode} | {v['pooled_tps']:.6f} | {ci_text} | {v['documents']} | {v['warm_captures']} |")
    lines += ['','## Retained pair comparisons (pointwise, post-selection)','', '| Input | Left vs right | TPS change |95% CI|n docs|','|---|---|---:|---|---:|']
    for n in LENGTHS:
        for row in summary.get('pairwise_retained',{}).get(str(n),[]):
            ci=row['ci95'];text='unavailable' if ci is None else f'[{ci[0]:+.4f}%,{ci[1]:+.4f}%]'
            lines.append(f"| {n} | {row['left']} vs {row['right']} | {row['point_percent']:+.4f}% | {text} | {row['documents']} |")
    lines += ['','## Saved figures and evidence','']
    for name in figures:lines += [f'[{name} PNG]({name}.png) · [SVG]({name}.svg)','',f'![{name}]({name}.png)','']
    lines += ['[Analysis source](analysis_source.py) · [Saved dependency](analysis_support.py) · [Summary](summary.json)','',
        '[Raw warm](../runtime.jsonl) · [Cold](../prime.jsonl) · [Owners](../owners.jsonl) · [Canonical](../reference.jsonl) · '
        '[Tree raw](../tree_raw.jsonl) · [Protocol](../protocol.json) · [Old data receipt](../data/DATASET_MANIFEST.json) · '
        '[Frozen union](../FROZEN_SELECTION.json) · [GPU telemetry](../gpu_diagnostics.jsonl)']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if summary.get('status')=='COMPLETE' and summary.get('authoritative_union'):
        value=dict(status='VERIFIED_FROZEN_UNION',authoritative=True,per_context=summary['stage2'],
            stage1_selection_sha256=summary.get('stage1_freeze_sha256'),
            future_contrasts=sum(len(v['union_modes']) for v in summary['stage2'].values()),goal_complete=False)
        target='VERIFIED_UNION.json'
    else:value=dict(status='PARTIAL',authoritative=False,per_context=summary.get('provisional_stage2'),goal_complete=False);target='PROVISIONAL_UNION.json'
    (output/target).write_text(json.dumps(value,indent=2)+'\n')
    receipt=dict(status=summary['status'],goal_complete=False,render_required=True,figures=figures,
        artifacts_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        note='Human render inspection and independent saved-raw verification remain required before handoff.')
    (output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=analyze(args.directory);write_report(summary,args.directory,args.output)
    print(json.dumps(dict(status=summary['status'],authoritative_union=summary.get('authoritative_union'),observed_counts=summary.get('observed_counts'),problems=summary.get('problems')),indent=2))


if __name__=='__main__':main()
