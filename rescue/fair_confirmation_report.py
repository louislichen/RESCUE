"""Offline, partial-safe audit of the frozen same-Qwen fair confirmation.

No torch/model imports. Statistical units are sixteen documents, never requests,
rounds, contexts or repeats. All eight comparisons share one bootstrap draw.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil

LENGTHS = (4096, 8192)
GROUPS = ('retrieval/eager', 'retrieval/resident', 'retrieval/graph', 'full_tree')
SEED = 20261024
DRAWS = 100000
DOCUMENTS = 16
REPEATS = 2
OUTPUT_LIMIT = 128
ASSETS = {
    'mapper_sha256': 'd08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
    'policy_sha256': '3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
    'cost_sha256': 'd92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1',
}
SELECTION_RULE = dict(original_split='validation', minimum_source_index=900,
    minimum_natural_tokens=16384, documents=16,
    order='original source_index ascending; first eligible sixteen',
    context_lengths=[4096, 8192],
    duplicate_rule='exclude any document SHA or exact 4K/8K prompt-prefix SHA, including within selected cohort',
    replacement='never use generation, EOS, speed or nonzero-injection outcomes to select or replace')
FIXED_CONTRACT = dict(teacher='Qwen2-7B-Instruct', student='Qwen2-1.5B-Instruct',
    target_dtype='float32', draft_dtype='bfloat16', prefill_precision='bfloat16',
    tf32=False, context_lengths=[4096, 8192], max_new_tokens=128, repeats=2,
    schedule='serial', graph_max_entries=128)
OWN_CONTRACT = dict(ASSETS, selector='hierarchical', working_budget=512, recent=64,
    coverage_rows=64, coverage_candidates=1024, coverage_batch=16, coarse_chunk_size=32,
    refresh_every=4, refresh_policy='periodic', injection_layers=[27], inject_tokens=64,
    injection_strength=.5, draft_execution='cuda_graph', coverage_execution='eager',
    length_policy='frozen old-cost causal2/4', mapper_required_nonzero=True,
    no_training=True, no_cost_promotion=True)
STATISTICS_CONTRACT = dict(bootstrap_seed=SEED, bootstrap_draws=DRAWS,
    independent_unit='whole document', primary='pooled token/time throughput ratio',
    paired_indices='same document resample for all arms and both contexts',
    plotted_interval='pointwise two-sided95 percentile [0.025,0.975]',
    formal_gate='ALL8 effects have positive point estimate AND positive simultaneous one-sided95 lower bound',
    simultaneous_lower_quantile=.00625, comparisons=8,
    filtering='no capture-free filtering, outcome selection, imputation or optional stopping')
EXECUTION_CONTRACT = dict(
    primary='Whole-generate outer includes handoff,input construction,prefill,reset,all legal lazy captures,decode,cleanup; excludes common model loading identically',
    owner_trace='At most one live private serving owner. Each context has five separately constructed and closed owners, each with one cold prime followed by32 warm requests; disclose finite trace, not fully prewarmed production speed',
    first_prime='Fixed old train document271, separately billed per context/arm, using the exact verified baseline input and canonical128 output; no extra AR, free warmup or quality Graph borrowing',
    order='Freeze seeded owner-block order and within-block document order. One owner executes all32 warm requests before close; this is not interleaving. Disclose block clock-drift limitation; reset per-request policy RNG',
    secondaries='Report cold constructor+prime and active-service constructor+prime+32warm+close separately; do not substitute them after observing primary',
    captures='Keep every legal capture in timing; no pre-capture union, test-aware warmup or capture-free filtering')
REQUIRED_SOURCE = ('rescue/fair_confirmation_data.py', 'rescue/fair_confirmation.py',
    'rescue/fair_confirmation_report.py', 'tests/test_rescue_fair_confirmation_data.py',
    'tests/test_rescue_fair_confirmation.py', 'tests/test_rescue_fair_confirmation_report.py',
    'exp/draft_graph/fair_confirmation_20261020_preregistration.json',
    'exp/draft_graph/run_fair_confirmation_20261020_v1.sh')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def near(a, b):
    return nonnegative(a) and nonnegative(b) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)


def digest_valid(value):
    return (isinstance(value, dict) and isinstance(value.get('shape'), list) and
        all(type(v) is int and v >= 0 for v in value['shape']) and
        isinstance(value.get('dtype'), str) and isinstance(value.get('sha256'), str) and
        len(value['sha256']) == 64 and all(c in '0123456789abcdef' for c in value['sha256']))


def pool_document(rows):
    """Pool both repeats' actual client tokens and full-generate outer time."""
    if (len(rows) != REPEATS or {r.get('repeat') for r in rows} != {0, 1} or
        len({(r.get('source_index'), r.get('length'), r.get('mode')) for r in rows}) != 1):
        raise ValueError('Exactly two repeats of one document/context/arm required')
    tokens = sum(r['tokens'] for r in rows)
    seconds = sum(r['outer_seconds'] for r in rows)
    if not positive(tokens) or not positive(seconds):
        raise ValueError('Positive saved client token/time pairs required')
    return dict(tokens=tokens, seconds=seconds, tps=tokens / seconds,
        accepted=sum(r['accepted'] for r in rows), blocks=sum(r['blocks'] for r in rows),
        proposed=None if any(r.get('proposed') is None for r in rows) else sum(r['proposed'] for r in rows),
        captures=sum(r.get('draft_runtime', {}).get('request', {}).get('captures', 0) for r in rows),
        capture_seconds=sum(r.get('draft_runtime', {}).get('request', {}).get('capture_host_seconds', 0.) for r in rows))


def synchronized_statistics(cells, own_modes, challengers, *, seed=SEED, draws=DRAWS):
    """One common document-index resample for all contexts, arms and contrasts.

    cells[context][mode] is in the SAME document order throughout. Each cell is
    already pooled over repeats. The point is a ratio of sums, not mean TPS or
    mean document percentage. Bonferroni's eight one-sided tests use .05/8.
    """
    import numpy as np
    if seed != SEED or draws != DRAWS:
        raise ValueError('Frozen seed20261024 and100000 bootstrap draws required')
    all_lists = [values for context in cells.values() for values in context.values()]
    sizes = {len(values) for values in all_lists}
    if len(sizes) != 1 or not sizes or not 0 < next(iter(sizes)) <= DOCUMENTS:
        raise ValueError('Common nonempty document alignment required')
    count = sizes.pop()
    if any(not positive(c.get('tokens')) or not positive(c.get('seconds')) for v in all_lists for c in v):
        raise ValueError('Invalid saved document token/time pair')
    index = np.random.default_rng(seed).integers(0, count, size=(draws, count)) if count > 1 else None
    result = dict(n=count, draws=draws, seed=seed, document_index_shared_across_all_eight=True,
        simultaneous_method='Bonferroni eight one-sided percentile-bootstrap lower bounds',
        simultaneous_lower_percentile=0.625, contexts={}, contrasts=[], statistical_gate=False)
    for context in map(str, LENGTHS):
        values = cells.get(context, {})
        own = own_modes.get(context)
        groups = challengers.get(context, {})
        if own not in values or set(groups) != set(GROUPS) or any(v not in values for v in groups.values()):
            raise ValueError('Both contexts and all four frozen challengers required')
        arrays = {}
        metrics = {}
        for mode, docs in values.items():
            token = np.array([d['tokens'] for d in docs], float)
            seconds = np.array([d['seconds'] for d in docs], float)
            point = float(token.sum() / seconds.sum())
            sample = token[index].sum(1) / seconds[index].sum(1) if index is not None else None
            arrays[mode] = sample
            metric = dict(documents=count, tokens=int(token.sum()), seconds=float(seconds.sum()),
                pooled_tps=point, tps_ci95=None if sample is None else [float(x) for x in np.percentile(sample, [2.5, 97.5])])
            for name, denominator in (('acceptance_length', 'blocks'), ('acceptance_rate', 'proposed')):
                if all(d.get(denominator) is not None and positive(d.get(denominator)) and nonnegative(d.get('accepted')) for d in docs):
                    num = np.array([d['accepted'] for d in docs], float)
                    den = np.array([d[denominator] for d in docs], float)
                    samples = num[index].sum(1) / den[index].sum(1) if index is not None else None
                    metric[name] = float(num.sum() / den.sum())
                    metric[name + '_ci95'] = None if samples is None else [float(x) for x in np.percentile(samples, [2.5, 97.5])]
                else:
                    metric[name] = None
                    metric[name + '_ci95'] = None
            metrics[mode] = metric
        result['contexts'][context] = metrics
        for group in GROUPS:
            mode = groups[group]
            point = 100. * (metrics[own]['pooled_tps'] / metrics[mode]['pooled_tps'] - 1.)
            samples = 100. * (arrays[own] / arrays[mode] - 1.) if index is not None else None
            result['contrasts'].append(dict(length=int(context), group=group, own_mode=own,
                baseline_mode=mode, documents=count, point_percent=point,
                pointwise_ci95=None if samples is None else [float(x) for x in np.percentile(samples, [2.5, 97.5])],
                simultaneous_one_sided_lower95=None if samples is None else float(np.percentile(samples, .625))))
    result['statistical_gate'] = (count == DOCUMENTS and len(result['contrasts']) == 8 and
        all(c['point_percent'] > 0 and c['simultaneous_one_sided_lower95'] is not None and
            c['simultaneous_one_sided_lower95'] > 0 for c in result['contrasts']))
    return result


def partial_context_statistics(cells, own_mode, challengers, length):
    """Display completed within-context pairs only; NEVER a formal8-way gate."""
    import numpy as np
    sizes = {len(v) for v in cells.values()}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        return {}, []
    count = sizes.pop()
    index = np.random.default_rng(SEED).integers(0, count, size=(DRAWS, count)) if count > 1 else None
    metrics, samples = {}, {}
    for mode, docs in cells.items():
        token = np.array([d['tokens'] for d in docs], float)
        seconds = np.array([d['seconds'] for d in docs], float)
        if not np.isfinite(token).all() or not np.isfinite(seconds).all() or (token <= 0).any() or (seconds <= 0).any():
            raise ValueError('Invalid partial paired document data')
        sample = token[index].sum(1) / seconds[index].sum(1) if index is not None else None
        samples[mode] = sample
        row = dict(documents=count, tokens=int(token.sum()), seconds=float(seconds.sum()),
            pooled_tps=float(token.sum()/seconds.sum()), tps_ci95=None if sample is None else [float(v) for v in np.percentile(sample, [2.5, 97.5])])
        for name, denominator in (('acceptance_length', 'blocks'), ('acceptance_rate', 'proposed')):
            if all(positive(d.get(denominator)) and nonnegative(d.get('accepted')) for d in docs):
                num = np.array([d['accepted'] for d in docs], float)
                den = np.array([d[denominator] for d in docs], float)
                distribution = num[index].sum(1)/den[index].sum(1) if index is not None else None
                row[name] = float(num.sum()/den.sum())
                row[name+'_ci95'] = None if distribution is None else [float(v) for v in np.percentile(distribution, [2.5, 97.5])]
            else:
                row[name], row[name+'_ci95'] = None, None
        metrics[mode] = row
    contrasts = []
    for group in GROUPS:
        mode = challengers[group]
        distribution = 100.*(samples[own_mode]/samples[mode]-1.) if index is not None else None
        contrasts.append(dict(length=length, group=group, own_mode=own_mode, baseline_mode=mode, documents=count,
            point_percent=100.*(metrics[own_mode]['pooled_tps']/metrics[mode]['pooled_tps']-1.),
            pointwise_ci95=None if distribution is None else [float(v) for v in np.percentile(distribution, [2.5, 97.5])],
            simultaneous_one_sided_lower95=None, diagnostic_partial_only=True))
    return metrics, contrasts


class Evidence:
    """Strict local reader. Missing evidence is an audit problem, not zero data."""
    def __init__(self, root):
        self.root = Path(root)
        self.problems = []

    def problem(self, text):
        if text not in self.problems:
            self.problems.append(text)

    def json(self, name, default=None):
        if default is None:
            default = {}
        try:
            value = json.loads((self.root / name).read_text())
            if not isinstance(value, type(default)):
                raise ValueError('Wrong JSON container')
            return value
        except (OSError, ValueError):
            self.problem('Missing/malformed ' + name)
            return default

    def rows(self, name):
        try:
            lines = (self.root / name).read_text().splitlines()
        except OSError:
            self.problem('Missing ' + name)
            return []
        result = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Not an object')
                result.append(row)
            except ValueError:
                self.problem(f'Malformed {name}:{number}')
        return result

    def pin(self, relative, expected):
        """Never dereference an external/source-server path supplied by a row."""
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            self.problem('Invalid local file/SHA pin: ' + str(relative))
            return False
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            self.problem('Pin escapes saved evidence root: ' + relative)
            return False
        if not path.is_file() or sha(path) != expected:
            self.problem('Missing/changed SHA evidence: ' + relative)
            return False
        return True


def prepared_dataset(evidence):
    """Portable recheck; never open saved remote/model/Parquet path strings."""
    data = Evidence(evidence.root / 'data')
    manifest = data.json('DATASET_MANIFEST.json')
    if (manifest.get('status') != 'PREPARED_NOT_EXECUTED' or manifest.get('selection_rule') != SELECTION_RULE or
        manifest.get('seed') != SEED or any(manifest.get(k) is not True for k in
            ('no_outcomes_read', 'no_model_generation', 'no_training'))):
        data.problem('Prepared unused-validation dataset contract differs')
    try:
        dates = [datetime.fromisoformat(manifest[k]) for k in
            ('source_frozen_at_utc', 'candidate_first_open_not_before_utc', 'prepared_at_utc')]
        if any(d.tzinfo is None for d in dates) or dates != sorted(dates):
            raise ValueError('Out-of-order or naive time')
    except (KeyError, TypeError, ValueError):
        data.problem('Candidate content access did not follow source/protocol freeze')
    if manifest.get('dataset_file') != 'confirmation.jsonl':
        data.problem('Unexpected prepared dataset filename')
    data.pin('confirmation.jsonl', manifest.get('dataset_sha256'))
    pins = manifest.get('receipt_sha256', {})
    required = ('source_freeze', 'exclusions', 'final_protocol', 'baseline_complete', 'baseline_protocol',
        'baseline_frozen_selection', 'baseline_verified_winners', 'baseline_verified_complete',
        'baseline_final_manifest', 'old_raw_source_manifest')
    if not isinstance(pins, dict) or not all('receipts/' + n + '.json' in pins for n in required):
        data.problem('Incomplete portable pre-data freeze receipts')
        pins = pins if isinstance(pins, dict) else {}
    for name, digest in pins.items():
        data.pin(name, digest)
    links = dict(source_freeze_sha256='source_freeze', final_protocol_sha256='final_protocol',
        selection_freeze_sha256='baseline_frozen_selection', verified_winners_sha256='baseline_verified_winners',
        baseline_verified_complete_sha256='baseline_verified_complete', exclusion_receipt_sha256='exclusions')
    for field, name in links.items():
        if manifest.get(field) != pins.get('receipts/' + name + '.json'):
            data.problem('Prepared manifest freeze-chain link differs: ' + field)
    gate = data.json('receipts/source_freeze.json')
    formal = data.json('receipts/final_protocol.json')
    proof = gate.get('baseline_freeze', {})
    if (gate.get('status') != 'FROZEN_BEFORE_DATA_ACCESS' or gate.get('candidate_content_accessed') is not False or
        gate.get('selection_rule') != SELECTION_RULE or gate.get('seed') != SEED or
        formal.get('status') != 'FROZEN_BEFORE_DATA_ACCESS' or formal.get('is_template') is not False or
        formal.get('selection_rule') != SELECTION_RULE or formal.get('baseline_freeze') != proof or
        gate.get('source_sha256') != manifest.get('source_sha256') or
        gate.get('tokenizer_sha256') != manifest.get('tokenizer_sha256') or
        gate.get('template_sha256') != manifest.get('template_sha256') or
        gate.get('raw_source', {}).get('expected_sha256') != manifest.get('source', {}).get('sha256')):
        data.problem('Portable source/protocol/template/baseline chain differs')
    if (formal.get('fixed') != FIXED_CONTRACT or formal.get('own_method') != OWN_CONTRACT or
        formal.get('statistics') != STATISTICS_CONTRACT or formal.get('execution') != EXECUTION_CONTRACT or
        formal.get('required_source_files') != list(REQUIRED_SOURCE)):
        data.problem('Formal pre-data precision/method/statistics/execution/source contract changed')
    template = gate.get('template', {})
    template_sha = hashlib.sha256(json.dumps(template, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if template_sha != manifest.get('template_sha256'):
        data.problem('Prepared prompt template bytes changed')
    excluded = data.json('receipts/exclusions.json')
    seen = set(excluded.get('document_sha256', []))
    prefixes = {str(n): set(excluded.get('input_sha256', {}).get(str(n), [])) for n in LENGTHS}
    if len(excluded.get('metadata_sources', [])) != 11 or excluded.get('status') != 'OLD_DATA_METADATA_ONLY':
        data.problem('All eleven old used/reserved cohorts are not excluded')
    rows = data.rows('confirmation.jsonl')
    details = manifest.get('selected', [])
    docs = []
    inputs = {}
    if len(rows) != DOCUMENTS or len(details) != DOCUMENTS:
        data.problem('Exactly sixteen prepared confirmation documents required')
    previous = 899
    for row, saved in zip(rows, details):
        index = row.get('source_index')
        digest = row.get('document_sha256')
        if (type(index) is not int or index <= previous or index < 900 or
            row.get('split') != 'confirmation' or row.get('original_split') != 'validation' or
            type(row.get('original_tokens')) is not int or row['original_tokens'] < 16384 or
            not isinstance(digest, str) or len(digest) != 64 or digest in seen):
            data.problem('Unseen natural validation document selection differs')
            continue
        previous = index
        docs.append(index)
        hashes = {}
        left, right, content = (row.get(k) for k in ('prefix_ids', 'suffix_ids', 'document_ids'))
        if (not all(isinstance(v, list) and all(type(t) is int and t >= 0 for t in v) for v in (left, right, content)) or
            dict(prefix_ids=left, suffix_ids=right) != template or len(content) != 16384):
            data.problem('Prepared natural tokens/template differ')
            continue
        for length in LENGTHS:
            take = length - len(left) - len(right)
            if not 0 < take <= len(content):
                data.problem('Natural document cannot supply exact input context')
                continue
            ids = left + content[:take] + right
            digest_input = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            hashes[str(length)] = digest_input
            if digest_input in prefixes[str(length)]:
                data.problem('Used/reserved/selected prompt-prefix duplication')
            prefixes[str(length)].add(digest_input)
            inputs[(index, length)] = digest_input
        expected = dict(source_index=index, original_split='validation', document_sha256=digest,
            original_tokens=row['original_tokens'], input_sha256=hashes)
        if saved != expected:
            data.problem('Saved selected-document/input fingerprints differ')
        seen.add(digest)
    skipped = manifest.get('skipped', [])
    selected_map = {row.get('source_index'): row for row in details}
    if not isinstance(skipped, list) or any(not isinstance(row, dict) or type(row.get('source_index')) is not int for row in skipped):
        data.problem('Malformed first-eligible scan receipts')
        skipped = []
    skipped_map = {row['source_index']: row for row in skipped}
    if (len(skipped_map) != len(skipped) or set(skipped_map) & set(selected_map) or
        set(skipped_map) | set(selected_map) != set(range(900, previous + 1))):
        data.problem('First-eligible scan omits/duplicates/adds original source indices')
    scan_docs = set(excluded.get('document_sha256', []))
    scan_prefix = {str(n): set(excluded.get('input_sha256', {}).get(str(n), [])) for n in LENGTHS}
    for index in range(900, previous + 1):
        if index in selected_map:
            item = selected_map[index]
            digest, hashes = item.get('document_sha256'), item.get('input_sha256', {})
            if digest in scan_docs or any(hashes.get(n) in scan_prefix[n] for n in scan_prefix):
                data.problem('Selected document/prefix already appears in the prior scan denylist')
            scan_docs.add(digest)
            for n in scan_prefix:
                scan_prefix[n].add(hashes.get(n))
            continue
        item = skipped_map.get(index, {})
        digest, reason = item.get('document_sha256'), item.get('reason')
        valid_sha = isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
        if not valid_sha:
            data.problem('Skipped scan row lacks document SHA')
        elif reason == 'used_reserved_or_selected_document_sha256':
            if digest not in scan_docs:
                data.problem('Document skip is not justified by prior used/reserved/selected metadata')
        elif reason == 'natural_tokens_below16384':
            if digest in scan_docs or type(item.get('original_tokens')) is not int or not 0 <= item['original_tokens'] < 16384:
                data.problem('Invalid natural-short-document scan receipt')
        elif reason == 'used_reserved_or_selected_input_prefix_sha256':
            hashes = item.get('input_sha256', {})
            if (digest in scan_docs or type(item.get('original_tokens')) is not int or item['original_tokens'] < 16384 or
                set(hashes) != set(scan_prefix) or any(not isinstance(v, str) or len(v) != 64 for v in hashes.values()) or
                not any(hashes[n] in scan_prefix[n] for n in scan_prefix)):
                data.problem('Prefix skip is not justified by the prior scan denylist')
        else:
            data.problem('Unknown/outcome-dependent scan skip reason')
    if proof.get('status') != 'BASELINE_VERIFIED' or proof.get('selection_freeze_sha256') != manifest.get('selection_freeze_sha256'):
        data.problem('No verified frozen baseline before new-data selection')
    evidence.problems.extend('Dataset: ' + p for p in data.problems)
    return dict(manifest=manifest, gate=gate, final_protocol=formal, baseline=proof,
        documents=docs, input_sha256=inputs, root=data.root)


def baseline_chain(evidence, protocol, proof):
    """Rehash the copied complete baseline tree without its remote path strings."""
    receipt = protocol.get('baseline_evidence', {})
    if receipt.get('root') != 'baseline_evidence':
        evidence.problem('Portable complete baseline evidence root missing')
        return
    base = Evidence(evidence.root / 'baseline_evidence')
    def relative(key):
        value = receipt.get(key)
        if not isinstance(value, str) or Path(value).is_absolute() or '..' in Path(value).parts:
            base.problem('Unsafe/missing portable baseline path: ' + key)
            return '__missing__'
        return value
    result_dir, analysis_dir = relative('baseline_results'), relative('analysis_dir')
    verified_path, final_path = relative('verified_complete'), relative('final_manifest')
    final = base.json(final_path)
    inventory = final.get('file_sha256', {})
    if not inventory:
        base.problem('Empty final baseline artifact manifest')
    for name, digest in inventory.items():
        base.pin(name, digest)
    evidence.pin(receipt.get('pin_index'), receipt.get('pin_index_sha256'))
    index = evidence.json(receipt.get('pin_index', '__missing__')).get('original_to_saved', {})
    if not index:
        evidence.problem('Original-to-portable baseline pin index missing')
    for value in index.values():
        evidence.pin(value.get('saved_path'), value.get('sha256'))
    complete = base.json(result_dir + '/COMPLETE.json')
    frozen = base.json(result_dir + '/FROZEN_SELECTION.json')
    baseline_protocol = base.json(result_dir + '/protocol.json')
    summary = base.json(analysis_dir + '/summary.json')
    winners = base.json(analysis_dir + '/VERIFIED_WINNERS.json')
    analyzed = base.json(analysis_dir + '/ANALYSIS_COMPLETE.json')
    verified = base.json(verified_path)
    if protocol.get('flash_attention') != baseline_protocol.get('flash_attention'):
        base.problem('Flash Python/extension binary receipt changed from the verified baseline')
    assets = protocol.get('assets_sha256', {})
    if not baseline_protocol.get('assets_sha256') or any(assets.get(path) != digest for path, digest in baseline_protocol.get('assets_sha256', {}).items()):
        base.problem('Model/tokenizer/old-prime asset receipts differ from the verified baseline')
    for name in ('mapper', 'policy', 'cost'):
        if assets.get(protocol.get('args', {}).get(name)) != ASSETS[name + '_sha256']:
            base.problem('Actual mandatory asset-path SHA differs: ' + name)
    counts = dict(reference_requests=6, prime_requests=76, timing_requests=456, owners=76)
    pipeline = base.json('PIPELINE_EXIT.json')
    if type(pipeline.get('exit_code')) is not int or pipeline.get('exit_code') != 0 or 'PIPELINE_EXIT.json' not in inventory:
        base.problem('Baseline pipeline did not finish with signed integer exit0')
    base.pin('PIPELINE_EXIT.json', proof.get('pipeline_exit_sha256'))
    if (complete.get('passed') is not True or complete.get('counts') != counts or complete.get('logical_requests') != 538 or
        complete.get('owner_closed') != 76 or complete.get('goal_complete') is not False or any(complete.get(k) is not True for k in
            ('sources_unchanged', 'assets_unchanged', 'dependencies_unchanged', 'all_canonical', 'backend_trajectories_equal'))):
        base.problem('Baseline538/76 execution/exactness/immutability receipt invalid')
    if (summary.get('status') != 'COMPLETE' or summary.get('problems') != [] or
        summary.get('winner_status') != 'VERIFIED_FROZEN' or summary.get('observed_counts') != counts or
        winners.get('status') != 'VERIFIED_FROZEN' or winners.get('authoritative') is not True or analyzed.get('status') != 'COMPLETE'):
        base.problem('Baseline strict rendered analysis/winner freeze incomplete')
    for name, digest in analyzed.get('artifacts_sha256', {}).items():
        base.pin(analysis_dir + '/' + name, digest)
    if not analyzed.get('artifacts_sha256'):
        base.problem('Baseline analysis artifact pins missing')
    for name in ('quality', 'tps', 'captures', 'capture_time', 'cold', 'service', 'selected_repeat1'):
        for suffix in ('.png', '.svg'):
            if analysis_dir + '/' + name + suffix not in inventory:
                base.problem('Baseline rendered figure omitted from final manifest')
    flags = ('passed', 'raw_independent_audit_passed', 'rendered_figures_inspected', 'strict_analysis_complete',
        'all538_requests_verified', 'all76_owners_verified', 'eight_frozen_choices_verified')
    if verified.get('status') != 'VERIFIED_COMPLETE' or any(verified.get(k) is not True for k in flags):
        base.problem('Independent baseline raw/render verification receipt missing')
    paths = dict(selection_freeze_sha256=result_dir + '/FROZEN_SELECTION.json',
        verified_winners_sha256=analysis_dir + '/VERIFIED_WINNERS.json', baseline_complete_sha256=result_dir + '/COMPLETE.json',
        analysis_summary_sha256=analysis_dir + '/summary.json', verified_complete_sha256=verified_path,
        final_manifest_sha256=final_path, baseline_protocol_sha256=result_dir + '/protocol.json')
    for field, path in paths.items():
        base.pin(path, proof.get(field))
    verified_links = dict(baseline_complete_sha256=proof.get('baseline_complete_sha256'),
        frozen_selection_sha256=proof.get('selection_freeze_sha256'), analysis_summary_sha256=proof.get('analysis_summary_sha256'),
        verified_winners_sha256=proof.get('verified_winners_sha256'), final_artifacts_manifest_sha256=proof.get('final_manifest_sha256'))
    if any(verified.get(k) != value for k, value in verified_links.items()):
        base.problem('Independent baseline verification SHA links differ')
    if (frozen.get('complete_request_count') != 538 or frozen.get('group_candidates') != 8 or
        frozen.get('no_new_confirmation_opened') is not True or frozen.get('protocol_sha256') != proof.get('baseline_protocol_sha256') or
        complete.get('frozen_selection_sha256') != proof.get('selection_freeze_sha256')):
        base.problem('Frozen baseline selection is not bound to complete execution')
    raw = base.rows(result_dir + '/runtime.jsonl')
    if len(raw) != 456:
        base.problem('Complete456-row baseline selection data absent')
    for length in LENGTHS:
        chosen = winners.get('winners', {}).get('selected', {}).get(str(length), {})
        for group in GROUPS:
            arm = proof.get('choices', {}).get(str(length), {}).get(group, {})
            mode = arm.get('mode')
            registered = frozen.get('per_context', {}).get(str(length), {}).get('groups', {}).get(group, {})
            if registered.get('arm') != arm or registered.get('mode') != mode or chosen.get(group) != mode:
                base.problem('Frozen eight baseline descriptors/modes changed')
            candidates = []
            for descriptor in baseline_protocol.get('arms', []):
                actual_group = ('full_tree' if descriptor.get('family') == 'full_tree' else 'retrieval/' +
                    ('graph' if descriptor.get('backend') == 'cuda_graph' else str(descriptor.get('backend'))))
                if actual_group != group:
                    continue
                rows = [r for r in raw if r.get('mode') == descriptor.get('mode') and r.get('length') == length]
                if len(rows) == 6 and all(positive(r.get('outer_seconds')) and positive(r.get('tokens')) for r in rows):
                    candidates.append((sum(r['tokens'] for r in rows) / sum(r['outer_seconds'] for r in rows), descriptor['mode']))
            if not candidates or min(candidates, key=lambda item: (-item[0], item[1]))[1] != mode:
                base.problem('Frozen winner differs from exact original6warm pooled score')
    for name, digest in baseline_protocol.get('source_sha256', {}).items():
        if name in protocol.get('source_sha256', {}) and (name.startswith('rescue/') or name.startswith('exp/')) and protocol['source_sha256'][name] != digest:
            base.problem('Shared algorithm/runtime source changed after baseline selection: ' + name)
    prime_refs = evidence.json('prime_references.json')
    old_refs = base.json(result_dir + '/references.json')
    if any(prime_refs.get(str(n)) != old_refs.get(f'271:{n}') or len(prime_refs.get(str(n), [])) != 128 or
        protocol.get('prime_input_sha256', {}).get(str(n)) != baseline_protocol.get('input_sha256', {}).get(f'271:{n}') for n in LENGTHS):
        base.problem('Old271 cold prime differs from the verified canonical input/output')
    evidence.problems.extend('Baseline: ' + p for p in base.problems)


def linear_trace_errors(row, *, own=False, fixed_k=None):
    errors = []
    trace = row.get('trace', [])
    if not isinstance(trace, list) or not trace or row.get('blocks') != len(trace):
        return ['Missing linear round trace']
    if row.get('accepted') != sum(t.get('accepted', 0) for t in trace) or row.get('proposed') != sum(t.get('proposed', 0) for t in trace):
        errors.append('Linear accepted/proposed totals disagree')
    previous = None
    for i, item in enumerate(trace):
        prefix, cap, proposed, accepted = (item.get(k) for k in ('prefix', 'cap', 'proposed', 'accepted'))
        if not all(type(v) is int for v in (prefix, cap, proposed, accepted)):
            errors.append('Noninteger linear trajectory')
            continue
        remaining = OUTPUT_LIMIT - (prefix - (row.get('length', 0) - 1))
        valid_cap = min(fixed_k, remaining) == cap if fixed_k is not None else cap in (min(2, remaining), min(4, remaining))
        if (not valid_cap or not 1 <= proposed <= cap <= (4 if own else 12) or
            not 0 <= accepted <= proposed or item.get('round_index') != i or
            item.get('terminal') is not (i == len(trace) - 1) or
            (i == 0 and prefix != row.get('length', 0) - 1) or
            (previous is not None and prefix != previous['prefix'] + previous['accepted'] + 1)):
            errors.append('Invalid cap/EOS/committed-prefix trajectory')
        previous = item
    if previous is not None:
        progress = previous.get('prefix', 0) - (row.get('length', 0) - 1)
        if row.get('tokens') not in (progress + previous.get('accepted', 0), progress + previous.get('accepted', 0) + 1):
            errors.append('Terminal committed count does not explain client token count')
    if not near(row.get('acceptance_rate'), row.get('accepted', 0) / max(1, row.get('proposed', 0))):
        errors.append('Incorrect saved linear acceptance rate')
    return errors


def tree_errors(row, arm=None):
    upstream = row.get('upstream', {})
    accepted = upstream.get('accept_length_list', [])
    raw = row.get('raw_total_generated')
    scored = row.get('scored_generated_before_eos')
    overshoot = row.get('generation_overshoot')
    good = (isinstance(accepted, list) and accepted and all(type(v) is int and v >= 0 for v in accepted) and
        row.get('accepted') == sum(accepted) and row.get('blocks') == len(accepted) and
        row.get('proposed') is None and row.get('acceptance_rate') is None and
        type(raw) is int and raw == sum(accepted) + len(accepted) and isinstance(scored, list) and
        type(overshoot) is int and raw - len(scored) == overshoot >= 0 and
        upstream.get('raw_total_generated') == raw and upstream.get('generation_overshoot') == overshoot and
        upstream.get('output_token_ids') == scored and upstream.get('total_generated') == len(scored) and
        scored[:row.get('tokens', 0)] == row.get('generated') and row.get('raw_tail_ids_available') is False and
        row.get('wrapper_generated') == row.get('generated') and row.get('wrapper_tokens') == row.get('tokens'))
    errors = [] if good else ['Tree raw acceptance/overshoot/client output accounting differs']
    if arm is not None:
        policy = dict(nodes=arm['nodes'], top_chunks=arm['top_chunks'], max_depth=10, threshold=.7,
            chunk_size=32, refresh_every=4, flash_prefill=True, hybrid_tree=True,
            prefill_precision='bfloat16', mapper=False, controller=False)
        flags = dict(target_use_flash_prefill=True, target_use_hybrid_tree_attn=True, draft_use_flash_prefill=True,
            use_retrieval_cache=True, retrieve_top_k=arm['top_chunks'], retrieval_chunk_size=32, retrieve_every_n_steps=4)
        if row.get('tree_policy') != policy or row.get('observed_full_config') != flags or any(v > 10 for v in accepted):
            errors.append('Actual full-tree Flash/hybrid/retrieval/depth policy differs')
    return errors


def quality_pair_errors(before, after):
    """Exact saved evidence, not only a runner's passed=True marker."""
    errors = []
    for field in ('generated', 'observed_rounds', 'queries', 'native_t', 'native_s'):
        if field not in before or field not in after or before[field] != after[field]:
            errors.append('Quality eager/execution mismatch: ' + field)
    if trace_signature(before) != trace_signature(after):
        errors.append('Quality eager/execution mismatch: logical round trajectory')
    for row in (before, after):
        for role in ('native_t', 'native_s'):
            cache = row.get(role, {})
            lengths, layers = cache.get('lengths', []), cache.get('layers', [])
            if len(lengths) != 28 or len(layers) != 28:
                errors.append('Missing all28 Native layers: ' + role)
                continue
            for length, layer in zip(lengths, layers):
                if type(length) is not int or length < row.get('length', 0) - 1 or any(
                    not digest_valid(layer.get(k)) or len(layer[k]['shape']) != 4 or layer[k]['shape'][2] != length
                    for k in ('key', 'value')):
                    errors.append('Invalid Native valid-span digest')
        if not row.get('queries') or any(not digest_valid(q.get('query')) for q in row['queries']):
            errors.append('Missing every-step independent Query digests')
    return errors


def observation_errors(row):
    errors = []
    rounds = row.get('observed_rounds', [])
    if len(rounds) != row.get('blocks'):
        errors.append('Not every actual verification round was observed')
    generated = []
    for index, observed in enumerate(rounds):
        if index >= len(row.get('trace', [])):
            errors.append('Observed extra round outside actual trace')
            continue
        trace = row['trace'][index]
        selected = observed.get('selected', [])
        if (not selected or len(selected) != len(set(selected)) or any(type(p) is not int or p < 0 or p >= observed.get('prefix', 0) for p in selected) or
            observed.get('progress') != len(generated) or observed.get('remaining') != 128-len(generated) or
            any(observed.get(k) != trace.get(k) for k in ('round_index', 'prefix', 'accepted'))):
            errors.append('Selected positions/observed state do not align with actual round')
        draft, checks = observed.get('draft', []), observed.get('checks', [])
        if len(draft) != trace.get('proposed') or len(checks) != len(draft)+1:
            errors.append('Missing actual per-round Draft/Target check tokens')
        accepted = 0
        for token, check in zip(draft, checks):
            if token != check:
                break
            accepted += 1
        if accepted != observed.get('accepted'):
            errors.append('Actual contiguous agreement differs from accepted count')
        generated.extend(draft[:accepted])
        if len(generated) < row.get('tokens', 0) and accepted < len(checks):
            generated.append(checks[accepted])
    if generated != row.get('generated'):
        errors.append('Raw observation commits differ from canonical client tokens')
    if len(row.get('queries', [])) != row.get('draft_forwards'):
        errors.append('Missing actual proposal/catchup per-step Query count')
    return errors


def trace_signature(row):
    fields = ('round_index', 'prefix', 'cap', 'proposed', 'accepted', 'terminal', 'workset_tokens')
    return [[item.get(k) for k in fields] for item in row.get('trace', [])]


def injection_errors(row):
    errors = []
    nonzero = False
    reconstructed = []
    rounds = row.get('observed_rounds', [])
    if len(rounds) != row.get('blocks'):
        errors.append('Missing every-round own-method quality observations')
    for index, observed in enumerate(rounds):
        if observed.get('progress') != len(reconstructed) or observed.get('remaining') != 128-len(reconstructed):
            errors.append('Observed generation progress is not the committed prefix')
        selected = observed.get('selected', [])
        if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or
            not all(type(v) is int and 0 <= v < observed.get('prefix', 0) for v in selected)):
            errors.append('Invalid absolute selected positions')
        vector = observed.get('budget_features')
        if vector is not None and (not isinstance(vector, list) or len(vector) != 12 or
            not all(type(v) in (int, float) and math.isfinite(v) for v in vector)):
            errors.append('Invalid actual twelve-dimensional M3 features')
        if index < len(row.get('trace', [])):
            trace = row['trace'][index]
            if any(observed.get(k) != trace.get(k) for k in ('round_index', 'prefix', 'accepted')):
                errors.append('Observed quality rounds differ from logical trace')
            draft, checks = observed.get('draft', []), observed.get('checks', [])
            if len(draft) != trace.get('proposed') or len(checks) != len(draft) + 1:
                errors.append('Missing actual draft/Target verification tokens')
            accepted = 0
            for draft_token, check in zip(draft, checks):
                if draft_token != check:
                    break
                accepted += 1
            if accepted != observed.get('accepted'):
                errors.append('Saved acceptance differs from contiguous token agreement')
            reconstructed.extend(draft[:accepted])
            if len(reconstructed) < row.get('tokens', 0) and accepted < len(checks):
                reconstructed.append(checks[accepted])
        for evidence in observed.get('injection_evidence', []):
            working, native = evidence.get('working_value', {}), evidence.get('native_value', {})
            positions, count = evidence.get('positions', []), evidence.get('different_elements')
            good = (evidence.get('layer') == 27 and isinstance(positions, list) and positions and
                len(set(positions)) == len(positions) and all(p in selected for p in positions) and
                digest_valid(working) and digest_valid(native) and working['shape'] == native['shape'] and
                working['dtype'] == native['dtype'] == 'torch.bfloat16' and len(working['shape']) == 2 and
                working['shape'][0] == len(positions) and type(count) is int and
                0 <= count <= math.prod(working['shape']) and evidence.get('native_matches_memo') is True)
            if not good:
                errors.append('Invalid same-BF16 actual working-V/native-V proof')
            elif (count > 0) == (working['sha256'] == native['sha256']):
                errors.append('Working-V difference count contradicts saved byte hashes')
            else:
                nonzero |= count > 0
    if not nonzero or row.get('nonzero_injection') is not True:
        errors.append('Missing genuine nonzero Mapper injection')
    if reconstructed != row.get('generated'):
        errors.append('Saved draft/check commits do not reconstruct exact client tokens')
    queries = row.get('queries', [])
    if len(queries) != row.get('draft_forwards') or any(
        type(q.get('logits')) is not bool or type(q.get('summary_requested')) is not bool or
        type(q.get('summary_count')) is not int or q['summary_count'] < 0 for q in queries):
        errors.append('Missing real per-step M3 summary schedule')
    cursor = 0
    for item in row.get('trace', []):
        for depth in range(item.get('proposed', 0)):
            if cursor >= len(queries) or queries[cursor].get('logits') is not True or queries[cursor].get('summary_requested') is not (depth < 2):
                errors.append('Causal first-two-step summary schedule differs')
            cursor += 1
        if item.get('accepted') == item.get('proposed') and not item.get('terminal'):
            if cursor >= len(queries) or queries[cursor].get('logits') is not False or queries[cursor].get('summary_requested') is not False:
                errors.append('Nonterminal catchup entered M3 summaries')
            cursor += 1
    if cursor != len(queries):
        errors.append('Query work cannot explain proposal/catchup schedule')
    return errors


def timing_errors(row, reference, arm, own_mode):
    errors = []
    if (not isinstance(reference, list) or not reference or row.get('generated') != reference or
        row.get('tokens') != len(reference) or not 0 < len(reference) <= OUTPUT_LIMIT or
        row.get('output_agrees') is not True):
        errors.append('Client-visible output differs from canonical Target AR')
    if (not positive(row.get('outer_seconds')) or not near(row.get('outer_tps'),
        row.get('tokens', 0) / row['outer_seconds'] if positive(row.get('outer_seconds')) else -1) or
        row.get('performance_eligible') is not True or 'timeline' in row or 'cost_samples' in row):
        errors.append('Invalid/profile-contaminated complete-generate outer timing')
    if positive(row.get('total_seconds')) and row.get('outer_seconds', 0) + 1e-6 < row['total_seconds']:
        errors.append('Outer clock excludes part of generate')
    if arm.get('family') == 'full_tree':
        errors.extend(tree_errors(row, arm))
    else:
        errors.extend(linear_trace_errors(row, own=row.get('mode') == own_mode,
            fixed_k=None if row.get('mode') == own_mode else arm.get('fixed_k')))
    if arm.get('family') != 'full_tree':
        execution = 'cuda_graph' if row.get('mode') == own_mode else arm.get('backend')
        delta = row.get('draft_runtime', {}).get('request', {})
        actual_execution = row.get('draft_runtime', {}).get('mode') if row.get('mode') == own_mode else row.get('execution_backend')
        if actual_execution != execution:
            errors.append('Execution backend differs from frozen arm')
        if execution == 'cuda_graph' and (delta.get('replays') != row.get('draft_forwards') or not positive(delta.get('replays'))):
            errors.append('Graph replay count differs from actual Draft forwards')
        if execution == 'resident' and (delta.get('resident_steps') != row.get('draft_forwards') or delta.get('captures') != 0):
            errors.append('Resident work/capture ledger differs')
        if execution != 'eager' and (type(delta.get('captures')) is not int or delta['captures'] < 0 or
            not nonnegative(delta.get('capture_host_seconds'))):
            errors.append('Missing real capture/replay accounting')
        if execution != 'eager' and row.get('runtime_stats', {}).get('max_graph_entries') != 128:
            errors.append('Actual resident/Graph owner entry limit differs from128')
        if execution != 'eager' and (delta.get('captures', 0) > 0) != (delta.get('capture_host_seconds', 0) > 0):
            errors.append('New capture count/time disagree')
        if row.get('mode') != own_mode and (row.get('projected_positions') != 0 or row.get('draft_shadow_probes') != 0):
            errors.append('Retrieval control executed unused Mapper/probes')
    return errors


def lifecycle_errors(owner, requests):
    errors = []
    prime = [r for r in requests if r.get('phase') == 'cold_prime']
    warm = [r for r in requests if r.get('phase') == 'warm']
    if len(prime) != 1 or len(warm) != DOCUMENTS * REPEATS:
        return ['Private owner lacks one cold prime and complete32warm requests']
    ids = [r['request_id'] for r in warm]
    seconds = sum(r['outer_seconds'] for r in warm)
    tokens = sum(r['tokens'] for r in warm)
    if (owner.get('prime_request_id') != prime[0].get('request_id') or owner.get('warm_request_ids') != ids or
        not near(owner.get('warm_outer_seconds_sum'), seconds) or owner.get('warm_tokens') != tokens):
        errors.append('Owner request/time/token ledger differs')
    constructor = owner.get('constructor_and_detach_seconds')
    close = owner.get('close_seconds')
    cold = prime[0]['outer_seconds']
    if not nonnegative(constructor) or not nonnegative(close):
        errors.append('Missing charged constructor/close intervals')
    elif not near(owner.get('accounted_service_seconds'), constructor + cold + seconds + close) or not near(
        owner.get('accounted_service_tps'), (tokens + prime[0]['tokens']) / (constructor + cold + seconds + close)):
        errors.append('Accounted-service interval excludes actual measured work')
    if owner.get('engine_closed') is not True or owner.get('hook_restored') is not True:
        errors.append('Private owner not fully closed/restored')
    signature = owner.get('owner_signature')
    if not signature or any(row.get('owner_signature') != signature for row in requests):
        errors.append('Cache/Graph owner addresses changed within private owner')
    closed = owner.get('closed_runtime_stats', {})
    final = owner.get('final_runtime_stats', {})
    if closed.get('entries') != 0 or final != warm[-1].get('runtime_stats'):
        errors.append('Graph entries not released at owner close')
    if final.get('mode') in ('resident', 'cuda_graph'):
        for name in ('captures', 'replays', 'resident_steps', 'capture_host_seconds'):
            if not near(final.get(name), sum(r.get('draft_runtime', {}).get('request', {}).get(name, 0) for r in requests)):
                errors.append('Lifetime/request runtime ledger differs: ' + name)
    return errors


def extra_geometries(choices):
    result = []
    for length in LENGTHS:
        groups = choices[str(length)]
        eager = groups['retrieval/eager']
        seen = {(eager['top_chunks'], eager['fixed_k'])}
        for group in ('retrieval/resident', 'retrieval/graph'):
            arm = groups[group]
            geometry = arm['top_chunks'], arm['fixed_k']
            if geometry not in seen:
                result.append(dict(length=length, top_chunks=geometry[0], fixed_k=geometry[1]))
                seen.add(geometry)
    return result


def registered_schedule(choices, documents):
    import random
    owners, warm = [], []
    for length in LENGTHS:
        rng = random.Random(SEED + length)
        modes = ['rescue_graph'] + [choices[str(length)][g]['mode'] for g in GROUPS]
        rng.shuffle(modes)
        for mode in modes:
            owners.append(dict(mode=mode, length=length))
            for repeat in (0, 1):
                order = list(documents)
                rng.shuffle(order)
                warm.extend(dict(mode=mode, length=length, source_index=d, repeat=repeat) for d in order)
    return owners, warm


def _analyze(directory):
    evidence = Evidence(directory)
    protocol = evidence.json('protocol.json')
    dataset = prepared_dataset(evidence)
    documents = dataset['documents']
    proof = dataset['baseline']
    choices = proof.get('choices', {})
    complete_choices = set(choices) == set(map(str, LENGTHS)) and all(
        isinstance(choices.get(str(n)), dict) and set(choices[str(n)]) == set(GROUPS) for n in LENGTHS)
    if not complete_choices:
        evidence.problem('All eight authoritative baseline choices missing')
        return dict(status='PARTIAL', verified_success=False, problems=evidence.problems,
            observed_counts={}, own_modes={}, challengers={}, statistics={})
    extras = extra_geometries(choices)
    expected = dict(reference_requests=32, own_quality_requests=64, baseline_quality_requests=8,
        matching_quality_requests=len(extras), prime_requests=10, timing_requests=320, owners=10)
    total = 434 + len(extras)
    own_modes = {str(n): 'rescue_graph' for n in LENGTHS}
    challengers = {str(n): {g: choices[str(n)][g]['mode'] for g in GROUPS} for n in LENGTHS}
    arms = {str(n): [dict(mode='rescue_graph', family='rescue', backend='cuda_graph')] +
        [choices[str(n)][g] for g in GROUPS] for n in LENGTHS}
    maps = {n: {a['mode']: a for a in values} for n, values in arms.items()}
    if (protocol.get('arms') != arms or protocol.get('modes_by_context') != {n: list(v) for n, v in maps.items()} or
        protocol.get('source_indices') != documents or protocol.get('lengths') != list(LENGTHS) or
        protocol.get('repeats') != 2 or protocol.get('max_new_tokens') != 128 or
        protocol.get('seed') != SEED or protocol.get('expected_counts') != expected or
        protocol.get('expected_requests') != total or protocol.get('extra_eager_reference_geometries') != extras):
        evidence.problem('Frozen434+R arm/cohort/context/repeat/count plan differs')
    if (proof.get('extra_eager_reference_geometries') != extras or proof.get('scientific_requests') != total or
        dataset['final_protocol'].get('expected_scientific_requests') != total):
        evidence.problem('Matching geometry requests changed after baseline freeze')
    if (protocol.get('baseline_freeze') != proof or protocol.get('frozen_choices') != choices or
        protocol.get('source_freeze_sha256') != dataset['manifest'].get('source_freeze_sha256') or
        protocol.get('final_protocol_sha256') != dataset['manifest'].get('final_protocol_sha256') or
        protocol.get('dataset_sha256') != dataset['manifest'].get('dataset_sha256') or
        protocol.get('dataset_manifest') != 'data/DATASET_MANIFEST.json' or
        protocol.get('portable_data_root') != 'data' or protocol.get('selection_rule') != SELECTION_RULE):
        evidence.problem('Executed confirmation is not bound to the pre-data freeze chain')
    evidence.pin('data/DATASET_MANIFEST.json', protocol.get('dataset_manifest_sha256'))
    gate_passed = evidence.json('GATE_PASSED.json')
    if (gate_passed.get('passed') is not True or gate_passed.get('confirmation_content_opened') is not False or
        gate_passed.get('baseline_freeze') != proof or gate_passed.get('expected_counts') != expected or
        gate_passed.get('expected_requests') != total or gate_passed.get('source_freeze_sha256') != dataset['manifest'].get('source_freeze_sha256')):
        evidence.problem('Scientific runner did not verify the source/baseline gate before data access')
    for argument, digest in (('freeze_receipt', dataset['manifest'].get('source_freeze_sha256')),
        ('dataset_manifest', protocol.get('dataset_manifest_sha256')), ('data_file', protocol.get('dataset_sha256'))):
        if protocol.get('assets_sha256', {}).get(protocol.get('args', {}).get(argument)) != digest:
            evidence.problem('Executed pre-data source/manifest/input asset pin differs: ' + argument)
    if (any(protocol.get(k) != v for k, v in ASSETS.items()) or protocol.get('mapper_frozen') is not True or
        protocol.get('policy_frozen') is not True or protocol.get('no_training') is not True or
        protocol.get('max_live_owners') != 1 or protocol.get('template_prefill_precision') != 'bfloat16'):
        evidence.problem('Nonzero Mapper/old-cost/policy/shared execution freeze differs')
    fixed_common = dict(max_new_tokens=128, max_draft=12, refresh_every=4, prefill_precision='bfloat16',
        schedule='serial', profile=False, working_mode='packed', optimized_execution=True,
        deferred_probes=True, attention_backend='sdpa', coverage_execution='eager')
    for length in LENGTHS:
        for arm in arms[str(length)]:
            config = protocol.get('configs', {}).get(str(length), {}).get(arm['mode'], {})
            if arm['family'] == 'full_tree':
                wanted = dict(nodes=arm['nodes'], top_chunks=arm['top_chunks'], max_depth=10, threshold=.7,
                    chunk_size=32, refresh_every=4, flash_prefill=True, hybrid_tree=True,
                    prefill_precision='bfloat16', mapper=False, controller=False)
            else:
                wanted = dict(fixed_common, draft_execution=arm['backend'])
                wanted.update(dict(selector='hierarchical', coverage_rows=64, refresh_policy='periodic',
                    budget=512, recent=64, coverage_candidates=1024, coverage_batch=16, coarse_chunk_size=32,
                    injection_layers=[27], inject_tokens=64, injection_strength=.5, length_policy='risk') if arm['family'] == 'rescue'
                    else dict(injection_layers=[], inject_tokens=0, injection_strength=0., length_policy='fixed', fixed_k=arm['fixed_k']))
            if any(config.get(k) != value for k, value in wanted.items()):
                evidence.problem('Frozen actual configuration differs: ' + arm['mode'] + '/' + str(length))
    environment = protocol.get('environment', {})
    if ('H100' not in environment.get('gpu_name', '') or environment.get('total_memory', 0) < 80_000_000_000 or
        environment.get('target_dtype') != 'torch.float32' or environment.get('draft_dtype') != 'torch.bfloat16' or
        environment.get('prefix_dtype') != 'torch.bfloat16' or environment.get('tf32') is not False):
        evidence.problem('H10080GB/FP32Target/BF16Draft+prefix/TF32off environment differs')
    inventory = protocol.get('source_sha256', {})
    if not inventory or inventory != dataset['manifest'].get('source_sha256'):
        evidence.problem('Executed source differs from source frozen before new-data access')
    if not set(REQUIRED_SOURCE) <= set(inventory) or len([n for n in inventory if n.startswith('exp/specextend_bridgekv/specextend/') and n.endswith('.py')]) < 27:
        evidence.problem('Pre-data source inventory omits final code/tests/preregistration/launcher/full tree')
    for name, digest in inventory.items():
        evidence.pin('source/' + name, digest)
    bindings = protocol.get('source_bindings', {})
    if not bindings or not protocol.get('tree_source_bindings'):
        evidence.problem('Missing actual imported code provenance')
    for name, value in bindings.items():
        relative = name.replace('.', '/') + '.py'
        if not str(value.get('path', '')).endswith('/' + relative) or inventory.get(relative) != value.get('sha256'):
            evidence.problem('Actual import/source SHA differs: ' + name)
    for value in protocol.get('tree_source_bindings', {}).values():
        if 'sha256' in value and inventory.get(value.get('path')) != value['sha256']:
            evidence.problem('Archived full-tree source provenance differs')
        if 'namespace_paths' in value and (not value['namespace_paths'] or any(
            not p.startswith('exp/specextend_bridgekv/specextend/') or '..' in Path(p).parts for p in value['namespace_paths'])):
            evidence.problem('Archived full-tree namespace escaped staged code')
    if not {'classic.model_classic', 'shared.modeling_llama_kv_target'} <= set(protocol.get('tree_source_bindings', {})):
        evidence.problem('Actual full-tree imported class/attention bindings missing')
    flash = protocol.get('flash_attention', {})
    if flash.get('version') != '2.8.3' or not str(flash.get('files', {}).get('flash_attn_2_cuda', {}).get('path', '')).endswith('.so'):
        evidence.problem('Official Flash2.8.3 binary receipt missing')
    baseline_chain(evidence, protocol, proof)
    # The root's exported evidence inventory is rehashed; it is not a model-file
    # reread or a claim that this CPU machine owns the remote model assets.
    outer_path = Path(directory).parent / 'EVIDENCE_MANIFEST.json'
    if outer_path.is_file():
        outer = Evidence(Path(directory).parent)
        values = outer.json('EVIDENCE_MANIFEST.json').get('file_sha256', {})
        if not values:
            outer.problem('Empty exported evidence manifest')
        for name, digest in values.items():
            outer.pin(name, digest)
        evidence.problems.extend('Export: ' + p for p in outer.problems)
    else:
        evidence.problem('Exported raw/source evidence manifest missing')
    canonical = evidence.rows('reference.jsonl')
    own_quality = evidence.rows('own_quality.jsonl')
    baseline_quality = evidence.rows('baseline_quality.jsonl')
    matching_quality = evidence.rows('matching_quality.jsonl') if extras or (Path(directory) / 'matching_quality.jsonl').exists() else []
    prime = evidence.rows('prime.jsonl')
    warm = evidence.rows('runtime.jsonl')
    owners = evidence.rows('owners.jsonl')
    quality_owners = evidence.rows('quality_owners.jsonl')
    observed = dict(reference_requests=len(canonical), own_quality_requests=len(own_quality),
        baseline_quality_requests=len(baseline_quality), matching_quality_requests=len(matching_quality),
        prime_requests=len(prime), timing_requests=len(warm), owners=len(owners))
    if observed != expected:
        evidence.problem('Complete434+R request/owner matrix missing or duplicated')
    requests = canonical + own_quality + baseline_quality + matching_quality + prime + warm
    ids = [r.get('request_id') for r in requests]
    indices = [r.get('execution_index') for r in requests]
    if any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
        evidence.problem('Scientific request UUID missing/reused')
    valid_indices = all(type(i) is int for i in indices) and len(set(indices)) == len(indices)
    if not valid_indices or sorted(indices) != list(range(total)):
        evidence.problem('Execution index ledger incomplete/duplicated')
    quality_rows = own_quality + baseline_quality + matching_quality
    if valid_indices and ((canonical and max(r['execution_index'] for r in canonical) >= 32) or
        (quality_rows and min(r['execution_index'] for r in quality_rows) < 32) or
        (quality_rows and prime and max(r['execution_index'] for r in quality_rows) >= min(r['execution_index'] for r in prime))):
        evidence.problem('References/complete quality gate did not precede timing')
    quality_gate = evidence.json('QUALITY_PASSED.json')
    if quality_gate.get('passed') is not True or any(quality_gate.get(k) is not True for k in
        ('own_all32_pairs', 'baseline_matching_geometry', 'actual_same_bfloat16_nonzero_mapper', 'quality_owners_closed')):
        evidence.problem('All quality gates must pass before timed owner construction')
    refs = evidence.json('references.json')
    prime_refs = evidence.json('prime_references.json')
    reference_map = {}
    for row in canonical:
        key = row.get('source_index'), row.get('length')
        generated = row.get('generated')
        if (key not in dataset['input_sha256'] or key in reference_map or not isinstance(generated, list) or
            not 0 < len(generated) <= OUTPUT_LIMIT or not all(type(v) is int and v >= 0 for v in generated) or
            refs.get(f'{key[0]}:{key[1]}') != generated):
            evidence.problem('Canonical32 independent reference cells inconsistent')
        reference_map[key] = generated
    if len(reference_map) != 32:
        evidence.problem('Canonical32 references incomplete')
    if protocol.get('input_sha256') != {f'{d}:{n}': v for (d, n), v in dataset['input_sha256'].items()}:
        evidence.problem('Executed inputs differ from exact prepared4K/8K fingerprints')
    eos = protocol.get('tokenizer_eos_token_id')
    if (type(eos) is not int or eos < 0 or type(protocol.get('vocab_size')) is not int or
        not eos < protocol['vocab_size'] or protocol.get('target_layers') != 28 or protocol.get('draft_layers') != 28 or
        protocol.get('graph_max_entries') != 128):
        evidence.problem('Frozen tokenizer EOS/vocabulary/28layer/Graph128 metadata invalid')
    for row in canonical:
        generated = row.get('generated', [])
        if len(generated) < OUTPUT_LIMIT and (eos is None or not generated or generated[-1] != eos):
            evidence.problem('Short canonical output lacks frozen tokenizer EOS evidence')
        if eos is not None and eos in generated and (generated[-1] != eos or eos in generated[:-1]):
            evidence.problem('Canonical EOS boundary changed')
    bad_cells = set()
    qown = {}
    qbaseline = {}
    qmatching = {}
    for row in quality_rows:
        key = row.get('source_index'), row.get('length')
        reference = reference_map.get(key)
        if row.get('generated') != reference or row.get('output_agrees') is not True or row.get('tokens') != len(reference or []):
            evidence.problem('Quality request differs from independent Target AR')
        if row.get('performance_eligible') is not False:
            evidence.problem('Instrumented quality was marked primary timing')
        if row.get('arm', {}).get('family') != 'full_tree':
            for problem in observation_errors(row) + quality_pair_errors(row, row):
                evidence.problem('Raw quality evidence: ' + problem)
            backend = row.get('arm', {}).get('backend')
            if backend in ('resident', 'cuda_graph') and row.get('runtime_stats', {}).get('max_graph_entries') != 128:
                evidence.problem('Quality owner actual Graph entry limit differs from128')
    for row in own_quality:
        key = row.get('source_index'), row.get('length'), row.get('mode')
        if key in qown or key[0] not in documents or key[1] not in LENGTHS or key[2] not in ('rescue_eager', 'rescue_graph') or row.get('phase') != 'own_quality':
            evidence.problem('Own64 quality matrix missing/duplicated')
        if row.get('arm') != dict(mode=key[2], family='rescue', backend='eager' if key[2]=='rescue_eager' else 'cuda_graph'):
            evidence.problem('Own quality did not execute the declared eager/Graph arm')
        qown[key] = row
        for problem in linear_trace_errors(row, own=True) + injection_errors(row):
            evidence.problem('Own quality: ' + problem)
    for length in LENGTHS:
        for doc in documents:
            before, after = qown.get((doc, length, 'rescue_eager')), qown.get((doc, length, 'rescue_graph'))
            if before is None or after is None:
                evidence.problem('Own eager/Graph full32 quality pairs incomplete')
                continue
            for problem in quality_pair_errors(before, after):
                evidence.problem('Own quality: ' + problem)
    for row in baseline_quality:
        length, mode = row.get('length'), row.get('mode')
        arm = maps.get(str(length), {}).get(mode)
        if not arm or arm['family'] == 'rescue' or row.get('source_index') != (documents[0] if documents else None) or row.get('phase') != 'baseline_sentinel' or (length, mode) in qbaseline:
            evidence.problem('Frozen baseline8 sentinel quality matrix differs')
            continue
        qbaseline[length, mode] = row
        if arm['family'] == 'full_tree':
            for problem in tree_errors(row, arm):
                evidence.problem('Baseline tree quality: ' + problem)
        else:
            for problem in linear_trace_errors(row, fixed_k=arm['fixed_k']):
                evidence.problem('Baseline quality: ' + problem)
            if row.get('nonzero_injection') is not False or any(q.get('summary_requested') is not False or q.get('summary_count') != 0 for q in row.get('queries', [])):
                evidence.problem('Retrieval sentinel secretly executed Mapper/M3')
            if arm['backend'] == 'eager':
                qmatching[length, arm['top_chunks'], arm['fixed_k']] = row
    for row in matching_quality:
        arm = row.get('arm', {})
        geometry = row.get('length'), arm.get('top_chunks'), arm.get('fixed_k')
        expected_geometry = dict(length=geometry[0], top_chunks=geometry[1], fixed_k=geometry[2])
        if expected_geometry not in extras or geometry in qmatching or row.get('phase') != 'matching_geometry_eager' or row.get('source_index') != (documents[0] if documents else None) or arm.get('backend') != 'eager':
            evidence.problem('Matching eager geometry quality changed after freeze')
        qmatching[geometry] = row
    for length in LENGTHS:
        for group in ('retrieval/resident', 'retrieval/graph'):
            arm = choices[str(length)][group]
            a = qmatching.get((length, arm['top_chunks'], arm['fixed_k']))
            b = qbaseline.get((length, arm['mode']))
            if a is None or b is None:
                evidence.problem('Selected retrieval geometry lacks exact eager quality reference')
            else:
                for problem in quality_pair_errors(a, b):
                    evidence.problem('Matching baseline quality: ' + problem)
    qo = {r.get('request_id'): r for r in quality_owners}
    if len(qo) != 72 + len(extras) or len(quality_owners) != len(qo):
        evidence.problem('All instrumented quality owner-close receipts missing')
    for row in quality_rows:
        owner = qo.get(row.get('request_id'), {})
        if (owner.get('owner_id') != row.get('owner_id') or owner.get('engine_closed') is not True or
            owner.get('hook_restored') is not True or owner.get('closed_runtime_stats', {}).get('entries') != 0 or
            owner.get('final_runtime_stats') != row.get('runtime_stats')):
            evidence.problem('Quality graph/cache owner survived into timing')
    pmap, wmap = {}, {}
    for phase, values in (('cold_prime', prime), ('warm', warm)):
        for row in values:
            mode, length, doc = row.get('mode'), row.get('length'), row.get('source_index')
            arm = maps.get(str(length), {}).get(mode)
            if arm is None:
                evidence.problem('Unknown timed arm/context')
                continue
            reference = prime_refs.get(str(length)) if phase == 'cold_prime' else reference_map.get((doc, length))
            errors = timing_errors(row, reference, arm, own_modes[str(length)])
            if row.get('arm') != arm or row.get('phase') != phase or any(row.get(k) is not False for k in ('observer_enabled', 'profile_enabled')):
                errors.append('Unfrozen arm or instrumented primary request')
            if any(row.get(k) is not (arm['family'] == 'rescue') for k in ('mapper_executed', 'controller_executed')):
                errors.append('Mapper/controller actual arm-execution receipt differs')
            if phase == 'cold_prime':
                if doc != 271 or (mode, length) in pmap:
                    errors.append('Fresh timing owner cold prime is not uniquely old271')
                pmap[mode, length] = row
                if arm['backend'] == 'cuda_graph':
                    delta, lifetime = row.get('draft_runtime', {}).get('request', {}), row.get('runtime_stats', {})
                    if not positive(delta.get('captures')) or any(not near(delta.get(k), lifetime.get(k)) for k in
                        ('captures', 'replays', 'resident_steps', 'capture_host_seconds')):
                        errors.append('Cold timing owner borrowed quality-stage captures')
            else:
                key = mode, length, doc, row.get('repeat')
                if doc not in documents or row.get('repeat') not in (0, 1) or key in wmap:
                    errors.append('Complete16x2 warm matrix duplicated/unregistered')
                wmap[key] = row
                target = qown.get((doc, length, 'rescue_graph')) if arm['family'] == 'rescue' else qbaseline.get((length, mode)) if documents and doc == documents[0] else None
                if target is not None and (trace_signature(target) != trace_signature(row) or any(target.get(k) != row.get(k) for k in
                    ('accepted', 'proposed', 'blocks', 'draft_forwards', 'selector_refreshes', 'working_gather_layers', 'projected_positions'))):
                    errors.append('Timing logical trajectory differs from its instrumented quality')
            for problem in errors:
                evidence.problem(f'{phase} {mode}/{doc}/{length}: ' + problem)
            if errors and phase == 'warm':
                bad_cells.add((mode, length, doc))
    expected_owner_keys = {(a['mode'], n) for n in LENGTHS for a in arms[str(n)]}
    expected_warm_keys = {(m, n, d, r) for m, n in expected_owner_keys for d in documents for r in (0, 1)}
    if set(pmap) != expected_owner_keys or set(wmap) != expected_warm_keys:
        evidence.problem('Complete10prime/320warm matrix not present')
    owner_plan, warm_plan = registered_schedule(choices, documents)
    if protocol.get('owner_order') != owner_plan or protocol.get('warm_order') != warm_plan:
        evidence.problem('Request plan differs from registered seed/document order')
    if [{k: row.get(k) for k in ('mode', 'length')} for row in prime] != owner_plan or [
        {k: row.get(k) for k in ('mode', 'length', 'source_index', 'repeat')} for row in warm] != warm_plan:
        evidence.problem('Observed owner/warm order differs from frozen plan')
    if valid_indices:
        timing_start = 104 + len(extras)
        for index, row in enumerate(prime):
            expected_index = timing_start + index * 33
            block = [r for r in warm if r.get('owner_id') == row.get('owner_id')]
            if row.get('execution_index') != expected_index or [r.get('execution_index') for r in block] != list(range(expected_index + 1, expected_index + 33)):
                evidence.problem('One-live-owner cold+32warm blocks were not executed in registered order')
    omap = {}
    owner_metrics, diagnostics, selection = {}, {}, {}
    for owner in owners:
        key = owner.get('mode'), owner.get('length')
        rows = sorted([r for r in prime + warm if r.get('owner_id') == owner.get('owner_id')], key=lambda r: r.get('execution_index', -1))
        if key in omap or key not in expected_owner_keys:
            evidence.problem('Unknown/duplicate private timing owner')
        omap[key] = owner
        if owner.get('maximum_live_owners') != 1:
            evidence.problem('Timing owner did not preserve single-live-owner bound')
        for problem in lifecycle_errors(owner, rows):
            evidence.problem('Owner ' + str(key) + ': ' + problem)
        if rows:
            cold_seconds = owner.get('constructor_and_detach_seconds', 0) + rows[0].get('outer_seconds', 0)
            owner_metrics.setdefault(str(key[1]), {})[key[0]] = dict(
                cold_complete_tps=rows[0].get('tokens', 0) / cold_seconds if positive(cold_seconds) else None,
                accounted_service_tps=owner.get('accounted_service_tps'))
    if set(omap) != expected_owner_keys:
        evidence.problem('All ten private owners not closed')
    if set(r.get('owner_id') for r in quality_owners) & set(r.get('owner_id') for r in owners):
        evidence.problem('Quality owner state was reused by a timing owner')
    for length in LENGTHS:
        for arm in arms[str(length)]:
            mode = arm['mode']
            rows = [r for r in warm if r.get('length') == length and r.get('mode') == mode]
            bydoc = {}
            for row in rows:
                bydoc.setdefault(row['source_index'], []).append(row)
            for doc, pair in bydoc.items():
                if len(pair) == 2 and (trace_signature(pair[0]) != trace_signature(pair[1]) or
                    pair[0].get('upstream', {}).get('accept_length_list') != pair[1].get('upstream', {}).get('accept_length_list')):
                    evidence.problem('Identical policy repeat changed its logical trajectory')
                    bad_cells.add((mode, length, doc))
            diag = dict(warm_captures=sum(r.get('draft_runtime', {}).get('request', {}).get('captures', 0) for r in rows),
                warm_capture_seconds=sum(r.get('draft_runtime', {}).get('request', {}).get('capture_host_seconds', 0.) for r in rows)) if rows else {}
            for stage in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds'):
                if rows and all(nonnegative(r.get(stage)) for r in rows):
                    diag[stage] = sum(r[stage] for r in rows) / len(rows)
            diagnostics.setdefault(str(length), {})[mode] = diag
        own_rows = [r for r in warm if r.get('length') == length and r.get('mode') == 'rescue_graph']
        selected_positions = [p / o['prefix'] for r in own_quality if r.get('length') == length and r.get('mode') == 'rescue_graph'
            for o in r.get('observed_rounds', []) if positive(o.get('prefix')) for p in o.get('selected', [])]
        selection[str(length)] = dict(cap_counts={str(cap): sum(t.get('cap') == cap for r in own_rows for t in r.get('trace', []))
            for cap in (1, 2, 3, 4)}, selected_relative_positions=selected_positions)
    tree_raw = evidence.rows('tree_raw.jsonl')
    tree_rows = [r for r in requests if r.get('arm', {}).get('family') == 'full_tree']
    raw_map = {r.get('request_id'): r for r in tree_raw}
    if len(raw_map) != len(tree_rows) or len(raw_map) != len(tree_raw):
        evidence.problem('Raw full-tree output ledger missing/duplicated')
    for row in tree_rows:
        original = raw_map.get(row.get('request_id'), {})
        raw = original.get('result', {})
        if (original.get('execution_index') != row.get('execution_index') or raw.get('generated') != row.get('wrapper_generated') or
            raw.get('tokens') != row.get('wrapper_tokens') or raw.get('upstream') != row.get('upstream')):
            evidence.problem('Normalized full-tree row differs from saved pre-check wrapper result')
    receipt = evidence.json('COMPLETE.json')
    if (receipt.get('passed') is not True or receipt.get('counts') != expected or receipt.get('expected_counts') != expected or
        receipt.get('logical_requests') != total or any(receipt.get(k) is not True for k in
            ('sources_unchanged', 'assets_unchanged', 'quality_gate_passed', 'all_canonical', 'owners_closed', 'analysis_required_locally')) or
        receipt.get('goal_complete') is not False):
        evidence.problem('Terminal complete/immutability/canonical/quality receipt invalid')
    if (Path(directory) / 'FAILURE.json').exists():
        evidence.problem('Failure marker exists; partial root must not be promoted')
    # A partial plot uses only full paired repeat cells common to ALL arms and
    # contexts. It never imputes absent requests or silently changes pairing.
    common = [d for d in documents if all((m, n, d) not in bad_cells and all((m, n, d, r) in wmap for r in (0, 1))
        for m, n in expected_owner_keys)]
    statistics = {}
    if common:
        cells = {str(n): {arm['mode']: [pool_document([wmap[arm['mode'], n, d, r] for r in (0, 1)]) for d in common]
            for arm in arms[str(n)]} for n in LENGTHS}
        statistics = synchronized_statistics(cells, own_modes, challengers)
    status = 'COMPLETE' if not evidence.problems else 'PARTIAL'
    per_context_documents = {str(n): [d for d in documents if all((a['mode'], n, d) not in bad_cells and
        all((a['mode'], n, d, r) in wmap for r in (0, 1)) for a in arms[str(n)])] for n in LENGTHS}
    if status == 'PARTIAL':
        # Do not hide an entirely completed4K subset when8K has not run yet.
        # These context-specific displays NEVER replace the synchronized formal
        # 16-document,8-contrast population or grant simultaneous significance.
        statistics = dict(n=len(common), seed=SEED, draws=DRAWS, contexts={}, contrasts=[], statistical_gate=False,
            document_index_shared_across_all_eight=False, partial_context_diagnostics=True,
            formal_intervals_available=False)
        for length in LENGTHS:
            docs = per_context_documents[str(length)]
            if not docs:
                continue
            cells = {a['mode']: [pool_document([wmap[a['mode'], length, d, r] for r in (0, 1)]) for d in docs]
                for a in arms[str(length)]}
            metrics, contrasts = partial_context_statistics(cells, own_modes[str(length)], challengers[str(length)], length)
            statistics['contexts'][str(length)] = metrics
            statistics['contrasts'].extend(contrasts)
    statistics['per_context_n'] = {n: len(v) for n, v in per_context_documents.items()}
    return dict(schema=1, status=status, verified_success=status == 'COMPLETE' and statistics.get('statistical_gate') is True,
        problems=evidence.problems, observed_counts=observed, expected_counts=expected,
        expected_requests=total, observed_requests=len(requests), documents=documents,
        paired_complete_documents=common, own_modes=own_modes, challengers=challengers, arms=arms,
        paired_complete_documents_by_context=per_context_documents,
        statistics=statistics, owner_metrics=owner_metrics, diagnostics=diagnostics,
        selection_diagnostics=selection, baseline_selection_sha256=proof.get('selection_freeze_sha256'),
        dataset_sha256=dataset['manifest'].get('dataset_sha256'),
        evidence_scope='Saved raw/source/hash/receipt audit; no model/Parquet reread or new GPU execution')


def analyze(directory):
    try:
        return _analyze(directory)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as error:
        return dict(schema=1, status='PARTIAL', verified_success=False,
            problems=['Malformed saved evidence; no completion or performance inference: ' + type(error).__name__ + ': ' + str(error)],
            statistics={}, own_modes={}, challengers={}, observed_counts={})


def render(summary, output):
    """Separate quality, performance and measured mechanisms; no invented data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stats = summary.get('statistics', {})
    contexts = stats.get('contexts', {})
    status = summary.get('status', 'PARTIAL')
    counts = stats.get('per_context_n', {str(n): stats.get('n', 0) for n in LENGTHS})
    caption = (f'{status} | H10080GB, Qwen2 FP32 Target / BF16 Draft+prefix; exact4K/8K, '
        f'max128 output,2 repeats/document; n4K={counts.get("4096",0)}/16,n8K={counts.get("8192",0)}/16 unused validation documents. '
        'Same-owner finite stream; all lazy captures charged; not production/Vicuna.')
    names = ('RESCUE', 'Retrieval eager', 'Retrieval resident', 'Retrieval Graph', 'Full tree')
    colors = ('#2563eb', '#9ca3af', '#6b7280', '#f59e0b', '#10b981')

    def modes(length):
        own = summary.get('own_modes', {}).get(str(length))
        challengers = summary.get('challengers', {}).get(str(length), {})
        return [own] + [challengers.get(group) for group in GROUPS]

    def finish(fig, name):
        fig.text(.01, .015, caption, fontsize=8, va='bottom', wrap=True)
        fig.tight_layout(rect=(0, .10, 1, .96))
        fig.savefig(output / (name + '.png'), dpi=160)
        fig.savefig(output / (name + '.svg'))
        plt.close(fig)

    def interval(ax, x, point, limits, color):
        if point is None:
            return
        ax.scatter([x], [point], color=color, zorder=4)
        if limits is not None:
            ax.vlines(x, limits[0], limits[1], color=color, linewidth=1.7)
            ax.hlines(limits, x - .055, x + .055, color=color, linewidth=1.2)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for col, length in enumerate(LENGTHS):
        values = contexts.get(str(length), {})
        for row, metric in enumerate(('acceptance_length', 'acceptance_rate')):
            ax = axes[row, col]
            for index, mode in enumerate(modes(length)[:4]):
                value = values.get(mode, {})
                interval(ax, index, value.get(metric), value.get(metric + '_ci95'), colors[index])
            ax.set_xticks(range(4), names[:4], rotation=15, ha='right')
            ax.set_title(f'{length} input tokens: linear quality')
            ax.set_ylabel('Accepted Draft tokens / round' if row == 0 else 'Accepted / proposed Draft tokens')
            ax.grid(axis='y', alpha=.2)
        ax = axes[2, col]
        mode = modes(length)[4]
        value = values.get(mode, {})
        interval(ax, 0, value.get('acceptance_length'), value.get('acceptance_length_ci95'), colors[4])
        ax.set_xticks([0], ['Full tree'])
        ax.set_title(f'{length} input tokens: tree raw unit (includes overshoot work)')
        ax.set_ylabel('Raw accepted tokens / raw round')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Quality only — document-ratio pointwise95% intervals; no invented tree acceptance rate')
    finish(fig, 'acceptance')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, length in enumerate(LENGTHS):
        ax = axes[0, col]
        values = contexts.get(str(length), {})
        for index, mode in enumerate(modes(length)):
            value = values.get(mode, {})
            interval(ax, index, value.get('pooled_tps'), value.get('tps_ci95'), colors[index])
        ax.set_xticks(range(5), names, rotation=18, ha='right')
        ax.set_title(f'{length} input tokens: complete generate outer')
        ax.set_ylabel('Client tokens / second')
        ax.grid(axis='y', alpha=.2)
        ax = axes[1, col]
        contrasts = [c for c in stats.get('contrasts', []) if c['length'] == length]
        for index, contrast in enumerate(contrasts):
            interval(ax, index, contrast['point_percent'], contrast['pointwise_ci95'], colors[index + 1])
            lower = contrast['simultaneous_one_sided_lower95']
            if lower is not None:
                ax.scatter([index], [lower], marker='D', facecolors='none', edgecolors='#dc2626', s=35,
                    label='Simultaneous one-sided95% lower' if index == 0 else None)
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_xticks(range(4), names[1:], rotation=18, ha='right')
        ax.set_ylabel('RESCUE pooled TPS improvement (%)')
        ax.set_title('Shared16-document resample;8 comparisons, Bonferroni lower0.625th percentile' if status == 'COMPLETE' else 'PARTIAL context-specific pointwise intervals; no formal simultaneous gate')
        if contrasts and status == 'COMPLETE':
            ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Performance only — frozen challengers;100000 document-paired bootstrap draws')
    finish(fig, 'tps')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, length in enumerate(LENGTHS):
        ledger = summary.get('owner_metrics', {}).get(str(length), {})
        for row, key in enumerate(('cold_complete_tps', 'accounted_service_tps')):
            ax = axes[row, col]
            for index, mode in enumerate(modes(length)):
                value = ledger.get(mode, {}).get(key)
                if positive(value):
                    ax.bar(index, value, color=colors[index])
            ax.set_xticks(range(5), names, rotation=18, ha='right')
            ax.set_ylabel('Client tokens / charged second')
            ax.set_title(f'{length}: ' + ('old doc271 cold request + constructor' if row == 0 else 'constructor + cold +32warm + close'))
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('Cold and accounted-service diagnostics — one owner/arm/context, descriptive; no document CI')
    finish(fig, 'cold_service')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    diagnostics = summary.get('diagnostics', {})
    for col, length in enumerate(LENGTHS):
        values = diagnostics.get(str(length), {})
        ax = axes[0, col]
        for index, mode in enumerate(modes(length)):
            row = values.get(mode, {})
            if 'warm_captures' in row:
                ax.bar(index, row['warm_captures'], color=colors[index])
                ax.text(index, row['warm_captures'], f"{row.get('warm_capture_seconds', 0):.3f}s", ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(5), names, rotation=18, ha='right')
        ax.set_ylabel('New captures across32warm requests')
        ax.set_title(f'{length}: all legal late capture costs retained')
        ax = axes[1, col]
        stage_colors = ('#60a5fa', '#fbbf24', '#c4b5fd')
        for offset, (stage, color) in enumerate(zip(('prefill_seconds', 'decode_seconds', 'cleanup_seconds'), stage_colors)):
            for index, mode in enumerate(modes(length)):
                value = values.get(mode, {}).get(stage)
                if nonnegative(value):
                    ax.bar(index + (offset - 1) * .22, value, width=.22, color=color,
                        label=stage.replace('_seconds', '') if index == 0 else None)
        ax.set_xticks(range(5), names, rotation=18, ha='right')
        ax.set_ylabel('Saved stage seconds / warm request')
        ax.set_title(f'{length}: available inner-stage diagnostics, not additive outer attribution')
        if ax.patches:
            ax.legend(fontsize=8)
    fig.suptitle('Mechanisms — timing is unprofiled; unavailable stages are omitted, not zero')
    finish(fig, 'latency_captures')

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, length in enumerate(LENGTHS):
        row = summary.get('selection_diagnostics', {}).get(str(length), {})
        caps = row.get('cap_counts', {})
        if any(caps.values()):
            axes[0, col].bar([str(i) for i in (1, 2, 3, 4)], [caps.get(str(i), 0) for i in (1, 2, 3, 4)], color=colors[0])
        else:
            axes[0, col].text(.5, .5, 'No completed warm rounds', transform=axes[0, col].transAxes, ha='center')
        axes[0, col].set_title(f'{length}: own-method actual cap decisions (all warm rounds)')
        axes[0, col].set_xlabel('Draft cap;1/3 may be remaining-budget boundaries')
        axes[0, col].set_ylabel('Observed round count')
        positions = row.get('selected_relative_positions', [])
        if positions:
            axes[1, col].hist(positions, bins=np.linspace(0, 1, 21), color=colors[0])
        axes[1, col].set_xlim(0, 1)
        axes[1, col].set_xlabel('Absolute selected position / current committed-prefix length')
        axes[1, col].set_ylabel('Quality-observed selected-position count')
        axes[1, col].set_title(f'{length}: independent quality requests, not timed observer work')
    fig.suptitle('Collected selection/action diagnostics — no causal feature-attribution claim')
    finish(fig, 'actions_positions')
    return ['acceptance', 'tps', 'cold_service', 'latency_captures', 'actions_positions']


def write_report(summary, directory, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    figure_names = render(summary, output)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    shutil.copyfile(Path(__file__), output / 'analysis_source.py')
    lines = [f"# {summary['status']}: same-Qwen fair confirmation", '',
        f"Audit completion: **{summary['status']}**. Formal speed/quality gate: **{summary.get('verified_success', False)}**.", '',
        'The primary endpoint pools both timing repeats within each document, then pools client tokens/time across16 documents. '
        'All8 contrasts share100000 bootstrap document-index draws (seed20261024). Pointwise two-sided95% intervals are descriptive; '
        'the formal gate uses8 Bonferroni one-sided95% lower bounds (0.625th bootstrap percentile), each strictly positive, '
        'and every point estimate must also be strictly positive. Completion is not the same as acceleration success.', '',
        'Scope: one H10080GB; same Qwen2 7B/1.5B weights; FP32 Target decode, BF16 Draft/prefix, TF32 off; '
        'exact4096/8192 inputs, max128 client outputs,2 repeats. Sixteen previously unused validation documents are a deterministic '
        'confirmation cohort, not a random official test. The original paused71/320 test is not resumed. '
        'Mapper and old-cost causal2/4 policy remain frozen and genuinely nonzero. '
        'All legal lazy captures and full-tree overshoot work are charged. No steady-state, production, original Vicuna or universal SOTA claim.', '',
        '| Input | Frozen challenger | RESCUE TPS change | Pointwise95% CI | Simultaneous one-sided95% lower |',
        '|---|---|---:|---|---:|']
    for c in summary.get('statistics', {}).get('contrasts', []):
        ci = c['pointwise_ci95']
        lines.append(f"| {c['length']} | {c['baseline_mode']} | {c['point_percent']:+.5f}% | " +
            (f"[{ci[0]:+.5f}%, {ci[1]:+.5f}%]" if ci is not None else 'unavailable') + ' | ' +
            (f"{c['simultaneous_one_sided_lower95']:+.5f}%" if c['simultaneous_one_sided_lower95'] is not None else 'unavailable') + ' |')
    lines += ['', 'The simultaneous bounds use a conservative Bonferroni percentile-bootstrap construction; '
        'this is the frozen empirical decision rule, not an exact finite-sample coverage theorem. '
        'Repeats, rounds, contexts and candidates are not independent documents.', '',
        'Owners execute seeded blocks rather than interleaving arms; a whole32-request warm block precedes close. '
        'Block clock drift remains a limitation. Cold prime uses old document271, one request per arm/context. Accounted-service throughput charges constructor/detach, '
        'cold prime, all32 warm requests and close; it is a non-contiguous measured service sum, not production session wall. '
        'These are secondary diagnostics, never substitutes for the primary endpoint. '
        'Full-tree quality is raw accepted tokens/raw verification round, including overshoot; no proposal acceptance rate or hidden tail token IDs are invented.', '',
        'The local audit rehashes saved artifacts, token arrays, scan/exclusion receipts and the frozen source/tokenizer/template chain. '
        'Original candidate Parquet and full report text are not copied or re-tokenized here; document full-text SHA and first-eligible '
        'selection are checked against the preserved pre-execution scan/source receipts, not claimed as a fresh local raw-corpus rerun.', '',
        '## Audit problems', '']
    lines += ['- ' + p for p in summary.get('problems', [])] or ['None.']
    lines += ['', '## Saved figures and raw evidence', '']
    for name in figure_names:
        lines += [f'[{name} PNG]({name}.png) · [SVG]({name}.svg)', '', f'![{name}]({name}.png)', '']
    lines += ['[Analysis source](analysis_source.py) · [Summary](summary.json)', '']
    for name in ('protocol.json', 'DATASET_MANIFEST.json', 'FINAL_FREEZE.json', 'GATE_PASSED.json', 'reference.jsonl',
        'own_quality.jsonl', 'baseline_quality.jsonl', 'matching_quality.jsonl', 'prime.jsonl', 'runtime.jsonl', 'owners.jsonl', 'tree_raw.jsonl', 'COMPLETE.json'):
        if (Path(directory) / name).is_file():
            lines.append(f'[{name}]({Path("..").as_posix()}/{name})')
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    receipt = dict(schema=1, status=summary['status'], verified_success=summary.get('verified_success', False),
        render_required=True, rendered_figures=figure_names, file_sha256=artifacts,
        note='Rendered output still requires human visual inspection and independent raw auditing before final handoff.')
    (output / 'ANALYSIS_COMPLETE.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    if summary.get('verified_success'):
        (output / 'VERIFIED_SUCCESS.json').write_text(json.dumps(dict(schema=1, status='VERIFIED_SUCCESS',
            evidence_audit_passed=True, statistical_gate_passed=True, summary_sha256=sha(output / 'summary.json'),
            analysis_receipt_sha256=sha(output / 'ANALYSIS_COMPLETE.json'),
            scope='Frozen same-Qwen,H100,4K/8K,max128,16-document confirmation only; final human artifact review required'), indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Saved confirmation results, never models or raw Parquet')
    parser.add_argument('--output', type=Path, required=True, help='Fresh analysis directory; never overwrite old figures')
    args = parser.parse_args()
    summary = analyze(args.directory)
    write_report(summary, args.directory, args.output)
    print(json.dumps(dict(status=summary['status'], verified_success=summary.get('verified_success', False),
        observed_counts=summary.get('observed_counts'), problems=summary.get('problems')), indent=2))


if __name__ == '__main__':
    main()
