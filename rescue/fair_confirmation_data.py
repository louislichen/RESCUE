"""Offline gates and fresh GovReport confirmation preparation.

Importing this module is stdlib-only.  Neither importing it nor freezing the
gate opens the candidate Parquet.  Candidate content is first accessed by
``prepare_confirmation`` after the complete baseline/figure/source chain has
been checked.  No runtime, outcome, password or remote-access API is used.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


SEED = 20261024
LENGTHS = (4096, 8192)
DOCUMENTS = 16
MIN_INDEX = 900
MIN_NATURAL_TOKENS = 16384
GROUPS = ('retrieval/eager', 'retrieval/resident', 'retrieval/graph', 'full_tree')
BASELINE_COUNTS = dict(reference_requests=6, prime_requests=76,
                       timing_requests=456, owners=76)
RULE = dict(original_split='validation', minimum_source_index=MIN_INDEX,
            minimum_natural_tokens=MIN_NATURAL_TOKENS, documents=DOCUMENTS,
            order='original source_index ascending; first eligible sixteen',
            context_lengths=list(LENGTHS), duplicate_rule='exclude any document SHA or exact 4K/8K prompt-prefix SHA, including within selected cohort',
            replacement='never use generation, EOS, speed or nonzero-injection outcomes to select or replace')
OLD_COHORT_FILES = (
    'exp/long_system/data/train.jsonl',
    'exp/long_system/data/development.jsonl',
    'exp/long_system/data/test.jsonl',
    'exp/module_opt/data_v1_20261018/fit.jsonl',
    'exp/module_opt/data_v1_20261018/tune.jsonl',
    'exp/module_opt/data_v1_20261018/warmup.jsonl',
    'exp/module_opt/data_v1_20261018/confirmation.jsonl',
    'exp/module_arch/data_v1_20261018/fit.jsonl',
    'exp/module_arch/data_v1_20261018/tune.jsonl',
    'exp/module_arch/data_v1_20261018/warmup.jsonl',
    'exp/module_arch/data_v1_20261018/confirmation.jsonl',
)
REQUIRED_NEW_SOURCE = (
    'rescue/fair_confirmation_data.py',
    'rescue/fair_confirmation.py',
    'rescue/fair_confirmation_report.py',
    'tests/test_rescue_fair_confirmation_data.py',
    'tests/test_rescue_fair_confirmation.py',
    'tests/test_rescue_fair_confirmation_report.py',
    'exp/draft_graph/fair_confirmation_20261020_preregistration.json',
    'exp/draft_graph/run_fair_confirmation_20261020_v1.sh',
)
TREE_PREFIX = 'exp/specextend_bridgekv/specextend/'
TREE_REQUIRED = ('classic/model_classic.py', 'classic/utils_classic.py',
                 'classic/modeling_llama_kv_draft.py', 'shared/opt_tree.py',
                 'shared/modeling_llama_kv_target.py')
FIXED = dict(teacher='Qwen2-7B-Instruct', student='Qwen2-1.5B-Instruct',
    target_dtype='float32', draft_dtype='bfloat16', prefill_precision='bfloat16',
    tf32=False, context_lengths=list(LENGTHS), max_new_tokens=128, repeats=2,
    schedule='serial', graph_max_entries=128)
OWN_METHOD = dict(mapper_sha256='d08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
    policy_sha256='3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
    cost_sha256='d92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1',
    selector='hierarchical', working_budget=512, recent=64, coverage_rows=64,
    coverage_candidates=1024, coverage_batch=16, coarse_chunk_size=32,
    refresh_every=4, refresh_policy='periodic', injection_layers=[27],
    inject_tokens=64, injection_strength=.5, draft_execution='cuda_graph',
    coverage_execution='eager', length_policy='frozen old-cost causal2/4',
    mapper_required_nonzero=True, no_training=True, no_cost_promotion=True)
STATISTICS = dict(bootstrap_seed=SEED, bootstrap_draws=100000,
    independent_unit='whole document', primary='pooled token/time throughput ratio',
    paired_indices='same document resample for all arms and both contexts',
    plotted_interval='pointwise two-sided95 percentile [0.025,0.975]',
    formal_gate='ALL8 effects have positive point estimate AND positive simultaneous one-sided95 lower bound',
    simultaneous_lower_quantile=.00625, comparisons=8,
    filtering='no capture-free filtering, outcome selection, imputation or optional stopping')
EXECUTION = dict(
    primary='Whole-generate outer includes handoff,input construction,prefill,reset,all legal lazy captures,decode,cleanup; excludes common model loading identically',
    owner_trace='At most one live private serving owner. Each context has five separately constructed and closed owners, each with one cold prime followed by32 warm requests; disclose finite trace, not fully prewarmed production speed',
    first_prime='Fixed old train document271, separately billed per context/arm, using the exact verified baseline input and canonical128 output; no extra AR, free warmup or quality Graph borrowing',
    order='Freeze seeded owner-block order and within-block document order. One owner executes all32 warm requests before close; this is not interleaving. Disclose block clock-drift limitation; reset per-request policy RNG',
    secondaries='Report cold constructor+prime and active-service constructor+prime+32warm+close separately; do not substitute them after observing primary',
    captures='Keep every legal capture in timing; no pre-capture union, test-aware warmup or capture-free filtering')
TOKENIZER_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                   'special_tokens_map.json', 'added_tokens.json', 'vocab.json',
                   'merges.txt', 'tokenizer.model', 'sentencepiece.bpe.model',
                   'chat_template.jinja')
OUTCOME_FIELDS = frozenset(('generated', 'output_token_ids', 'accept_length_list',
    'trace', 'accepted', 'proposed', 'outer_seconds', 'runtime_stats',
    'observed_rounds', 'cost_samples', 'performance_eligible'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def input_sha(ids):
    # Match the previously frozen experiment input_sha256 serialization.
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), 'Expected JSON object: ' + str(path))
    return value


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def time_value(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, 'A timezone-aware UTC receipt is required')
    return result.astimezone(timezone.utc)


def _relative(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    require(path.is_relative_to(root), 'Artifact escapes its frozen root: ' + str(path))
    return str(path.relative_to(root))


def extra_eager_geometries(choices):
    """Same-geometry eager references, deduplicated without new selection."""
    result = []
    for length in LENGTHS:
        values = choices[str(length)]
        eager = values['retrieval/eager']
        existing = {(eager['top_chunks'], eager['fixed_k'])}
        for group in ('retrieval/resident', 'retrieval/graph'):
            arm = values[group]
            geometry = (arm['top_chunks'], arm['fixed_k'])
            if geometry not in existing:
                result.append(dict(length=length, top_chunks=geometry[0], fixed_k=geometry[1]))
                existing.add(geometry)
    return result


def validate_baseline_freeze(baseline_results, analysis_dir, verified_complete, final_manifest):
    """Pure, offline baseline gate.  Booleans alone never establish completion.

    Manifest paths are relative to ``final_manifest.parent``.  Every required
    raw/source/analysis file must be covered, and all manifest entries are
    rehashed.  The independently authored VERIFIED_COMPLETE binds this
    manifest without a circular self-hash.
    """
    results, analysis = Path(baseline_results).resolve(), Path(analysis_dir).resolve()
    verified_path, manifest_path = Path(verified_complete).resolve(), Path(final_manifest).resolve()
    root = manifest_path.parent
    require(results.is_dir() and analysis.is_dir(), 'Complete baseline results and strict local analysis required')
    _relative(root, results); _relative(root, analysis); _relative(root, verified_path)
    exit_path = root / 'PIPELINE_EXIT.json'
    exit_receipt = read_json(exit_path) if exit_path.is_file() else {}
    require(type(exit_receipt.get('exit_code')) is int and exit_receipt['exit_code'] == 0,
            'Baseline pipeline must have exited successfully before confirmation freeze')
    require(not (results / 'FAILURE.json').exists(), 'Baseline failure marker exists')
    names = dict(baseline_complete=results / 'COMPLETE.json',
                 frozen_selection=results / 'FROZEN_SELECTION.json',
                 baseline_protocol=results / 'protocol.json',
                 analysis_summary=analysis / 'summary.json',
                 verified_winners=analysis / 'VERIFIED_WINNERS.json',
                 analysis_complete=analysis / 'ANALYSIS_COMPLETE.json')
    require(all(p.is_file() for p in names.values()) and verified_path.is_file() and manifest_path.is_file(),
            'Baseline has no complete, independently verified freeze chain')
    complete, frozen, protocol, summary, winners, analyzed = (read_json(names[k]) for k in names)
    require(complete.get('passed') is True and complete.get('counts') == BASELINE_COUNTS and
            complete.get('logical_requests') == 538 and complete.get('owner_closed') == 76 and
            complete.get('goal_complete') is False, 'Baseline538/76 completion receipt invalid')
    require(all(complete.get(k) is True for k in ('sources_unchanged', 'assets_unchanged',
            'dependencies_unchanged', 'all_canonical', 'backend_trajectories_equal')),
            'Baseline immutability/exactness receipt invalid')
    require(summary.get('status') == 'COMPLETE' and summary.get('problems') == [] and
            summary.get('winner_status') == 'VERIFIED_FROZEN' and summary.get('observed_counts') == BASELINE_COUNTS,
            'Strict baseline analysis is not complete')
    require(winners.get('status') == 'VERIFIED_FROZEN' and winners.get('authoritative') is True and
            analyzed.get('status') == 'COMPLETE', 'Only verified, authoritative winners may open confirmation')
    require(frozen.get('complete_request_count') == 538 and frozen.get('group_candidates') == 8 and
            frozen.get('no_new_confirmation_opened') is True and
            frozen.get('protocol_sha256') == sha256(names['baseline_protocol']) and
            complete.get('frozen_selection_sha256') == sha256(names['frozen_selection']),
            'Frozen selection is not bound to the complete baseline protocol')
    verified = read_json(verified_path)
    flags = ('passed', 'raw_independent_audit_passed', 'rendered_figures_inspected',
             'strict_analysis_complete', 'all538_requests_verified', 'all76_owners_verified',
             'eight_frozen_choices_verified')
    require(verified.get('status') == 'VERIFIED_COMPLETE' and all(verified.get(k) is True for k in flags),
            'Independent raw audit and actual seven-figure inspection are required')
    digests = {k: sha256(p) for k, p in names.items()}
    links = dict(baseline_complete_sha256=digests['baseline_complete'],
                 frozen_selection_sha256=digests['frozen_selection'],
                 analysis_summary_sha256=digests['analysis_summary'],
                 verified_winners_sha256=digests['verified_winners'],
                 final_artifacts_manifest_sha256=sha256(manifest_path))
    require(all(verified.get(k) == v for k, v in links.items()), 'Independent audit SHA links differ')
    manifest = read_json(manifest_path); inventory = manifest.get('file_sha256', {})
    require(isinstance(inventory, dict) and inventory, 'Final artifact manifest is empty')
    for name, digest in inventory.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and valid_sha(digest),
                'Unsafe/malformed final artifact manifest entry')
        path = (root / relative).resolve(); _relative(root, path)
        require(path.is_file() and sha256(path) == digest, 'Final baseline artifact changed: ' + name)
    required = set()
    for directory in (results, analysis):
        for path in directory.rglob('*'):
            if path.is_file() and not any(part.startswith('.') or part == '__pycache__' for part in path.relative_to(directory).parts):
                if path.resolve() not in (verified_path, manifest_path):
                    required.add(_relative(root, path))
    required.add(_relative(root, exit_path))
    require(required <= set(inventory), 'Final manifest omits raw/source/analysis artifacts')
    require(any('/source/' in '/' + name for name in required), 'Frozen executed source is missing')
    figures = ('quality', 'tps', 'captures', 'capture_time', 'cold', 'service', 'selected_repeat1')
    require(all((analysis / (name + suffix)).is_file() for name in figures for suffix in ('.png', '.svg')),
            'All seven rendered figure pairs are required')
    artifacts = analyzed.get('artifacts_sha256', {})
    require(isinstance(artifacts, dict) and artifacts, 'Strict analysis has no artifact hashes')
    for name, digest in artifacts.items():
        path = analysis / name; _relative(analysis, path)
        require(path.is_file() and sha256(path) == digest, 'Strict analysis artifact changed: ' + name)
    choices, modes = {}, {}
    for length in LENGTHS:
        key = str(length); chosen = winners.get('winners', {}).get('selected', {}).get(key, {})
        require(set(chosen) == set(GROUPS), 'Four challengers must be frozen at each context')
        choices[key], modes[key] = {}, {}
        for group in GROUPS:
            row = frozen.get('per_context', {}).get(key, {}).get('groups', {}).get(group, {})
            arm = row.get('arm', {}); mode = row.get('mode')
            require(mode == chosen[group] and isinstance(arm, dict) and arm.get('mode') == mode,
                    'Authoritative winner differs from frozen arm')
            require(arm.get('top_chunks') in (16, 32), 'Unknown frozen retrieval width')
            if group == 'full_tree':
                require(arm.get('family') == 'full_tree' and arm.get('nodes') in (8, 32, 50, 64) and
                        arm.get('max_depth') == 10 and arm.get('threshold') == .7 and
                        arm.get('backend') == 'upstream_flash_hybrid', 'Unknown frozen tree geometry')
            else:
                backend = 'cuda_graph' if group.endswith('/graph') else group.split('/')[1]
                require(arm.get('family') == 'retrieval' and arm.get('fixed_k') in (1, 2, 4, 8, 12) and
                        arm.get('backend') == backend, 'Unknown frozen linear geometry/backend')
            choices[key][group], modes[key][group] = copy.deepcopy(arm), mode
    extra = extra_eager_geometries(choices)
    return dict(schema=1, status='BASELINE_VERIFIED', choices=choices, selected_modes=modes,
        extra_eager_reference_geometries=extra, extra_eager_reference_count=len(extra),
        scientific_requests=434 + len(extra), selection_freeze_sha256=digests['frozen_selection'],
        verified_winners_sha256=digests['verified_winners'], baseline_complete_sha256=digests['baseline_complete'],
        analysis_summary_sha256=digests['analysis_summary'], verified_complete_sha256=sha256(verified_path),
        final_manifest_sha256=sha256(manifest_path), baseline_protocol_sha256=digests['baseline_protocol'],
        pipeline_exit_sha256=sha256(exit_path))


def prompt_ids(row, length):
    left, right, document = row['prefix_ids'], row['suffix_ids'], row['document_ids']
    require(all(isinstance(v, list) and all(type(t) is int and t >= 0 for t in v) for v in (left, right, document)),
            'Token arrays must contain nonnegative integer IDs')
    take = length - len(left) - len(right)
    require(0 < take <= len(document), 'Natural document cannot supply exact context')
    return left + document[:take] + right


def _reject_outcome_fields(value):
    if isinstance(value, dict):
        require(not OUTCOME_FIELDS.intersection(value), 'Runtime/outcome records are forbidden as exclusion metadata')
        for child in value.values():
            _reject_outcome_fields(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                _reject_outcome_fields(child)


def exclusion_receipt(metadata_paths):
    """Read only explicit old tokenized cohort data, never inference outputs."""
    require(len(metadata_paths) == len(OLD_COHORT_FILES), 'All eleven known used/reserved GovReport cohorts are required')
    sources, documents = [], set()
    prefixes = {str(n): set() for n in LENGTHS}
    found = set()
    for value in metadata_paths:
        path = Path(value).resolve()
        matches = [name for name in OLD_COHORT_FILES if path.as_posix().endswith('/' + name)]
        require(len(matches) == 1 and matches[0] not in found and path.is_file(),
                'Only the explicit eleven old data JSONL paths are permitted')
        found.add(matches[0]); before = sha256(path)
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line); require(isinstance(row, dict), 'Old data row must be an object')
            _reject_outcome_fields(row)
            digest = row.get('document_sha256')
            require(valid_sha(digest) and type(row.get('source_index')) is int and row['source_index'] >= 0 and
                    isinstance(row.get('split'), str), 'Old cohort document metadata invalid')
            record = dict(source_index=row['source_index'], split=row['split'], document_sha256=digest,
                          input_sha256={})
            for n in LENGTHS:
                take = n - len(row.get('prefix_ids', [])) - len(row.get('suffix_ids', []))
                if 0 < take <= len(row.get('document_ids', [])):
                    record['input_sha256'][str(n)] = input_sha(prompt_ids(row, n))
                    prefixes[str(n)].add(record['input_sha256'][str(n)])
            documents.add(digest); rows.append(record)
        require(rows and sha256(path) == before, 'Empty/changed old cohort metadata')
        sources.append(dict(path=str(path), relative_name=matches[0], sha256=before, rows=rows))
    return dict(schema=1, status='OLD_DATA_METADATA_ONLY', metadata_sources=sources,
        document_sha256=sorted(documents), input_sha256={n: sorted(v) for n, v in prefixes.items()},
        reserved_test_scope='Whole old long_system/data/test.jsonl is excluded; no paused outcomes were accessed',
        scope='All eleven known used/reserved GovReport cohorts. Earlier HotpotQA experiments are a different corpus; this is not a claim about foundation-model pretraining contamination')


def _validate_protocol_contract(protocol, baseline, *, template=False):
    require(protocol.get('seed') == SEED and protocol.get('selection_rule') == RULE and
            protocol.get('fixed') == FIXED and protocol.get('own_method') == OWN_METHOD and
            protocol.get('statistics') == STATISTICS and
            protocol.get('execution') == EXECUTION and
            protocol.get('required_source_files') == list(REQUIRED_NEW_SOURCE),
            'Frozen selection/method/precision/statistics/source contract changed')
    require(all(protocol.get('counts', {}).get(k) == v for k, v in dict(
        reference=32, own_paired_quality=64, baseline_sentinel_quality=8,
        cold_prime=10, warm=320, base_scientific_requests=434).items()),
        'Frozen scientific request categories changed')
    if not template:
        require(protocol.get('status') == 'FROZEN_BEFORE_DATA_ACCESS' and protocol.get('is_template') is False and
                protocol.get('baseline_freeze') == baseline and
                protocol.get('expected_scientific_requests') == baseline['scientific_requests'] and
                protocol.get('expected_extra_eager_references') == baseline['extra_eager_reference_count'],
                'Final protocol must bind the verified choices/counts before candidate access')


def make_final_protocol(template_path, baseline):
    template = read_json(template_path)
    require(template.get('status') == 'TEMPLATE_NOT_EXECUTED' and template.get('is_template') is True,
            'Only the unexecuted confirmation template may be finalized')
    require(baseline.get('status') == 'BASELINE_VERIFIED', 'Complete independently verified baseline required')
    _validate_protocol_contract(template, baseline, template=True)
    result = copy.deepcopy(template)
    result.update(status='FROZEN_BEFORE_DATA_ACCESS', is_template=False, seed=SEED,
        finalized_at_utc=utc_now(), baseline_freeze=copy.deepcopy(baseline),
        selection_rule=copy.deepcopy(RULE), expected_scientific_requests=baseline['scientific_requests'],
        expected_extra_eager_references=baseline['extra_eager_reference_count'])
    return result


def _source_inventory(root, expected_tree):
    root = Path(root).resolve()
    require(all((root / name).is_file() for name in REQUIRED_NEW_SOURCE), 'All final runner/report/tests/preregistration sources must exist before data access')
    require(isinstance(expected_tree, dict) and len(expected_tree) >= 27 and
            all(TREE_PREFIX + name in expected_tree for name in TREE_REQUIRED),
            'The complete gated archived tree source is required')
    actual_tree = {_relative(root, p): sha256(p) for p in (root / TREE_PREFIX).rglob('*.py')}
    require(actual_tree == expected_tree, 'Full-tree source paths/SHA must exactly match the gated baseline')
    paths = list((root / 'rescue').glob('*.py')) + list((root / 'tests').glob('test_rescue*.py'))
    paths += list((root / 'exp/specextend_bridgekv/specextend').rglob('*.py'))
    paths += [root / name for name in REQUIRED_NEW_SOURCE]
    paths += [root / 'exp/attention_horizon/selection.py', root / 'exp/selective_kv/bridge.py']
    require(all(path.is_file() for path in paths), 'Incomplete frozen code dependency inventory')
    return {_relative(root, path): sha256(path) for path in sorted(set(paths))}


def _tokenizer_inventory(root):
    root = Path(root).resolve()
    require((root / 'tokenizer_config.json').is_file() and any((root / n).is_file() for n in ('tokenizer.json', 'tokenizer.model', 'sentencepiece.bpe.model')),
            'An explicit complete local tokenizer is required')
    return {name: sha256(root / name) for name in TOKENIZER_FILES if (root / name).is_file()}


def freeze_gate(output, *, baseline_results, analysis_dir, verified_complete, final_manifest,
                metadata_paths, raw_source_manifest, raw_source, tokenizer_root,
                source_root, final_protocol):
    """Create a fresh source/exclusion gate; this NEVER opens ``raw_source``."""
    output = Path(output).resolve(); require(not output.exists(), 'Fresh gate root required; never overwrite/resume')
    output.mkdir(parents=True)
    try:
        arguments = dict(baseline_results=str(Path(baseline_results).resolve()), analysis_dir=str(Path(analysis_dir).resolve()),
            verified_complete=str(Path(verified_complete).resolve()), final_manifest=str(Path(final_manifest).resolve()))
        baseline = validate_baseline_freeze(**arguments)
        final_protocol = Path(final_protocol).resolve(); protocol = read_json(final_protocol)
        _validate_protocol_contract(protocol, baseline)
        exclusions = exclusion_receipt(metadata_paths)
        raw_manifest = Path(raw_source_manifest).resolve(); require(raw_manifest.name == 'manifest.json' and
            raw_manifest.as_posix().endswith('/exp/module_arch/data_v1_20261018/manifest.json'),
            'Expected raw SHA must come from the old module_arch data manifest')
        old_source = read_json(raw_manifest); expected = old_source.get('source_sha256')
        require(valid_sha(expected), 'Old manifest has no candidate source SHA')
        raw_source = Path(raw_source)
        require(raw_source.is_absolute() and raw_source.name == 'validation-00000-of-00001.parquet',
                'Explicit original validation Parquet required')
        baseline_protocol = read_json(Path(baseline_results) / 'protocol.json')
        templates = [s for s in exclusions['metadata_sources'] if s['sha256'] == baseline_protocol.get('data_file_sha256')]
        require(templates, 'No old metadata file matches the baseline prompt-template source')
        first = next(json.loads(line) for line in Path(templates[0]['path']).read_text().splitlines() if line.strip())
        template = dict(prefix_ids=first['prefix_ids'], suffix_ids=first['suffix_ids'])
        require(all(isinstance(v, list) and v and all(type(t) is int and t >= 0 for t in v) for v in template.values()),
                'Prompt template tokens invalid')
        source_root, tokenizer_root = Path(source_root).resolve(), Path(tokenizer_root).resolve()
        expected_tree = {name: digest for name, digest in baseline_protocol.get('source_sha256', {}).items()
                         if name.startswith(TREE_PREFIX)}
        sources = _source_inventory(source_root, expected_tree); tokenizer = _tokenizer_inventory(tokenizer_root)
        write_json(output / 'EXCLUSIONS.json', exclusions)
        gate = dict(schema=1, status='FROZEN_BEFORE_DATA_ACCESS', frozen_at_utc=utc_now(), seed=SEED,
            candidate_content_accessed=False, selection_rule=copy.deepcopy(RULE), baseline_arguments=arguments,
            baseline_freeze=baseline, final_protocol=dict(path=str(final_protocol), sha256=sha256(final_protocol)),
            source_root=str(source_root), source_sha256=sources, unchanged_tree_sha256=expected_tree,
            tokenizer_root=str(tokenizer_root), tokenizer_sha256=tokenizer,
            template=template, template_sha256=json_sha(template), template_source_sha256=templates[0]['sha256'],
            exclusion_receipt=dict(path=str(output / 'EXCLUSIONS.json'), sha256=sha256(output / 'EXCLUSIONS.json')),
            raw_source=dict(path=str(raw_source), split='validation', format='parquet', expected_sha256=expected,
                old_manifest_path=str(raw_manifest), old_manifest_sha256=sha256(raw_manifest)),
            no_training=True, scope='Fresh project-confirmation validation subset; not a random official test sample')
        write_json(output / 'GATE_SOURCE_FREEZE.json', gate)
        return gate
    except BaseException as error:
        write_json(output / 'GATE_FAILURE.json', dict(status='FAILED_BEFORE_DATA_ACCESS', error_type=type(error).__name__, error=str(error)))
        raise


def validate_source_freeze(gate_path):
    gate_path = Path(gate_path).resolve(); gate = read_json(gate_path)
    require(gate.get('status') == 'FROZEN_BEFORE_DATA_ACCESS' and gate.get('candidate_content_accessed') is False and
            gate.get('seed') == SEED and gate.get('selection_rule') == RULE, 'Unfrozen/changed data-access gate')
    require(time_value(gate['frozen_at_utc']) <= time_value(utc_now()), 'Gate freeze time is in the future')
    require(validate_baseline_freeze(**gate['baseline_arguments']) == gate['baseline_freeze'], 'Baseline chain changed after source freeze')
    protocol_path = Path(gate['final_protocol']['path'])
    require(sha256(protocol_path) == gate['final_protocol']['sha256'], 'Final protocol changed after freeze')
    protocol = read_json(protocol_path)
    _validate_protocol_contract(protocol, gate['baseline_freeze'])
    baseline_protocol = read_json(Path(gate['baseline_arguments']['baseline_results']) / 'protocol.json')
    expected_tree = {name: digest for name, digest in baseline_protocol.get('source_sha256', {}).items()
                     if name.startswith(TREE_PREFIX)}
    require(expected_tree == gate.get('unchanged_tree_sha256'), 'Frozen tree dependency declaration changed')
    require(_source_inventory(gate['source_root'], expected_tree) == gate['source_sha256'], 'Final source changed before candidate access')
    require(_tokenizer_inventory(gate['tokenizer_root']) == gate['tokenizer_sha256'], 'Frozen tokenizer changed')
    require(json_sha(gate['template']) == gate['template_sha256'], 'Prompt template changed')
    excluded_path = Path(gate['exclusion_receipt']['path'])
    require(sha256(excluded_path) == gate['exclusion_receipt']['sha256'], 'Exclusion receipt changed')
    exclusions = read_json(excluded_path)
    require(exclusion_receipt([s['path'] for s in exclusions['metadata_sources']]) == exclusions,
            'Old used/reserved metadata changed after freeze')
    raw = gate['raw_source']
    require(sha256(raw['old_manifest_path']) == raw['old_manifest_sha256'] and
            read_json(raw['old_manifest_path']).get('source_sha256') == raw['expected_sha256'], 'Old raw-source manifest changed')
    return gate, exclusions


def _load_tokenizer(root):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)


def _raw_rows(path):
    import pyarrow.parquet as parquet
    with parquet.ParquetFile(path) as source:
        index = 0
        for batch in source.iter_batches(columns=['report']):
            for row in batch.to_pylist():
                if index >= MIN_INDEX:
                    yield index, row['report']
                index += 1


def prepare_confirmation(gate_path):
    """First candidate open occurs here, after all independent freeze gates."""
    gate_path = Path(gate_path).resolve()
    gate, exclusions = validate_source_freeze(gate_path)
    output = gate_path.parent / 'data'
    require(not output.exists(), 'Fresh confirmation data root required; never resume or replace')
    output.mkdir()
    started = utc_now()
    write_json(output / 'DATA_PREP_STARTED.json', dict(started_at_utc=started, source_freeze_sha256=sha256(gate_path), candidate_content_accessed=False))
    try:
        raw = Path(gate['raw_source']['path'])
        before = sha256(raw)  # FIRST candidate-file open, strictly after gate.
        require(before == gate['raw_source']['expected_sha256'], 'Candidate Parquet source SHA differs from old manifest')
        tokenizer = _load_tokenizer(gate['tokenizer_root'])
        seen_documents = set(exclusions['document_sha256'])
        seen_prefixes = {n: set(exclusions['input_sha256'][n]) for n in map(str, LENGTHS)}
        selected, details, skipped = [], [], []
        iterator = _raw_rows(raw); previous = MIN_INDEX - 1
        try:
            for index, text in iterator:
                require(type(index) is int and index == previous + 1, 'Candidate original indices must cover every source row from900')
                previous = index; require(isinstance(text, str) and text, 'Candidate report must be nonempty text')
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest in seen_documents:
                    skipped.append(dict(source_index=index, document_sha256=digest,
                        reason='used_reserved_or_selected_document_sha256')); continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                require(isinstance(ids, list) and all(type(t) is int and t >= 0 for t in ids), 'Tokenizer returned invalid token IDs')
                if len(ids) < MIN_NATURAL_TOKENS:
                    skipped.append(dict(source_index=index, document_sha256=digest, original_tokens=len(ids),
                        reason='natural_tokens_below16384')); continue
                row = dict(split='confirmation', original_split='validation', source_index=index,
                    document_sha256=digest, original_tokens=len(ids), document_ids=ids[:MIN_NATURAL_TOKENS], **gate['template'])
                hashes = {str(n): input_sha(prompt_ids(row, n)) for n in LENGTHS}
                if any(hashes[n] in seen_prefixes[n] for n in hashes):
                    skipped.append(dict(source_index=index, document_sha256=digest, original_tokens=len(ids), input_sha256=hashes,
                        reason='used_reserved_or_selected_input_prefix_sha256')); continue
                selected.append(row)
                details.append(dict(source_index=index, original_split='validation', document_sha256=digest,
                    original_tokens=len(ids), input_sha256=hashes))
                seen_documents.add(digest)
                for n, digest_n in hashes.items(): seen_prefixes[n].add(digest_n)
                if len(selected) == DOCUMENTS: break
        finally:
            if hasattr(iterator, 'close'): iterator.close()
        require(len(selected) == DOCUMENTS, 'Fewer than16 eligible unseen naturally long documents; no replacement/resampling')
        require(sha256(raw) == before, 'Candidate Parquet changed during preparation')
        validate_source_freeze(gate_path)
        data_path = output / 'confirmation.jsonl'
        with data_path.open('x') as handle:
            for row in selected: handle.write(json.dumps(row, allow_nan=False) + '\n')
        receipts = output / 'receipts'; receipts.mkdir()
        copies = dict(source_freeze=gate_path, exclusions=Path(gate['exclusion_receipt']['path']),
            final_protocol=Path(gate['final_protocol']['path']),
            baseline_complete=Path(gate['baseline_arguments']['baseline_results']) / 'COMPLETE.json',
            baseline_protocol=Path(gate['baseline_arguments']['baseline_results']) / 'protocol.json',
            baseline_frozen_selection=Path(gate['baseline_arguments']['baseline_results']) / 'FROZEN_SELECTION.json',
            baseline_verified_winners=Path(gate['baseline_arguments']['analysis_dir']) / 'VERIFIED_WINNERS.json',
            baseline_verified_complete=Path(gate['baseline_arguments']['verified_complete']),
            baseline_final_manifest=Path(gate['baseline_arguments']['final_manifest']),
            old_raw_source_manifest=Path(gate['raw_source']['old_manifest_path']))
        receipt_sha = {}
        for name, source in copies.items():
            target = receipts / (name + '.json'); shutil.copyfile(source, target)
            receipt_sha[str(target.relative_to(output))] = sha256(target)
        manifest = dict(schema=1, status='PREPARED_NOT_EXECUTED', selection_rule=copy.deepcopy(RULE), seed=SEED,
            source_freeze_sha256=sha256(gate_path), final_protocol_sha256=gate['final_protocol']['sha256'],
            selection_freeze_sha256=gate['baseline_freeze']['selection_freeze_sha256'],
            verified_winners_sha256=gate['baseline_freeze']['verified_winners_sha256'],
            baseline_verified_complete_sha256=gate['baseline_freeze']['verified_complete_sha256'],
            source_frozen_at_utc=gate['frozen_at_utc'], candidate_first_open_not_before_utc=started, prepared_at_utc=utc_now(),
            source=dict(path=str(raw), sha256=before, original_split='validation'),
            source_sha256=gate['source_sha256'], tokenizer_sha256=gate['tokenizer_sha256'], template_sha256=gate['template_sha256'],
            exclusion_receipt_sha256=gate['exclusion_receipt']['sha256'], selected=details, skipped=skipped,
            dataset_file='confirmation.jsonl', dataset_sha256=sha256(data_path), receipt_sha256=receipt_sha,
            scope='Sixteen previously unused project-confirmation validation reports, deterministic index/length-filtered subset; not a random official test sample, no generation-dependent selection',
            offline_limit='Saved scan receipts and frozen source/SHA are auditable offline; raw Parquet and full text are not copied or re-tokenized by the local report',
            no_outcomes_read=True, no_model_generation=True, no_training=True)
        write_json(output / 'DATASET_MANIFEST.json', manifest)
        verify_dataset_manifest(output)
        return manifest
    except BaseException as error:
        write_json(output / 'DATA_PREP_FAILURE.json', dict(status='FAILED_PRESERVE_PARTIAL', error_type=type(error).__name__, error=str(error), scope='No resampling, replacement or partial-root reuse'))
        raise


def verify_dataset_manifest(data_root):
    """Portable saved-data checks; never open model/raw/old-data paths."""
    root = Path(data_root).resolve(); manifest = read_json(root / 'DATASET_MANIFEST.json')
    require(manifest.get('status') == 'PREPARED_NOT_EXECUTED' and manifest.get('selection_rule') == RULE and
            manifest.get('seed') == SEED and manifest.get('no_outcomes_read') is True and
            manifest.get('no_model_generation') is True, 'Confirmation dataset contract invalid')
    require(not (root / 'DATA_PREP_FAILURE.json').exists(), 'Failed preparation cannot authorize inference')
    times = [time_value(manifest[k]) for k in ('source_frozen_at_utc', 'candidate_first_open_not_before_utc', 'prepared_at_utc')]
    require(times == sorted(times), 'Candidate access preceded source freeze')
    require(manifest.get('dataset_file') == 'confirmation.jsonl' and sha256(root / 'confirmation.jsonl') == manifest.get('dataset_sha256'), 'Prepared dataset changed')
    for name, digest in manifest.get('receipt_sha256', {}).items():
        path = root / name; _relative(root, path)
        require(valid_sha(digest) and sha256(path) == digest, 'Portable freeze receipt changed')
    links = dict(source_freeze_sha256='receipts/source_freeze.json', final_protocol_sha256='receipts/final_protocol.json',
        selection_freeze_sha256='receipts/baseline_frozen_selection.json', verified_winners_sha256='receipts/baseline_verified_winners.json',
        baseline_verified_complete_sha256='receipts/baseline_verified_complete.json', exclusion_receipt_sha256='receipts/exclusions.json')
    require(all(manifest.get(key) == sha256(root / name) for key, name in links.items()), 'Dataset is not linked to its portable freeze chain')
    gate = read_json(root / 'receipts/source_freeze.json'); protocol = read_json(root / 'receipts/final_protocol.json')
    require(protocol.get('status') == 'FROZEN_BEFORE_DATA_ACCESS' and protocol.get('baseline_freeze') == gate.get('baseline_freeze') and
            gate.get('source_sha256') == manifest.get('source_sha256') and gate.get('tokenizer_sha256') == manifest.get('tokenizer_sha256') and
            gate.get('template_sha256') == manifest.get('template_sha256') and
            gate.get('raw_source', {}).get('expected_sha256') == manifest.get('source', {}).get('sha256'), 'Portable source/protocol chain differs')
    exclusions = read_json(root / 'receipts/exclusions.json')
    docs = set(exclusions['document_sha256']); prefixes = {n: set(v) for n, v in exclusions['input_sha256'].items()}
    rows = [json.loads(line) for line in (root / 'confirmation.jsonl').read_text().splitlines() if line.strip()]
    require(len(rows) == DOCUMENTS and len(manifest.get('selected', [])) == DOCUMENTS, 'Exactly16 confirmation documents required')
    previous = MIN_INDEX - 1; selected_by_index = {}
    for row, saved in zip(rows, manifest['selected']):
        require(row.get('split') == 'confirmation' and row.get('original_split') == 'validation' and type(row.get('source_index')) is int and
                row['source_index'] > previous and row['source_index'] >= MIN_INDEX and type(row.get('original_tokens')) is int and
                row['original_tokens'] >= MIN_NATURAL_TOKENS and len(row.get('document_ids', [])) == MIN_NATURAL_TOKENS and
                valid_sha(row.get('document_sha256')), 'Prepared cohort selection differs')
        previous = row['source_index']; digest = row['document_sha256']
        hashes = {str(n): input_sha(prompt_ids(row, n)) for n in LENGTHS}
        require(saved == dict(source_index=row['source_index'], original_split='validation', document_sha256=digest,
            original_tokens=row['original_tokens'], input_sha256=hashes), 'Selected manifest differs from exact prepared inputs')
        require(dict(prefix_ids=row['prefix_ids'], suffix_ids=row['suffix_ids']) == gate['template'], 'Prepared prompt template changed')
        selected_by_index[row['source_index']] = saved
    skipped = manifest.get('skipped', [])
    require(isinstance(skipped, list) and all(isinstance(r, dict) and type(r.get('source_index')) is int for r in skipped),
            'Malformed first-eligible scan receipts')
    skipped_by_index = {r['source_index']: r for r in skipped}
    require(len(skipped_by_index) == len(skipped) and not set(skipped_by_index).intersection(selected_by_index) and
            set(skipped_by_index) | set(selected_by_index) == set(range(MIN_INDEX, previous + 1)),
            'First-eligible scan has a missing, duplicated or extra source index')
    for index in range(MIN_INDEX, previous + 1):
        record = selected_by_index.get(index)
        if record is not None:
            digest, hashes = record['document_sha256'], record['input_sha256']
            require(digest not in docs and all(hashes[n] not in prefixes[n] for n in hashes),
                    'Used/reserved/duplicate confirmation document or prefix')
            docs.add(digest)
            for n, digest_n in hashes.items(): prefixes[n].add(digest_n)
            continue
        record = skipped_by_index[index]; digest = record.get('document_sha256'); reason = record.get('reason')
        require(valid_sha(digest), 'Skipped source row has no document SHA')
        if reason == 'used_reserved_or_selected_document_sha256':
            require(digest in docs, 'Skipped document is not in the used/reserved/selected denylist')
        elif reason == 'natural_tokens_below16384':
            require(digest not in docs and type(record.get('original_tokens')) is int and
                    0 <= record['original_tokens'] < MIN_NATURAL_TOKENS, 'Invalid short-document skip receipt')
        elif reason == 'used_reserved_or_selected_input_prefix_sha256':
            hashes = record.get('input_sha256', {})
            require(digest not in docs and type(record.get('original_tokens')) is int and record['original_tokens'] >= MIN_NATURAL_TOKENS and
                    set(hashes) == set(map(str, LENGTHS)) and all(valid_sha(v) for v in hashes.values()) and
                    any(hashes[n] in prefixes[n] for n in hashes), 'Skipped prefix is not in the used/reserved/selected denylist')
        else:
            require(False, 'Unregistered outcome-dependent or unknown skip reason')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze')
    for name in ('output', 'baseline-results', 'analysis-dir', 'verified-complete', 'final-manifest',
                 'raw-source-manifest', 'raw-source', 'tokenizer-root', 'source-root', 'final-protocol'):
        freeze.add_argument('--' + name, type=Path, required=True)
    freeze.add_argument('--metadata-file', type=Path, action='append', required=True)
    prepare = sub.add_parser('prepare'); prepare.add_argument('--gate', type=Path, required=True)
    verify = sub.add_parser('verify'); verify.add_argument('--data-root', type=Path, required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    if command == 'freeze':
        args['metadata_paths'] = args.pop('metadata_file'); result = freeze_gate(**args)
    elif command == 'prepare': result = prepare_confirmation(args['gate'])
    else: result = verify_dataset_manifest(args['data_root'])
    print(json.dumps(dict(status=result['status'], candidate_content_accessed=result.get('candidate_content_accessed'),
        selected_documents=len(result.get('selected', []))), indent=2))


if __name__ == '__main__':
    main()
