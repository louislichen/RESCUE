"""Synthetic CPU program tests, not experimental evidence or GPU results."""
import copy
import importlib.util
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'rescue/fair_confirmation_report.py'
spec = importlib.util.spec_from_file_location('fair_confirmation_report', MODULE)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def statistical_fixture(n=16, own_scale=.90):
    cells, own, baselines = {}, {}, {}
    for length in report.LENGTHS:
        key = str(length)
        own[key] = 'rescue_graph'
        baselines[key] = dict(zip(report.GROUPS, ('re', 'rr', 'rg', 'tree')))
        cells[key] = {}
        for i, mode in enumerate(('rescue_graph', 're', 'rr', 'rg', 'tree')):
            scale = own_scale if i == 0 else 1 + i / 10
            cells[key][mode] = [dict(tokens=256, seconds=(4 + d / 10) * scale,
                accepted=100 + d, proposed=None if mode == 'tree' else 200, blocks=80) for d in range(n)]
    return cells, own, baselines


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def dump_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def full_confirmation_fixture(root):
    """Synthetic434+R schema, including complete saved baseline/pre-data chain."""
    from dataclasses import asdict
    from rescue.fair_confirmation import own_config
    from rescue.baseline_selection import linear_config, tree_policy
    from rescue import fair_confirmation_data as data_contract
    baseline_spec = importlib.util.spec_from_file_location('synthetic_baseline_fixture', MODULE.parents[1]/'tests/test_rescue_baseline_selection_report.py')
    helper = importlib.util.module_from_spec(baseline_spec)
    baseline_spec.loader.exec_module(helper)
    root = Path(root)
    base = root/'baseline_evidence'
    bresults = base/'results'
    original = helper.full_grid_fixture(bresults)
    bprotocol = original['protocol']
    bprotocol['input_sha256'] = {f'271:{n}': str(n).zfill(64) for n in report.LENGTHS}
    dump(bresults/'protocol.json', bprotocol)
    frozen = json.loads((bresults/'FROZEN_SELECTION.json').read_text())
    frozen['protocol_sha256'] = report.sha(bresults/'protocol.json')
    dump(bresults/'FROZEN_SELECTION.json', frozen)
    bcomplete = json.loads((bresults/'COMPLETE.json').read_text())
    bcomplete['frozen_selection_sha256'] = report.sha(bresults/'FROZEN_SELECTION.json')
    dump(bresults/'COMPLETE.json', bcomplete)
    ban = bresults/'analysis_final_v1'
    dump(ban/'summary.json', dict(status='COMPLETE', problems=[], winner_status='VERIFIED_FROZEN', observed_counts=bcomplete['counts']))
    dump(ban/'VERIFIED_WINNERS.json', dict(status='VERIFIED_FROZEN', authoritative=True,
        winners=dict(selected=original['expected_winners'])))
    for name in ('quality', 'tps', 'captures', 'capture_time', 'cold', 'service', 'selected_repeat1'):
        for suffix in ('.png', '.svg'):
            (ban/(name+suffix)).write_text('synthetic program-test artifact, not rendered scientific evidence\n')
    dump(ban/'ANALYSIS_COMPLETE.json', dict(status='COMPLETE', artifacts_sha256={p.name: report.sha(p) for p in ban.iterdir()}))
    manifest_path = base/'FINAL_ARTIFACTS_MANIFEST.json'
    dump(base/'PIPELINE_EXIT.json', dict(exit_code=0))
    dump(manifest_path, dict(file_sha256={str(p.relative_to(base)): report.sha(p) for p in base.rglob('*') if p.is_file()}))
    verified = dict(status='VERIFIED_COMPLETE', **{k: True for k in ('passed', 'raw_independent_audit_passed',
        'rendered_figures_inspected', 'strict_analysis_complete', 'all538_requests_verified', 'all76_owners_verified', 'eight_frozen_choices_verified')},
        baseline_complete_sha256=report.sha(bresults/'COMPLETE.json'), frozen_selection_sha256=report.sha(bresults/'FROZEN_SELECTION.json'),
        analysis_summary_sha256=report.sha(ban/'summary.json'), verified_winners_sha256=report.sha(ban/'VERIFIED_WINNERS.json'),
        final_artifacts_manifest_sha256=report.sha(manifest_path))
    dump(base/'VERIFIED_COMPLETE.json', verified)
    index = {str(Path('/synthetic/baseline')/p.relative_to(base)): dict(saved_path=str(p.relative_to(root)), sha256=report.sha(p))
        for p in base.rglob('*') if p.is_file()}
    dump(base/'PIN_INDEX.json', dict(original_to_saved=index))
    choices = {str(n): {g: frozen['per_context'][str(n)]['groups'][g]['arm'] for g in report.GROUPS} for n in report.LENGTHS}
    extras = report.extra_geometries(choices)
    proof = dict(status='BASELINE_VERIFIED', choices=choices,
        selected_modes={n: {g: a['mode'] for g, a in groups.items()} for n, groups in choices.items()},
        extra_eager_reference_geometries=extras, extra_eager_reference_count=len(extras), scientific_requests=434+len(extras),
        selection_freeze_sha256=report.sha(bresults/'FROZEN_SELECTION.json'), verified_winners_sha256=report.sha(ban/'VERIFIED_WINNERS.json'),
        baseline_complete_sha256=report.sha(bresults/'COMPLETE.json'), analysis_summary_sha256=report.sha(ban/'summary.json'),
        verified_complete_sha256=report.sha(base/'VERIFIED_COMPLETE.json'), final_manifest_sha256=report.sha(manifest_path),
        baseline_protocol_sha256=report.sha(bresults/'protocol.json'), pipeline_exit_sha256=report.sha(base/'PIPELINE_EXIT.json'))
    sources = dict(bprotocol['source_sha256'])
    for name in sources:
        path = root/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((bresults/'source'/name).read_bytes())
    tree_names = ['exp/specextend_bridgekv/specextend/synthetic_'+str(i)+'.py' for i in range(27)]
    for name in list(data_contract.REQUIRED_NEW_SOURCE)+tree_names:
        path = root/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Synthetic program-test source, not measured GPU code.\n')
        sources[name] = report.sha(path)
    data = root/'data'
    data.mkdir()
    template = dict(prefix_ids=[1], suffix_ids=[2])
    template_sha = hashlib.sha256(json.dumps(template, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    rows = [dict(split='confirmation', original_split='validation', source_index=900+i,
        document_sha256=hashlib.sha256(f'synthetic-new:{i}'.encode()).hexdigest(), original_tokens=16384,
        document_ids=[10+i]*16384, **template) for i in range(16)]
    details = [dict(source_index=r['source_index'], original_split='validation', document_sha256=r['document_sha256'],
        original_tokens=r['original_tokens'], input_sha256={str(n): hashlib.sha256(json.dumps([1]+r['document_ids'][:n-2]+[2]).encode()).hexdigest() for n in report.LENGTHS}) for r in rows]
    dump_rows(data/'confirmation.jsonl', rows)
    formal = dict(status='FROZEN_BEFORE_DATA_ACCESS', is_template=False, seed=report.SEED, selection_rule=report.SELECTION_RULE,
        baseline_freeze=proof, expected_scientific_requests=434+len(extras), fixed=data_contract.FIXED,
        own_method=data_contract.OWN_METHOD, statistics=data_contract.STATISTICS, execution=data_contract.EXECUTION,
        required_source_files=list(data_contract.REQUIRED_NEW_SOURCE))
    excluded = dict(status='OLD_DATA_METADATA_ONLY', metadata_sources=[dict(relative_name=f'synthetic/{i}') for i in range(11)],
        document_sha256=['0'*64], input_sha256={'4096': [], '8192': []})
    gate = dict(status='FROZEN_BEFORE_DATA_ACCESS', candidate_content_accessed=False, seed=report.SEED,
        selection_rule=report.SELECTION_RULE, baseline_freeze=proof, source_sha256=sources, tokenizer_sha256={'tokenizer.json': 'a'*64},
        template=template, template_sha256=template_sha, raw_source=dict(expected_sha256='b'*64), frozen_at_utc='2026-10-20T01:00:00+00:00')
    copies = dict(source_freeze=gate, exclusions=excluded, final_protocol=formal, baseline_complete=bcomplete,
        baseline_protocol=bprotocol, baseline_frozen_selection=frozen, baseline_verified_winners=json.loads((ban/'VERIFIED_WINNERS.json').read_text()),
        baseline_verified_complete=verified, baseline_final_manifest=json.loads(manifest_path.read_text()), old_raw_source_manifest=dict(source_sha256='b'*64))
    for name, value in copies.items():
        dump(data/'receipts'/(name+'.json'), value)
    pins = {str(p.relative_to(data)): report.sha(p) for p in (data/'receipts').iterdir()}
    dmanifest = dict(status='PREPARED_NOT_EXECUTED', selection_rule=report.SELECTION_RULE, seed=report.SEED,
        source_freeze_sha256=pins['receipts/source_freeze.json'], final_protocol_sha256=pins['receipts/final_protocol.json'],
        selection_freeze_sha256=proof['selection_freeze_sha256'], verified_winners_sha256=proof['verified_winners_sha256'],
        baseline_verified_complete_sha256=proof['verified_complete_sha256'], exclusion_receipt_sha256=pins['receipts/exclusions.json'],
        source_frozen_at_utc='2026-10-20T01:00:00+00:00', candidate_first_open_not_before_utc='2026-10-20T01:01:00+00:00',
        prepared_at_utc='2026-10-20T01:02:00+00:00', source=dict(sha256='b'*64), source_sha256=sources,
        tokenizer_sha256=gate['tokenizer_sha256'], template_sha256=template_sha, selected=details,
        dataset_file='confirmation.jsonl', dataset_sha256=report.sha(data/'confirmation.jsonl'), receipt_sha256=pins,
        skipped=[],
        no_outcomes_read=True, no_model_generation=True, no_training=True)
    dump(data/'DATASET_MANIFEST.json', dmanifest)
    expected = dict(reference_requests=32, own_quality_requests=64, baseline_quality_requests=8,
        matching_quality_requests=len(extras), prime_requests=10, timing_requests=320, owners=10)
    arms = {str(n): [dict(mode='rescue_graph', family='rescue', backend='cuda_graph')]+[choices[str(n)][g] for g in report.GROUPS] for n in report.LENGTHS}
    docs = [r['source_index'] for r in rows]
    owner_order, warm_order = report.registered_schedule(choices, docs)
    protocol = dict(arms=arms, modes_by_context={n: [a['mode'] for a in values] for n, values in arms.items()},
        source_indices=docs, lengths=list(report.LENGTHS), repeats=2, max_new_tokens=128, seed=report.SEED,
        expected_counts=expected, expected_requests=434+len(extras), extra_eager_reference_geometries=extras,
        tokenizer_eos_token_id=100000, vocab_size=151936, target_layers=28, draft_layers=28, graph_max_entries=128,
        baseline_freeze=proof, frozen_choices=choices, source_freeze_sha256=dmanifest['source_freeze_sha256'],
        final_protocol_sha256=dmanifest['final_protocol_sha256'], dataset_sha256=dmanifest['dataset_sha256'],
        dataset_manifest='data/DATASET_MANIFEST.json', portable_data_root='data', selection_rule=report.SELECTION_RULE,
        dataset_manifest_sha256=report.sha(data/'DATASET_MANIFEST.json'), **report.ASSETS, mapper_frozen=True,
        policy_frozen=True, no_training=True, max_live_owners=1, template_prefill_precision='bfloat16',
        configs={str(n): {a['mode']: asdict(own_config()) if a['family']=='rescue' else asdict(linear_config(a)) if a['family']=='retrieval' else tree_policy(a) for a in values} for n, values in ((int(n), v) for n, v in arms.items())},
        environment=bprotocol['environment'], source_sha256=sources, source_bindings=bprotocol['source_bindings'],
        tree_source_bindings=bprotocol['tree_source_bindings'], flash_attention=bprotocol['flash_attention'],
        baseline_evidence=dict(root='baseline_evidence', baseline_results='results', analysis_dir='results/analysis_final_v1',
            verified_complete='VERIFIED_COMPLETE.json', final_manifest='FINAL_ARTIFACTS_MANIFEST.json',
            pin_index='baseline_evidence/PIN_INDEX.json', pin_index_sha256=report.sha(base/'PIN_INDEX.json')),
        owner_order=owner_order, warm_order=warm_order,
        input_sha256={f'{r["source_index"]}:{n}': r['input_sha256'][str(n)] for r in details for n in report.LENGTHS},
        prime_input_sha256={str(n): bprotocol['input_sha256'][f'271:{n}'] for n in report.LENGTHS})
    protocol['args'] = {name: '/synthetic/'+name for name in ('mapper', 'policy', 'cost', 'freeze_receipt', 'dataset_manifest', 'data_file')}
    protocol['assets_sha256'] = dict(bprotocol['assets_sha256'], **{
        protocol['args'][name]: digest for name, digest in (
            ('mapper', report.ASSETS['mapper_sha256']), ('policy', report.ASSETS['policy_sha256']),
            ('cost', report.ASSETS['cost_sha256']), ('freeze_receipt', dmanifest['source_freeze_sha256']),
            ('dataset_manifest', report.sha(data/'DATASET_MANIFEST.json')), ('data_file', dmanifest['dataset_sha256']))})
    dump(root/'protocol.json', protocol)
    dump(root/'GATE_PASSED.json', dict(passed=True, confirmation_content_opened=False, baseline_freeze=proof,
        expected_counts=expected, expected_requests=434+len(extras), source_freeze_sha256=dmanifest['source_freeze_sha256']))
    reference = list(range(128))
    dump(root/'prime_references.json', {str(n): reference for n in report.LENGTHS})
    refs = {f'{d}:{n}': reference for n in report.LENGTHS for d in docs}
    dump(root/'references.json', refs)
    state = dict(protocol=protocol, documents=docs, arms=arms, choices=choices, extras=extras,
        expected=expected, reference=reference, owner_order=owner_order, warm_order=warm_order)
    populate_confirmation_rows(root, state)
    return state


def populate_confirmation_rows(root, state):
    from rescue.baseline_selection import tree_policy
    reference, docs = state['reference'], state['documents']
    ledgers = {name: [] for name in ('reference', 'own_quality', 'baseline_quality', 'matching_quality', 'prime', 'runtime')}
    quality_owners, owners, tree_raw = [], [], []
    index = 0

    def register(name, row):
        nonlocal index
        row.update(request_id=f'synthetic-confirmation:{index}', execution_index=index)
        index += 1
        ledgers[name].append(row)
        if row.get('arm', {}).get('family') == 'full_tree':
            tree_raw.append(dict(request_id=row['request_id'], execution_index=row['execution_index'],
                result=dict(generated=reference, tokens=128, upstream=row['upstream'])))
        return row

    def linear(arm, length):
        own = arm['family'] == 'rescue'
        trace, observations, queries = [], [], []
        progress = 0
        while progress < 128:
            requested = (2 if len(trace) % 2 == 0 else 4) if own else arm['fixed_k']
            cap = min(requested, 128-progress)
            count = cap-1
            draft = reference[progress:progress+count]+[999]
            checks = (reference+[128])[progress:progress+cap+1]
            prefix = length-1+progress
            selected = list(range(64 if own else arm['top_chunks']*32))
            trace.append(dict(round_index=len(trace), prefix=prefix, cap=cap, proposed=cap,
                accepted=count, terminal=progress+cap == 128, workset_tokens=len(selected)))
            value_digest = dict(shape=[64, 256], dtype='torch.bfloat16', sha256='a'*64)
            native_digest = dict(value_digest, sha256='b'*64)
            proof = [dict(layer=27, positions=selected[:64], working_value=value_digest,
                native_value=native_digest, different_elements=1, native_matches_memo=True)] if own else []
            observations.append(dict(round_index=len(observations), prefix=prefix, progress=progress,
                remaining=128-progress, selected=selected, draft=draft, checks=checks, accepted=count,
                budget_features=[0.]*11+[.2] if own and cap>=2 else None,
                injection_evidence=proof, summary_count=1 if own else 0))
            for depth in range(cap):
                queries.append(dict(logits=True, query=dict(shape=[1, 4, 1, 8], dtype='torch.bfloat16', sha256='c'*64),
                    summary_requested=own and depth<2, summary_count=int(own and depth<2)))
            progress += cap
        valid_length = length+128
        native = dict(lengths=[valid_length]*28, layers=[dict(key=dict(shape=[1, 2, valid_length, 8], dtype='torch.bfloat16', sha256='d'*64),
            value=dict(shape=[1, 2, valid_length, 8], dtype='torch.bfloat16', sha256='e'*64)) for _ in range(28)])
        accepted = sum(t['accepted'] for t in trace)
        return dict(generated=reference.copy(), tokens=128, trace=trace, blocks=len(trace), accepted=accepted,
            proposed=128, acceptance_rate=accepted/128, draft_forwards=128, selector_refreshes=(len(trace)+3)//4,
            working_gather_layers=len(trace)*28, projected_positions=len(trace)*64 if own else 0,
            draft_shadow_probes=0, baseline_algorithm='qwen_chunk_retrieval' if not own else None,
            observed_rounds=observations, queries=queries, native_t=native, native_s=copy.deepcopy(native),
            nonzero_injection=own)

    def tree(arm, length):
        upstream = dict(output_token_ids=reference.copy(), total_generated=128,
            raw_total_generated=132, generation_overshoot=4, accept_length_list=[3]*33)
        return dict(generated=reference.copy(), tokens=128, wrapper_generated=reference.copy(), wrapper_tokens=128,
            upstream=upstream, raw_total_generated=132, generation_overshoot=4, scored_generated_before_eos=reference.copy(),
            raw_tail_ids_available=False, accepted=99, blocks=33, proposed=None, acceptance_rate=None,
            tree_policy=tree_policy(arm), observed_full_config=dict(target_use_flash_prefill=True,
                target_use_hybrid_tree_attn=True, draft_use_flash_prefill=True, use_retrieval_cache=True,
                retrieve_top_k=arm['top_chunks'], retrieval_chunk_size=32, retrieve_every_n_steps=4))

    def counters(arm, request_count, cold, forwards=128):
        backend = arm['backend']
        graph, resident = backend=='cuda_graph', backend=='resident'
        delta = dict(captures=2 if cold and graph else 0, replays=forwards if graph else 0,
            resident_steps=forwards if resident else 0, capture_host_seconds=.2 if cold and graph else 0.)
        stats = dict(mode=backend, entries=2 if graph else 0, captures=2 if graph else 0,
            replays=forwards*request_count if graph else 0, resident_steps=forwards*request_count if resident else 0,
            capture_host_seconds=.2 if graph else 0., max_graph_entries=128 if graph or resident else None)
        return delta, stats

    for length in report.LENGTHS:
        for doc in docs:
            register('reference', dict(mode='canonical_ar', phase='reference', source_index=doc, length=length,
                generated=reference.copy(), performance_eligible=False))
    own_quality = {}
    baseline_quality = {}

    def quality(arm, doc, length, ledger, phase):
        row = tree(arm, length) if arm['family']=='full_tree' else linear(arm, length)
        _, stats = counters(arm, 1, True)
        owner_id = f'synthetic-quality-owner:{index}'
        row.update(mode=arm['mode'], arm=arm, source_index=doc, length=length, phase=phase,
            performance_eligible=False, output_agrees=True, runtime_stats=stats, owner_id=owner_id)
        register(ledger, row)
        quality_owners.append(dict(request_id=row['request_id'], owner_id=owner_id,
            final_runtime_stats=copy.deepcopy(stats), closed_runtime_stats=dict(stats, entries=0), engine_closed=True, hook_restored=True))
        return row

    for length in report.LENGTHS:
        for doc in docs:
            quality(dict(mode='rescue_eager', family='rescue', backend='eager'), doc, length, 'own_quality', 'own_quality')
            arm = state['arms'][str(length)][0]
            own_quality[doc, length] = quality(arm, doc, length, 'own_quality', 'own_quality')
        for group in report.GROUPS:
            arm = state['choices'][str(length)][group]
            baseline_quality[arm['mode'], length] = quality(arm, docs[0], length, 'baseline_quality', 'baseline_sentinel')
        for geometry in state['extras']:
            if geometry['length'] != length:
                continue
            arm = dict(mode=f"matching_retrieval{geometry['top_chunks']}_k{geometry['fixed_k']}_eager", family='retrieval',
                backend='eager', top_chunks=geometry['top_chunks'], fixed_k=geometry['fixed_k'])
            quality(arm, docs[0], length, 'matching_quality', 'matching_geometry_eager')
    for owner_index, descriptor in enumerate(state['owner_order']):
        mode, length = descriptor['mode'], descriptor['length']
        arm = next(a for a in state['arms'][str(length)] if a['mode']==mode)
        owner_id = f'synthetic-timing-owner:{owner_index}'
        signature = dict(owner_identity=owner_index+5000)
        owned = []
        requests = [dict(mode=mode, length=length, source_index=271, phase='cold_prime')]+[
            dict(r, phase='warm') for r in state['warm_order'] if r['mode']==mode and r['length']==length]
        group_index = next((i for i, g in enumerate(report.GROUPS) if state['choices'][str(length)][g]['mode']==mode), -1)
        factor = .7 if arm['family']=='rescue' else 1.+group_index*.1
        for count, request in enumerate(requests, 1):
            cold = request['phase']=='cold_prime'
            row = tree(arm, length) if arm['family']=='full_tree' else linear(arm, length)
            for field in ('observed_rounds', 'queries', 'native_t', 'native_s', 'nonzero_injection'):
                row.pop(field, None)
            delta, stats = counters(arm, count, cold)
            seconds = factor*(4 if cold else 2.+(request['source_index']-900)/100)
            row.update(request, arm=arm, owner_id=owner_id, owner_signature=signature,
                performance_eligible=True, output_agrees=True, observer_enabled=False, profile_enabled=False,
                mapper_executed=arm['family']=='rescue', controller_executed=arm['family']=='rescue',
                outer_seconds=seconds, outer_tps=128/seconds, total_seconds=seconds-.01,
                prefill_seconds=.1, decode_seconds=seconds-.12, cleanup_seconds=.01, runtime_stats=stats)
            if arm['family']=='retrieval':
                row['execution_backend'] = arm['backend']
            if arm['backend'] in ('resident', 'cuda_graph'):
                row['draft_runtime'] = dict(mode=arm['backend'], request=delta, lifetime=copy.deepcopy(stats))
            if cold:
                row.update(constructor_and_detach_seconds=.02, cold_complete_outer_seconds=.02+seconds)
            register('prime' if cold else 'runtime', row)
            owned.append(row)
        final = owned[-1]['runtime_stats']
        seconds = .02+sum(r['outer_seconds'] for r in owned)+.001
        owners.append(dict(mode=mode, length=length, arm=arm, owner_id=owner_id, owner_signature=signature,
            maximum_live_owners=1, prime_request_id=owned[0]['request_id'], warm_request_ids=[r['request_id'] for r in owned[1:]],
            constructor_and_detach_seconds=.02, close_seconds=.001, cold_prime_outer_seconds=owned[0]['outer_seconds'],
            prime_tokens=128, warm_tokens=32*128, warm_outer_seconds_sum=sum(r['outer_seconds'] for r in owned[1:]),
            accounted_service_seconds=seconds, accounted_service_tps=33*128/seconds,
            final_runtime_stats=copy.deepcopy(final), closed_runtime_stats=dict(final, entries=0), engine_closed=True, hook_restored=True))
    assert index == 434+len(state['extras'])
    for name, values in ledgers.items():
        dump_rows(root/(name+'.jsonl'), values)
    dump_rows(root/'owners.jsonl', owners)
    dump_rows(root/'quality_owners.jsonl', quality_owners)
    dump_rows(root/'tree_raw.jsonl', tree_raw)
    dump(root/'QUALITY_PASSED.json', dict(passed=True, own_all32_pairs=True, baseline_matching_geometry=True,
        actual_same_bfloat16_nonzero_mapper=True, quality_owners_closed=True))
    dump(root/'COMPLETE.json', dict(passed=True, counts=state['expected'], expected_counts=state['expected'],
        logical_requests=index, sources_unchanged=True, assets_unchanged=True, quality_gate_passed=True,
        all_canonical=True, owners_closed=True, analysis_required_locally=True, goal_complete=False))
    dump(root.parent/'EVIDENCE_MANIFEST.json', dict(file_sha256={str(p.relative_to(root.parent)): report.sha(p)
        for p in root.rglob('*') if p.is_file()}))
    state['ledgers'] = ledgers


class FairStatisticsTests(unittest.TestCase):
    def test_ratio_of_sums_not_mean_tps(self):
        cells, own, groups = statistical_fixture()
        cells['4096']['rescue_graph'][0]['seconds'] *= 10
        result = report.synchronized_statistics(cells, own, groups)
        docs = cells['4096']['rescue_graph']
        expected = sum(d['tokens'] for d in docs) / sum(d['seconds'] for d in docs)
        self.assertEqual(result['contexts']['4096']['rescue_graph']['pooled_tps'], expected)
        self.assertNotAlmostEqual(expected, np.mean([d['tokens']/d['seconds'] for d in docs]))

    def test_independent_reconstruction_shared_indices_and_bonferroni(self):
        cells, own, groups = statistical_fixture()
        cells['4096']['rescue_graph'][0]['seconds'] *= 1.7
        cells['8192'] = copy.deepcopy(cells['4096'])
        result = report.synchronized_statistics(cells, own, groups)
        index = np.random.default_rng(20261024).integers(0, 16, size=(100000, 16))
        own_seconds = np.array([d['seconds'] for d in cells['4096']['rescue_graph']])
        other_seconds = np.array([d['seconds'] for d in cells['4096']['re']])
        distribution = 100 * (other_seconds[index].sum(1) / own_seconds[index].sum(1) - 1)
        first, corresponding = result['contrasts'][0], result['contrasts'][4]
        np.testing.assert_allclose(first['pointwise_ci95'], np.percentile(distribution, [2.5, 97.5]), rtol=0, atol=1e-12)
        self.assertAlmostEqual(first['simultaneous_one_sided_lower95'], np.percentile(distribution, .625), places=12)
        self.assertEqual(first['pointwise_ci95'], corresponding['pointwise_ci95'])
        self.assertEqual(first['simultaneous_one_sided_lower95'], corresponding['simultaneous_one_sided_lower95'])
        self.assertLessEqual(first['simultaneous_one_sided_lower95'], first['pointwise_ci95'][0])

    def test_all_eight_required_and_partial_never_passes(self):
        cells, own, groups = statistical_fixture()
        self.assertTrue(report.synchronized_statistics(cells, own, groups)['statistical_gate'])
        cells['8192']['rescue_graph'] = copy.deepcopy(cells['8192']['re'])
        self.assertFalse(report.synchronized_statistics(cells, own, groups)['statistical_gate'])
        partial = statistical_fixture(15)
        self.assertFalse(report.synchronized_statistics(*partial)['statistical_gate'])

    def test_repeat_pool_and_frozen_bootstrap(self):
        rows = [dict(mode='x', length=4096, source_index=900, repeat=i, tokens=128,
            outer_seconds=seconds, accepted=10, proposed=30, blocks=20) for i, seconds in enumerate((2., 4.))]
        self.assertEqual(report.pool_document(rows)['tps'], 256/6)
        with self.assertRaises(ValueError):
            report.pool_document(rows[:1])
        with self.assertRaises(ValueError):
            report.synchronized_statistics(*statistical_fixture(), draws=10000)
        with self.assertRaises(ValueError):
            report.synchronized_statistics(*statistical_fixture(), seed=1)


class FairEvidenceTests(unittest.TestCase):
    def test_eos_before_second_draft_allows_none_features_but_not_fake_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = full_confirmation_fixture(Path(tmp)/'results')
            row = copy.deepcopy(state['ledgers']['own_quality'][0])
            eos = state['protocol']['tokenizer_eos_token_id']
            row.update(generated=[eos], tokens=1, proposed=1, accepted=1, blocks=1,
                acceptance_rate=1., draft_forwards=1)
            row['trace'] = [dict(row['trace'][0], proposed=1, accepted=1, terminal=True)]
            row['queries'] = row['queries'][:1]
            row['observed_rounds'] = [dict(row['observed_rounds'][0], draft=[eos], checks=[eos, 0],
                accepted=1, budget_features=None)]
            self.assertEqual(report.linear_trace_errors(row, own=True), [])
            self.assertEqual(report.injection_errors(row), [])
            row['tokens'] = 100
            self.assertTrue(report.linear_trace_errors(row, own=True))

    def test_saved_real_rescue_row_has_no_baseline_only_execution_field(self):
        # Regression against a previously completed, already public-to-this-task
        # archive. This does not read the new confirmation cohort or run a model.
        path = MODULE.parents[1]/'exp/draft_graph/out_h100_cost_recalibration_v2_20261020/results/runtime.jsonl'
        if not path.exists():
            self.skipTest('Historical saved-row regression archive not staged')
        row = next(json.loads(line) for line in path.read_text().splitlines() if json.loads(line).get('mode')=='oldrisk')
        self.assertNotIn('execution_backend', row)
        # The new runner adds this observed owner setting, not a new engine key.
        row['runtime_stats']['max_graph_entries'] = 128
        self.assertEqual(report.timing_errors(row, row['generated'],
            dict(mode='oldrisk', family='rescue', backend='cuda_graph'), 'oldrisk'), [])

    def test_full438_fixture_and_no_missing_or_tampered_pair_can_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'results'
            state = full_confirmation_fixture(root)
            result = report.analyze(root)
            self.assertEqual(result['status'], 'COMPLETE', result.get('problems'))
            self.assertTrue(result['verified_success'])
            self.assertEqual(result['observed_requests'], 438)
            self.assertEqual(result['statistics']['n'], 16)
            self.assertEqual(len(result['statistics']['contrasts']), 8)
            self.assertEqual(result['paired_complete_documents'], list(range(900, 916)))
            rows = copy.deepcopy(state['ledgers']['runtime'])
            removed = rows.pop()
            dump_rows(root/'runtime.jsonl', rows)
            partial = report.analyze(root)
            self.assertEqual(partial['status'], 'PARTIAL')
            self.assertFalse(partial['verified_success'])
            self.assertEqual(partial['statistics']['n'], 15)
            dump_rows(root/'runtime.jsonl', state['ledgers']['runtime'])
            quality = copy.deepcopy(state['ledgers']['own_quality'])
            quality[1]['queries'][0]['query']['sha256'] = 'f'*64
            dump_rows(root/'own_quality.jsonl', quality)
            changed = report.analyze(root)
            self.assertEqual(changed['status'], 'PARTIAL')
            self.assertFalse(changed['verified_success'])
            self.assertTrue(any('queries' in p for p in changed['problems']))

    def test_partial_complete4k_is_not_hidden_by_missing8k(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'results'
            state = full_confirmation_fixture(root)
            dump_rows(root/'runtime.jsonl', [r for r in state['ledgers']['runtime'] if r['length']==4096])
            result = report.analyze(root)
            self.assertEqual(result['status'], 'PARTIAL')
            self.assertFalse(result['verified_success'])
            self.assertEqual(result['statistics']['n'], 0)
            self.assertEqual(result['statistics']['per_context_n'], {'4096':16, '8192':0})
            self.assertEqual(len(result['statistics']['contexts']['4096']), 5)
            self.assertNotIn('8192', result['statistics']['contexts'])
            self.assertEqual(len(result['statistics']['contrasts']), 4)
            self.assertTrue(all(c['simultaneous_one_sided_lower95'] is None for c in result['statistics']['contrasts']))

    def test_short_eos_and_eight_gate_are_not_assumed_from_complete(self):
        cells, own, groups = statistical_fixture()
        stats = report.synchronized_statistics(cells, own, groups)
        stats['contrasts'][0]['simultaneous_one_sided_lower95'] = -1.
        # Main success is never inferred from a completion filename alone.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump(root/'COMPLETE.json', dict(passed=True))
            self.assertEqual(report.analyze(root)['status'], 'PARTIAL')
        geometry = {str(n): {g: dict(mode=str(g), family='retrieval', top_chunks=16, fixed_k=4,
            backend='eager' if g.endswith('/eager') else 'resident' if g.endswith('/resident') else 'cuda_graph')
            for g in report.GROUPS} for n in report.LENGTHS}
        self.assertEqual(report.extra_geometries(geometry), [])

    def test_local_sha_pin_rejects_escape_and_changed_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / 'freeze.json'
            file.write_text('{}')
            evidence = report.Evidence(root)
            self.assertTrue(evidence.pin('freeze.json', report.sha(file)))
            self.assertFalse(evidence.pin('../outside.json', 'a'*64))
            self.assertFalse(evidence.pin('freeze.json', 'b'*64))
            self.assertEqual(len(evidence.problems), 2)

    def test_tree_keeps_raw_overshoot_and_no_invented_rate(self):
        row = dict(generated=[1, 2], tokens=2, wrapper_generated=[1, 2], wrapper_tokens=2,
            raw_total_generated=3, generation_overshoot=1, scored_generated_before_eos=[1, 2],
            raw_tail_ids_available=False, accepted=2, blocks=1, proposed=None, acceptance_rate=None,
            upstream=dict(accept_length_list=[2], raw_total_generated=3, generation_overshoot=1,
                output_token_ids=[1, 2], total_generated=2))
        self.assertEqual(report.tree_errors(row), [])
        row['acceptance_rate'] = .8
        self.assertTrue(report.tree_errors(row))

    def test_render_synthetic_figures_and_standalone_source(self):
        cells, own, groups = statistical_fixture()
        summary = dict(status='PARTIAL', verified_success=False, problems=['synthetic program fixture'],
            statistics=report.synchronized_statistics(cells, own, groups), own_modes=own,
            challengers=groups, owner_metrics={}, diagnostics={}, selection_diagnostics={})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'analysis'
            receipt = report.write_report(summary, Path(tmp), output)
            self.assertEqual(len(receipt['rendered_figures']), 5)
            for name in receipt['rendered_figures']:
                self.assertGreater((output/(name+'.png')).stat().st_size, 10000)
                self.assertGreater((output/(name+'.svg')).stat().st_size, 1000)
            self.assertEqual(report.sha(MODULE), report.sha(output/'analysis_source.py'))
            self.assertFalse((output/'VERIFIED_SUCCESS.json').exists())
            self.assertIn('PARTIAL', (output/'REPORT.md').read_text())


if __name__ == '__main__':
    unittest.main()
