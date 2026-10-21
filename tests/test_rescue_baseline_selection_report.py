"""Synthetic CPU tests of selection math, not model/GPU performance results."""
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rescue.baseline_selection_report import (
    BACKENDS, DOCS, LENGTHS, SEED, analyze, menu, pool_document,
    pooled_bootstrap, select_winners, write_report,
)

FIGURES = {'quality', 'tps', 'captures', 'cold', 'service',
           'selected_repeat1', 'capture_time'}


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def dump_rows(path, values):
    path.write_text(''.join(json.dumps(value, allow_nan=False)+'\n' for value in values))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_grid_fixture(directory):
    """Complete synthetic538 contract; no model or GPU is constructed."""
    from rescue.baseline_selection import (
        arms, linear_config, schedule, select_winners as runner_winners, tree_policy,
    )
    from rescue.baseline_selection_report import DATA_SHA, EXPECTED

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    descriptors = arms()
    arm_map = {arm['mode']: arm for arm in descriptors}
    owner_plan, warm_plan = schedule()
    expected_winners = {
        '4096': dict(zip(('retrieval/eager', 'retrieval/resident', 'retrieval/graph', 'full_tree'),
            ('retrieval16_k2_eager', 'retrieval32_k4_resident',
             'retrieval16_k8_graph', 'tree32_n50'))),
        '8192': dict(zip(('retrieval/eager', 'retrieval/resident', 'retrieval/graph', 'full_tree'),
            ('retrieval32_k1_eager', 'retrieval16_k12_resident',
             'retrieval32_k4_graph', 'tree16_n8'))),
    }
    factors = dict(zip(('retrieval/eager', 'retrieval/resident', 'retrieval/graph', 'full_tree'),
                       (.7, .6, .4, .8)))
    reference = list(range(128))
    canonical, primes, warm, owners, raw_tree = [], [], [], [], []
    logical_index = 0

    def register(rows, row):
        nonlocal logical_index
        row.update(request_id=f'synthetic-request:{logical_index}', execution_index=logical_index)
        logical_index += 1
        rows.append(row)
        return row

    for length in LENGTHS:
        for doc in DOCS:
            register(canonical, dict(mode='canonical_ar', phase='reference',
                source_index=doc, length=length, generated=reference.copy(),
                performance_eligible=False))

    def runtime(arm, count, cold, closed=False):
        backend = arm['backend']
        graph, resident = backend == 'cuda_graph', backend == 'resident'
        delta = dict(captures=2 if cold and graph else 0,
            replays=128 if graph else 0, resident_steps=128 if resident else 0,
            capture_host_seconds=.3 if cold and graph else 0.,
            shadow_updates=128 if graph or resident else 0)
        stats = dict(mode=backend, entries=0 if closed or not (graph or resident) else 2,
            captures=2 if graph else 0, replays=128*count if graph else 0,
            resident_steps=128*count if resident else 0,
            capture_host_seconds=.3 if graph else 0.,
            shadow_updates=128*count if graph or resident else 0,
            graph_keys=[] if closed or not (graph or resident) else [[513, True], [514, False]])
        return delta, stats

    def linear(arm, length, seconds):
        traces, progress = [], 0
        while progress < 128:
            cap = min(arm['fixed_k'], 128-progress)
            traces.append(dict(round_index=len(traces), prefix=length-1+progress,
                cap=cap, proposed=cap, accepted=cap-1,
                terminal=progress+cap == 128, workset_tokens=arm['top_chunks']*32))
            progress += cap
        blocks = len(traces)
        accepted = 128-blocks
        total = seconds-.005
        return dict(generated=reference.copy(), tokens=128, accepted=accepted,
            proposed=128, blocks=blocks, trace=traces, acceptance_rate=accepted/128,
            draft_forwards=128, selector_refreshes=(blocks+3)//4,
            working_gather_layers=28*blocks, working_full_gather_layers=28*blocks,
            working_gathered_position_layers=arm['top_chunks']*32*28*blocks,
            working_incremental_binds=0, target_probe_rounds=(blocks+3)//4,
            draft_shadow_probes=0, projected_positions=0, rpc_calls=0,
            execution_backend=arm['backend'], baseline_algorithm='qwen_chunk_retrieval',
            total_seconds=total, total_tps=128/total,
            prefill_seconds=.1, decode_seconds=total-.105, cleanup_seconds=.005)

    def tree(arm, seconds):
        upstream = dict(output_token_ids=reference.copy(), total_generated=128,
            raw_total_generated=132, generation_overshoot=4, accept_length_list=[3]*33)
        return dict(generated=reference.copy(), wrapper_generated=reference.copy(),
            wrapper_tokens=128, tokens=128, upstream=upstream,
            scored_generated_before_eos=reference.copy(), raw_total_generated=132,
            generation_overshoot=4, raw_tail_ids_available=False, blocks=33,
            accepted=99, proposed=None, acceptance_rate=None, tree_policy=tree_policy(arm),
            observed_full_config=dict(target_use_flash_prefill=True,
                target_use_hybrid_tree_attn=True, draft_use_flash_prefill=True,
                use_retrieval_cache=True, retrieve_top_k=arm['top_chunks'],
                retrieval_chunk_size=32, retrieve_every_n_steps=4),
            total_seconds=seconds-.005, total_tps=128/(seconds-.005))

    for owner_index, entry in enumerate(owner_plan):
        mode, length = entry['mode'], entry['length']
        arm = arm_map[mode]
        group_for_mode = next((key for key, value in expected_winners[str(length)].items()
                               if value == mode), None)
        factor = factors[group_for_mode] if group_for_mode else 1.+descriptors.index(arm)/100
        context_scale = 1. if length == 4096 else 1.4
        owner_id = f'synthetic-owner:{owner_index}'
        pointer = 10000*(owner_index+1)
        signature = (dict(owner_identity=pointer, engine_identity=pointer+1,
            base_model_identity=1, draft_model_identity=2, prefill_precision='bfloat16')
            if arm['family'] == 'full_tree' else {
                cache: dict(identity=pointer+i, key_ptr=pointer+10+i,
                    value_ptr=pointer+20+i, capacity=16384)
                for i, cache in enumerate(('native_t', 'native_s', 'working'))})
        requests = [dict(mode=mode, length=length, source_index=271, phase='cold_prime')]+[
            dict(value, phase='warm') for value in warm_plan
            if value['mode'] == mode and value['length'] == length]
        owned_warm = []
        for count, request in enumerate(requests, 1):
            cold = request['phase'] == 'cold_prime'
            doc_scale = (1., 1.3, 2.)[DOCS.index(request['source_index'])]
            seconds = (6. if cold else (2., 5.)[request['repeat']]*doc_scale)*factor*context_scale
            row = tree(arm, seconds) if arm['family'] == 'full_tree' else linear(arm, length, seconds)
            if arm['family'] == 'full_tree':
                stats = dict(mode='upstream_flash_hybrid', entries=0,
                    captures=0, replays=0, resident_steps=0,
                    capture_host_seconds=0., closed=False)
            else:
                delta, stats = runtime(arm, count, cold)
                if arm['backend'] != 'eager':
                    row['draft_runtime'] = dict(mode=arm['backend'], request=delta,
                        lifetime=copy.deepcopy(stats), max_graph_entries=128,
                        retain_query=False)
            row.update(request, arm=copy.deepcopy(arm), owner_id=owner_id,
                owner_signature=signature, runtime_stats=stats,
                output_agrees=True, performance_eligible=True,
                observer_enabled=False, profile_enabled=False,
                mapper_executed=False, controller_executed=False,
                outer_seconds=seconds, outer_tps=128/seconds)
            if cold:
                row.update(constructor_and_detach_seconds=.02,
                           cold_complete_outer_seconds=.02+seconds)
            register(primes if cold else warm, row)
            if arm['family'] == 'full_tree':
                raw_tree.append(dict(request_id=row['request_id'],
                    execution_index=row['execution_index'], owner_id=owner_id,
                    **request, outer_seconds=seconds,
                    result=dict(generated=reference.copy(), tokens=128,
                        upstream=copy.deepcopy(row['upstream']),
                        total_seconds=row['total_seconds'], total_tps=row['total_tps'],
                        observed_full_config=copy.deepcopy(row['observed_full_config']))))
            if cold:
                prime = row
            else:
                owned_warm.append(row)
        warm_seconds = sum(row['outer_seconds'] for row in owned_warm)
        service_seconds = .02+prime['outer_seconds']+warm_seconds+.001
        final_stats = copy.deepcopy(owned_warm[-1]['runtime_stats'])
        closed_stats = copy.deepcopy(final_stats)
        closed_stats.update(entries=0)
        if arm['family'] == 'full_tree':
            closed_stats['closed'] = True
        else:
            closed_stats['graph_keys'] = []
        owners.append(dict(mode=mode, arm=arm, length=length, owner_id=owner_id,
            owner_signature=signature, maximum_live_owners=1,
            constructor_and_detach_seconds=.02, prime_request_id=prime['request_id'],
            cold_prime_outer_seconds=prime['outer_seconds'], prime_tokens=128,
            warm_tokens=6*128, warm_outer_seconds_sum=warm_seconds,
            warm_request_ids=[row['request_id'] for row in owned_warm], close_seconds=.001,
            accounted_service_seconds=service_seconds, accounted_service_tokens=7*128,
            accounted_service_tps=7*128/service_seconds,
            owner_lifetime_wall_seconds=service_seconds+1.,
            final_runtime_stats=final_stats, closed_runtime_stats=closed_stats,
            engine_closed=True, hook_restored=True))
    assert logical_index == 538
    assert len(raw_tree) == 112
    sources = {}
    for name in ('rescue/synthetic.py',
                 'exp/specextend_bridgekv/specextend/classic/model_classic.py',
                 'exp/specextend_bridgekv/specextend/shared/modeling_llama_kv_target.py'):
        path = root/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Synthetic contract fixture; no GPU source execution.\n')
        sources[name] = digest(path)
    assets = {'/synthetic/tune.jsonl': DATA_SHA}
    protocol = dict(schema=1, seed=SEED, source_indices=list(DOCS), lengths=list(LENGTHS),
        repeats=2, max_new_tokens=128, arms=descriptors,
        modes=[arm['mode'] for arm in descriptors], expected_counts=EXPECTED,
        expected_requests=538, owner_order=owner_plan, warm_order=warm_plan,
        no_training=True, mapper_executed=False, controller_executed=False,
        data_file_sha256=DATA_SHA, development_file_sha256=DATA_SHA,
        original_corpus_roles={str(doc): 'train' for doc in DOCS},
        document_sha256={str(doc): hashlib.sha256(f'synthetic:{doc}'.encode()).hexdigest() for doc in DOCS},
        configs={arm['mode']: asdict(linear_config(arm)) if arm['family']=='retrieval'
                 else tree_policy(arm) for arm in descriptors},
        source_sha256=sources,
        source_bindings={'rescue.synthetic': dict(path='/synthetic/rescue/synthetic.py',
                            sha256=sources['rescue/synthetic.py'])},
        tree_source_bindings={name: dict(path=path, sha256=sources[path]) for name,path in (
            ('classic.model_classic', 'exp/specextend_bridgekv/specextend/classic/model_classic.py'),
            ('shared.modeling_llama_kv_target', 'exp/specextend_bridgekv/specextend/shared/modeling_llama_kv_target.py'))},
        assets_sha256=assets,
        environment=dict(gpu_name='NVIDIA H100', total_memory=85_000_000_000,
            target_dtype='torch.float32', draft_dtype='torch.bfloat16',
            prefix_dtype='torch.bfloat16', tf32=False),
        flash_attention=dict(version='2.8.3', read_only=True,
            files={'flash_attn_2_cuda': dict(path='/synthetic/flash_attn_2_cuda.so', sha256='f'*64)}),
        graph_max_entries=128, max_live_owners=1, observer=False, profile=False,
        retrieval_retain_query=False, template_prefill_precision='bfloat16')
    dump_json(root/'protocol.json', protocol)
    dump_json(root/'references.json', {f'{doc}:{length}': reference.copy() for doc in DOCS for length in LENGTHS})
    for name, values in (('reference.jsonl',canonical), ('prime.jsonl',primes),
                         ('runtime.jsonl',warm), ('owners.jsonl',owners), ('tree_raw.jsonl',raw_tree)):
        dump_rows(root/name, values)
    frozen = dict(runner_winners(warm, descriptors), source_sha256=sources,
        assets_sha256=assets, protocol_sha256=digest(root/'protocol.json'),
        complete_request_count=538, no_new_confirmation_opened=True)
    dump_json(root/'FROZEN_SELECTION.json', frozen)
    dump_json(root/'COMPLETE.json', dict(passed=True, counts=EXPECTED,
        expected_counts=EXPECTED, logical_requests=538, sources_unchanged=True,
        assets_unchanged=True, dependencies_unchanged=True, all_canonical=True,
        backend_trajectories_equal=True, owner_closed=76,
        frozen_selection_sha256=digest(root/'FROZEN_SELECTION.json'), goal_complete=False))
    return dict(protocol=protocol, canonical=canonical, primes=primes,
        warm=warm, owners=owners, tree_raw=raw_tree,
        expected_winners=expected_winners, winner_factors=factors)


def two_repeats(doc=271, length=4096, mode='retrieval16_k4_graph'):
    return [dict(source_index=doc, length=length, mode=mode, repeat=repeat,
        tokens=tokens, outer_seconds=seconds, accepted=accepted,
        proposed=proposed, blocks=blocks,
        draft_runtime=dict(request=dict(captures=repeat,
            capture_host_seconds=.1*repeat)))
        for repeat, tokens, seconds, accepted, proposed, blocks in
        ((0, 80, 2., 40, 80, 20), (1, 120, 8., 60, 120, 30))]


def metric_grid():
    """All38 entries in both contexts; synthetic scores only."""
    return {str(length): {arm['mode']: dict(documents=3, pooled_tps=10.,
        accounted_tps=5.) for arm in menu()} for length in LENGTHS}


class BaselineSelectionMathTests(unittest.TestCase):
    def test_exact_thirty_retrieval_plus_eight_tree_menu(self):
        from rescue.baseline_selection import arms as runner_arms
        arms = menu()
        self.assertEqual(len(arms), 38)
        self.assertEqual(arms, runner_arms())
        self.assertEqual(len({a['mode'] for a in arms}), 38)
        retrieval = [a for a in arms if a['family'] == 'retrieval']
        tree = [a for a in arms if a['family'] == 'full_tree']
        self.assertEqual(len(retrieval), 30)
        self.assertEqual(len(tree), 8)
        self.assertEqual({(a['top_chunks'], a['fixed_k'], a['backend']) for a in retrieval},
            {(top, k, 'cuda_graph' if backend == 'graph' else backend)
             for top in (16, 32) for k in (1, 2, 4, 8, 12) for backend in BACKENDS})
        self.assertEqual({(a['top_chunks'], a['nodes']) for a in tree},
            {(top, nodes) for top in (16, 32) for nodes in (8, 32, 50, 64)})
        self.assertTrue(all(a['max_depth'] == 10 and a['threshold'] == .7 for a in tree))

    def test_document_cell_pools_tokens_time_and_retains_real_capture_cost(self):
        result = pool_document(two_repeats())
        self.assertEqual(result['tokens'], 200)
        self.assertEqual(result['seconds'], 10.)
        self.assertEqual(result['tps'], 20.)
        self.assertNotEqual(result['tps'], (80/2.+120/8.)/2)
        self.assertEqual(result['accepted'], 100)
        self.assertEqual(result['blocks'], 50)
        self.assertEqual(result['proposed'], 200)
        self.assertEqual(result['captures'], 1)
        self.assertAlmostEqual(result['capture_seconds'], .1)
        tree = two_repeats(mode='tree16_n8')
        for row in tree:
            row['proposed'] = None
        self.assertIsNone(pool_document(tree)['proposed'])

    def test_missing_duplicate_or_cross_document_context_arm_pairs_are_rejected(self):
        original = two_repeats()
        invalid = [original[:1], original+[copy.deepcopy(original[0])],
                   [original[0], copy.deepcopy(original[0])]]
        for field, value in (('source_index', 272), ('length', 8192),
                             ('mode', 'retrieval32_k4_graph')):
            changed = copy.deepcopy(original)
            changed[1][field] = value
            invalid.append(changed)
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                pool_document(rows)

    def test_bootstrap_point_is_global_token_time_ratio_not_mean_document_tps(self):
        cells = [dict(tokens=100, seconds=1.), dict(tokens=200, seconds=10.),
                 dict(tokens=300, seconds=40.)]
        result = pooled_bootstrap(cells)
        self.assertEqual(result['n'], 3)
        self.assertAlmostEqual(result['point'], 600/51)
        self.assertNotAlmostEqual(result['point'], (100.+20.+7.5)/3)
        # Three draws of the same complete document have probability 1/27;
        # the frozen percentile interval therefore reaches these endpoints.
        self.assertEqual(result['ci95'], [7.5, 100.])

    def test_paired_bootstrap_reuses_document_indices_and_recomputes_ratios(self):
        cells = [dict(tokens=128, seconds=1.), dict(tokens=128, seconds=3.),
                 dict(tokens=64, seconds=6.)]
        other = [dict(tokens=128, seconds=2.), dict(tokens=128, seconds=4.),
                 dict(tokens=64, seconds=3.)]
        result = pooled_bootstrap(cells, other=other)
        self.assertEqual(result['n'], 3)
        self.assertAlmostEqual(result['point'], -10.)
        self.assertAlmostEqual(result['ci95'][0], -50.)
        self.assertAlmostEqual(result['ci95'][1], 100.)
        mean_paired_percent = sum(100*((a['tokens']/a['seconds'])/
            (b['tokens']/b['seconds'])-1) for a, b in zip(cells, other))/3
        self.assertNotAlmostEqual(result['point'], mean_paired_percent)
        identical = pooled_bootstrap(cells, other=copy.deepcopy(cells))
        self.assertEqual(identical, dict(n=3, point=0., ci95=[0., 0.]))

    def test_bootstrap_frozen_seed_draws_n1_and_no_cross_context_pseudoreplication(self):
        for length in LENGTHS:
            cells = [pool_document(two_repeats(doc=d, length=length)) for d in DOCS]
            result = pooled_bootstrap(cells)
            self.assertEqual(result['n'], 3)
            self.assertEqual(result, pooled_bootstrap(cells, seed=SEED))
        self.assertEqual(pooled_bootstrap([]), dict(n=0, point=None, ci95=None))
        one = pooled_bootstrap([dict(tokens=128, seconds=4.)])
        self.assertEqual(one, dict(n=1, point=32., ci95=None))
        paired_one = pooled_bootstrap([dict(tokens=128, seconds=4.)],
                                     other=[dict(tokens=128, seconds=8.)])
        self.assertEqual(paired_one, dict(n=1, point=100., ci95=None))
        for cells, kwargs in (([dict(tokens=1, seconds=1.)], dict(draws=200)),
                              ([dict(tokens=1, seconds=float('nan'))], {}),
                              ([dict(tokens=0, seconds=1.)], {}),
                              ([dict(tokens=1, seconds=1.)], dict(other=[]))):
            with self.subTest(cells=cells, kwargs=kwargs), self.assertRaises(ValueError):
                pooled_bootstrap(cells, **kwargs)

    def test_selects_one_winner_per_backend_and_tree_in_each_context(self):
        metrics = metric_grid()
        expected = {}
        for length in LENGTHS:
            chosen = dict(zip(('retrieval/eager', 'retrieval/resident', 'retrieval/graph', 'full_tree'),
                ('retrieval32_k12_eager', 'retrieval16_k8_resident',
                 'retrieval32_k4_graph', 'tree16_n50')))
            for mode in chosen.values():
                metrics[str(length)][mode]['pooled_tps'] = 20.
            # A larger score on an incomplete two-document cell is ineligible.
            metrics[str(length)]['retrieval16_k1_eager'].update(documents=2, pooled_tps=999.)
            expected[str(length)] = chosen
        result = select_winners(metrics, menu())
        self.assertEqual(result['selected'], expected)
        self.assertEqual(sum(len(v) for v in result['selected'].values()), 8)
        self.assertEqual(set(result['overall_family']), {'4096', '8192'})

    def test_only_exact_score_ties_use_lexical_mode_not_isclose(self):
        arms = [a for a in menu() if a['mode'] in
                ('retrieval16_k1_eager', 'retrieval32_k1_eager')]
        metrics = {'4096': {a['mode']: dict(documents=3, pooled_tps=20.,
                                        accounted_tps=10.) for a in arms}}
        result = select_winners(metrics, arms)
        self.assertEqual(result['selected']['4096']['retrieval/eager'], 'retrieval16_k1_eager')
        metrics['4096']['retrieval32_k1_eager']['pooled_tps'] += 1e-12
        self.assertTrue(math.isclose(20., metrics['4096']['retrieval32_k1_eager']['pooled_tps']))
        result = select_winners(metrics, arms)
        self.assertEqual(result['selected']['4096']['retrieval/eager'], 'retrieval32_k1_eager')
        self.assertEqual(result['selected']['8192'], {})

    def test_descriptive_service_winner_never_replaces_primary_warm_winner(self):
        metrics = metric_grid()
        for length in LENGTHS:
            values = metrics[str(length)]
            values['retrieval32_k4_graph']['pooled_tps'] = 30.
            values['retrieval16_k1_eager']['accounted_tps'] = 999.
            values['tree32_n8']['pooled_tps'] = 30.
            values['tree16_n64']['accounted_tps'] = 999.
        result = select_winners(metrics, menu())
        for length in LENGTHS:
            self.assertEqual(result['overall_family'][str(length)],
                             dict(retrieval='retrieval32_k4_graph', full_tree='tree32_n8'))
            self.assertEqual(result['descriptive_service'][str(length)],
                             dict(retrieval='retrieval16_k1_eager', full_tree='tree16_n64'))
            self.assertEqual(result['selected'][str(length)]['retrieval/graph'], 'retrieval32_k4_graph')

    def test_empty_and_truncated_partial_render_seven_figures_and_standalone_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(analyze(root)['status'], 'PARTIAL')
            (root/'runtime.jsonl').write_text('{"unfinished":')
            summary = analyze(root)
            self.assertEqual(summary['status'], 'PARTIAL')
            self.assertTrue(any('Malformed runtime.jsonl' in v for v in summary['problems']))
            output = root/'analysis'
            summary = write_report(root, output)
            names = FIGURES
            self.assertEqual(summary['status'], 'PARTIAL')
            self.assertEqual({p.stem for p in output.glob('*.png')}, names)
            self.assertEqual({p.stem for p in output.glob('*.svg')}, names)
            for name in names:
                self.assertTrue((output/(name+'.png')).read_bytes().startswith(b'\x89PNG'))
                self.assertIn('<svg', (output/(name+'.svg')).read_text())
            text = (output/'REPORT.md').read_text()
            self.assertIn('PARTIAL', text)
            self.assertIn('Exactly equal scores use lexical', text)
            receipt = json.loads((output/'ANALYSIS_COMPLETE.json').read_text())
            self.assertEqual(receipt['status'], 'PARTIAL')
            self.assertTrue((output/'PROVISIONAL_RANKING.json').is_file())
            self.assertFalse((output/'VERIFIED_WINNERS.json').exists())
            for name, wanted in receipt['artifacts_sha256'].items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), wanted)
            rerun = root/'standalone_rerun'
            process = subprocess.run([sys.executable, str(output/'analysis_source.py'),
                str(root), '--output', str(rerun)], cwd=output,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            saved = json.loads((rerun/'summary.json').read_text())
            self.assertEqual(saved['status'], 'PARTIAL')
            self.assertEqual(saved['observed_counts'], summary['observed_counts'])
            self.assertEqual({p.stem for p in rerun.glob('*.png')}, names)
            self.assertTrue((rerun/'PROVISIONAL_RANKING.json').is_file())
            self.assertFalse((rerun/'VERIFIED_WINNERS.json').exists())


class BaselineSelectionFullSchemaTests(unittest.TestCase):
    def test_complete_538_contract_eight_winners_and_document_ratio_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = full_grid_fixture(tmp)
            summary = analyze(tmp)
            self.assertEqual(summary['status'], 'COMPLETE', summary['problems'])
            self.assertEqual(summary['problems'], [])
            self.assertEqual(summary['observed_counts'], dict(reference_requests=6,
                prime_requests=76, timing_requests=456, owners=76))
            self.assertEqual(sum(len(fixture[k]) for k in ('canonical', 'primes', 'warm')), 538)
            self.assertEqual(len(fixture['tree_raw']), 112)
            self.assertEqual(summary['winner_status'], 'VERIFIED_FROZEN')
            self.assertEqual(summary['winners']['selected'], fixture['expected_winners'])
            self.assertEqual(sum(len(v) for v in summary['winners']['selected'].values()), 8)
            self.assertEqual(summary['bootstrap_draws'], 10000)
            self.assertEqual(summary['bootstrap_seed'], SEED)
            for length in LENGTHS:
                context_scale = 1. if length == 4096 else 1.4
                self.assertEqual(len(summary['metrics'][str(length)]), 38)
                for mode, value in summary['metrics'][str(length)].items():
                    self.assertEqual(value['documents'], 3)
                    self.assertEqual(set(value['per_document']), set(map(str, DOCS)))
                    self.assertEqual(value['warm_tokens'], 768)
                for group, mode in fixture['expected_winners'][str(length)].items():
                    factor = fixture['winner_factors'][group]*context_scale
                    value = summary['metrics'][str(length)][mode]
                    # Each document pools two requests before the bootstrap;
                    # the point is the six-request ratio, never mean doc TPS.
                    self.assertAlmostEqual(value['pooled_tps'], 768/(7*4.3*factor))
                    self.assertNotAlmostEqual(value['pooled_tps'], value['mean_document_tps'])
                    self.assertAlmostEqual(value['pooled_tps_ci95'][0], 256/(14*factor))
                    self.assertAlmostEqual(value['pooled_tps_ci95'][1], 256/(7*factor))
                    self.assertAlmostEqual(value['observed_repeat1']['tps'], 384/(5*4.3*factor))
                    if group == 'full_tree':
                        self.assertEqual(value['acceptance_length'], 3.)
                        self.assertEqual(value['acceptance_length_ci95'], [3., 3.])
                        self.assertIsNone(value['acceptance_rate'])
                        self.assertIn('RAW', value['acceptance_scope'])
            for row in fixture['tree_raw']:
                self.assertEqual(row['result']['upstream']['raw_total_generated'], 132)
                self.assertEqual(row['result']['upstream']['generation_overshoot'], 4)

    def test_missing_warm_pair_is_partial_without_imputed_or_verified_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = full_grid_fixture(root)
            victim = next(row for row in fixture['warm']
                if row['mode'] == 'retrieval16_k8_graph' and row['length'] == 4096
                and row['source_index'] == 288 and row['repeat'] == 1)
            dump_rows(root/'runtime.jsonl', [row for row in fixture['warm'] if row is not victim])
            summary = write_report(root, root/'analysis')
            self.assertEqual(summary['status'], 'PARTIAL')
            self.assertEqual(summary['winner_status'], 'PROVISIONAL_ONLY')
            self.assertEqual(summary['observed_counts']['timing_requests'], 455)
            self.assertEqual(summary['usable_documents']['4096'][victim['mode']], [])
            self.assertNotIn(victim['mode'], summary['metrics']['4096'])
            self.assertEqual(summary['metrics']['8192'][victim['mode']]['documents'], 3)
            self.assertTrue(any('456 warm cells incomplete' in p for p in summary['problems']))
            self.assertFalse((root/'analysis'/'VERIFIED_WINNERS.json').exists())
            provisional = json.loads((root/'analysis'/'PROVISIONAL_RANKING.json').read_text())
            self.assertEqual(provisional['status'], 'PROVISIONAL_ONLY')
            self.assertFalse(provisional['authoritative'])

    def test_backend_trajectory_tamper_is_partial_even_when_row_is_internally_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = full_grid_fixture(root)
            victim = next(row for row in fixture['warm']
                if row['mode'] == 'retrieval16_k8_graph' and row['length'] == 4096
                and row['source_index'] == 288 and row['repeat'] == 1)
            # A different valid workset still breaks the same-policy contract;
            # this cannot be explained by backend-only capture/timing changes.
            victim['trace'][0]['workset_tokens'] -= 1
            dump_rows(root/'runtime.jsonl', fixture['warm'])
            summary = analyze(root)
            self.assertEqual(summary['status'], 'PARTIAL')
            self.assertEqual(summary['winner_status'], 'PROVISIONAL_ONLY')
            self.assertTrue(any('Same strategy repeat/backend logical trajectory differs' in p
                                for p in summary['problems']))
            self.assertFalse(any('Actual retrieval working set differs' in p for p in summary['problems']))
            self.assertNotIn(288, summary['usable_documents']['4096'][victim['mode']])

    def test_complete_renders_seven_figures_and_binds_verified_winner_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = full_grid_fixture(root)
            output = root/'analysis'
            summary = write_report(root, output)
            self.assertEqual(summary['status'], 'COMPLETE', summary['problems'])
            self.assertEqual({p.stem for p in output.glob('*.png')}, FIGURES)
            self.assertEqual({p.stem for p in output.glob('*.svg')}, FIGURES)
            for name in FIGURES:
                self.assertTrue((output/(name+'.png')).read_bytes().startswith(b'\x89PNG'))
                self.assertIn('<svg', (output/(name+'.svg')).read_text())
            verified = json.loads((output/'VERIFIED_WINNERS.json').read_text())
            self.assertEqual(verified['status'], 'VERIFIED_FROZEN')
            self.assertTrue(verified['authoritative'])
            self.assertEqual(verified['winners']['selected'], fixture['expected_winners'])
            self.assertFalse((output/'PROVISIONAL_RANKING.json').exists())
            report = (output/'REPORT.md').read_text()
            self.assertIn('RAW accepted/round including overshoot', report)
            self.assertIn('not held-out', report)
            self.assertIn('research goal is not thereby complete', report)
            self.assertIn('Constructor/cold/service views are secondary', report)
            receipt = json.loads((output/'ANALYSIS_COMPLETE.json').read_text())
            self.assertEqual(receipt['status'], 'COMPLETE')
            self.assertTrue(receipt['rendered_inspection_required'])
            for name, wanted in receipt['artifacts_sha256'].items():
                self.assertEqual(digest(output/name), wanted)


if __name__ == '__main__':
    unittest.main()
