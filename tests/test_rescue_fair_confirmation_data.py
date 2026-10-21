"""Synthetic stdlib-only gate tests; no real datasets, models or GPU requests."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from rescue import fair_confirmation_data as data


PROJECT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT / 'exp/draft_graph/fair_confirmation_20261020_preregistration.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def synthetic_environment(root):
    """Fabricated receipt chain only; fake PNG bytes are never real evidence."""
    old = root / 'oldrepo'; metadata = []
    for i, name in enumerate(data.OLD_COHORT_FILES):
        path = old / name; path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(source_index=i, split='test' if name.endswith('/test.jsonl') else 'train',
            document_sha256=hashlib.sha256(('old_doc' + str(i)).encode()).hexdigest(),
            original_tokens=16384, document_ids=[42] * 16384, prefix_ids=[1], suffix_ids=[2])
        path.write_text(json.dumps(row) + '\n'); metadata.append(path)
    raw = root / 'raw/validation-00000-of-00001.parquet'
    raw.parent.mkdir(); raw.write_bytes(b'SYNTHETIC_NOT_A_REAL_PARQUET')
    raw_manifest = old / 'exp/module_arch/data_v1_20261018/manifest.json'
    write(raw_manifest, dict(source_sha256=data.sha256(raw)))

    base = root / 'baseline'; results = base / 'results'; analysis = results / 'analysis_final'
    analysis.mkdir(parents=True)
    tree_names = list(data.TREE_REQUIRED) + [f'shared/synthetic_{i}.py' for i in range(27-len(data.TREE_REQUIRED))]
    inventory = {}
    for name in tree_names:
        relative = data.TREE_PREFIX + name
        path = results / 'source' / relative
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# synthetic source only\n')
        inventory[relative] = data.sha256(path)
    tune = old / 'exp/module_opt/data_v1_20261018/tune.jsonl'
    protocol = dict(source_sha256=inventory, data_file_sha256=data.sha256(tune))
    write(results / 'protocol.json', protocol)
    selected, contexts = {}, {}
    for length in data.LENGTHS:
        selected[str(length)], contexts[str(length)] = {}, dict(groups={})
        for group in data.GROUPS:
            if group == 'full_tree':
                mode = 'tree16_n8'
                arm = dict(mode=mode, family='full_tree', top_chunks=16, nodes=8,
                    backend='upstream_flash_hybrid', max_depth=10, threshold=.7)
            else:
                backend = group.split('/')[1]
                top, k = ((16, 1) if backend == 'eager' else (16, 2) if backend == 'resident' else (32, 4))
                mode = f'retrieval{top}_k{k}_{backend}'
                arm = dict(mode=mode, family='retrieval', top_chunks=top, fixed_k=k,
                    backend='cuda_graph' if backend == 'graph' else backend)
            selected[str(length)][group] = mode
            contexts[str(length)]['groups'][group] = dict(mode=mode, arm=arm)
    frozen = dict(group_candidates=8, complete_request_count=538,
        no_new_confirmation_opened=True, protocol_sha256=data.sha256(results / 'protocol.json'),
        per_context=contexts)
    write(results / 'FROZEN_SELECTION.json', frozen)
    complete = dict(passed=True, counts=data.BASELINE_COUNTS, logical_requests=538, owner_closed=76,
        goal_complete=False, sources_unchanged=True, assets_unchanged=True,
        dependencies_unchanged=True, all_canonical=True, backend_trajectories_equal=True,
        frozen_selection_sha256=data.sha256(results / 'FROZEN_SELECTION.json'))
    write(results / 'COMPLETE.json', complete)
    # These explicit synthetic files exercise coverage, not a scientific audit.
    for name in ('reference.jsonl', 'prime.jsonl', 'runtime.jsonl', 'owners.jsonl', 'tree_raw.jsonl'):
        (results / name).write_text('{"synthetic_fixture_only":true}\n')
    write(analysis / 'summary.json', dict(status='COMPLETE', problems=[], winner_status='VERIFIED_FROZEN', observed_counts=data.BASELINE_COUNTS))
    write(analysis / 'VERIFIED_WINNERS.json', dict(status='VERIFIED_FROZEN', authoritative=True, winners=dict(selected=selected)))
    (analysis / 'REPORT.md').write_text('SYNTHETIC TEST FIXTURE, NOT AN EXPERIMENT\n')
    (analysis / 'analysis_source.py').write_text('# synthetic analysis only\n')
    for name in ('quality', 'tps', 'captures', 'capture_time', 'cold', 'service', 'selected_repeat1'):
        for extension in ('.png', '.svg'):
            (analysis / (name + extension)).write_bytes(b'SYNTHETIC NON-RENDERED TEST FIXTURE')
    write(analysis / 'ANALYSIS_COMPLETE.json', dict(status='COMPLETE', rendered_inspection_required=True,
        artifacts_sha256={p.name:data.sha256(p) for p in analysis.iterdir() if p.is_file()}))
    manifest = base / 'FINAL_ARTIFACTS_MANIFEST.json'
    write(base / 'PIPELINE_EXIT.json', dict(exit_code=0, scope='synthetic completed process only'))
    file_hashes = {str(p.relative_to(base)):data.sha256(p) for p in results.rglob('*') if p.is_file()}
    file_hashes['PIPELINE_EXIT.json'] = data.sha256(base / 'PIPELINE_EXIT.json')
    write(manifest, dict(file_sha256=file_hashes))
    verified = base / 'VERIFIED_COMPLETE.json'
    write(verified, dict(status='VERIFIED_COMPLETE', passed=True, raw_independent_audit_passed=True,
        rendered_figures_inspected=True, strict_analysis_complete=True, all538_requests_verified=True,
        all76_owners_verified=True, eight_frozen_choices_verified=True,
        baseline_complete_sha256=data.sha256(results / 'COMPLETE.json'),
        frozen_selection_sha256=data.sha256(results / 'FROZEN_SELECTION.json'),
        analysis_summary_sha256=data.sha256(analysis / 'summary.json'),
        verified_winners_sha256=data.sha256(analysis / 'VERIFIED_WINNERS.json'),
        final_artifacts_manifest_sha256=data.sha256(manifest)))
    baseline_args = dict(baseline_results=results, analysis_dir=analysis,
                         verified_complete=verified, final_manifest=manifest)
    proof = data.validate_baseline_freeze(**baseline_args)
    run = root / 'new_run'; source = run / 'after'
    for name in data.REQUIRED_NEW_SOURCE:
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('_preregistration.json'): shutil.copyfile(TEMPLATE, path)
        else: path.write_text('# synthetic final source only\n')
    for name in ('exp/attention_horizon/selection.py', 'exp/selective_kv/bridge.py'):
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# synthetic dependency\n')
    for name in inventory:
        target = source / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(results / 'source' / name, target)
    tokenizer = root / 'tokenizer'; tokenizer.mkdir()
    write(tokenizer / 'tokenizer_config.json', dict(synthetic=True))
    write(tokenizer / 'tokenizer.json', dict(synthetic=True))
    final_protocol = run / 'final_protocol.json'
    write(final_protocol, data.make_final_protocol(TEMPLATE, proof))
    gate_args = dict(**baseline_args, metadata_paths=metadata, raw_source_manifest=raw_manifest,
        raw_source=raw, tokenizer_root=tokenizer, source_root=source, final_protocol=final_protocol)
    return dict(root=root, old=old, results=results, analysis=analysis, manifest=manifest,
        verified=verified, baseline_args=baseline_args, proof=proof, raw=raw,
        metadata=metadata, source=source, final_protocol=final_protocol,
        gate_args=gate_args, gate_root=run / 'data_gate')


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == 'short': return [500] * 16383
        if text == 'duplicate-prefix': return [42] * 16384
        return [1000 + int(text.removeprefix('eligible'))] * 16384


def candidates(count=16):
    return [(900, 'old_doc0'), (901, 'short'), (902, 'duplicate-prefix')] + [
        (903+i, 'eligible'+str(i)) for i in range(count)]


class BaselineAndDataGateTests(unittest.TestCase):
    def test_baseline_requires_independent_audit_not_analysis_inspection_required_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = synthetic_environment(Path(tmp)); proof = env['proof']
            self.assertEqual(proof['extra_eager_reference_count'], 4)
            self.assertEqual(proof['scientific_requests'], 438)
            verified = data.read_json(env['verified']); verified['rendered_figures_inspected'] = False
            write(env['verified'], verified)
            with self.assertRaisesRegex(RuntimeError, 'actual seven-figure'):
                data.validate_baseline_freeze(**env['baseline_args'])

    def test_manifest_binds_all_raw_source_figures_and_rejects_changed_bytes(self):
        for target in ('results/runtime.jsonl', 'results/source/'+data.TREE_PREFIX+'shared/opt_tree.py',
                       'results/analysis_final/tps.png'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                env = synthetic_environment(Path(tmp))
                (env['manifest'].parent / target).write_text('changed')
                with self.assertRaises(RuntimeError): data.validate_baseline_freeze(**env['baseline_args'])

    def test_successful_request_receipts_without_exit0_cannot_authorize_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = synthetic_environment(Path(tmp))
            write(env['manifest'].parent/'PIPELINE_EXIT.json', dict(exit_code=1))
            with self.assertRaisesRegex(RuntimeError, 'pipeline must have exited'):
                data.validate_baseline_freeze(**env['baseline_args'])

    def test_matching_geometry_references_deduplicate_without_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            choices = copy.deepcopy(synthetic_environment(Path(tmp))['proof']['choices'])
            for length in choices:
                for group in ('retrieval/resident', 'retrieval/graph'):
                    choices[length][group].update(top_chunks=16, fixed_k=1)
            self.assertEqual(data.extra_eager_geometries(choices), [])
            choices['4096']['retrieval/graph'].update(top_chunks=32, fixed_k=12)
            self.assertEqual(data.extra_eager_geometries(choices), [dict(length=4096, top_chunks=32, fixed_k=12)])

    def test_freeze_reads_only_old_metadata_and_never_opens_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = synthetic_environment(Path(tmp)); original = data.sha256
            def guarded(path):
                self.assertNotEqual(Path(path).resolve(), env['raw'].resolve(), 'Candidate opened during freeze')
                return original(path)
            with patch.object(data, 'sha256', side_effect=guarded):
                gate = data.freeze_gate(env['gate_root'], **env['gate_args'])
            self.assertFalse(gate['candidate_content_accessed'])
            self.assertEqual(len(data.read_json(env['gate_root']/'EXCLUSIONS.json')['metadata_sources']), 11)
            self.assertFalse((env['gate_root']/'data').exists())
            with self.assertRaisesRegex(RuntimeError, 'Fresh gate'):
                data.freeze_gate(env['gate_root'], **env['gate_args'])

    def test_missing_launcher_tree_or_changed_statistics_cannot_open_data(self):
        for corruption in ('launcher', 'tree', 'statistics', 'prime', 'order'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                env = synthetic_environment(Path(tmp))
                if corruption == 'launcher':
                    (env['source']/data.REQUIRED_NEW_SOURCE[-1]).unlink()
                elif corruption == 'tree':
                    (env['source']/data.TREE_PREFIX/'shared/opt_tree.py').unlink()
                elif corruption == 'statistics':
                    protocol = data.read_json(env['final_protocol'])
                    protocol['statistics']['simultaneous_lower_quantile'] = .025
                    write(env['final_protocol'], protocol)
                else:
                    protocol = data.read_json(env['final_protocol'])
                    protocol['execution']['first_prime' if corruption == 'prime' else 'order'] = 'new-doc prime or interleaved timing is not this protocol'
                    write(env['final_protocol'], protocol)
                with patch.object(data, '_raw_rows', side_effect=AssertionError('raw opened')) as reader:
                    with self.assertRaises(RuntimeError): data.freeze_gate(env['gate_root'], **env['gate_args'])
                    reader.assert_not_called()
                self.assertTrue((env['gate_root']/'GATE_FAILURE.json').is_file())
                self.assertFalse((env['gate_root']/'GATE_SOURCE_FREEZE.json').exists())

    def test_outcome_records_or_incomplete_old_cohorts_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = synthetic_environment(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, 'eleven'):
                data.exclusion_receipt(env['metadata'][:-1])
            path = env['metadata'][0]; row = json.loads(path.read_text()); row['generated'] = [1,2]
            path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(RuntimeError, 'Runtime/outcome'):
                data.exclusion_receipt(env['metadata'])

    def test_prepare_selects_first16_only_by_index_length_and_exclusion_then_verifies_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = synthetic_environment(Path(tmp)); data.freeze_gate(env['gate_root'], **env['gate_args'])
            gate_path = env['gate_root']/'GATE_SOURCE_FREEZE.json'
            with patch.object(data, '_load_tokenizer', return_value=Tokenizer()), \
                 patch.object(data, '_raw_rows', return_value=iter(candidates())):
                manifest = data.prepare_confirmation(gate_path)
            self.assertEqual([r['source_index'] for r in manifest['selected']], list(range(903,919)))
            self.assertEqual([r['reason'] for r in manifest['skipped']], [
                'used_reserved_or_selected_document_sha256', 'natural_tokens_below16384',
                'used_reserved_or_selected_input_prefix_sha256'])
            self.assertEqual(manifest['status'], 'PREPARED_NOT_EXECUTED')
            self.assertEqual(manifest['selection_freeze_sha256'], env['proof']['selection_freeze_sha256'])
            copied = Path(tmp)/'portable'; shutil.copytree(env['gate_root']/'data', copied)
            with patch.object(data, 'validate_baseline_freeze', side_effect=AssertionError('external gate read')), \
                 patch.object(data, 'validate_source_freeze', side_effect=AssertionError('external source read')):
                self.assertEqual(data.verify_dataset_manifest(copied), manifest)
            with self.assertRaisesRegex(RuntimeError, 'Fresh confirmation'):
                data.prepare_confirmation(gate_path)

    def test_changed_source_or_baseline_rejects_before_first_candidate_read(self):
        for corruption in ('source', 'baseline'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                env = synthetic_environment(Path(tmp)); data.freeze_gate(env['gate_root'], **env['gate_args'])
                path = env['source']/data.REQUIRED_NEW_SOURCE[1] if corruption == 'source' else env['analysis']/'tps.svg'
                path.write_text('changed after freeze')
                with patch.object(data, '_load_tokenizer', side_effect=AssertionError('tokenizer loaded')), \
                     patch.object(data, '_raw_rows', side_effect=AssertionError('raw opened')) as reader:
                    with self.assertRaises(RuntimeError): data.prepare_confirmation(env['gate_root']/'GATE_SOURCE_FREEZE.json')
                    reader.assert_not_called()
                self.assertFalse((env['gate_root']/'data').exists())

    def test_short_cohort_failure_is_preserved_and_never_resampled(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = synthetic_environment(Path(tmp)); data.freeze_gate(env['gate_root'], **env['gate_args'])
            with patch.object(data, '_load_tokenizer', return_value=Tokenizer()), \
                 patch.object(data, '_raw_rows', return_value=iter(candidates(15))):
                with self.assertRaisesRegex(RuntimeError, 'Fewer than16'):
                    data.prepare_confirmation(env['gate_root']/'GATE_SOURCE_FREEZE.json')
            root = env['gate_root']/'data'
            self.assertTrue((root/'DATA_PREP_FAILURE.json').exists())
            self.assertFalse((root/'confirmation.jsonl').exists())
            self.assertFalse((root/'DATASET_MANIFEST.json').exists())
            with self.assertRaisesRegex(RuntimeError, 'Fresh confirmation'):
                data.prepare_confirmation(env['gate_root']/'GATE_SOURCE_FREEZE.json')

    def test_prepared_token_or_receipt_mutation_is_detected_without_outcome_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = synthetic_environment(Path(tmp)); data.freeze_gate(env['gate_root'], **env['gate_args'])
            with patch.object(data, '_load_tokenizer', return_value=Tokenizer()), \
                 patch.object(data, '_raw_rows', return_value=iter(candidates())):
                data.prepare_confirmation(env['gate_root']/'GATE_SOURCE_FREEZE.json')
            root = env['gate_root']/'data'; target = root/'confirmation.jsonl'
            target.write_text(target.read_text().replace('1000', '9999', 1))
            with self.assertRaisesRegex(RuntimeError, 'dataset changed'):
                data.verify_dataset_manifest(root)

    def test_scan_hole_short_or_nonduplicate_skip_cannot_fake_first16(self):
        for corruption in ('hole', 'short', 'nonduplicate', 'only8k'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                env = synthetic_environment(Path(tmp)); data.freeze_gate(env['gate_root'], **env['gate_args'])
                with patch.object(data, '_load_tokenizer', return_value=Tokenizer()), \
                     patch.object(data, '_raw_rows', return_value=iter(candidates())):
                    data.prepare_confirmation(env['gate_root']/'GATE_SOURCE_FREEZE.json')
                root = env['gate_root']/'data'; manifest = data.read_json(root/'DATASET_MANIFEST.json')
                if corruption == 'hole': manifest['skipped'].pop(0)
                elif corruption == 'short': manifest['skipped'][1]['original_tokens'] = 16384
                elif corruption == 'nonduplicate': manifest['skipped'][0]['document_sha256'] = 'f'*64
                else:
                    rows = [json.loads(line) for line in (root/'confirmation.jsonl').read_text().splitlines()]
                    rows[0]['document_ids'] = rows[0]['document_ids'][:8192]
                    (root/'confirmation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    manifest['dataset_sha256'] = data.sha256(root/'confirmation.jsonl')
                write(root/'DATASET_MANIFEST.json', manifest)
                with self.assertRaises(RuntimeError): data.verify_dataset_manifest(root)


if __name__ == '__main__':
    unittest.main()
