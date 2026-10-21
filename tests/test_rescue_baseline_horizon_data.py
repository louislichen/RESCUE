"""Synthetic old-input tests only; never load scientific data/models/outcomes."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from rescue import baseline_horizon_data as data


def row(index, split='train'):
    return dict(source_index=index, split=split,
        document_sha256=hashlib.sha256(f'SYNTHETIC_DOCUMENT_{index}'.encode()).hexdigest(),
        original_tokens=8222, document_ids=[1000 + index] * 8192,
        prefix_ids=list(range(24)), suffix_ids=list(range(24, 30)))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def fixture(root):
    values = {
        data.TRAIN: [row(i) for i in data.TRAIN_INDICES],
        data.OPT_CONFIRMATION: [row(i, 'confirmation') for i in data.OPT_INDICES],
        data.ARCH_CONFIRMATION: [row(i, 'confirmation') for i in data.ARCH_INDICES],
        data.TUNE: [row(i) for i in data.TUNE_INDICES],
    }
    paths = {name: root / name for name in data.COHORTS}
    for name in data.COHORTS:
        write_rows(paths[name], values[name])
    return paths


def hashes(paths):
    return {name: data.sha256(path) for name, path in paths.items()}


def load(paths):
    return data.load_horizon_documents(paths, expected_source_sha256=hashes(paths),
        tokenizer_sha256={'tokenizer.json': 'a' * 64})


def change(paths, name, transform):
    rows = [json.loads(line) for line in paths[name].read_text().splitlines()]
    transform(rows)
    write_rows(paths[name], rows)


def reseal(manifest):
    manifest['receipt_sha256'] = data.json_sha({k: v for k, v in manifest.items()
                                               if k != 'receipt_sha256'})


class HorizonDataTests(unittest.TestCase):
    def test_exact_extraction_matches_real_baseline_context_without_torch_import(self):
        path = Path(__file__).resolve().parents[1] / 'rescue/round_overhead_run.py'
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'context')
        namespace = {'require': data.require}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        for index in data.TRAIN_INDICES + data.OPT_INDICES + (604,):
            for n in data.LENGTHS:
                item = row(index)
                self.assertEqual(data.prompt_ids(item, n), namespace['context'](item, n))

    def test_fixed16_order_exact_context_hash_and_intentional_prime_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(Path(tmp)); before = hashes(paths)
            docs, prime, manifest = load(paths)
            order = data.TRAIN_INDICES + data.OPT_INDICES + (604,)
            self.assertEqual(tuple(docs), order)
            self.assertEqual(docs[271], prime)
            self.assertEqual(manifest['source_indices'], list(order))
            self.assertEqual(len(manifest['input_sha256']), 32)
            self.assertEqual(manifest['prime_input_sha256'], {
                str(n): manifest['input_sha256'][f'271:{n}'] for n in data.LENGTHS})
            self.assertEqual(manifest['selected_documents'][-1]['source_identity'],
                dict(cohort=data.ARCH_CONFIRMATION, row_ordinal=0, source_index=604, split='confirmation',
                     original_corpus_split='validation', original_shard='validation-00000-of-00001.parquet',
                     preparation_source='rescue/module_arch_data.py'))
            self.assertEqual(manifest['selected_documents'][0]['source_identity']['original_corpus_split'], 'train')
            self.assertEqual(manifest['prime']['source_identity']['original_shard'], 'train-00000-of-00002.parquet')
            for index, item in docs.items():
                for n in data.LENGTHS:
                    expected = item['prefix_ids'] + item['document_ids'][:n-30] + item['suffix_ids']
                    self.assertEqual(data.prompt_ids(item, n), expected)
                    self.assertEqual(len(expected), n)
                    self.assertEqual(manifest['input_sha256'][f'{index}:{n}'],
                        hashlib.sha256(json.dumps(expected).encode()).hexdigest())
            self.assertEqual(before, hashes(paths))
            self.assertEqual(manifest['selection_rule']['replacement_allowed'], False)
            self.assertFalse(manifest['scope']['retokenized_raw_text'])

    def test_production_default_hashes_cannot_accept_synthetic_or_replaced_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                data.load_horizon_documents(paths)
            pinned = hashes(paths)
            paths[data.TRAIN].write_text(paths[data.TRAIN].read_text() + '\n')
            with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                data.load_horizon_documents(paths, expected_source_sha256=pinned)

    def test_train_count_order_and_identity_fail_without_replacement(self):
        for alteration in ('missing', 'extra', 'reorder', 'replacement'):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as tmp:
                paths = fixture(Path(tmp))
                def alter(rows):
                    if alteration == 'missing': rows.pop()
                    if alteration == 'extra': rows.append(row(999))
                    if alteration == 'reorder': rows.reverse()
                    if alteration == 'replacement': rows[0] = row(999)
                change(paths, data.TRAIN, alter)
                with self.assertRaisesRegex(RuntimeError, 'count/order/source'):
                    load(paths)

    def test_supplemental_and_tune_indices_are_exact_and_arch_is_first604(self):
        for cohort in (data.OPT_CONFIRMATION, data.ARCH_CONFIRMATION, data.TUNE):
            with self.subTest(cohort=cohort), tempfile.TemporaryDirectory() as tmp:
                paths = fixture(Path(tmp))
                change(paths, cohort, lambda rows: rows.reverse())
                with self.assertRaisesRegex(RuntimeError, 'count/order/source'):
                    load(paths)

    def test_paths_are_only_four_explicit_absolute_old_jsonl_cohorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(Path(tmp)); pinned = hashes(paths)
            bad = dict(paths); bad[data.TRAIN] = Path(tmp) / 'runtime.jsonl'
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('must not open')):
                with self.assertRaisesRegex(RuntimeError, 'locked cohort'):
                    data.load_horizon_documents(bad, expected_source_sha256=pinned)
            bad = dict(paths); bad[data.TRAIN] = Path(data.TRAIN)
            with self.assertRaisesRegex(RuntimeError, 'absolute'):
                data.load_horizon_documents(bad, expected_source_sha256=pinned)
            bad = dict(paths); bad['fresh/validation.parquet'] = Path(tmp) / 'fresh.parquet'
            with self.assertRaisesRegex(RuntimeError, 'four explicit'):
                data.load_horizon_documents(bad, expected_source_sha256=pinned)

    def test_wrong_split_and_outcome_records_are_rejected(self):
        for corruption in ('split', 'outcome'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                paths = fixture(Path(tmp))
                def alter(rows):
                    if corruption == 'split': rows[0]['split'] = 'test'
                    else: rows[0]['auxiliary'] = dict(generated=[123])
                change(paths, data.TRAIN, alter)
                with self.assertRaisesRegex(RuntimeError, 'split|outcome'):
                    load(paths)

    def test_natural_and_stored_lengths_are_not_satisfied_by_padding(self):
        for corruption in ('natural', 'stored', 'longer_than_natural'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                paths = fixture(Path(tmp))
                def alter(rows):
                    if corruption == 'natural': rows[0]['original_tokens'] = 8191
                    if corruption == 'stored': rows[0]['document_ids'] = rows[0]['document_ids'][:8191]
                    if corruption == 'longer_than_natural': rows[0]['original_tokens'] = 8192; rows[0]['document_ids'].append(999)
                change(paths, data.TRAIN, alter)
                with self.assertRaisesRegex(RuntimeError, 'too short|exceeds natural'):
                    load(paths)

    def test_invalid_token_types_and_negative_ids_are_rejected(self):
        for value in (True, -1, 1.5):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                paths = fixture(Path(tmp))
                change(paths, data.TRAIN, lambda rows: rows[0]['document_ids'].__setitem__(0, value))
                with self.assertRaisesRegex(RuntimeError, 'token arrays'):
                    load(paths)

    def test_prompt_template_and_cold271_body_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(Path(tmp))
            change(paths, data.OPT_CONFIRMATION, lambda rows: rows[0]['suffix_ids'].__setitem__(0, 999))
            with self.assertRaisesRegex(RuntimeError, 'prompt template'):
                load(paths)
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(Path(tmp))
            change(paths, data.TUNE, lambda rows: rows[0]['document_ids'].__setitem__(-1, 999))
            with self.assertRaisesRegex(RuntimeError, 'Cold prime271'):
                load(paths)

    def test_duplicate_document_body_and_context_prefix_fail_closed(self):
        for corruption in ('document_sha', 'stored_body', '4k_prefix'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                paths = fixture(Path(tmp))
                def alter(rows):
                    if corruption == 'document_sha': rows[1]['document_sha256'] = rows[0]['document_sha256']
                    if corruption == 'stored_body': rows[1]['document_ids'] = list(rows[0]['document_ids'])
                    if corruption == '4k_prefix': rows[1]['document_ids'][:4096] = rows[0]['document_ids'][:4096]
                change(paths, data.TRAIN, alter)
                with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
                    load(paths)

    def test_pure_validator_performs_no_file_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(Path(tmp)); pinned = hashes(paths)
            docs, prime, manifest = load(paths)
            with patch.object(Path, 'open', side_effect=AssertionError('no file access')), \
                 patch.object(Path, 'read_bytes', side_effect=AssertionError('no file access')), \
                 patch.object(Path, 'read_text', side_effect=AssertionError('no file access')):
                proof = data.validate_horizon_manifest(manifest, docs, prime, expected_source_sha256=pinned)
            self.assertEqual(proof['status'], 'VERIFIED_OLD_DATA_ONLY')
            self.assertEqual(proof['input_count'], 32)

    def test_receipt_and_derived_input_tamper_rejected_even_with_new_self_digest(self):
        for corruption in ('input', 'cohort', 'tokenizer', 'scope', 'prime_overlap'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                paths = fixture(Path(tmp)); pinned = hashes(paths)
                docs, prime, original = load(paths); manifest = copy.deepcopy(original)
                if corruption == 'input': manifest['input_sha256']['211:4096'] = 'b' * 64
                if corruption == 'cohort': manifest['selected_documents'][0]['source_identity']['cohort'] = data.OPT_CONFIRMATION
                if corruption == 'tokenizer': manifest['tokenizer_inventory_bound'] = False
                if corruption == 'scope': manifest['scope']['fresh_confirmation_opened'] = True
                if corruption == 'prime_overlap': manifest['prime_in_warm_documents'] = False
                reseal(manifest)
                with self.assertRaises(RuntimeError):
                    data.validate_horizon_manifest(manifest, docs, prime, expected_source_sha256=pinned)
            self.assertNotEqual(manifest, original)

    def test_changed_document_body_or_unsealed_receipt_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(Path(tmp)); pinned = hashes(paths)
            docs, prime, manifest = load(paths)
            docs[211]['document_ids'][-1] = 999
            with self.assertRaisesRegex(RuntimeError, 'stored token receipt'):
                data.validate_horizon_manifest(manifest, docs, prime, expected_source_sha256=pinned)
            docs, prime, manifest = load(paths)
            manifest['frozen_at_utc'] = 'changed'
            with self.assertRaisesRegex(RuntimeError, 'receipt SHA256'):
                data.validate_horizon_manifest(manifest, docs, prime, expected_source_sha256=pinned)

    def test_portable_four_source_copies_rebuild_receipt_and_detect_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); paths = fixture(root / 'original'); pinned = hashes(paths)
            _, _, manifest = load(paths)
            copies = {name: root / 'results/data' / name for name in data.COHORTS}
            for name, target in copies.items():
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(paths[name], target)
            proof = data.verify_horizon_sources(manifest, copies, expected_source_sha256=pinned)
            self.assertEqual(proof['source_files_verified'], 4)
            self.assertNotEqual(proof['verified_source_paths'][data.TRAIN], manifest['metadata_sources'][data.TRAIN]['path'])
            copies[data.TRAIN].write_text(copies[data.TRAIN].read_text() + '\n')
            with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                data.verify_horizon_sources(manifest, copies, expected_source_sha256=pinned)

    def test_mutation_during_read_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(Path(tmp)); pinned = hashes(paths)
            with patch.object(data, 'sha256', return_value='f' * 64):
                with self.assertRaisesRegex(RuntimeError, 'changed during'):
                    data.load_horizon_documents(paths, expected_source_sha256=pinned)


if __name__ == '__main__':
    unittest.main()
