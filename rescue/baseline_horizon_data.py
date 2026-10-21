"""Frozen, old-tokenized-data-only inputs for baseline horizon development.

This module imports neither Torch nor a tokenizer and never opens raw text,
Parquet, model outputs, or a prospective confirmation cohort.  The extraction
is deliberately identical to ``round_overhead_run.context``.  Document text
SHA values are inherited metadata, not claims of independently rehashing text.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


LENGTHS = (4096, 8192)
MIN_NATURAL_TOKENS = 8192
TRAIN = 'exp/long_system/data/train.jsonl'
OPT_CONFIRMATION = 'exp/module_opt/data_v1_20261018/confirmation.jsonl'
ARCH_CONFIRMATION = 'exp/module_arch/data_v1_20261018/confirmation.jsonl'
TUNE = 'exp/module_opt/data_v1_20261018/tune.jsonl'
COHORTS = (TRAIN, OPT_CONFIRMATION, ARCH_CONFIRMATION, TUNE)
TRAIN_INDICES = (211, 213, 220, 226, 233, 236, 246, 248, 253, 271, 272, 288)
OPT_INDICES = (303, 306, 327)
ARCH_INDICES = (604, 613, 614)
TUNE_INDICES = (271, 272, 288)
SOURCE_SHA256 = {
    TRAIN: 'ae1e3f58a663dd505ddc9d1f1fdb87af55cad1d967c6b771f696bcbbadce7083',
    OPT_CONFIRMATION: '03eb1b126b424cc99b3586fb42594430ed74706f3ade587dc7102abfa5df90bb',
    ARCH_CONFIRMATION: '007fd9247bf8d77dc53eef51469c84c8a66d29bb2d9577815349c4601ecb29f9',
    TUNE: '85b2eb1a81d1617b2d158c069c420c452459a8bdb90da128f257220cc27ded09',
}
CORPUS_ORIGINS = {
    TRAIN: dict(original_corpus_split='train', original_shard='train-00000-of-00002.parquet',
                preparation_source='rescue/long_data.py'),
    TUNE: dict(original_corpus_split='train', original_shard='train-00000-of-00002.parquet',
               preparation_source='rescue/module_opt_data.py -> rescue/long_data.py'),
    OPT_CONFIRMATION: dict(original_corpus_split='validation', original_shard='validation-00000-of-00001.parquet',
                          preparation_source='rescue/module_opt_data.py'),
    ARCH_CONFIRMATION: dict(original_corpus_split='validation', original_shard='validation-00000-of-00001.parquet',
                           preparation_source='rescue/module_arch_data.py'),
}
RULE = {
    'status': 'OLD_DATA_ONLY_DEVELOPMENT_NOT_HELD_OUT',
    'train': 'All twelve rows, in locked existing file order',
    'module_opt': 'All three rows303/306/327, in locked file order',
    'module_arch': 'Only first row604; never substitute613/614',
    'prime': 'Existing tune271; identical body and prompt to warm document271',
    'contexts': list(LENGTHS),
    'minimum_natural_tokens': MIN_NATURAL_TOKENS,
    'replacement_allowed': False,
    'selection_uses_outcomes_or_eos': False,
}
SCOPE = {
    'retokenized_raw_text': False,
    'fresh_confirmation_opened': False,
    'runtime_or_paused_outcomes_read': False,
    'model_generation_performed': False,
    'training_performed': False,
    'document_text_sha256': 'Inherited old metadata; raw text is not reopened',
    'original_corpus_identity': 'Split/shard inherited from the named old preparation source; caller freezes that source, no raw file reopened',
    'tokenizer': 'Pretokenized old IDs; caller separately binds the actual tokenizer inventory',
    'cohort': 'Twelve historical train documents plus four repeatedly reused development documents',
    'prime_overlap': '271 is one of the sixteen warm documents, intentionally primed for every configuration',
}
OUTCOME_FIELDS = frozenset((
    'generated', 'output_token_ids', 'request_id', 'outer_seconds',
    'total_seconds', 'total_tps', 'accepted', 'proposed', 'trace',
    'performance_eligible', 'runtime_stats', 'capture_host_seconds',
    'output_agrees', 'execution_index', 'owner_id',
))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def input_sha(ids):
    """Match the original baseline's json.dumps(ids) byte representation."""
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def prompt_ids(row, length):
    require(length in LENGTHS, 'Only the frozen4K/8K contexts are permitted')
    arrays = [row.get(name) for name in ('prefix_ids', 'document_ids', 'suffix_ids')]
    require(all(isinstance(a, list) and all(type(t) is int and t >= 0 for t in a)
                for a in arrays), 'Invalid pretokenized token arrays')
    prefix, document, suffix = arrays
    count = length - len(prefix) - len(suffix)
    require(0 < count <= len(document), 'Natural document cannot supply exact context')
    result = prefix + document[:count] + suffix
    require(len(result) == length, 'Context length is not exact')
    return result


def _reject_outcomes(value):
    if isinstance(value, dict):
        require(not OUTCOME_FIELDS.intersection(value), 'Runtime/outcome fields are forbidden')
        for child in value.values():
            _reject_outcomes(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                _reject_outcomes(child)


def _validate_row(row, split):
    require(isinstance(row, dict), 'Old metadata rows must be objects')
    _reject_outcomes(row)
    require(type(row.get('source_index')) is int and row['source_index'] >= 0 and
            row.get('split') == split, 'Wrong source index/split in old cohort')
    require(valid_sha(row.get('document_sha256')), 'Missing inherited document SHA256')
    require(type(row.get('original_tokens')) is int and
            row['original_tokens'] >= MIN_NATURAL_TOKENS, 'Natural document is too short')
    prompt_ids(row, LENGTHS[-1])
    require(row['prefix_ids'] and row['suffix_ids'], 'Missing common prompt template')
    require(MIN_NATURAL_TOKENS <= len(row['document_ids']) <= row['original_tokens'],
            'Stored document is too short or exceeds natural length')


def _expected(expected_source_sha256, expected_train_indices):
    hashes = dict(SOURCE_SHA256 if expected_source_sha256 is None else expected_source_sha256)
    indices = tuple(TRAIN_INDICES if expected_train_indices is None else expected_train_indices)
    require(set(hashes) == set(COHORTS) and all(valid_sha(v) for v in hashes.values()),
            'Exactly four explicitly pinned old cohort SHA256 values are required')
    require(len(indices) == 12 and len(set(indices)) == 12 and
            all(type(i) is int and i >= 0 for i in indices), 'Exactly twelve pinned train indices required')
    require(not set(indices).intersection(OPT_INDICES + (ARCH_INDICES[0],)),
            'Train and supplemental source indices overlap')
    require(271 in indices, 'Warm train271 must be present to disclose prime overlap')
    return hashes, indices


def _source_rows(metadata_paths, hashes, train_indices):
    require(isinstance(metadata_paths, dict) and set(metadata_paths) == set(COHORTS),
            'Only the four explicit old metadata paths are permitted')
    result, sources = {}, {}
    expected = {TRAIN: train_indices, OPT_CONFIRMATION: OPT_INDICES,
                ARCH_CONFIRMATION: ARCH_INDICES, TUNE: TUNE_INDICES}
    for name in COHORTS:
        supplied = Path(metadata_paths[name])
        require(supplied.is_absolute(), 'Metadata paths must be explicit absolute paths')
        path = supplied.resolve()
        require(path.as_posix().endswith('/' + name) and path.is_file(),
                'Old metadata path does not match its locked cohort')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == hashes[name], 'Old source file SHA256 mismatch: ' + name)
        require(raw.endswith(b'\n'), 'Old JSONL must end at a complete newline')
        lines = raw.decode('utf-8').splitlines()
        require(all(line.strip() for line in lines), 'Blank old JSONL rows are not permitted')
        rows = [json.loads(line) for line in lines]
        require(tuple(r.get('source_index') if isinstance(r, dict) else None for r in rows) == expected[name],
                'Locked cohort count/order/source indices changed: ' + name)
        split = 'train' if name in (TRAIN, TUNE) else 'confirmation'
        for row in rows:
            _validate_row(row, split)
        require(sha256(path) == hashes[name], 'Old source changed during its read')
        result[name] = rows
        sources[name] = dict(path=str(path), sha256=hashes[name], rows=len(rows),
                             source_indices=list(expected[name]), split=split)
    return result, sources


def _detail(row, cohort, ordinal):
    return dict(source_identity=dict(cohort=cohort, row_ordinal=ordinal,
                                      source_index=row['source_index'], split=row['split'],
                                      **CORPUS_ORIGINS[cohort]),
                document_sha256=row['document_sha256'], original_tokens=row['original_tokens'],
                stored_document_tokens=len(row['document_ids']),
                stored_document_ids_sha256=input_sha(row['document_ids']),
                input_sha256={str(n): input_sha(prompt_ids(row, n)) for n in LENGTHS})


def _selection(rows, train_indices):
    selected = [(row, TRAIN, i) for i, row in enumerate(rows[TRAIN])]
    selected += [(row, OPT_CONFIRMATION, i) for i, row in enumerate(rows[OPT_CONFIRMATION])]
    selected.append((rows[ARCH_CONFIRMATION][0], ARCH_CONFIRMATION, 0))
    documents = {row['source_index']: copy.deepcopy(row) for row, _, _ in selected}
    prime = copy.deepcopy(rows[TUNE][0])
    require(tuple(documents) == train_indices + OPT_INDICES + (604,) and len(documents) == 16,
            'The fixed sixteen development identities changed')
    template = dict(prefix_ids=prime['prefix_ids'], suffix_ids=prime['suffix_ids'])
    for row in documents.values():
        require(dict(prefix_ids=row['prefix_ids'], suffix_ids=row['suffix_ids']) == template,
                'Old cohorts do not share the exact tune271 prompt template')
    require(all(documents[271][key] == prime[key] for key in (
        'document_sha256', 'document_ids', 'prefix_ids', 'suffix_ids', 'original_tokens')),
        'Cold prime271 differs from warm document271')
    require(len({r['document_sha256'] for r in documents.values()}) == 16,
            'Duplicate document SHA256 among the sixteen inputs')
    require(len({input_sha(r['document_ids']) for r in documents.values()}) == 16,
            'Duplicate stored document bodies among the sixteen inputs')
    for n in LENGTHS:
        require(len({input_sha(prompt_ids(r, n)) for r in documents.values()}) == 16,
                'Duplicate4K/8K input prefix among the sixteen documents')
    return documents, prime, template, [_detail(*item) for item in selected]


def _receipt_sha(manifest):
    return json_sha({k: v for k, v in manifest.items() if k != 'receipt_sha256'})


def load_horizon_documents(metadata_paths, *, expected_source_sha256=None,
                           expected_train_indices=None, tokenizer_sha256=None):
    """Return ``(documents_by_index, prime271_row, pre_generation_manifest)``.

    Defaults pin the audited production four-file cohort.  Explicit alternative
    SHA/identity expectations exist for synthetic tests only; a scientific
    runner must bind the production defaults in its own frozen protocol.
    ``tokenizer_sha256`` is a caller-verified inventory, not a tokenizer load.
    No file is created; caller persists this receipt before any generation.
    """
    hashes, indices = _expected(expected_source_sha256, expected_train_indices)
    rows, sources = _source_rows(metadata_paths, hashes, indices)
    documents, prime, template, details = _selection(rows, indices)
    tokenizer = copy.deepcopy(dict(tokenizer_sha256 or {}))
    require(all(isinstance(k, str) and k and valid_sha(v) for k, v in tokenizer.items()),
            'Invalid caller tokenizer inventory SHA256')
    manifest = dict(schema=1, status='FROZEN_OLD_DATA_ONLY_BEFORE_GENERATION',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        selection_rule=copy.deepcopy(RULE), scope=copy.deepcopy(SCOPE),
        original_corpus_provenance=copy.deepcopy(CORPUS_ORIGINS),
        source_sha256=hashes, metadata_sources=sources,
        expected_train_indices=list(indices), source_indices=list(documents), document_count=16,
        selected_documents=details, prime=_detail(prime, TUNE, 0),
        prime_in_warm_documents=True, prime_source_index=271,
        prompt_template=template, template_sha256=json_sha(template),
        input_sha256={f'{i}:{n}': input_sha(prompt_ids(row, n))
                      for i, row in documents.items() for n in LENGTHS},
        prime_input_sha256={str(n): input_sha(prompt_ids(prime, n)) for n in LENGTHS},
        tokenizer_sha256=tokenizer, tokenizer_inventory_bound=bool(tokenizer))
    manifest['receipt_sha256'] = _receipt_sha(manifest)
    validate_horizon_manifest(manifest, documents, prime,
        expected_source_sha256=hashes, expected_train_indices=indices)
    return documents, prime, manifest


def validate_horizon_manifest(manifest, documents, prime_row, *,
                              expected_source_sha256=None, expected_train_indices=None):
    """Pure, in-memory validation; no paths are opened and no code is imported."""
    hashes, indices = _expected(expected_source_sha256, expected_train_indices)
    require(isinstance(manifest, dict) and manifest.get('schema') == 1 and
            manifest.get('status') == 'FROZEN_OLD_DATA_ONLY_BEFORE_GENERATION', 'Invalid old-data receipt status')
    require(manifest.get('receipt_sha256') == _receipt_sha(manifest), 'Old-data receipt SHA256 changed')
    require(manifest.get('selection_rule') == RULE and manifest.get('scope') == SCOPE and
            manifest.get('original_corpus_provenance') == CORPUS_ORIGINS,
            'Old-data selection/scope contract changed')
    try:
        timestamp = datetime.fromisoformat(manifest['frozen_at_utc'])
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError('Invalid old-data freeze timestamp') from exc
    require(timestamp.utcoffset() is not None, 'Old-data freeze timestamp requires timezone')
    expected_order = indices + OPT_INDICES + (604,)
    require(isinstance(documents, dict) and tuple(documents) == expected_order and
            manifest.get('source_indices') == list(expected_order) and
            manifest.get('expected_train_indices') == list(indices) and manifest.get('document_count') == 16,
            'Fixed sixteen identities/order changed')
    require(manifest.get('source_sha256') == hashes, 'Pinned old source SHA256 map changed')
    sources = manifest.get('metadata_sources', {})
    require(isinstance(sources, dict) and set(sources) == set(COHORTS), 'Missing old source receipt')
    expected_rows = {TRAIN: indices, OPT_CONFIRMATION: OPT_INDICES,
                     ARCH_CONFIRMATION: ARCH_INDICES, TUNE: TUNE_INDICES}
    for name, order in expected_rows.items():
        source = sources[name]
        require(source.get('sha256') == hashes[name] and source.get('rows') == len(order) and
                source.get('source_indices') == list(order) and
                source.get('split') == ('train' if name in (TRAIN, TUNE) else 'confirmation') and
                isinstance(source.get('path'), str) and Path(source['path']).is_absolute() and
                source['path'].endswith('/' + name), 'Invalid old source identity receipt')
    rows = {TRAIN: [documents[i] for i in indices],
            OPT_CONFIRMATION: [documents[i] for i in OPT_INDICES],
            ARCH_CONFIRMATION: [documents[604]], TUNE: [prime_row]}
    for name, records in rows.items():
        for row in records:
            _validate_row(row, 'train' if name in (TRAIN, TUNE) else 'confirmation')
            require(row['source_index'] in documents and documents[row['source_index']] == row,
                    'Row dictionary source index/body does not match its key')
    require(prime_row.get('source_index') == 271, 'Cold prime must be old tune271')
    _, prime, template, details = _selection(rows, indices)
    require(manifest.get('selected_documents') == details and manifest.get('prime') == _detail(prime, TUNE, 0),
            'Document/source identity or stored token receipt changed')
    require(manifest.get('prompt_template') == template and manifest.get('template_sha256') == json_sha(template),
            'Common template changed')
    require(manifest.get('input_sha256') == {f'{i}:{n}': input_sha(prompt_ids(row, n))
        for i, row in documents.items() for n in LENGTHS} and
        manifest.get('prime_input_sha256') == {str(n): input_sha(prompt_ids(prime, n)) for n in LENGTHS},
        'Exact4K/8K input SHA256 changed')
    require(manifest.get('prime_in_warm_documents') is True and manifest.get('prime_source_index') == 271,
            'Prime overlap must be explicitly disclosed')
    tokenizer = manifest.get('tokenizer_sha256')
    require(isinstance(tokenizer, dict) and all(isinstance(k, str) and k and valid_sha(v)
        for k, v in tokenizer.items()) and manifest.get('tokenizer_inventory_bound') is bool(tokenizer),
        'Invalid tokenizer inventory receipt')
    return dict(status='VERIFIED_OLD_DATA_ONLY', document_count=16, source_indices=list(expected_order),
        input_count=32, prime_source_index=271, prime_in_warm_documents=True,
        receipt_sha256=manifest['receipt_sha256'])


def verify_horizon_sources(manifest, metadata_paths, *, expected_source_sha256=None,
                            expected_train_indices=None):
    """Rebuild from four pinned old sources, including portable nested copies.

    Copy files under ``results/data/<cohort-relative-path>``; the original
    absolute locations remain in the receipt.  No raw tokenizer data is needed.
    """
    hashes, indices = _expected(expected_source_sha256, expected_train_indices)
    rows, sources = _source_rows(metadata_paths, hashes, indices)
    documents, prime, _, _ = _selection(rows, indices)
    proof = validate_horizon_manifest(manifest, documents, prime,
        expected_source_sha256=hashes, expected_train_indices=indices)
    return dict(**proof, source_files_verified=4,
                verified_source_paths={name: item['path'] for name, item in sources.items()})
