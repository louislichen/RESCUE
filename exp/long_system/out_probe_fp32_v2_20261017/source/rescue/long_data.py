"""Prepare nested real-document contexts; never repeat text to create length."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .loading import DEFAULT_TEACHER


def context(item, length):
    left, right = item['prefix_ids'], item['suffix_ids']
    take = length-len(left)-len(right)
    if take < 1 or take > len(item['document_ids']):
        raise ValueError('Document does not support the requested context length')
    return left+item['document_ids'][:take]+right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('/mnt/8T/cl/rescue_experiment_cache/long_system_v1/raw'))
    parser.add_argument('--output', type=Path, default=Path('exp/long_system/data'))
    parser.add_argument('--train', type=int, default=12)
    parser.add_argument('--development', type=int, default=6)
    parser.add_argument('--test', type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh data directory')
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TEACHER, local_files_only=True)
    prefix = tokenizer.encode('<|im_start|>system\nYou summarize reports accurately.\n<|im_end|>\n'
        '<|im_start|>user\nWrite a concise summary of the following report.\n\nReport:\n', add_special_tokens=False)
    suffix = tokenizer.encode('\n<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False)
    maximum = 16384-len(prefix)-len(suffix)
    used, selected, sources = set(), {}, {}
    for split, name, count, skip in (
        ('train', 'train-00000-of-00002.parquet', args.train, 200),
        ('development', 'validation-00000-of-00001.parquet', args.development, 200),
        ('test', 'test-00000-of-00001.parquet', args.test, 200)):
        path = args.raw/name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = pq.read_table(path).to_pylist()
        result = []
        for index, row in enumerate(data):
            if index < skip:
                continue
            document = row.get('report', row.get('document'))
            if not isinstance(document, str):
                raise ValueError(f'Unknown document schema: {row.keys()}')
            digest = hashlib.sha256(document.encode()).hexdigest()
            if digest in used:
                continue
            ids = tokenizer.encode(document, add_special_tokens=False)
            if len(ids) < maximum:
                continue
            result.append(dict(split=split, source_index=index, document_sha256=digest,
                original_tokens=len(ids), document_ids=ids[:maximum], prefix_ids=prefix, suffix_ids=suffix))
            used.add(digest)
            print(split, len(result), '/', count, 'source', index, 'tokens', len(ids), flush=True)
            if len(result) == count:
                break
        if len(result) != count:
            raise ValueError(f'Insufficient naturally long documents in {split}: {len(result)}/{count}')
        selected[split] = result
    args.output.mkdir(parents=True)
    hashes = {}
    for split, rows in selected.items():
        path = args.output/(split+'.jsonl')
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(source='https://huggingface.co/datasets/ccdv/govreport-summarization',
        raw_sha256=sources, data_sha256=hashes, counts={s: len(r) for s, r in selected.items()},
        context_lengths=[2048, 4096, 8192, 16384], max_new_tokens=256,
        selection='First naturally >=16K-token reports after row 200, selected without generation or model results',
        splits='Original corpus train/validation/test respectively; document SHA256 disjoint',
        truncation='Nested report prefixes plus identical instruction; no repeated, padded or synthetic documents',
        quality='Generation matches a matched Target reference; truncated-prefix outputs are not scored against full-report summaries')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
