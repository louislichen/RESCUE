#!/usr/bin/env python3
"""Build fresh, title-disjoint HotpotQA splits for attention diagnostics."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from pathlib import Path

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'exp' / 'attn_adapter_transfer'))
from common import sample_to_prompt


def source_rows():
    # Read Arrow batches without scanner worker threads. Some installed
    # pyarrow versions abort during shutdown of a partially consumed remote
    # dataset scanner; single-thread ParquetFile streaming avoids that path.
    filesystem = HfFileSystem()
    paths = sorted(filesystem.glob('datasets/hotpotqa/hotpot_qa/distractor/train-*.parquet'))
    if not paths:
        raise RuntimeError('No HotpotQA train parquet files found')
    for path in paths:
        with filesystem.open(path, 'rb') as source:
            parquet = pq.ParquetFile(source)
            for batch in parquet.iter_batches(batch_size=128, use_threads=False):
                yield from batch.to_pylist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--train', type=int, default=24)
    p.add_argument('--validation', type=int, default=12)
    p.add_argument('--test', type=int, default=30)
    p.add_argument('--skip', type=int, default=1000)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--min-tokens', type=int, default=600)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    tt = AutoTokenizer.from_pretrained('/mnt/8T/cl/models/Qwen2-7B-Instruct', local_files_only=True)
    st = AutoTokenizer.from_pretrained('/mnt/8T/cl/models/Qwen2-1.5B-Instruct', local_files_only=True)
    old = ROOT / 'exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl'
    prior = [json.loads(line) for line in old.read_text().splitlines()]
    old_questions = {r['question'] for r in prior}
    prior_titles = set()
    for r in prior:
        prior_titles.update(re.findall(r'\[Doc \d+\]\n([^\n]+)', r['prompt']))
    counts = {'train': a.train, 'validation': a.validation, 'test': a.test}
    splits = {k: [] for k in counts}
    # Avoid shared source documents across splits and previously used documents.
    used_titles = set(prior_titles)
    used_questions = set(old_questions)
    stats = {'seen': 0, 'prior_or_duplicate': 0, 'length': 0, 'tokenization': 0}
    url = 'hf://datasets/hotpotqa/hotpot_qa/distractor/train-*.parquet'
    data = source_rows()
    iterator = iter(data)
    for source_index, sample in enumerate(iterator):
        if source_index < a.skip:
            continue
        stats['seen'] += 1
        prompt, gold, question = sample_to_prompt(sample, 'all')
        titles = set(re.findall(r'\[Doc \d+\]\n([^\n]+)', prompt))
        if question in used_questions or titles & used_titles:
            stats['prior_or_duplicate'] += 1
            continue
        enc = st(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc['input_ids']
        if not a.min_tokens <= len(ids) <= a.max_tokens:
            stats['length'] += 1
            continue
        if tt(prompt, add_special_tokens=False)['input_ids'] != ids:
            stats['tokenization'] += 1
            continue
        split = next((k for k, v in splits.items() if len(v) < counts[k]), None)
        if split is None:
            break
        sf = sample.get('supporting_facts', {})
        support_titles = set(sf.get('title', [])) if isinstance(sf, dict) else {x[0] for x in sf}
        evidence_spans = []
        for m in re.finditer(r'\[Doc \d+\]\n([^\n]+)\n', prompt):
            end = prompt.find('\n\n', m.end())
            if end < 0:
                end = len(prompt)
            if m.group(1) in support_titles:
                evidence_spans.append((m.start(), end))
        evidence = [i for i, (s, e) in enumerate(enc['offset_mapping'])
                    if e > s and any(s < b and e > aa for aa, b in evidence_spans)]
        row = dict(sample_id=str(sample.get('id', source_index)), source_index=source_index,
                   split=split, prompt=prompt, gold=gold, question=question, ids=ids,
                   titles=sorted(titles), evidence_positions=evidence,
                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        splits[split].append(row)
        used_titles.update(titles)
        used_questions.add(question)
        print(f"{split} {len(splits[split])}/{counts[split]} source={source_index} tokens={len(ids)}", flush=True)
        if all(len(splits[k]) == counts[k] for k in counts):
            break
    if any(len(splits[k]) != counts[k] for k in counts):
        raise RuntimeError(f'Insufficient fresh samples: {[len(v) for v in splits.values()]}')
    for split, rows in splits.items():
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    manifest = dict(args=vars(a), source=url, exclusion='Prior 300 questions and all their document titles; all new splits title-disjoint',
                    prior_sha256=hashlib.sha256(old.read_bytes()).hexdigest(), stats=stats,
                    counts=counts, split_sha256={k: hashlib.sha256((out / f'{k}.jsonl').read_bytes()).hexdigest() for k in splits})
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    # Close the partially consumed Arrow stream before Python finalization.
    # Otherwise some pyarrow/datasets versions release worker state too late.
    iterator.close()
    del iterator, data
    gc.collect()


if __name__ == '__main__':
    main()
