"""Reuse stage-two train/validation; acquire an untouched, title-disjoint test."""
import gc
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'exp/attention_selection'))
from prepare_data import source_rows, sample_to_prompt
from transformers import AutoTokenizer


def main():
    out = Path(__file__).parent / 'data'
    out.mkdir(exist_ok=True)
    olddir = ROOT / 'exp/attention_selection/data_fresh'
    old = [json.loads(line) for split in ('train', 'validation', 'test')
           for line in (olddir / f'{split}.jsonl').read_text().splitlines()]
    prior_path = ROOT / 'exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl'
    prior = [json.loads(line) for line in prior_path.read_text().splitlines()]
    used_questions = {r['question'] for r in old + prior}
    used_titles = {t for r in old for t in r['titles']}
    used_titles.update(t for r in prior for t in re.findall(r'\[Doc \d+\]\n([^\n]+)', r['prompt']))
    tt = AutoTokenizer.from_pretrained('/mnt/8T/cl/models/Qwen2-7B-Instruct', local_files_only=True)
    st = AutoTokenizer.from_pretrained('/mnt/8T/cl/models/Qwen2-1.5B-Instruct', local_files_only=True)
    result = []
    data = source_rows()
    for source_index, sample in enumerate(data):
        if source_index < 1200:
            continue
        prompt, gold, question = sample_to_prompt(sample, 'all')
        titles = set(re.findall(r'\[Doc \d+\]\n([^\n]+)', prompt))
        if question in used_questions or titles & used_titles:
            continue
        ids = st(prompt, add_special_tokens=False)['input_ids']
        if not 600 <= len(ids) <= 2048 or tt(prompt, add_special_tokens=False)['input_ids'] != ids:
            continue
        result.append(dict(sample_id=str(sample.get('id', source_index)), source_index=source_index,
                           split='test', prompt=prompt, gold=gold, question=question, ids=ids,
                           titles=sorted(titles), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        used_questions.add(question)
        used_titles.update(titles)
        print(f'fresh test {len(result)}/30 source={source_index} n={len(ids)}', flush=True)
        if len(result) == 30:
            break
    data.close()
    gc.collect()
    assert len(result) == 30
    for split in ('train', 'validation'):
        (out / f'{split}.jsonl').write_bytes((olddir / f'{split}.jsonl').read_bytes())
    (out / 'test.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result))
    manifest = dict(counts={'train': 24, 'validation': 12, 'test': 30},
                    train_validation='Reused stage-two training/validation; no new test documents previously examined',
                    exclusion='Prior 300 and all 66 stage-two documents/questions; title-disjoint across all splits',
                    source='hf://datasets/hotpotqa/hotpot_qa/distractor/train-*.parquet', skip=1200,
                    hashes={s: hashlib.sha256((out / f'{s}.jsonl').read_bytes()).hexdigest()
                            for s in ('train', 'validation', 'test')})
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
