"""New title-disjoint HotpotQA splits, excluding all earlier experiments."""
import gc
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'exp/attention_selection'))
from prepare_data import source_rows, sample_to_prompt
from transformers import AutoTokenizer


def main():
    out = Path(__file__).parent/'data'
    out.mkdir(exist_ok=True)
    old_paths = [ROOT/'exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl']
    old_paths += [p for directory in ['exp/attention_selection/data_fresh', 'exp/selective_kv/data']
                  for p in (ROOT/directory).glob('*.jsonl')]
    old = [json.loads(line) for p in old_paths for line in p.read_text().splitlines()]
    questions = {r['question'] for r in old}
    titles = {t for r in old for t in re.findall(r'\[Doc \d+\]\n([^\n]+)', r['prompt'])}
    tt = AutoTokenizer.from_pretrained('/mnt/8T/cl/models/Qwen2-7B-Instruct', local_files_only=True)
    st = AutoTokenizer.from_pretrained('/mnt/8T/cl/models/Qwen2-1.5B-Instruct', local_files_only=True)
    counts = dict(train=48, validation=16, test=30)
    result = {key: [] for key in counts}
    stream = source_rows()
    for source_index, sample in enumerate(stream):
        if source_index < 1500:
            continue
        prompt, gold, question = sample_to_prompt(sample, 'all')
        current_titles = set(re.findall(r'\[Doc \d+\]\n([^\n]+)', prompt))
        if question in questions or current_titles & titles:
            continue
        ids = st(prompt, add_special_tokens=False)['input_ids']
        if not 600 <= len(ids) <= 2048 or tt(prompt, add_special_tokens=False)['input_ids'] != ids:
            continue
        split = next((s for s in counts if len(result[s]) < counts[s]), None)
        if split is None:
            break
        result[split].append(dict(split=split, source_index=source_index, question=question,
            gold=gold, prompt=prompt, ids=ids, titles=sorted(current_titles),
            sample_id=str(sample.get('id', source_index)), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        questions.add(question)
        titles.update(current_titles)
        print(f'{split} {len(result[split])}/{counts[split]} source={source_index} n={len(ids)}', flush=True)
        if all(len(result[s]) == counts[s] for s in counts):
            break
    stream.close()
    gc.collect()
    assert all(len(result[s]) == counts[s] for s in counts)
    for split, data in result.items():
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data))
    manifest = dict(counts=counts, skip=1500, source='hf://datasets/hotpotqa/hotpot_qa/distractor/train-*.parquet',
        exclusion='Prior 300 questions and all attention-selection/selective-V documents, including their test titles',
        previous_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in old_paths},
        hashes={s:hashlib.sha256((out/f'{s}.jsonl').read_bytes()).hexdigest() for s in counts})
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
