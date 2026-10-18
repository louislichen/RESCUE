"""Freeze train/tuning and new natural-document confirmation inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from .loading import DEFAULT_TEACHER,ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists(): p.error('Use a fresh directory')
    previous=ROOT/'exp/long_system/data'
    old={s:[json.loads(l) for l in (previous/f'{s}.jsonl').read_text().splitlines()] for s in ('train','development','test')}
    seen={r['document_sha256'] for rr in old.values() for r in rr}
    fit=old['train'][:6];tune=old['train'][-3:]
    tokenizer=AutoTokenizer.from_pretrained(DEFAULT_TEACHER,local_files_only=True)
    template=old['train'][0]
    take=16384-len(template['prefix_ids'])-len(template['suffix_ids'])
    raw=Path('/mnt/8T/cl/rescue_experiment_cache/long_system_v1/raw/validation-00000-of-00001.parquet')
    confirmation=[]
    for i,r in enumerate(pq.read_table(raw).to_pylist()):
        if i<300: continue
        text=r['report'];digest=hashlib.sha256(text.encode()).hexdigest()
        if digest in seen: continue
        ids=tokenizer.encode(text,add_special_tokens=False)
        if len(ids)<take: continue
        confirmation.append(dict(split='confirmation',source_index=i,document_sha256=digest,
            original_tokens=len(ids),document_ids=ids[:take],prefix_ids=template['prefix_ids'],suffix_ids=template['suffix_ids']))
        seen.add(digest)
        if len(confirmation)==3: break
    assert len(confirmation)==3
    args.output.mkdir(parents=True)
    hashes={}
    for split,items in dict(fit=fit,tune=tune,confirmation=confirmation,warmup=old['train'][6:7]).items():
        q=args.output/f'{split}.jsonl';q.write_text(''.join(json.dumps(r)+'\n' for r in items));hashes[split]=hashlib.sha256(q.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(dict(data_sha256=hashes,
        source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),selection='First three naturally long validation documents after index300, excluding all previous document hashes; no generation-dependent selection',
        fit_ids=[r['source_index'] for r in fit],tune_ids=[r['source_index'] for r in tune],
        confirmation_ids=[r['source_index'] for r in confirmation],scope='New development confirmation pilot, not the paused test matrix'),indent=2)+'\n')
    print((args.output/'manifest.json').read_text())


if __name__=='__main__': main()
