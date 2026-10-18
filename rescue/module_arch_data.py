"""New development inputs for internal M2/M3 architecture optimization."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .loading import DEFAULT_TEACHER,ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Use a new data directory')
    source=ROOT/'exp/module_opt/data_v1_20261018'
    groups={s:[json.loads(l) for l in (source/f'{s}.jsonl').read_text().splitlines()] for s in ('fit','tune','warmup')}
    previous=[*sorted((ROOT/'exp/long_system/data').glob('*.jsonl')),
              *sorted((ROOT/'exp/module_opt/data_v1_20261018').glob('*.jsonl'))]
    # Also exclude the objective-study development cohort, whose full report
    # SHA is reconstructed from the original validation rows below.
    seen={r['document_sha256'] for path in previous for r in
          (json.loads(l) for l in path.read_text().splitlines())}
    tokenizer=AutoTokenizer.from_pretrained(DEFAULT_TEACHER,local_files_only=True)
    template=groups['fit'][0];take=16384-len(template['prefix_ids'])-len(template['suffix_ids'])
    raw=Path('/mnt/8T/cl/rescue_experiment_cache/long_system_v1/raw/validation-00000-of-00001.parquet')
    confirmation=[]
    for i,row in enumerate(pq.read_table(raw).to_pylist()):
        digest=hashlib.sha256(row['report'].encode()).hexdigest()
        if i in (229,231,251):seen.add(digest)
        if i<600 or digest in seen:continue
        ids=tokenizer.encode(row['report'],add_special_tokens=False)
        if len(ids)<take:continue
        candidate=dict(split='confirmation',source_index=i,document_sha256=digest,
            original_tokens=len(ids),document_ids=ids[:take],prefix_ids=template['prefix_ids'],suffix_ids=template['suffix_ids'])
        # Reject exact nested token-prefix collisions as well as full reports.
        def prefix(r,n):return r['prefix_ids']+r['document_ids'][:n-len(r['prefix_ids'])-len(r['suffix_ids'])]+r['suffix_ids']
        old=[json.loads(l) for path in previous for l in path.read_text().splitlines()]
        if any(prefix(candidate,n)==prefix(r,n) for n in (8192,16384) for r in old):continue
        confirmation.append(candidate);seen.add(digest)
        if len(confirmation)==3:break
    assert len(confirmation)==3
    groups['confirmation']=confirmation;args.output.mkdir(parents=True)
    hashes={}
    for name,rows in groups.items():
        path=args.output/f'{name}.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows));hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=dict(data_sha256=hashes,source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        selection='First three naturally long validation reports after index600, no generation-dependent selection, excluding old report hashes/nested prefixes',
        ids={s:[r['source_index'] for r in rows] for s,rows in groups.items()},
        scope='Three-document fresh development confirmation; not the paused test or broad generalization')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
