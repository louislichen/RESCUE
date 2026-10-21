"""Independently re-pool a live horizon prefix; never freezes final choices."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def need(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root,analysis):
    root,analysis=Path(root).resolve(),Path(analysis).resolve()
    need(analysis.is_relative_to(root),'Analysis must remain inside snapshot')
    read=lambda name:json.loads((root/name).read_text())
    rows=lambda name:[json.loads(line) for line in (root/name).read_text().splitlines() if line.strip()]
    source=Path(__file__).read_bytes()
    snapshot=read('SNAPSHOT_COLLECTION.json');results=root/'results'
    need(snapshot['status']=='PARTIAL_READ_ONLY_PREFIX_SNAPSHOT' and snapshot['read_only'] is True and
        snapshot['snapshot_is_atomic'] is False,'Not an explicitly non-atomic live prefix snapshot')
    for name,info in snapshot['files'].items():
        path=root/name
        need(not Path(name).is_absolute() and '..' not in Path(name).parts and path.resolve().is_relative_to(root) and
            path.is_file() and not path.is_symlink() and sha(path)==info['sha256'] and path.stat().st_size==info['bytes'],
             'Snapshot SHA/size mismatch: '+name)
    protocol=read('results/protocol.json');summary=json.loads((analysis/'summary.json').read_text())
    need(summary['status']=='PARTIAL' and summary['authoritative_union'] is False and summary['goal_complete'] is False and
        not (root/'VERIFIED_COMPLETE.json').exists() and not (analysis/'VERIFIED_UNION.json').exists(),
         'Live prefix must not claim authoritative completion')
    for name,digest in protocol['source_sha256'].items():
        need(sha(root/'after'/name)==sha(results/'source'/name)==digest,'Frozen executed source changed')
    metadata={}
    for name in ('config.json','tokenizer_config.json','tokenizer.json'):
        path=root/'asset_metadata/Qwen2-7B-Instruct'/name
        need(protocol['assets_sha256'][protocol['args']['teacher']+'/'+name]==sha(path),'Target metadata not frozen')
        metadata[name]=json.loads(path.read_text())
    token=metadata['tokenizer_config.json']['eos_token']
    if isinstance(token,dict):token=token['content']
    ids=[v['id'] for v in metadata['tokenizer.json']['added_tokens'] if v['content']==token]
    need(len(ids)==1 and protocol['tokenizer_eos_token_id']==ids[0],'EOS not derived from saved tokenizer')
    eos=ids[0];refs,prime,warm,owners=[rows('results/'+n+'.jsonl') for n in ('reference','prime','runtime','owners')]
    counts=dict(reference_requests=len(refs),prime_requests=len(prime),timing_requests=len(warm),owners=len(owners))
    need(counts==snapshot['counts']==summary['observed_counts'],'Partial raw counts differ')
    allrows=refs+prime+warm
    need(len({r['request_id'] for r in allrows})==len(allrows) and sorted(r['execution_index'] for r in allrows)==list(range(len(allrows))),
         'Live prefix request UUID/order not bijective')
    refmap={(r['source_index'],r['length']):r['generated'] for r in refs}
    for generated in refmap.values():
        need(0<len(generated)<=128 and (len(generated)==128 or generated[-1]==eos) and
            (eos not in generated or (generated[-1]==eos and eos not in generated[:-1])),'Canonical EOS boundary invalid')
    for row in prime+warm:
        need(row['generated']==refmap[row['source_index'],row['length']] and row['tokens']==len(row['generated']) and
            row['output_agrees'] is True and row['performance_eligible'] is True and
            row['mapper_executed'] is row['controller_executed'] is row['observer_enabled'] is row['profile_enabled'] is False and
            math.isfinite(row['outer_seconds']) and row['outer_seconds']>0 and math.isclose(row['outer_tps'],row['tokens']/row['outer_seconds']),
             'Partial canonical/unprofiled/full-clock gate failed')
        if row['arm']['family']=='retrieval':
            need(row['accepted']==sum(t['accepted'] for t in row['trace']) and row['proposed']==sum(t['proposed'] for t in row['trace']) and
                row['blocks']==len(row['trace']),'Round acceptance sums differ')
    checks={}
    for n,metrics in summary['metrics'].items():
        for mode,value in metrics.items():
            saved=[r for r in warm if r['mode']==mode and r['length']==int(n)]
            docs=[i for i in protocol['source_indices'] if {r['repeat'] for r in saved if r['source_index']==i}=={0,1}]
            selected=[r for r in saved if r['source_index'] in docs]
            tokens=sum(r['tokens'] for r in selected);seconds=sum(r['outer_seconds'] for r in selected)
            point=sum(r['accepted'] for r in selected)/sum(r['blocks'] for r in selected)
            need(value['documents']==len(docs) and value['paired_document_indices']==docs and value['warm_requests']==len(selected) and
                value['actual_raw_warm_requests']==len(saved) and value['warm_tokens']==tokens and math.isclose(value['warm_seconds'],seconds) and
                math.isclose(value['pooled_tps'],tokens/seconds) and math.isclose(value['acceptance_length']['point'],point) and
                value['warm_captures']==sum(r.get('draft_runtime',{}).get('request',{}).get('captures',0) for r in saved),
                 'Partial metric substituted missing repeats or filtered captures')
            checks[f'{n}:{mode}']=dict(paired_documents=len(docs),paired_warm=len(selected),all_saved_warm=len(saved),pooled_tps=tokens/seconds)
    for owner in owners:
        block=sorted((r for r in prime+warm if r['owner_id']==owner['owner_id']),key=lambda r:r['execution_index'])
        need(len(block)==33 and owner['warm_request_ids']==[r['request_id'] for r in block[1:]] and owner['prime_request_id']==block[0]['request_id'] and
            owner['engine_closed'] is owner['hook_restored'] is True and owner['maximum_live_owners']==1 and owner['closed_runtime_stats']['entries']==0,
             'Claimed completed owner lacks full shared32/close ledger')
    need(Path(__file__).read_bytes()==source,'Verifier source changed during run')
    receipt=dict(status='PARTIAL_AUDIT_PASSED',scientific_experiment_complete=False,goal_complete=False,
        verified_at_utc=datetime.now(timezone.utc).isoformat(),counts=counts,canonical_sd_requests=len(prime)+len(warm),
        snapshot_files_verified=len(snapshot['files']),independently_recomputed_partial_metrics=checks,
        eos_derived_from_saved_tokenizer=eos,summary_sha256=sha(analysis/'summary.json'),
        audit_source_sha256=hashlib.sha256(source).hexdigest(),
        limits='Non-atomic live prefix only; no complete matrix/union, final success, tensor equality, CI independence or production-speed claim')
    with (root/'INDEPENDENT_PARTIAL_AUDIT_SOURCE.py').open('xb') as handle:handle.write(source)
    with (root/'INDEPENDENT_PARTIAL_AUDIT.json').open('x') as handle:json.dump(receipt,handle,indent=2);handle.write('\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--analysis-dir',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(verify(args.root,args.analysis_dir),indent=2))
