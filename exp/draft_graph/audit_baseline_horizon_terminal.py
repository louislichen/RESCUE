"""Independent stdlib saved-record audit of the COMPLETE2540 horizon study.

No runner/report module is imported. Shared validators come from the separate
stage1 independent auditor and are saved/pinned with the audit for replay.
No model/corpus/remote access occurs; recorded pointers are not KV tensor proof.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re

SUPPORT_PATH = Path(__file__).with_name('audit_baseline_terminal.py')
if not SUPPORT_PATH.is_file():
    SUPPORT_PATH = Path(__file__).with_name('INDEPENDENT_RAW_AUDIT_SUPPORT.py')
AUDIT_SOURCE_BYTES = Path(__file__).read_bytes()
SUPPORT_SOURCE_BYTES = SUPPORT_PATH.read_bytes()
spec = importlib.util.spec_from_file_location('independent_stage1_validators', SUPPORT_PATH)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
if SUPPORT_PATH.read_bytes() != SUPPORT_SOURCE_BYTES:
    raise RuntimeError('Independent support changed while loading')
need, digest, local, read, lines, finite, near = s.need, s.digest, s.local, s.read, s.lines, s.finite, s.near
COUNTS = dict(reference_requests=32, prime_requests=76, timing_requests=2432, owners=76)
DOCS = (211,213,220,226,233,236,246,248,253,271,272,288,303,306,327,604)
LENGTHS, GROUPS = s.LENGTHS, s.GROUPS
DATA_SHA = {
    'exp/long_system/data/train.jsonl':'ae1e3f58a663dd505ddc9d1f1fdb87af55cad1d967c6b771f696bcbbadce7083',
    'exp/module_opt/data_v1_20261018/confirmation.jsonl':'03eb1b126b424cc99b3586fb42594430ed74706f3ade587dc7102abfa5df90bb',
    'exp/module_arch/data_v1_20261018/confirmation.jsonl':'007fd9247bf8d77dc53eef51469c84c8a66d29bb2d9577815349c4601ecb29f9',
    'exp/module_opt/data_v1_20261018/tune.jsonl':'85b2eb1a81d1617b2d158c069c420c452459a8bdb90da128f257220cc27ded09',
}
MANDATORY = {
    'PIPELINE_EXIT.json','RUN_STARTED.json','LAUNCH_MANIFEST.json',
    'results/protocol.json','results/COMPLETE.json','results/FROZEN_SELECTION.json',
    'results/reference.jsonl','results/references.json','results/prime.jsonl','results/prime_references.json',
    'results/runtime.jsonl','results/owners.jsonl','results/tree_raw.jsonl','results/gpu_diagnostics.jsonl',
    'results/GATE_PASSED.json','results/data/DATASET_MANIFEST.json','results/data/documents.jsonl','results/data/prime.json',
    'results/stage1_evidence/results/FROZEN_SELECTION.json','results/stage1_evidence/results/references.json',
    *{'results/data/'+name for name in DATA_SHA},
    *{'asset_metadata/Qwen2-7B-Instruct/'+name for name in ('config.json','tokenizer_config.json','tokenizer.json')},
}


def verify_collection(root,collection,protocol=None):
    """Every consumed scientific byte must belong to the terminal inventory."""
    inventory=collection.get('files',{});need(isinstance(inventory,dict) and inventory,'Empty terminal inventory')
    required=set(MANDATORY)
    if protocol is not None:
        sources=protocol.get('source_sha256',{});need(isinstance(sources,dict) and sources,'Empty executed source inventory')
        required.update(parent+'/'+name for name in sources for parent in ('after','results/source'))
    need(required<=set(inventory),'Terminal inventory omits mandatory raw/source/proof artifacts')
    for name,info in inventory.items():
        path=local(root,name)
        need(type(info.get('bytes')) is int and info['bytes']>=0 and path.stat().st_size==info['bytes'] and
            digest(path)==info.get('sha256'),'Collected SHA/size mismatch: '+name)


def linear(row,arm):
    s.linear(row,arm)
    commits=sum(step['accepted']+1 for step in row['trace'])
    need(row['tokens'] in (commits,commits-1),'Short-EOS output does not align with terminal accepted/bonus commits')


def group(arm):
    return 'full_tree' if arm['family']=='full_tree' else 'retrieval/'+('graph' if arm['backend']=='cuda_graph' else arm['backend'])


def plan():
    rng=random.Random(20261025);sequence=[]
    for repeat in (0,1):
        documents=list(DOCS);rng.shuffle(documents)
        sequence.extend(dict(source_index=i,repeat=repeat) for i in documents)
    owners=[];requests=[]
    for length in LENGTHS:
        modes=[a['mode'] for a in s.menu()];random.Random(20261025+length).shuffle(modes)
        for mode in modes:
            owners.append(dict(mode=mode,length=length))
            requests.append(dict(mode=mode,length=length,source_index=271,phase='cold_prime'))
            requests.extend(dict(mode=mode,length=length,**r,phase='warm',warm_index=j) for j,r in enumerate(sequence))
    return sequence,owners,requests


def pooled(rows):
    tokens=sum(r['tokens'] for r in rows);seconds=sum(r['outer_seconds'] for r in rows)
    return dict(tokens=tokens,seconds=seconds,tps=tokens/seconds,
        captures=sum(r.get('draft_runtime',{}).get('request',{}).get('captures',0) for r in rows),
        capture_seconds=sum(r.get('draft_runtime',{}).get('request',{}).get('capture_host_seconds',0.) for r in rows))


def selection(warm,stage1):
    result={}
    for length in LENGTHS:
        candidates=[]
        for arm in s.menu():
            rows=[r for r in warm if r['length']==length and r['mode']==arm['mode']]
            need(len(rows)==32,'Incomplete selection cell')
            metric=pooled(rows)
            candidates.append(dict(mode=arm['mode'],arm=arm,warm_requests=32,warm_tokens=metric['tokens'],
                warm_seconds=metric['seconds'],warm_tps=metric['tps'],request_ids=[r['request_id'] for r in rows],
                prefix6=pooled(rows[:6]),prefix32=metric))
        best=lambda values:min(values,key=lambda v:(-v['warm_tps'],v['mode']))
        groups={g:best([r for r in candidates if group(r['arm'])==g]) for g in GROUPS}
        original=copy.deepcopy(stage1['per_context'][str(length)]['groups'])
        union={g:[original[g]]+([] if original[g]['mode']==groups[g]['mode'] else [groups[g]]) for g in GROUPS}
        result[str(length)]=dict(groups=groups,stage1_groups=original,union_by_group=union,
            union_modes=list(dict.fromkeys(r['mode'] for g in GROUPS for r in union[g])),
            ranking=sorted(candidates,key=lambda r:(-r['warm_tps'],r['mode'])),overall=best(candidates),
            overall_family_winner={f:best([r for r in candidates if r['arm']['family']==f]) for f in ('retrieval','full_tree')})
    return result


def data_audit(root,protocol):
    values={}
    for name,sha in DATA_SHA.items():
        need(protocol['data_source_paths'].get(name)=='data/'+name,'Old metadata path changed')
        path=local(root,'results/data/'+name);need(digest(path)==sha,'Old metadata SHA changed')
        values[name]=lines(root,'results/data/'+name)
    train,opt,arch,tune=[values[n] for n in DATA_SHA]
    need(tuple(r['source_index'] for r in train)==DOCS[:12] and
        tuple(r['source_index'] for r in opt)==(303,306,327) and
        tuple(r['source_index'] for r in arch)==(604,613,614) and
        tuple(r['source_index'] for r in tune)==(271,272,288),'Old cohort identities/order changed')
    documents=train+opt+arch[:1];prime=tune[0]
    need(lines(root,'results/data/documents.jsonl')==documents and read(root,'results/data/prime.json')==prime and
         prime==next(r for r in documents if r['source_index']==271),'Selected inputs/prime271 were replaced')
    seen=set()
    for row in documents:
        need(row['document_sha256'] not in seen and row['original_tokens']>=8192 and len(row['document_ids'])>=8192,
             'Repeated or insufficient natural document')
        seen.add(row['document_sha256'])
        for n in LENGTHS:
            available=n-len(row['prefix_ids'])-len(row['suffix_ids'])
            ids=row['prefix_ids']+row['document_ids'][:available]+row['suffix_ids']
            need(available>0 and len(ids)==n and hashlib.sha256(json.dumps(ids).encode()).hexdigest()==
                 protocol['input_sha256'][f"{row['source_index']}:{n}"],'Exact4K/8K input SHA mismatch')
    need(protocol.get('prime_in_warm_cohort') is True and protocol.get('prime_source_index')==271,
         'Prime overlap not disclosed')
    need(digest(local(root,'results/'+protocol['data_manifest']))==protocol['data_manifest_sha256'],
         'Frozen old data receipt SHA changed')


def owner_audit(owner,block,expected,sequence):
    first,rest=block[0],block[1:]
    need(all(owner.get(k)==v for k,v in expected.items()) and re.fullmatch('[0-9a-f]{32}',owner['owner_id']),
         'Owner identity/geometry mismatch')
    need(len(rest)==32 and [{k:r[k] for k in ('source_index','repeat')} for r in rest]==sequence and
        owner['prime_request_id']==first['request_id'] and owner['warm_request_ids']==[r['request_id'] for r in rest],
         'Owner cold/shared32 request ledger mismatch')
    need(all(r['owner_id']==owner['owner_id'] and r['owner_signature']==owner['owner_signature'] for r in block) and
        owner['maximum_live_owners']==1 and owner['engine_closed'] is owner['hook_restored'] is True and
        owner['closed_runtime_stats']['entries']==0 and owner['final_runtime_stats']==rest[-1]['runtime_stats'],
         'Private cache identity/closed runtime mismatch')
    metric=pooled(rest)
    need(owner['warm_tokens']==metric['tokens'] and near(owner['warm_outer_seconds_sum'],metric['seconds']) and
        owner['prime_tokens']==first['tokens'] and near(owner['cold_prime_outer_seconds'],first['outer_seconds']) and
        near(owner['constructor_and_detach_seconds'],first['constructor_and_detach_seconds']) and
        near(first['cold_complete_outer_seconds'],owner['constructor_and_detach_seconds']+first['outer_seconds']),
         'Owner raw token/time sums mismatch')
    service=owner['constructor_and_detach_seconds']+first['outer_seconds']+metric['seconds']+owner['close_seconds']
    need(finite(owner['close_seconds']) and near(owner['accounted_service_seconds'],service) and
        owner['accounted_service_tokens']==metric['tokens']+first['tokens'] and
        near(owner['accounted_service_tps'],(metric['tokens']+first['tokens'])/service) and
        owner['owner_lifetime_wall_seconds']+1e-8>=service,'Cold/service timing omitted or double-counted')
    if first['arm']['family']=='retrieval' and first['arm']['backend']!='eager':
        total=dict.fromkeys(('captures','replays','resident_steps','shadow_updates','capture_host_seconds'),0)
        for row in block:
            for key in total:
                delta=row['draft_runtime']['request'][key]
                need(finite(delta) and (key=='capture_host_seconds' or type(delta) is int),'Invalid runtime delta')
                total[key]+=delta
                need(near(row['runtime_stats'][key],total[key]) and near(row['draft_runtime']['lifetime'][key],total[key]),
                     'Runtime cold/late capture ledger mismatch')
    return metric


def audit(root,analysis):
    root,analysis=Path(root).resolve(),Path(analysis).resolve()
    need(analysis.is_relative_to(root),'Analysis escapes explicit local export')
    names=('INDEPENDENT_RAW_AUDIT.json','INDEPENDENT_RAW_AUDIT_SOURCE.py','INDEPENDENT_RAW_AUDIT_SUPPORT.py')
    need(not any((root/n).exists() for n in names),'Never overwrite audit evidence')
    collection=read(root,'TERMINAL_COLLECTION.json');collection_sha=digest(local(root,'TERMINAL_COLLECTION.json'))
    need(collection.get('status')=='TERMINAL_COLLECTED_NOT_YET_INDEPENDENTLY_AUDITED' and
        collection.get('counts')==COUNTS and type(collection.get('pipeline_exit_code')) is int and collection.get('pipeline_exit_code')==0 and all(collection.get(k) is True for k in
        ('read_only','process_and_tmux_absent','complete_file_set_rechecked','all_file_bytes_rechecked')),
         'Not an immutable terminal export')
    verify_collection(root,collection)
    exit_code=read(root,'PIPELINE_EXIT.json').get('exit_code')
    need(type(exit_code) is int and exit_code==0 and not (root/'FAILURE.json').exists() and not (root/'results/FAILURE.json').exists(),
         'Failed scientific pipeline')
    protocol,complete,frozen=[read(root,'results/'+n+'.json') for n in ('protocol','COMPLETE','FROZEN_SELECTION')]
    verify_collection(root,collection,protocol)
    need(complete.get('counts')==complete.get('expected_counts')==COUNTS and complete.get('logical_requests')==2540 and
        complete.get('owner_closed')==76 and complete.get('goal_complete') is False and all(complete.get(k) is True for k in
        ('passed','sources_unchanged','assets_unchanged','dependencies_unchanged','all_canonical','backend_trajectories_equal',
         'data_unchanged','stage1_verified_unchanged')),'Incomplete2540 immutable execution gate')
    sequence,owner_plan,requests=plan();arms={a['mode']:a for a in s.menu()}
    need(protocol['source_indices']==list(DOCS) and protocol['arms']==s.menu() and protocol['lengths']==list(LENGTHS) and
        protocol['seed']==20261025 and protocol['repeats']==2 and protocol['max_new_tokens']==128 and
        protocol['expected_counts']==COUNTS and protocol['expected_requests']==2540 and
        protocol['shared_warm_sequence']==sequence and protocol['owner_order']==owner_plan and
        protocol['warm_order']==[{k:v for k,v in r.items() if k not in ('phase','warm_index')} for r in requests if r['phase']=='warm'],
         'Frozen cohort/menu/shared order changed')
    need(protocol['max_live_owners']==1 and protocol['graph_max_entries']==128 and protocol['no_training'] is True and
        all(protocol[k] is False for k in ('mapper_executed','controller_executed','observer','profile','retrieval_retain_query')) and
        protocol['template_prefill_precision']=='bfloat16','Wrong execution/Mapper/precision contract')
    need(frozen['protocol_sha256']==digest(root/'results/protocol.json') and complete['frozen_selection_sha256']==digest(root/'results/FROZEN_SELECTION.json') and
        frozen['source_sha256']==protocol['source_sha256'] and frozen['assets_sha256']==protocol['assets_sha256'] and
        frozen['complete_request_count']==2540 and frozen['no_new_confirmation_opened'] is True,'Frozen union not bound to complete inputs')
    launch=read(root,'LAUNCH_MANIFEST.json')
    for name,sha in protocol['source_sha256'].items():
        need(launch['source_sha256'].get(name)==sha and digest(local(root,'after/'+name))==sha and
            digest(local(root,'results/source/'+name))==sha,'Executed source differs: '+name)
    env=protocol['environment']
    need('GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9' in env['nvidia_smi_gpu'] and
        env['target_dtype']=='torch.float32' and env['draft_dtype']==env['prefix_dtype']=='torch.bfloat16' and
        env['tf32'] is False and protocol['flash_attention']['version']=='2.8.3','Hardware/precision/dependency differs')
    for section in ('source_bindings','tree_source_bindings'):
        for record in protocol[section].values():
            if isinstance(record,dict) and 'path' in record and 'sha256' in record:
                matches=[p for p in protocol['source_sha256'] if record['path']==p or record['path'].endswith('/'+p)]
                need(len(matches)==1 and protocol['source_sha256'][matches[0]]==record['sha256'],'Imported source binding changed')
    data_audit(root,protocol)
    eos=s.tokenizer_eos(root,protocol,collection);eos_id=eos['eos_token_id']
    need(protocol['tokenizer_eos_token_id']==eos_id,'Recorded EOS differs from actual exported tokenizer')
    refs,prime,warm,owners,tree_raw=[lines(root,'results/'+n+'.jsonl') for n in ('reference','prime','runtime','owners','tree_raw')]
    need(tuple(map(len,(refs,prime,warm,owners,tree_raw)))==(32,76,2432,76,528),'Raw counts incomplete')
    allrows=refs+prime+warm
    need(len({r['request_id'] for r in allrows})==2540 and all(re.fullmatch('[0-9a-f]{32}',r['request_id']) for r in allrows) and
        sorted(r['execution_index'] for r in allrows)==list(range(2540)),'UUID/execution ledger not bijective')
    refmap={}
    for j,row in enumerate(refs):
        need(row['execution_index']==j and row['phase']=='reference' and row['mode']=='canonical_ar' and row['performance_eligible'] is False,
             'Independent AR did not precede timing')
        s.canonical(row['generated'],eos_id);key=row['length'],row['source_index']
        need(key not in refmap,'Duplicate AR input');refmap[key]=row['generated']
    need(set(refmap)=={(n,i) for n in LENGTHS for i in DOCS} and read(root,'results/references.json')==
        {f'{i}:{n}':ids for (n,i),ids in refmap.items()},'AR reference map differs')
    stage1=read(root,'results/stage1_evidence/results/FROZEN_SELECTION.json')
    stage1_ref=read(root,'results/stage1_evidence/results/references.json')
    stage1_sha=digest(local(root,'results/stage1_evidence/results/FROZEN_SELECTION.json'))
    need(frozen['stage1_selection_sha256']==complete['stage1_selection_sha256']==protocol['stage1_selection_sha256']==
        protocol['stage1_freeze']['selection_freeze_sha256']==stage1_sha,'Stage1 choice linkage changed')
    need(all(refmap[n,271]==stage1_ref[f'271:{n}'] for n in LENGTHS) and read(root,'results/prime_references.json')==
        {str(n):refmap[n,271] for n in LENGTHS},'Old271 canonical reused without equal reference')
    sd=sorted(prime+warm,key=lambda r:r['execution_index']);raw_map={r['request_id']:r for r in tree_raw};trajectories={}
    need(len(raw_map)==528,'Duplicate raw tree UUID')
    for row,expected in zip(sd,requests):
        need(all(row.get(k)==v for k,v in expected.items()),'Request differs from exact owner plan')
        arm=arms[row['mode']]
        need(row['arm']==arm and row['generated']==refmap[row['length'],row['source_index']] and row['tokens']==len(row['generated']) and
            row['output_agrees'] is row['performance_eligible'] is True and all(row[k] is False for k in
            ('observer_enabled','profile_enabled','mapper_executed','controller_executed')) and
            'timeline' not in row and 'cost_samples' not in row,'Canonical/unprofiled request failed')
        need(finite(row['outer_seconds'],True) and finite(row['total_seconds'],True) and row['outer_seconds']+1e-8>=row['total_seconds'] and
            near(row['outer_tps'],row['tokens']/row['outer_seconds']),'Invalid full-generate clock')
        if arm['family']=='retrieval':linear(row,arm)
        else:s.tree(row,raw_map[row['request_id']],arm,eos_id)
        key=(arm['family'],arm['top_chunks'],arm.get('fixed_k',arm.get('nodes')),row['length'],row['source_index'])
        trace=s.trace_signature(row,arm)
        need(key not in trajectories or trajectories[key]==trace,'Cold/repeat/backend trajectory/work mismatch')
        trajectories[key]=trace
    need(set(raw_map)=={r['request_id'] for r in sd if r['arm']['family']=='full_tree'},'Tree raw coverage mismatch')
    need(len({o['owner_id'] for o in owners})==76,'Duplicate private owner')
    metrics={}
    for j,(owner,expected) in enumerate(zip(owners,owner_plan)):
        metrics[expected['length'],expected['mode']]=owner_audit(owner,sd[j*33:(j+1)*33],expected,sequence)
    telemetry=lines(root,'results/gpu_diagnostics.jsonl');need(len(telemetry)==152,'GPU diagnostic pairs incomplete')
    fields=['uuid','temperature.gpu','clocks.current.sm','clocks.current.memory','power.draw','power.limit','utilization.gpu','memory.used','memory.total']
    for j,owner in enumerate(owners):
        for offset,phase in enumerate(('owner_start','owner_end')):
            row=telemetry[2*j+offset];value=row['telemetry']
            need(row['phase']==phase and row['owner_id']==owner['owner_id'] and row['mode']==owner['mode'] and row['length']==owner['length'] and
                row['execution_index']==32+j*33+offset*33 and value['read_only'] is True and value['primary_clock_included'] is False and
                value['query_fields']==fields and set(value['values'])==set(fields) and finite(value['wall_time_unix'],True) and
                value['values']['uuid']=='GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9','GPU telemetry ownership/clock scope mismatch')
    recomputed=selection(warm,stage1);relative=analysis.relative_to(root).as_posix()
    summary,union,analyzed=[read(root,relative+'/'+n+'.json') for n in ('summary','VERIFIED_UNION','ANALYSIS_COMPLETE')]
    need(summary['status']=='COMPLETE' and summary['problems']==[] and summary['observed_counts']==COUNTS and summary['authoritative_union'] is True and
        union['status']=='VERIFIED_FROZEN_UNION' and union['authoritative'] is True and analyzed['status']=='COMPLETE',
         'Strict analysis not complete')
    need(frozen['per_context']==summary['stage2']==union['per_context']==recomputed and frozen['union_candidates']==
        union['future_contrasts']==sum(len(v['union_modes']) for v in recomputed.values()),'Retained stage1+stage2 union/ranking differs from raw')
    for (n,mode),metric in metrics.items():
        actual=summary['metrics'][str(n)][mode];rows=[r for r in warm if r['mode']==mode and r['length']==n]
        proposed=None if arms[mode]['family']=='full_tree' else sum(r['proposed'] for r in rows)
        need(actual['documents']==16 and actual['warm_requests']==32 and actual['warm_tokens']==metric['tokens'] and
            near(actual['warm_seconds'],metric['seconds']) and near(actual['pooled_tps'],metric['tps']) and
            near(actual['acceptance_length']['point'],sum(r['accepted'] for r in rows)/sum(r['blocks'] for r in rows)) and
            actual['warm_captures']==metric['captures'] and near(actual['warm_capture_seconds'],metric['capture_seconds']),
             'Reported raw quality/TPS/capture/whole-document count mismatch')
        owner=next(o for o in owners if o['length']==n and o['mode']==mode)
        need((actual['acceptance_rate']['point'] is None if proposed is None else
            near(actual['acceptance_rate']['point'],sum(r['accepted'] for r in rows)/proposed)) and
            near(actual['cold_complete_tps'],owner['prime_tokens']/(owner['constructor_and_detach_seconds']+owner['cold_prime_outer_seconds'])) and
            near(actual['accounted_service_tps'],owner['accounted_service_tps']),
            'Reported acceptance rate/cold/service differs from raw owner ledger')
    for name,sha in analyzed['artifacts_sha256'].items():
        need(digest(local(root,relative+'/'+name))==sha,'Analyzed artifact changed: '+name)
    verify_collection(root,collection,protocol)
    need(digest(local(root,'TERMINAL_COLLECTION.json'))==collection_sha,'Terminal collection receipt changed during audit')
    need(Path(__file__).read_bytes()==AUDIT_SOURCE_BYTES and SUPPORT_PATH.read_bytes()==SUPPORT_SOURCE_BYTES,
        'Independent audit/source support changed after loading')
    source,dependency=AUDIT_SOURCE_BYTES,SUPPORT_SOURCE_BYTES
    receipt=dict(status='INDEPENDENT_RAW_AUDIT_PASSED',passed=True,audited_at_utc=datetime.now(timezone.utc).isoformat(),
        counts=COUNTS,logical_requests=2540,all2540_requests_verified=True,all76_owners_verified=True,retained_union_verified=True,
        canonical_sd_requests=2508,tree_raw_records=528,independent_ar_requests=32,telemetry_records=152,
        horizon_complete_sha256=digest(root/'results/COMPLETE.json'),horizon_protocol_sha256=digest(root/'results/protocol.json'),
        frozen_selection_sha256=digest(root/'results/FROZEN_SELECTION.json'),analysis_summary_sha256=digest(analysis/'summary.json'),
        verified_union_sha256=digest(analysis/'VERIFIED_UNION.json'),stage1_selection_sha256=stage1_sha,
        independently_verified_tokenizer_eos=eos,independently_recomputed_union=recomputed,
        audit_source_file=names[1],audit_source_sha256=hashlib.sha256(source).hexdigest(),
        dependency_file=names[2],dependency_sha256=hashlib.sha256(dependency).hexdigest(),
        terminal_collection_sha256=collection_sha,terminal_files_verified=len(collection['files']),
        scope='Independent saved-record development audit, not new confirmation, figure appearance, tensor equality or goal completion',goal_complete=False)
    for name,payload in ((names[1],source),(names[2],dependency),(names[0],(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode())):
        with (root/name).open('xb') as handle:handle.write(payload)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--analysis-dir',type=Path,required=True)
    args=parser.parse_args();result=audit(args.root,args.analysis_dir)
    print(json.dumps(dict(status=result['status'],counts=result['counts'],goal_complete=False),indent=2))
