"""CPU synthetic program tests only; no corpus/model/scientific data opened."""
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from rescue import fair_union_confirmation_report as report
from rescue import fair_union_confirmation as runner

PROJECT=Path(__file__).resolve().parents[1]


def load_test(name):
    spec=importlib.util.spec_from_file_location('union_test_'+name,PROJECT/'tests'/('test_rescue_'+name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def rows(path,values):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(v)+'\n' for v in values))


def raw_fixture(root,maximum=False,eos=False,choices=None):
    """Actual runner menu/schedule/configs, synthetic every-step saved evidence."""
    if choices is None:choices=runner.frozen_choices(load_test('fair_union_confirmation').selection(maximum))
    docs=list(range(3,19));sequence,owner_plan,warm_plan=runner.timing_schedule(choices,docs)
    arms={str(n):[dict(report.OWN),*choices[str(n)].values()] for n in report.LENGTHS}
    reference=[4,100000] if eos else list(range(128));limit=len(reference)
    ledgers={k:[] for k in ('reference','own_quality','baseline_quality','matching_quality','prime','runtime')}
    qo,owners,tree_raw=[],[],[];index=0
    def register(name,row):
        nonlocal index
        row.update(request_id=f'union:{index:032x}',execution_index=index);index+=1;ledgers[name].append(row)
        if row.get('arm',{}).get('family')=='full_tree':
            tree_raw.append(dict(request_id=row['request_id'],execution_index=row['execution_index'],
                result=dict(generated=row['wrapper_generated'],tokens=row['wrapper_tokens'],upstream=row['upstream'])))
        return row
    def linear(arm,n,ref):
        own=arm['family']=='rescue';trace,obs,queries=[],[],[];progress=0
        while progress<len(ref):
            cap=min((2 if len(trace)%2==0 else 4) if own else arm['fixed_k'],128-progress)
            proposed=min(cap,len(ref)-progress);accepted=proposed-1;prefix=n-1+progress
            selected=list(range(64 if own else arm['top_chunks']*32))
            draft=ref[progress:progress+accepted]+[999];checks=(ref+[128]*13)[progress:progress+proposed+1]
            trace.append(dict(round_index=len(trace),prefix=prefix,cap=cap,proposed=proposed,accepted=accepted,
                terminal=progress+proposed==len(ref),workset_tokens=len(selected)))
            working=dict(shape=[64,256],dtype='torch.bfloat16',sha256='a'*64);native=dict(working,sha256='b'*64)
            proof=[dict(layer=27,positions=selected[:64],working_value=working,native_value=native,different_elements=1,native_matches_memo=True)] if own else []
            obs.append(dict(round_index=len(obs),prefix=prefix,progress=progress,remaining=128-progress,selected=selected,
                draft=draft,checks=checks,accepted=accepted,budget_features=[0.]*11+[.2] if own and proposed>=2 else None,injection_evidence=proof))
            for depth in range(proposed):queries.append(dict(logits=True,query=dict(shape=[1,4,1,8],dtype='torch.bfloat16',sha256='c'*64),
                summary_requested=own and depth<2,summary_count=int(own and depth<2)))
            progress+=proposed
        length=n+len(ref);native=dict(lengths=[length]*28,layers=[dict(key=dict(shape=[1,2,length,8],dtype='torch.bfloat16',sha256='d'*64),
            value=dict(shape=[1,2,length,8],dtype='torch.bfloat16',sha256='e'*64)) for _ in range(28)])
        accepted=sum(t['accepted'] for t in trace);proposed=sum(t['proposed'] for t in trace)
        return dict(generated=ref.copy(),tokens=len(ref),trace=trace,blocks=len(trace),accepted=accepted,proposed=proposed,
            acceptance_rate=accepted/proposed,draft_forwards=proposed,selector_refreshes=(len(trace)+3)//4,working_gather_layers=len(trace)*28,
            projected_positions=len(trace)*64 if own else 0,draft_shadow_probes=0,observed_rounds=obs,queries=queries,
            native_t=native,native_s=copy.deepcopy(native),nonzero_injection=own)
    def tree(arm,n,ref):
        # Full-tree raw accepted work includes overshoot; scored output retains
        # canonical EOS, without inventing unavailable raw tail token IDs.
        blocks=(len(ref)+3)//4;accepted=[3]*blocks;raw=blocks*4;scored=ref.copy()
        upstream=dict(output_token_ids=scored,total_generated=len(scored),raw_total_generated=raw,generation_overshoot=raw-len(scored),accept_length_list=accepted)
        return dict(generated=ref.copy(),tokens=len(ref),wrapper_generated=ref.copy(),wrapper_tokens=len(ref),upstream=upstream,
            raw_total_generated=raw,generation_overshoot=raw-len(ref),scored_generated_before_eos=scored,raw_tail_ids_available=False,
            accepted=sum(accepted),blocks=blocks,proposed=None,acceptance_rate=None,tree_policy=runner.baseline.tree_policy(arm),
            observed_full_config=dict(target_use_flash_prefill=True,target_use_hybrid_tree_attn=True,draft_use_flash_prefill=True,
                use_retrieval_cache=True,retrieve_top_k=arm['top_chunks'],retrieval_chunk_size=32,retrieve_every_n_steps=4))
    def runtime(arm,count,cold,forwards):
        graph,resident=arm['backend']=='cuda_graph',arm['backend']=='resident'
        delta=dict(captures=2 if cold and graph else 0,replays=forwards if graph else 0,resident_steps=forwards if resident else 0,
            capture_host_seconds=.2 if cold and graph else 0.)
        stat=dict(mode=arm['backend'],entries=2 if graph else 0,captures=2 if graph else 0,replays=forwards*count if graph else 0,
            resident_steps=forwards*count if resident else 0,capture_host_seconds=.2 if graph else 0.,max_graph_entries=128 if graph or resident else None)
        return delta,stat
    for n in report.LENGTHS:
        for d in docs:register('reference',dict(source_index=d,length=n,mode='canonical_ar',phase='reference',generated=reference.copy(),performance_eligible=False))
    def quality(arm,d,n,name,phase,targets=None):
        row=tree(arm,n,reference) if arm['family']=='full_tree' else linear(arm,n,reference)
        _,stats=runtime(arm,1,True,row.get('draft_forwards',0));oid=f'quality:{index}'
        row.update(mode=arm['mode'],arm=arm,source_index=d,length=n,phase=phase,owner_id=oid,performance_eligible=False,
            output_agrees=True,runtime_stats=stats,equivalence_targets=targets or [])
        register(name,row);qo.append(dict(request_id=row['request_id'],owner_id=oid,engine_closed=True,hook_restored=True,
            final_runtime_stats=copy.deepcopy(stats),closed_runtime_stats=dict(stats,entries=0)));return row
    extras=runner.extra_geometries(choices)
    for n in report.LENGTHS:
        for d in docs:
            quality(dict(mode='rescue_eager',family='rescue',backend='eager'),d,n,'own_quality','own_quality')
            quality(report.OWN,d,n,'own_quality','own_quality')
        for arm in choices[str(n)].values():quality(arm,docs[0],n,'baseline_quality','baseline_sentinel')
        for g in extras:
            if g['length']!=n:continue
            arm=dict(mode=f"matching_retrieval{g['top_chunks']}_k{g['fixed_k']}_eager",family='retrieval',backend='eager',top_chunks=g['top_chunks'],fixed_k=g['fixed_k'])
            targets=[a['mode'] for a in choices[str(n)].values() if a['family']=='retrieval' and report.geometry(a)==report.geometry(arm)]
            quality(arm,docs[0],n,'matching_quality','matching_geometry_eager',targets)
    for oi,owner in enumerate(owner_plan):
        mode,n=owner['mode'],owner['length'];arm=next(a for a in arms[str(n)] if a['mode']==mode);oid=f'timing:{oi}';sig=dict(identity=oi+1000);owned=[];forwards_total=0
        plan=[dict(mode=mode,length=n,source_index=271,phase='cold_prime')]+[dict(v,mode=mode,length=n,phase='warm',warm_index=j) for j,v in enumerate(sequence)]
        for count,item in enumerate(plan,1):
            cold=count==1;ref=list(range(128)) if cold else reference
            row=tree(arm,n,ref) if arm['family']=='full_tree' else linear(arm,n,ref)
            for k in ('observed_rounds','queries','native_t','native_s','nonzero_injection'):row.pop(k,None)
            f=row.get('draft_forwards',0);delta,stats=runtime(arm,count,cold,f);forwards_total+=f
            stats.update(replays=forwards_total if arm['backend']=='cuda_graph' else 0,resident_steps=forwards_total if arm['backend']=='resident' else 0)
            seconds=(.6 if arm['family']=='rescue' else 1.)*(3. if cold else 2.+docs.index(item['source_index'])/20)
            row.update(item,arm=arm,owner_id=oid,owner_signature=sig,runtime_stats=stats,performance_eligible=True,output_agrees=True,
                observer_enabled=False,profile_enabled=False,mapper_executed=arm['family']=='rescue',controller_executed=arm['family']=='rescue',
                outer_seconds=seconds,outer_tps=len(ref)/seconds,total_seconds=seconds-.01,prefill_seconds=.1,decode_seconds=seconds-.12,cleanup_seconds=.01)
            if arm['family']=='retrieval':row['execution_backend']=arm['backend']
            if arm['backend'] in ('resident','cuda_graph'):row['draft_runtime']=dict(mode=arm['backend'],request=delta,lifetime=copy.deepcopy(stats))
            register('prime' if cold else 'runtime',row);owned.append(row)
        seconds=.02+sum(r['outer_seconds'] for r in owned)+.001;final=owned[-1]['runtime_stats']
        owners.append(dict(mode=mode,length=n,owner_id=oid,owner_signature=sig,maximum_live_owners=1,prime_request_id=owned[0]['request_id'],
            warm_request_ids=[r['request_id'] for r in owned[1:]],constructor_and_detach_seconds=.02,close_seconds=.001,prime_tokens=128,
            cold_prime_outer_seconds=owned[0]['outer_seconds'],warm_tokens=sum(r['tokens'] for r in owned[1:]),warm_outer_seconds_sum=sum(r['outer_seconds'] for r in owned[1:]),
            accounted_service_seconds=seconds,accounted_service_tps=sum(r['tokens'] for r in owned)/seconds,engine_closed=True,hook_restored=True,
            final_runtime_stats=copy.deepcopy(final),closed_runtime_stats=dict(final,entries=0)))
    expected=runner.expected_counts(choices);m=runner.validate_choices(choices)
    assert index==runner.scientific_requests(choices)
    for name,values in ledgers.items():rows(root/(name+'.jsonl'),values)
    rows(root/'owners.jsonl',owners);rows(root/'quality_owners.jsonl',qo);rows(root/'tree_raw.jsonl',tree_raw)
    dump(root/'references.json',{f'{d}:{n}':reference for d in docs for n in report.LENGTHS});dump(root/'prime_references.json',{str(n):list(range(128)) for n in report.LENGTHS})
    dump(root/'QUALITY_PASSED.json',dict(passed=True,own_all32_pairs=True,baseline_matching_geometry=True,actual_same_bfloat16_nonzero_mapper=True,
        quality_owners_closed=True,comparisons=m,matching_geometry_count=len(extras)))
    protocol=dict(arms=arms,arms_by_context=arms,modes_by_context={n:[a['mode'] for a in values] for n,values in arms.items()},
        source_indices=docs,lengths=list(report.LENGTHS),repeats=2,max_new_tokens=128,seed=report.SEED,frozen_choices=choices,comparisons=m,
        expected_counts=expected,expected_requests=index,extra_eager_reference_geometries=extras,owner_order=owner_plan,warm_order=warm_plan,
        shared_warm_sequence=sequence,tokenizer_eos_token_id=100000,vocab_size=151936,target_layers=28,draft_layers=28,graph_max_entries=128,
        configs={n:{a['mode']:asdict(runner.legacy.own_config()) if a['family']=='rescue' else asdict(runner.baseline.linear_config(a)) if a['family']=='retrieval' else runner.baseline.tree_policy(a) for a in values} for n,values in arms.items()})
    dataset=dict(documents=docs,input_sha256={(d,n):'a'*64 for d in docs for n in report.LENGTHS})
    protocol['input_sha256']={f'{d}:{n}':v for (d,n),v in dataset['input_sha256'].items()}
    dump(root/'protocol.json',protocol)
    dump(root/'COMPLETE.json',dict(passed=True,counts=expected,expected_counts=expected,logical_requests=index,comparisons=m,matching_geometry_count=len(extras),
        **{k:True for k in ('sources_unchanged','assets_unchanged','dependencies_unchanged','quality_gate_passed','all_canonical','owners_closed','analysis_required_locally','data_unchanged','dual_study_freeze_unchanged')},goal_complete=False))
    return dict(protocol=protocol,dataset=dataset,choices=choices,ledgers=ledgers,expected=expected,owners=owners)


def seal_study(root,analysis,stage2):
    """Full synthetic byte-bound root/render/independent/rerun contract."""
    result=root/'results';figures=('quality','tps','captures','capture_time','retained_tps','cold_service','cumulative','latency','gpu_telemetry') if stage2 else ('quality','tps','captures','capture_time','cold','service','selected_repeat1')
    for name in figures:
        for suffix in ('.png','.svg'):
            (analysis/(name+suffix)).write_text('SYNTHETIC PROGRAM FIXTURE; NOT A REAL SCIENTIFIC FIGURE\n')
    dump(analysis/'RENDER_INSPECTION.json',dict(actually_inspected_by_root=True,rendered_pngs_inspected=list(figures),
        figure_sha256={name+suffix:report.sha(analysis/(name+suffix)) for name in figures for suffix in ('.png','.svg')}))
    source=root/'INDEPENDENT_RAW_AUDIT_SOURCE.py';source.write_text('# Synthetic test audit source only.\n')
    audit=dict(status='INDEPENDENT_RAW_AUDIT_PASSED',all76_owners_verified=True,audit_source_file=source.name,audit_source_sha256=report.sha(source),
        frozen_selection_sha256=report.sha(result/'FROZEN_SELECTION.json'),analysis_summary_sha256=report.sha(analysis/'summary.json'))
    audit.update({'all2540_requests_verified' if stage2 else 'all538_requests_verified':True,
        'retained_union_verified' if stage2 else 'eight_frozen_choices_verified':True,
        'horizon_complete_sha256' if stage2 else 'baseline_complete_sha256':report.sha(result/'COMPLETE.json'),
        'horizon_protocol_sha256' if stage2 else 'baseline_protocol_sha256':report.sha(result/'protocol.json')})
    if stage2:audit['verified_union_sha256']=report.sha(analysis/'VERIFIED_UNION.json')
    dump(root/'INDEPENDENT_RAW_AUDIT.json',audit)
    if stage2:
        (analysis/'analysis_support.py').write_text('# Synthetic test dependency only.\n')
        dump(root/'ANALYSIS_RERUN_VERIFIED.json',dict(standalone_summary_exact_rerun=True,summary_sha256=report.sha(analysis/'summary.json'),
            saved_analysis_sha256=report.sha(analysis/'analysis_source.py'),dependency_sha256={'analysis_support.py':report.sha(analysis/'analysis_support.py')}))
    dump(analysis/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',artifacts_sha256={p.name:report.sha(p) for p in analysis.iterdir() if p.is_file() and p.name!='ANALYSIS_COMPLETE.json'}))
    dump(root/'FINAL_ARTIFACTS_MANIFEST.json',dict(file_sha256={str(p.relative_to(root)):report.sha(p) for p in root.rglob('*') if p.is_file() and p.name not in ('FINAL_ARTIFACTS_MANIFEST.json','VERIFIED_COMPLETE.json')}))
    verified=dict(status='VERIFIED_COMPLETE',passed=True,raw_independent_audit_passed=True,rendered_figures_inspected=True,strict_analysis_complete=True,
        all76_owners_verified=True,standalone_summary_exact_rerun=True,frozen_selection_sha256=report.sha(result/'FROZEN_SELECTION.json'),
        analysis_summary_sha256=report.sha(analysis/'summary.json'),final_artifacts_manifest_sha256=report.sha(root/'FINAL_ARTIFACTS_MANIFEST.json'),
        independent_raw_audit_sha256=report.sha(root/'INDEPENDENT_RAW_AUDIT.json'),render_inspection_sha256=report.sha(analysis/'RENDER_INSPECTION.json'))
    if stage2:verified.update(all2540_requests_verified=True,retained_union_verified=True,horizon_complete_sha256=report.sha(result/'COMPLETE.json'),
        verified_union_sha256=report.sha(analysis/'VERIFIED_UNION.json'),standalone_rerun_receipt_sha256=report.sha(root/'ANALYSIS_RERUN_VERIFIED.json'))
    else:verified.update(all538_requests_verified=True,eight_frozen_choices_verified=True,baseline_complete_sha256=report.sha(result/'COMPLETE.json'),verified_winners_sha256=report.sha(analysis/'VERIFIED_WINNERS.json'))
    dump(root/'VERIFIED_COMPLETE.json',verified)


def complete_fixture(root,maximum=False):
    """Actual production data gate/prepare API; all content is synthetic."""
    from rescue import fair_union_confirmation_data as data
    helper=load_test('fair_union_confirmation_data');env=helper.fixture(root/'preparation')
    p1=json.loads((env['results']/'protocol.json').read_text());first=json.loads((env['results']/'FROZEN_SELECTION.json').read_text())
    environment=dict(p1['environment'],gpu_name='NVIDIA H100 80GB HBM3',total_memory=85_899_345_920)
    flash=dict(version='2.8.3',files={'flash_attn_2_cuda':dict(path='/synthetic/flash_attn_2_cuda.so',sha256='e'*64)})
    menu=data._menu();p1.update(arms=menu,environment=environment,flash_attention=flash,input_sha256={f'271:{n}':hashlib.sha256(json.dumps([1]+[42]*(n-2)+[2]).encode()).hexdigest() for n in report.LENGTHS})
    dump(env['results']/'protocol.json',p1)
    oldwarm=[]
    for n in report.LENGTHS:
        for arm in menu:
            chosen=first['per_context'][str(n)]['groups'][data._group(arm)]['mode']==arm['mode']
            for rep in (0,1):
                for doc in (271,272,288):oldwarm.append(dict(mode=arm['mode'],length=n,tokens=128,outer_seconds=1. if chosen else 2.,source_index=doc,repeat=rep))
    rows(env['results']/'runtime.jsonl',oldwarm);dump(env['results']/'references.json',{f'271:{n}':list(range(128)) for n in report.LENGTHS})
    first['protocol_sha256']=report.sha(env['results']/'protocol.json');dump(env['results']/'FROZEN_SELECTION.json',first)
    complete=json.loads((env['results']/'COMPLETE.json').read_text());complete['frozen_selection_sha256']=report.sha(env['results']/'FROZEN_SELECTION.json');dump(env['results']/'COMPLETE.json',complete)
    seal_study(env['stage1'],env['analysis'],False)
    proof1=data.old.validate_baseline_freeze(**{k:env['gate_args'][name] for k,name in (('baseline_results','stage1_results'),('analysis_dir','stage1_analysis'),('verified_complete','stage1_verified'),('final_manifest','stage1_manifest'))})
    p2=json.loads((env['r2']/'protocol.json').read_text());p2.update(environment=environment,flash_attention=flash,stage1_freeze=proof1,stage1_selection_sha256=proof1['selection_freeze_sha256'])
    dump(env['r2']/'protocol.json',p2)
    warm=[json.loads(v) for v in (env['r2']/'runtime.jsonl').read_text().splitlines()]
    if not maximum:
        for row in warm:
            chosen=first['per_context'][str(row['length'])]['groups'][data._group(row['arm'])]['mode']==row['mode']
            row['outer_seconds']=1. if chosen else 2.
        rows(env['r2']/'runtime.jsonl',warm)
    computed=data._recompute(warm,p2,first)
    frozen=json.loads((env['r2']/'FROZEN_SELECTION.json').read_text());frozen.update(per_context=computed,protocol_sha256=report.sha(env['r2']/'protocol.json'),stage1_selection_sha256=proof1['selection_freeze_sha256'])
    dump(env['r2']/'FROZEN_SELECTION.json',frozen)
    complete=json.loads((env['r2']/'COMPLETE.json').read_text());complete.update(stage1_selection_sha256=proof1['selection_freeze_sha256'],frozen_selection_sha256=report.sha(env['r2']/'FROZEN_SELECTION.json'));dump(env['r2']/'COMPLETE.json',complete)
    m=sum(len(c['union_modes']) for c in computed.values())
    dump(env['a2']/'summary.json',dict(status='COMPLETE',problems=[],observed_counts=data.STAGE2_COUNTS,authoritative_union=True,stage2=computed))
    dump(env['a2']/'VERIFIED_UNION.json',dict(status='VERIFIED_FROZEN_UNION',authoritative=True,per_context=computed,future_contrasts=m))
    seal_study(env['stage2'],env['a2'],True)
    proof=data.validate_union_freeze(**env['args'])
    formal=data.make_final_protocol(env['union_source']/data.REQUIRED_NEW_SOURCE[-2],proof);dump(env['formal'],formal)
    # Real pure preparation APIs with a synthetic tokenizer and synthetic text
    # iterator. These are program tests, never candidate-corpus access.
    data.freeze_gate(env['union_gate'],**env['gate_args'])
    with patch.object(data,'_load_tokenizer',return_value=helper.Tokenizer()),patch.object(data,'_raw_rows',side_effect=lambda p:iter(helper.candidates())):
        prepared=data.prepare_confirmation(env['union_gate']/'GATE_SOURCE_FREEZE.json')
    gate=json.loads((env['union_gate']/'GATE_SOURCE_FREEZE.json').read_text());result=root/'results'
    state=raw_fixture(result,choices=proof['choices'])
    shutil=__import__('shutil');shutil.copytree(env['union_gate']/'data',result/'data')
    stage1=runner.copy_development_evidence(gate['baseline_arguments'],result,'stage1_evidence')
    stage2=runner.copy_development_evidence(gate['horizon_arguments'],result,'stage2_evidence')
    for name in gate['source_sha256']:
        target=result/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((env['union_source']/name).read_bytes())
    protocol=state['protocol'];sources=gate['source_sha256']
    protocol.update(stage1_evidence=stage1,stage2_evidence=stage2,union_freeze=proof,source_freeze_sha256=prepared['source_freeze_sha256'],
        dataset_manifest_sha256=report.sha(result/'data/DATASET_MANIFEST.json'),dataset_sha256=prepared['dataset_sha256'],dataset_manifest='data/DATASET_MANIFEST.json',
        portable_data_root='data',selection_rule=prepared['selection_rule'],final_protocol_sha256=prepared['final_protocol_sha256'],**report.ASSETS,
        mapper_frozen=True,policy_frozen=True,no_training=True,max_live_owners=1,template_prefill_precision='bfloat16',environment=environment,
        source_sha256=sources,flash_attention=flash,
        source_bindings={name:dict(path='/synthetic/after/'+name.replace('.','/')+'.py',sha256=sources[name.replace('.','/')+'.py']) for name in
            ('rescue.fair_union_confirmation','rescue.fair_union_confirmation_data','rescue.fair_union_confirmation_report')},
        tree_source_bindings={name:dict(path='exp/specextend_bridgekv/specextend/'+relative,sha256=sources['exp/specextend_bridgekv/specextend/'+relative]) for name,relative in
            (('classic.model_classic','classic/model_classic.py'),('shared.modeling_llama_kv_target','shared/modeling_llama_kv_target.py'))},
        prime_input_sha256={str(n):p1['input_sha256'][f'271:{n}'] for n in report.LENGTHS},
        input_sha256={f"{v['source_index']}:{n}":v['input_sha256'][str(n)] for v in prepared['selected'] for n in report.LENGTHS})
    protocol['args']={name:'/synthetic/'+name for name in ('mapper','policy','cost','freeze_receipt','dataset_manifest','data_file')}
    protocol['assets_sha256']=dict(p1['assets_sha256'],**{protocol['args'][name]:digest for name,digest in (
        ('mapper',report.ASSETS['mapper_sha256']),('policy',report.ASSETS['policy_sha256']),('cost',report.ASSETS['cost_sha256']),
        ('freeze_receipt',prepared['source_freeze_sha256']),('dataset_manifest',protocol['dataset_manifest_sha256']),('data_file',prepared['dataset_sha256']))})
    dump(result/'protocol.json',protocol);dump(result/'GATE_PASSED.json',dict(passed=True,union_freeze=proof,source_freeze_sha256=prepared['source_freeze_sha256'],
        expected_counts=state['expected'],expected_requests=protocol['expected_requests'],comparisons=m,extra_eager_reference_geometries=runner.extra_geometries(proof['choices']),confirmation_content_opened=False))
    dump(root/'EVIDENCE_MANIFEST.json',dict(file_sha256={str(p.relative_to(root)):report.sha(p) for p in result.rglob('*') if p.is_file()}))
    state.update(root=result,proof=proof);return state


class UnionReportTests(unittest.TestCase):
    def test_full_production_preparation_contract_min_max_chain_and_formal_gate(self):
        for maximum in (False,True):
            with self.subTest(maximum=maximum),tempfile.TemporaryDirectory() as tmp:
                state=complete_fixture(Path(tmp),maximum);value=report.analyze(state['root'])
                self.assertEqual(value['status'],'COMPLETE',value.get('problems'));self.assertTrue(value['verified_success'])
                self.assertEqual(value['comparisons'],16 if maximum else 8)
                self.assertEqual(value['statistics']['comparisons'],state['proof']['M'])
                self.assertEqual(value['observed_counts'],state['proof']['expected_counts'])
                self.assertEqual(value['expected_requests'],state['proof']['scientific_requests'])
                self.assertEqual(len(value['statistics']['contrasts']),state['proof']['M'])

    def test_full_chain_missing_union_m_scan_and_independent_receipts_cannot_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=complete_fixture(Path(tmp),True);root=state['root']
            names=('protocol.json','runtime.jsonl','data/DATASET_MANIFEST.json','stage2_evidence/results/analysis_final_v1/RENDER_INSPECTION.json')
            saved={name:(root/name).read_bytes() for name in names}
            for kind in ('opponent','M','union','scan','render'):
                with self.subTest(kind=kind):
                    if kind=='opponent':rows(root/'runtime.jsonl',[r for r in state['ledgers']['runtime'] if r['length']!=8192 or r['mode']!=next(iter(state['choices']['8192']))])
                    if kind=='M':
                        p=copy.deepcopy(state['protocol']);p['comparisons']=8;dump(root/'protocol.json',p)
                    if kind=='union':
                        p=copy.deepcopy(state['protocol']);p['frozen_choices']['4096'].pop(next(iter(p['frozen_choices']['4096'])));dump(root/'protocol.json',p)
                    if kind=='scan':
                        p=json.loads(saved['data/DATASET_MANIFEST.json']);p['skipped'].pop();dump(root/'data/DATASET_MANIFEST.json',p)
                    if kind=='render':
                        p=json.loads(saved[names[-1]]);p['actually_inspected_by_root']=False;dump(root/names[-1],p)
                    value=report.analyze(root);self.assertEqual(value['status'],'PARTIAL');self.assertFalse(value['verified_success']);self.assertTrue(value['problems'])
                    if kind=='opponent':
                        self.assertEqual(value['statistics']['per_context_n'],{'4096':16,'8192':0})
                        self.assertIn('4096',value['statistics']['contexts']);self.assertNotIn('8192',value['statistics']['contexts'])
                        self.assertTrue(all(c['simultaneous_one_sided_lower95'] is None for c in value['statistics']['contrasts']))
                    for name,content in saved.items():(root/name).write_bytes(content)

    def test_render_five_separate_families_and_saved_standalone_no_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=complete_fixture(Path(tmp),True);root=state['root']
            rows(root/'runtime.jsonl',[r for r in state['ledgers']['runtime'] if r['length']==4096])
            value=report.analyze(root);self.assertEqual(value['status'],'PARTIAL')
            output=root/'analysis';receipt=report.write_report(value,root,output)
            self.assertEqual(set(receipt['rendered_figures']),{'acceptance','tps','cold_service','latency_captures','actions_selection'})
            for name in receipt['rendered_figures']:
                self.assertGreater((output/(name+'.png')).stat().st_size,1000);self.assertGreater((output/(name+'.svg')).stat().st_size,1000)
            self.assertFalse((output/'VERIFIED_SUCCESS.json').exists())
            with self.assertRaises(FileExistsError):report.write_report(value,root,output)
            code="import runpy,sys; m=runpy.run_path('analysis_source.py'); assert 'torch' not in sys.modules; assert m['analyze']('..')['status']=='PARTIAL'"
            run=subprocess.run([__import__('sys').executable,'-c',code],cwd=output,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)

    def test_min_max_actual_runner_counts_schedule_and_every_raw_quality_owner(self):
        for maximum in (False,True):
            with self.subTest(maximum=maximum),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);state=raw_fixture(root,maximum);reader=report.Evidence(root)
                value=report.audit_execution(reader,state['protocol'],state['dataset'],state['choices'])
                self.assertEqual(reader.problems,[]);self.assertEqual(value['observed_counts'],state['expected'])
                self.assertEqual(value['observed_requests'],434 if not maximum else 714)
                self.assertEqual(value['paired_complete_documents'],state['dataset']['documents'])
                self.assertEqual(report.expected_counts(state['choices']),runner.expected_counts(state['choices']))
                self.assertEqual(report.registered_schedule(state['choices'],state['dataset']['documents']),
                    (state['protocol']['owner_order'],state['protocol']['warm_order'],state['protocol']['shared_warm_sequence']))

    def test_shared_bootstrap_dynamic_m_and_ratio_of_sums_independently(self):
        for maximum in (False,True):
            choices=runner.frozen_choices(load_test('fair_union_confirmation').selection(maximum))
            challengers={n:{m:m for m in values} for n,values in choices.items()};own={n:'rescue_graph' for n in choices};cells={}
            for n,values in choices.items():
                cells[n]={mode:[dict(tokens=128+2*d,seconds=(1.+d**.7)*(1. if mode=='rescue_graph' else 1.4+d/100),accepted=20,blocks=10,proposed=40) for d in range(16)] for mode in ['rescue_graph',*values]}
            value=report.synchronized_statistics(cells,own,challengers);m=sum(len(v) for v in choices.values())
            self.assertTrue(value['statistical_gate']);self.assertEqual(value['simultaneous_lower_quantile'],.05/m)
            index=np.random.default_rng(20261026).integers(0,16,size=(100000,16));n='4096';base=next(iter(choices[n]))
            t=np.array([d['tokens'] for d in cells[n]['rescue_graph']]);a=np.array([d['seconds'] for d in cells[n]['rescue_graph']]);b=np.array([d['seconds'] for d in cells[n][base]])
            dist=100.*((t[index].sum(1)/a[index].sum(1))/(t[index].sum(1)/b[index].sum(1))-1.)
            contrast=value['contrasts'][0]
            self.assertAlmostEqual(contrast['point_percent'],100.*(b.sum()/a.sum()-1.))
            np.testing.assert_allclose(contrast['pointwise_ci95'],np.percentile(dist,[2.5,97.5]),rtol=0,atol=1e-12)
            self.assertAlmostEqual(contrast['simultaneous_one_sided_lower95'],np.quantile(dist,.05/m))
            with self.assertRaises(ValueError):report.synchronized_statistics(cells,own,challengers,draws=10000)
            cells['8192'].pop(next(iter(choices['8192'])))
            with self.assertRaises(ValueError):report.synchronized_statistics(cells,own,challengers)

    def test_positive_point_alone_is_not_formal_union_success(self):
        choices=runner.frozen_choices(load_test('fair_union_confirmation').selection())
        challenges={n:{m:m for m in values} for n,values in choices.items()};own={n:'rescue_graph' for n in choices};cells={}
        for n,values in choices.items():
            cells[n]={mode:[dict(tokens=256,seconds=10.,accepted=20,blocks=10,proposed=40) for _ in range(16)] for mode in ['rescue_graph',*values]}
            cells[n]['rescue_graph']=[dict(tokens=256,seconds=8. if d else 35.,accepted=20,blocks=10,proposed=40) for d in range(16)]
        value=report.synchronized_statistics(cells,own,challenges)
        self.assertTrue(all(c['point_percent']>0 for c in value['contrasts']))
        self.assertTrue(any(c['simultaneous_one_sided_lower95']<=0 for c in value['contrasts']));self.assertFalse(value['statistical_gate'])

    def test_evidence_reader_rejects_outside_archive_before_opening(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader=report.Evidence(tmp)
            with patch.object(Path,'read_text',side_effect=AssertionError('outside path opened')):
                self.assertEqual(reader.json('../outside.json'),{});self.assertEqual(reader.rows('/outside.jsonl'),[])
            self.assertEqual(len(reader.problems),2)

    def test_missing_opponent_m_tamper_raw_schema_and_eos_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=raw_fixture(root,True,eos=True)
            reader=report.Evidence(root);value=report.audit_execution(reader,state['protocol'],state['dataset'],state['choices'])
            self.assertEqual(reader.problems,[]);self.assertEqual(value['paired_complete_documents'],state['dataset']['documents'])
            for kind in ('opponent','M','quality','tree','eos','owner'):
                with self.subTest(kind=kind):
                    saved={p:p.read_bytes() for p in root.glob('*.json*')};protocol=copy.deepcopy(state['protocol'])
                    if kind=='opponent':rows(root/'runtime.jsonl',[r for r in state['ledgers']['runtime'] if r['mode']!='tree32_n64'])
                    if kind=='M':protocol['comparisons']=8
                    if kind=='quality':
                        q=copy.deepcopy(state['ledgers']['own_quality']);q[0]['native_s']['layers'][1]['value']['sha256']='f'*64;rows(root/'own_quality.jsonl',q)
                    if kind=='tree':
                        q=copy.deepcopy(state['ledgers']['runtime']);next(r for r in q if r['arm']['family']=='full_tree')['wrapper_generated']=[999];rows(root/'runtime.jsonl',q)
                    if kind=='eos':
                        q=copy.deepcopy(state['ledgers']['reference']);q[0]['generated']=[4,5];rows(root/'reference.jsonl',q)
                    if kind=='owner':
                        q=copy.deepcopy(state['owners']);q[0]['accounted_service_seconds']-=1;rows(root/'owners.jsonl',q)
                    reader=report.Evidence(root);report.audit_execution(reader,protocol,state['dataset'],state['choices'])
                    if kind=='M':
                        # Protocol multiplicity is the metadata gate, never a
                        # parameter used to weaken the independently counted M.
                        self.assertNotEqual(protocol['comparisons'],sum(len(v) for v in state['choices'].values()))
                    else:self.assertTrue(reader.problems,kind)
                    for path,content in saved.items():path.write_bytes(content)
