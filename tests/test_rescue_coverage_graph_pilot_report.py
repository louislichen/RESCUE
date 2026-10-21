"""Synthetic saved-record audit fixtures; these are not model measurements."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rescue.coverage_graph_pilot_report import EXPECTED, LENGTHS, MODES, analyze, write_report
from rescue.draft_graph_benchmark_report import ASSETS


def save(path, rows):
    path.write_text(''.join(json.dumps(v)+'\n' for v in rows))


def fixture(directory):
    directory=Path(directory);tokens=list(range(128))
    canonical=[];quality=[];primes=[];warm=[];profiles=[];owners=[];contexts=[]

    def digest(shape):
        return dict(shape=shape,dtype='torch.bfloat16',sha256='a'*64)

    def stats(calls, captures, coverage=False, identity=None):
        result=dict(mode='cuda_graph',entries=captures,captures=captures,replays=calls*(8 if coverage else 128),capture_host_seconds=.02*captures)
        if coverage:
            result.update(owner_identity=identity,max_entries=64,eager_calls=0,input_copies=calls*8,shape_keys=[['pool',i] for i in range(captures)])
        else:
            result.update(resident_steps=0,shadow_updates=calls*128,graph_keys=[['step',i] for i in range(captures)])
        return result

    def row(mode,length,name,phase,index,calls=1,seconds=2.,new_coverage=0):
        owner_id=f'owner:{mode}:{length}' if phase != 'correctness' else f'quality-owner:{mode}:{length}'
        serial=length+(100 if mode == MODES[1] else 0)+(1000 if phase == 'correctness' else 0)
        pointer={k:dict(identity=serial+j,key_ptr=serial*100+j,value_ptr=serial*100+10+j,capacity=length*2)
                 for j,k in enumerate(('native_t','native_s','working'))}
        pointer.update(runtime=dict(identity=serial+4,bound_ptrs=[serial+5]*4),coverage_runtime_identity=serial+6 if mode == MODES[1] else None)
        value=dict(mode=mode,length=length,source_index=303,phase=phase,request_id=name,execution_index=index,
            owner_id=owner_id,owner_signature=pointer,generated=tokens.copy(),tokens=128,output_agrees=True,
            trace=[dict(round_index=i,prefix=length-1+4*i,cap=4,proposed=4,accepted=3,terminal=i==31,workset_tokens=512) for i in range(32)],
            blocks=32,accepted=96,proposed=128,draft_forwards=128,selector_refreshes=8,
            projected_positions=64,static_affine_positions=64,selection_similarity_elements=1048576,
            working_gathered_position_layers=512*28*32,prefill_seconds=.2,decode_seconds=seconds-.21,
            cleanup_seconds=.01,total_seconds=seconds,total_tps=128/seconds,decode_tps=128/(seconds-.21),
            performance_eligible=phase in ('warm','cold_prime'))
        draft=stats(calls,2);draft_delta=dict(captures=2 if calls==1 else 0,replays=128,resident_steps=0,
            capture_host_seconds=.04 if calls==1 else 0.,shadow_updates=128)
        value.update(runtime_stats=draft,draft_runtime=dict(mode='cuda_graph',request=draft_delta,lifetime=draft))
        if mode == MODES[1]:
            captures=2+(1 if calls>=2 else 0)+(1 if calls>=4 else 0)
            coverage=stats(calls,captures,True,pointer['coverage_runtime_identity'])
            delta=dict(captures=2 if calls==1 else new_coverage,replays=8,capture_host_seconds=.04 if calls==1 else .02*new_coverage)
            value.update(coverage_runtime_stats=coverage,coverage_runtime=dict(mode='cuda_graph',request=delta,lifetime=coverage))
        else:
            value['coverage_runtime_stats']=dict(mode='eager',owner_identity=None,max_entries=0,entries=0,captures=0,replays=0,capture_host_seconds=0.,eager_calls=0,input_copies=0,shape_keys=[])
        if phase in ('warm','cold_prime'):
            value.update(outer_seconds=seconds+.01,outer_tps=128/(seconds+.01))
        if phase == 'correctness':
            value.update(observed_rounds=[dict(round_index=i,prefix=length-1+4*i,progress=4*i,remaining=128-4*i,
                selected=list(range(64)),budget_features=[0.]*11+[.1],draft=[4*i,4*i+1,4*i+2,9999],
                checks=[4*i,4*i+1,4*i+2,4*i+3,9998],accepted=3) for i in range(32)],
                queries=[dict(logits=True,query=digest([1,12,1,128])) for _ in range(128)],
                native_t=dict(lengths=[length+128]*28,layers=[dict(key=digest([1,4,length+128,128]),value=digest([1,4,length+128,128])) for _ in range(28)]),
                native_s=dict(lengths=[length+127]*28,layers=[dict(key=digest([1,2,length+127,128]),value=digest([1,2,length+127,128])) for _ in range(28)]),nonzero_injection=True)
        if phase == 'profile':
            value.update(diagnostic_only=True,profile_enabled=True,profile_outer_seconds=seconds+.04,
                timeline=dict(stages=dict(selection=dict(calls=32,host_seconds=.1 if mode==MODES[1] else .3,stream_seconds=.15 if mode==MODES[1] else .4),
                    target_verify=dict(calls=32,host_seconds=1.,stream_seconds=1.4)),scope='Diagnostic overlapping queue/host intervals'),
                cost_samples=dict(student_step=[.001]*128,verify=[dict(n=length,k=4,seconds=.001)]*32,overhead=[.001]*32))
        return value

    for j,length in enumerate(LENGTHS):
        canonical.append(dict(mode='canonical_ar',source_index=303,length=length,generated=tokens.copy(),request_id=f'ar:{length}',execution_index=j))
        for i,mode in enumerate(MODES):
            quality.append(row(mode,length,f'q:{mode}:{length}','correctness',2+j*2+i))
        offset=6+j*6
        for i,mode in enumerate(MODES):
            factor=1. if i==0 else .9
            prime=row(mode,length,f'p:{mode}:{length}','cold_prime',offset+i,seconds=3.*factor)
            prime.update(constructor_and_detach_seconds=.02,cold_complete_outer_seconds=.02+prime['outer_seconds']);primes.append(prime)
            group=[]
            for repeat in range(2):
                value=row(mode,length,f'w:{mode}:{length}:{repeat}','warm',offset+2+repeat*2+i,
                    calls=repeat+2,seconds=(2.+.1*repeat)*factor,new_coverage=1 if repeat==0 else 0)
                value['repeat']=repeat;warm.append(value);group.append(value)
            profile=row(mode,length,f'profile:{mode}:{length}','profile',18+j*2+i,calls=4,seconds=2.*factor,new_coverage=1)
            profiles.append(profile)
            active=.02+prime['outer_seconds']+sum(r['outer_seconds'] for r in group)+.01
            closed_draft=copy.deepcopy(profile['runtime_stats']);closed_draft.update(entries=0,graph_keys=[])
            closed_coverage=copy.deepcopy(profile['coverage_runtime_stats']);closed_coverage.update(entries=0,shape_keys=[])
            owners.append(dict(mode=mode,length=length,owner_id=prime['owner_id'],owner_signature=prime['owner_signature'],
                constructor_and_detach_seconds=.02,prime_request_id=prime['request_id'],prime_tokens=128,
                cold_prime_outer_seconds=prime['outer_seconds'],warm_tokens=256,warm_outer_seconds_sum=sum(r['outer_seconds'] for r in group),
                warm_request_ids=[r['request_id'] for r in group],profile_request_ids=[profile['request_id']],profile_tokens=128,
                profile_outer_seconds_sum=profile['profile_outer_seconds'],close_seconds=.01,owner_lifetime_wall_seconds=50.,
                accounted_service_seconds=active,accounted_service_tokens=384,accounted_service_tps=384/active,
                profile_excluded_from_service=True,hook_restored=True,engine_closed=True,
                final_runtime_stats=profile['runtime_stats'],final_coverage_runtime_stats=profile['coverage_runtime_stats'],
                closed_runtime_stats=closed_draft,closed_coverage_runtime_stats=closed_coverage))
        selected=[r for r in primes+warm if r['length']==length]
        contexts.append(dict(length=length,timing_client_tokens=128*6,
            unprofiled_block_wall_seconds=sum(r['outer_seconds'] for r in selected)+.04+1.))
    warm.sort(key=lambda r:r['execution_index'])
    configs={m:dict(selector='hierarchical',budget=512,recent=64,coverage_rows=64,coverage_batch=16,refresh_every=4,
        refresh_policy='periodic',injection_layers=[27],inject_tokens=64,injection_strength=.5,prefill_precision='bfloat16',
        max_new_tokens=128,max_draft=12,length_policy='risk',schedule='serial',profile=False,draft_execution='cuda_graph',
        coverage_execution='cuda_graph' if m==MODES[1] else 'eager') for m in MODES}
    protocol=dict(source_index=303,lengths=list(LENGTHS),max_new_tokens=128,repeats=2,modes=list(MODES),expected_counts=EXPECTED,
        expected_requests=22,mapper_frozen=True,policy_frozen=True,no_training=True,**ASSETS,
        assets_sha256={'/synthetic/'+name:ASSETS[key] for name,key in (('full_final.pt','mapper_sha256'),('budget_selected.pt','policy_sha256'),('cost.json','cost_sha256'))},
        configs=configs,draft_graph_max_entries=128,coverage_graph_max_entries=64,retain_query=True,
        environment=dict(gpu_name='NVIDIA H100',total_memory=85_000_000_000,target_dtype='torch.float32',draft_dtype='torch.bfloat16',prefix_dtype='torch.bfloat16',tf32=False),
        source_sha256={},warm_order=[{k:r[k] for k in ('mode','source_index','length','repeat')} for r in warm])
    source=directory/'source/rescue/synthetic.py';source.parent.mkdir(parents=True,exist_ok=True);source.write_text('# synthetic unit test, not model data\n')
    protocol['source_sha256']['rescue/synthetic.py']=hashlib.sha256(source.read_bytes()).hexdigest()
    for name,value in (('protocol.json',protocol),('references.json',{str(n):tokens for n in LENGTHS}),
        ('COMPLETE.json',dict(passed=True,counts=EXPECTED,sources_unchanged=True,assets_unchanged=True,profile_excluded_from_service=True))):
        (directory/name).write_text(json.dumps(value))
    data=dict(canonical=canonical,quality=quality,primes=primes,warm=warm,profiles=profiles,owners=owners,contexts=contexts,protocol=protocol)
    for key,name in (('canonical','reference.jsonl'),('quality','correctness.jsonl'),('primes','prime.jsonl'),('warm','runtime.jsonl'),('profiles','profiles.jsonl'),('owners','owners.jsonl'),('contexts','contexts.jsonl')):
        save(directory/name,data[key])
    return data


class CoveragePilotReportTests(unittest.TestCase):
    def test_complete22_has_no_ci_and_profile_is_not_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp);result=analyze(tmp)
        self.assertEqual(result['status'],'COMPLETE',result['problems'])
        self.assertEqual(result['observed_counts'],EXPECTED)
        self.assertIsNone(result['confidence_intervals'])
        self.assertEqual(result['usable_repeats'],{'4096':[0,1],'8192':[0,1]})
        self.assertEqual(result['metrics']['4096'][MODES[0]]['acceptance_length'],3.)
        self.assertAlmostEqual(result['metrics']['4096'][MODES[0]]['warm_tps'],256/4.12)
        self.assertEqual(result['diagnostics']['4096'][MODES[1]]['coverage_warm_captures'],1)

    def test_profile_time_cannot_enter_service_or_primary_tps(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=fixture(tmp);path=Path(tmp)
            owner=data['owners'][0];owner['accounted_service_seconds']+=owner['profile_outer_seconds_sum']
            owner['accounted_service_tps']=512/owner['accounted_service_seconds']
            save(path/'owners.jsonl',data['owners']);result=analyze(tmp)
        self.assertEqual(result['status'],'PARTIAL')
        self.assertTrue(any('service accounting' in v for v in result['problems']))
        self.assertEqual(result['usable_repeats']['4096'],[])

    def test_profiles_must_follow_all_warm_requests_and_have_no_outer_timing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=fixture(tmp);path=Path(tmp)
            data['profiles'][0]['outer_seconds']=.001
            data['profiles'][0]['performance_eligible']=True
            data['profiles'][0]['execution_index'],data['warm'][-1]['execution_index']=data['warm'][-1]['execution_index'],data['profiles'][0]['execution_index']
            save(path/'profiles.jsonl',data['profiles']);save(path/'runtime.jsonl',data['warm']);result=analyze(tmp)
        self.assertEqual(result['status'],'PARTIAL')
        self.assertTrue(any('Profiles ran before' in v for v in result['problems']))
        self.assertTrue(any('leaked into primary' in v for v in result['problems']))

    def test_exact_feature_native_query_and_token_evidence_are_checked(self):
        for field in ('features','native','query','token'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                data=fixture(tmp);row=data['quality'][1]
                if field=='features':row['observed_rounds'][0]['budget_features'][0]=.1
                elif field=='native':row['native_s']['layers'][0]['key']['sha256']='b'*64
                elif field=='query':row['queries'][0]['query']['sha256']='b'*64
                else:row['generated'][0]=999
                save(Path(tmp)/'correctness.jsonl',data['quality']);result=analyze(tmp)
                self.assertEqual(result['status'],'PARTIAL')
                self.assertEqual(result['usable_repeats']['4096'],[])

    def test_missing_warm_cell_is_partial_and_only_complete_other_context_is_plotted(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=fixture(tmp);data['warm']=data['warm'][1:]
            save(Path(tmp)/'runtime.jsonl',data['warm']);result=analyze(tmp)
        self.assertEqual(result['status'],'PARTIAL')
        self.assertEqual(result['usable_repeats']['4096'],[])
        self.assertEqual(result['usable_repeats']['8192'],[0,1])

    def test_capture_owner_lifetime_and_release_are_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=fixture(tmp)
            data['warm'][1]['coverage_runtime_stats']['captures']+=1
            data['owners'][1]['closed_coverage_runtime_stats']['entries']=1
            save(Path(tmp)/'runtime.jsonl',data['warm']);save(Path(tmp)/'owners.jsonl',data['owners']);result=analyze(tmp)
        self.assertEqual(result['status'],'PARTIAL')
        self.assertTrue(any('lifetime counters' in v for v in result['problems']))
        self.assertTrue(any('retained graph entries' in v for v in result['problems']))

    def test_source_tamper_suppresses_all_performance_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp);(Path(tmp)/'source/rescue/synthetic.py').write_text('# changed\n')
            result=analyze(tmp)
        self.assertEqual(result['status'],'PARTIAL')
        self.assertEqual(result['usable_repeats'],{'4096':[],'8192':[]})

    def test_three_figure_sets_render_and_analysis_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp);path=Path(tmp);output=path/'analysis'
            result=write_report(path,output)
            self.assertEqual(result['status'],'COMPLETE',result['problems'])
            for name in ('acceptance','tps','diagnostics'):
                self.assertTrue((output/(name+'.png')).read_bytes().startswith(b'\x89PNG'))
                self.assertIn('<svg',(output/(name+'.svg')).read_text())
            self.assertTrue((output/'analysis_source.py').is_file())
            self.assertTrue((output/'analysis_support.py').is_file())
            self.assertIn('No 95%',(output/'REPORT.md').read_text())
            rerun=path/'saved_rerun'
            subprocess.run([sys.executable,str(output/'analysis_source.py'),str(path),
                '--output',str(rerun)],capture_output=True,text=True,check=True,timeout=30)
            self.assertEqual(json.loads((rerun/'summary.json').read_text())['status'],'COMPLETE')
            self.assertEqual((output/'analysis_support.py').read_bytes(),(rerun/'analysis_support.py').read_bytes())
            artifacts=json.loads((output/'ANALYSIS_COMPLETE.json').read_text())['artifacts_sha256']
            self.assertEqual(artifacts['analysis_support.py'],hashlib.sha256((output/'analysis_support.py').read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):write_report(path,output)


if __name__ == '__main__':
    unittest.main()
