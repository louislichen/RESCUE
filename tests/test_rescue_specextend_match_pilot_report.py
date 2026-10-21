"""Synthetic CPU accounting/audit fixtures, never model performance claims."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rescue.specextend_match_pilot_report import (ARMS,EXPECTED,FULL_FLAGS,
    LENGTHS,PINS,QUALITY_ARMS,TREE,analyze,plot,write_report)


def save_rows(path, values):
    path.write_text(''.join(json.dumps(v)+'\n' for v in values))


def fixture(directory, warm_capture=True):
    directory=Path(directory);directory.mkdir(exist_ok=True,parents=True)
    tokens=list(range(128)); quality=[];primes=[];runtime=[];owners=[];canonical=[];contexts=[]

    def digest(shape):
        return dict(shape=shape,dtype='torch.bfloat16',sha256='a'*64)

    def row(mode,length,request_id,seconds=2.):
        value=dict(mode=mode,length=length,source_index=303,request_id=request_id,
            generated=tokens.copy(),tokens=128,output_agrees=True,total_seconds=seconds,
            total_tps=128/seconds,blocks=32,accepted=96,proposed=128)
        if mode == 'full32_n8':
            accepted=[3]*32+[2]
            value.update(upstream=dict(output_token_ids=tokens.copy(),total_generated=128,
                raw_total_generated=131,generation_overshoot=3,accept_length_list=accepted),
                scored_generated_before_eos=tokens.copy(),raw_total_generated=131,generation_overshoot=3,
                raw_tail_ids_available=False,tree_policy=TREE,observed_full_config=FULL_FLAGS,
                blocks=33,accepted=sum(accepted),proposed=None)
        else:
            value.update(trace=[dict(round_index=i,prefix=length-1+4*i,cap=4,proposed=4,
                accepted=3,terminal=i == 31,workset_tokens=512) for i in range(32)],
                prefill_seconds=.2,decode_seconds=seconds-.21,cleanup_seconds=.01,
                draft_forwards=128,selector_refreshes=8,draft_shadow_probes=0,
                working_gathered_position_layers=512*28*32,
                projected_positions=64 if mode.startswith('rescue') else 0,
                selection_similarity_elements=4096 if mode.startswith('rescue') else 0)
        return value

    def runtime_stats(mode,count,captures,host):
        return dict(captures=captures,replays=128*count if mode.endswith('graph') else 0,
            resident_steps=128*count if mode.endswith('resident') else 0,
            capture_host_seconds=host,shadow_updates=128*count if mode.endswith(('graph','resident')) else 0)

    plan=[dict(mode=m,source_index=303,length=l,repeat=r) for l in LENGTHS for r in range(2) for m in ARMS]
    for length in LENGTHS:
        canonical.append(dict(mode='canonical_ar',source_index=303,length=length,
            generated=tokens.copy(),request_id=f'ar:{length}'))
        for mode in QUALITY_ARMS:
            value=row(mode,length,f'q:{mode}:{length}')
            value.update(phase='correctness',owner_id=f'qowner:{mode}:{length}')
            if mode != 'full32_n8':
                observed=[dict(round_index=i,prefix=length-1+4*i,selected=[0,1],
                    budget_features=[0.]*11+[.1] if mode.startswith('rescue') else None,
                    draft=[4*i,4*i+1,4*i+2,10000],checks=[4*i,4*i+1,4*i+2,4*i+3,0],accepted=3) for i in range(32)]
                value.update(observed_rounds=observed,
                    queries=[dict(logits=True,query=digest([1,12,1,128])) for _ in range(128)],
                    native_t=dict(lengths=[length+128]*28,layers=[dict(key=digest([1,4,length+128,128]),value=digest([1,4,length+128,128])) for _ in range(28)]),
                    native_s=dict(lengths=[length+127]*28,layers=[dict(key=digest([1,2,length+127,128]),value=digest([1,2,length+127,128])) for _ in range(28)]),
                    nonzero_injection=mode.startswith('rescue'),
                    runtime_stats=runtime_stats(mode,1,2 if mode.endswith('graph') else 0,.4 if mode.endswith('graph') else 0.))
            quality.append(value)
        for mode in ARMS:
            owner_id=f'owner:{mode}:{length}';pointer=dict(native_t=owner_id,native_s=owner_id,working=owner_id)
            factor=dict(zip(ARMS,[.8,1.,.95,.7,1.2]))[mode]
            prime=row(mode,length,f'prime:{mode}:{length}',4.*factor)
            prime.update(phase='cold_prime',owner_id=owner_id,outer_seconds=4.*factor+.01,
                outer_tps=128/(4.*factor+.01),constructor_and_detach_seconds=.02,
                cold_complete_outer_seconds=4.*factor+.03)
            cap=2 if mode.endswith('graph') else 0;host=.4 if cap else 0.
            if mode != 'full32_n8':
                prime.update(owner_signature=pointer,runtime_stats=runtime_stats(mode,1,cap,host),
                    draft_runtime=dict(request=runtime_stats(mode,1,cap,host),retain_query=False,max_graph_entries=128))
            primes.append(prime); group=[]
            for repeat in range(2):
                duration=[2.,3.][repeat]*factor+.01
                new_capture=int(warm_capture and mode.endswith('graph') and repeat == 0)
                capture_host=.1 if new_capture else 0.
                value=row(mode,length,f'warm:{mode}:{length}:{repeat}',duration-.01)
                value.update(phase='warm',owner_id=owner_id,repeat=repeat,
                    outer_seconds=duration,outer_tps=128/duration)
                if mode != 'full32_n8':
                    cap+=new_capture;host+=capture_host
                    value.update(owner_signature=pointer,runtime_stats=runtime_stats(mode,repeat+2,cap,host),
                        draft_runtime=dict(request=runtime_stats(mode,1,new_capture,capture_host),retain_query=False,max_graph_entries=128))
                group.append(value);runtime.append(value)
            active=.02+prime['outer_seconds']+sum(r['outer_seconds'] for r in group)+.001
            owners.append(dict(mode=mode,length=length,owner_id=owner_id,owner_signature=pointer if mode != 'full32_n8' else None,
                constructor_and_detach_seconds=.02,prime_request_id=prime['request_id'],
                cold_prime_outer_seconds=prime['outer_seconds'],prime_tokens=128,warm_tokens=256,
                warm_request_ids=[v['request_id'] for v in group],warm_outer_seconds_sum=sum(v['outer_seconds'] for v in group),
                close_seconds=.001,owner_lifetime_wall_seconds=active*6,accounted_service_seconds=active,
                accounted_service_tps=384/active,hook_restored=True))
        active=sum(v['accounted_service_seconds'] for v in owners if v['length'] == length)
        contexts.append(dict(length=length,tokens=5*384,context_wall_seconds=active+1.))
    lookup={(r['mode'],r['length'],r['repeat']):r for r in runtime}
    runtime=[lookup[r['mode'],r['length'],r['repeat']] for r in plan]
    protocol=dict(source_index=303,lengths=list(LENGTHS),max_new_tokens=128,repeats=2,
        arms=list(ARMS),quality_arms=list(QUALITY_ARMS),expected_counts=EXPECTED,expected_requests=44,
        retrieval=dict(top_chunks=32,chunk_size=32,fixed_k=4,refresh_every=4,mapper=False),tree=TREE,
        graph_max_entries=128,rescue_retain_query=True,retrieval_retain_query=dict(correctness=True,timing=False),
        no_training=True,mapper_frozen=True,policy_frozen=True,**PINS,
        assets_sha256={'/synthetic/full_final.pt':PINS['mapper_sha256'],'/synthetic/budget_selected.pt':PINS['policy_sha256'],'/synthetic/cost.json':PINS['cost_sha256']},
        config=dict(selector='hierarchical',coverage_rows=64,refresh_every=4,injection_layers=[27],inject_tokens=64,
            injection_strength=.5,prefill_precision='bfloat16',max_new_tokens=128,max_draft=12,length_policy='risk',schedule='serial',profile=False),
        environment=dict(gpu_name='NVIDIA H100',total_memory=85_000_000_000,target_dtype='torch.float32',draft_dtype='torch.bfloat16',prefix_dtype='torch.bfloat16',tf32=False),
        source_sha256={},tree_source_bindings={},warm_order=plan)
    for name,module in (('rescue/synthetic.py',None),('exp/specextend_bridgekv/specextend/classic/model_classic.py','classic.model_classic'),('exp/specextend_bridgekv/specextend/shared/modeling_llama_kv_target.py','shared.modeling_llama_kv_target')):
        path=directory/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# synthetic fixture only\n')
        digest=hashlib.sha256(path.read_bytes()).hexdigest();protocol['source_sha256'][name]=digest
        if module:
            protocol['tree_source_bindings'][module]=dict(path=name,sha256=digest)
    for name,value in (('protocol.json',protocol),('references.json',{str(l):tokens for l in LENGTHS}),
                       ('COMPLETE.json',dict(passed=True,counts=EXPECTED,sources_unchanged=True,assets_unchanged=True))):
        (directory/name).write_text(json.dumps(value))
    for name,values in (('reference.jsonl',canonical),('correctness.jsonl',quality),('prime.jsonl',primes),
                        ('runtime.jsonl',runtime),('owners.jsonl',owners),('contexts.jsonl',contexts)):
        save_rows(directory/name,values)
    return dict(protocol=protocol,quality=quality,primes=primes,runtime=runtime,owners=owners,canonical=canonical)


class MatchedPilotReportTests(unittest.TestCase):
    def test_complete44_has_no_document_ci_and_formal_goal_remains_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp);summary=analyze(tmp)
        self.assertEqual(summary['status'],'COMPLETE',summary['problems'])
        self.assertEqual(summary['observed_counts'],EXPECTED)
        self.assertIsNone(summary['confidence_intervals'])
        self.assertIn('PARTIAL',summary['goal_status'])
        self.assertEqual(summary['documents'],1)
        self.assertEqual(summary['usable_repeats'],{'4096':[0,1],'8192':[0,1]})
        metric=summary['metrics']['4096']['retrieval32_k4_eager']
        self.assertAlmostEqual(metric['warm_tps'],256/5.02)
        self.assertEqual(metric['acceptance_length'],3.)
        self.assertIsNone(summary['metrics']['4096']['full32_n8']['acceptance_rate'])
        self.assertAlmostEqual(summary['metrics']['4096']['full32_n8']['acceptance_length'],98/33)

    def test_legal_new_warm_capture_is_kept_and_not_subtracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=fixture(tmp,warm_capture=True);summary=analyze(tmp)
        self.assertEqual(summary['status'],'COMPLETE',summary['problems'])
        metric=summary['metrics']['4096']['retrieval32_k4_graph']
        self.assertEqual(metric['warm_captures'],1)
        self.assertAlmostEqual(metric['warm_capture_seconds'],.05)
        owner=next(o for o in data['owners'] if o['mode']=='retrieval32_k4_graph' and o['length']==4096)
        self.assertAlmostEqual(metric['accounted_service_tps'],384/owner['accounted_service_seconds'])
        self.assertAlmostEqual(metric['cold_complete_tps'],128/(.02+2.81))

    def test_full_overshoot_and_eos_tails_are_distinct_and_no_raw_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp);summary=analyze(tmp)
        full=summary['metrics']['4096']['full32_n8']
        self.assertEqual(full['generation_overshoot'],6)
        self.assertEqual(full['eos_tail_tokens'],0)
        self.assertIsNone(full['acceptance_rate'])

    def test_wrong_full_client_token_is_not_hid_by_overshoot(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=fixture(tmp);row=next(r for r in data['runtime'] if r['mode']=='full32_n8')
            row['generated'][-1]=999
            save_rows(Path(tmp)/'runtime.jsonl',data['runtime']);summary=analyze(tmp)
        self.assertEqual(summary['status'],'PARTIAL')
        self.assertTrue(any('Canonical client-token' in v for v in summary['problems']))
        self.assertEqual(summary['usable_repeats']['4096'],[1])

    def test_family_native_query_and_nonzero_evidence_recomputed(self):
        for kind in ('native','query','nonzero'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                data=fixture(tmp);index=next(i for i,r in enumerate(data['quality']) if r['mode']=='rescue_graph')
                row=copy.deepcopy(data['quality'][index]);data['quality'][index]=row
                if kind=='native':row['native_s']['layers'][0]['key']['sha256']='b'*64
                elif kind=='query':row['queries'][0]['query']['sha256']='b'*64
                else:
                    for obs in row['observed_rounds']:obs['budget_features'][-1]=0.
                save_rows(Path(tmp)/'correctness.jsonl',data['quality']);summary=analyze(tmp)
                self.assertEqual(summary['status'],'PARTIAL')
                self.assertTrue(any('family quality' in v or 'actual nonzero' in v for v in summary['problems']))

    def test_cold_owner_must_capture_and_timing_retrieval_must_not_keep_query(self):
        for kind in ('cold','retain'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                data=fixture(tmp)
                if kind=='cold':
                    row=next(r for r in data['primes'] if r['mode']=='retrieval32_k4_graph')
                    row['draft_runtime']['request']['captures']=0
                    save_rows(Path(tmp)/'prime.jsonl',data['primes'])
                else:
                    row=next(r for r in data['runtime'] if r['mode']=='retrieval32_k4_graph')
                    row['draft_runtime']['retain_query']=True
                    save_rows(Path(tmp)/'runtime.jsonl',data['runtime'])
                summary=analyze(tmp)
                self.assertEqual(summary['status'],'PARTIAL')

    def test_raw_counts_sources_uuid_and_full_binding_are_not_just_boolean_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=fixture(tmp);path=Path(tmp)
            row=next(r for r in data['runtime'] if r['mode']=='full32_n8');row['upstream']['raw_total_generated']=132
            data['canonical'][0]['request_id']=data['runtime'][0]['request_id']
            save_rows(path/'runtime.jsonl',data['runtime']);save_rows(path/'reference.jsonl',data['canonical'])
            (path/'source/rescue/synthetic.py').write_text('# tampered\n')
            summary=analyze(path)
        self.assertEqual(summary['status'],'PARTIAL')
        for phrase in ('raw accepted/round','Frozen source SHA','request UUID'):
            self.assertTrue(any(phrase in v for v in summary['problems']),phrase)

    def test_partial_keeps_complete_context_without_missing_pair_imputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=fixture(tmp);save_rows(Path(tmp)/'runtime.jsonl',data['runtime'][:-1]);summary=analyze(tmp)
        self.assertEqual(summary['status'],'PARTIAL')
        self.assertEqual(summary['usable_repeats']['4096'],[0,1])
        self.assertEqual(summary['usable_repeats']['8192'],[])

    def test_truncated_jsonl_keeps_completed_records_but_marks_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp);path=Path(tmp)/'runtime.jsonl'
            path.write_text(path.read_text()+'{"unfinished":')
            summary=analyze(tmp)
        self.assertEqual(summary['status'],'PARTIAL')
        self.assertEqual(summary['observed_counts']['timing_requests'],20)
        self.assertEqual(summary['usable_repeats']['4096'],[0,1])

    def test_three_figures_render_complete_and_empty_partial_with_explicit_colors(self):
        for populated in (False,True):
            with self.subTest(populated=populated),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)
                if populated:fixture(path)
                summary=analyze(path);output=path/'analysis';output.mkdir()
                from matplotlib.axes import Axes
                original=Axes.bar;colors=[]
                def record(axis,*args,**kwargs):
                    colors.append(kwargs.get('color'));return original(axis,*args,**kwargs)
                with patch.object(Axes,'bar',record):names=plot(summary,output)
                write_report(path,output,summary,names)
                self.assertEqual(names,['quality','tps','latency_captures'])
                if populated:self.assertTrue(colors and all(v is not None for v in colors))
                for name in names:
                    self.assertTrue((output/(name+'.png')).read_bytes().startswith(b'\x89PNG'))
                    self.assertIn('<svg',(output/(name+'.svg')).read_text())
                text=(output/'REPORT.md').read_text()
                for required in ('PARTIAL','NOT95%CI','non-contiguous','not exposed','upstream','No baseline or method search'):
                    self.assertIn(required.lower(),text.lower())


if __name__ == '__main__':
    unittest.main()
