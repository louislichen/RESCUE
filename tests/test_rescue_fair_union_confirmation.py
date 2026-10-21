"""Union/count/lifecycle tests using synthetic controls, never fresh data."""
import copy
from contextlib import contextmanager,ExitStack
import json
from pathlib import Path
import tempfile
from types import ModuleType,SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import uuid

import torch

from rescue import fair_union_confirmation as runner


def selection(maximum=False):
    menu={a['mode']:a for a in runner.baseline.arms()}
    modes=(('retrieval16_k1_eager','retrieval32_k1_eager'),('retrieval16_k2_resident','retrieval32_k2_resident'),
        ('retrieval16_k4_graph','retrieval32_k4_graph'),('tree16_n8','tree32_n64')) if maximum else (
        ('retrieval16_k4_eager',),('retrieval16_k4_resident',),('retrieval16_k4_graph',),('tree16_n8',))
    result={}
    for n in runner.LENGTHS:
        groups={g:[dict(mode=m,arm=copy.deepcopy(menu[m])) for m in values] for g,values in zip(runner.GROUPS,modes)}
        result[str(n)]=dict(union_by_group=groups,union_modes=[m for values in modes for m in values])
    return dict(per_context=result)


class UnionHelpers(unittest.TestCase):
    def test_actual_dual_gate_metadata_run_layer_copies_data_and_freezes_counts_before_execute(self):
        """No models: exercise run() orchestration with synthetic authenticated inputs."""
        from rescue import fair_union_confirmation_data as data
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'inputs';inputs.mkdir();fresh=inputs/'prepared';fresh.mkdir()
            first=root/'stage1';second=root/'stage2';first.mkdir();second.mkdir()
            args=SimpleNamespace(output=root/'results',teacher=inputs/'teacher',student=inputs/'student',mapper=inputs/'mapper',policy=inputs/'policy',
                cost=inputs/'cost',prime_file=inputs/'prime',freeze_receipt=inputs/'gate',data_file=fresh/'confirmation.jsonl',dataset_manifest=fresh/'DATASET_MANIFEST.json')
            args.teacher.mkdir();args.student.mkdir();weights=[args.teacher/'weights.bin',args.student/'weights.bin']
            for path in weights+[args.mapper,args.policy,args.cost,args.prime_file,args.freeze_receipt,inputs/'formal',inputs/'exclusions']:path.write_text('synthetic frozen bytes\n')
            choices=runner.frozen_choices(selection());expected=runner.expected_counts(choices)
            proof=dict(status='UNION_VERIFIED',choices=choices,M=8,R=0,expected_counts=expected,scientific_requests=434,extra_eager_reference_geometries=[])
            model_assets={str(p):runner.sha256(p) for p in weights+[args.prime_file]}
            for directory in (first,second):
                (directory/'protocol.json').write_text(json.dumps({'assets_sha256':model_assets}));(directory/'references.json').write_text('{}')
            (second/'FROZEN_SELECTION.json').write_text(json.dumps(selection()))
            docs=[dict(source_index=i,prefix_ids=[1],suffix_ids=[2],document_ids=[3]*8) for i in range(16)]
            args.data_file.write_text(''.join(json.dumps(v)+'\n' for v in docs))
            manifest=dict(dataset_file='confirmation.jsonl',union_freeze=proof,source_freeze_sha256=runner.sha256(args.freeze_receipt),source_sha256={},
                selected=[{'source_index':i} for i in range(16)],dataset_sha256=runner.sha256(args.data_file),selection_rule={'synthetic':True})
            args.dataset_manifest.write_text(json.dumps(manifest))
            arguments=lambda p:dict(baseline_results=str(p),analysis_dir=str(p/'analysis'),verified_complete=str(p/'VERIFIED'),final_manifest=str(p/'MANIFEST'))
            gate=dict(source_root=str(Path(runner.__file__).resolve().parents[1]),source_sha256={},union_freeze=proof,baseline_arguments=arguments(first),
                horizon_arguments=arguments(second),final_protocol=dict(path=str(inputs/'formal'),sha256=runner.sha256(inputs/'formal')),
                exclusion_receipt={'path':str(inputs/'exclusions')},tokenizer_root=str(args.teacher),tokenizer_sha256={})
            def execute(actual,prepared):
                self.assertTrue((actual.output/'GATE_PASSED.json').exists());self.assertTrue((actual.output/'data/confirmation.jsonl').exists())
                self.assertEqual(prepared['choices'],choices);self.assertEqual(prepared['counts'],dict.fromkeys(expected,0))
                self.assertEqual(prepared['metadata']['union_freeze'],proof)
                return dict(passed=True,counts=prepared['counts'],goal_complete=False)
            with patch.object(data,'validate_source_freeze',return_value=(gate,{})) as validate,patch.object(data,'verify_dataset_manifest',return_value=manifest),\
                 patch.object(runner,'validate_minimum_sources'),patch.object(runner,'copy_development_evidence',side_effect=lambda a,o,n:{'root':n}),\
                 patch.object(runner,'model_files',side_effect=lambda p:[p/'weights.bin']),patch.object(runner,'MAPPER_SHA256',runner.sha256(args.mapper)),\
                 patch.object(runner,'POLICY_SHA256',runner.sha256(args.policy)),patch.object(runner,'COST_SHA256',runner.sha256(args.cost)),\
                 patch.object(runner,'execute_study',side_effect=execute),patch.object(runner,'progress'):
                runner.run(args);self.assertEqual(validate.call_count,2)
            self.assertTrue((args.output/'COMPLETE.json').is_file());self.assertFalse((args.output/'FAILURE.json').exists())

    def test_new_source_bindings_include_all_three_modules_and_reject_foreign_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();modules={}
            for name in ('rescue.fair_union_confirmation','rescue.fair_union_confirmation_data','rescue.fair_union_confirmation_report'):
                path=root/Path(*name.split('.')).with_suffix('.py');path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# source fixture\n')
                modules[name]=SimpleNamespace(__file__=str(path))
            with patch.object(runner.legacy,'source_bindings',return_value={}),patch.object(runner.importlib,'import_module',side_effect=lambda n:modules[n]):
                self.assertEqual(len(runner.source_bindings(root)),3)
                modules['rescue.fair_union_confirmation_report'].__file__=str(root/'foreign/report.py')
                with self.assertRaisesRegex(RuntimeError,'foreign module'):runner.source_bindings(root)

    def test_dual_study_gate_precedes_any_fresh_dataset_read_hash_or_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=SimpleNamespace(output=root/'fresh',**{name:root/'assets'/name for name in (
                'teacher','student','mapper','policy','cost','freeze_receipt','dataset_manifest','data_file','prime_file')})
            fake=ModuleType('rescue.fair_union_confirmation_data')
            def fail(*a):raise RuntimeError('dual study incomplete')
            fake.validate_source_freeze=fail;fake.verify_dataset_manifest=lambda *a: self.fail('fresh dataset opened')
            with patch.dict(sys.modules,{'rescue.fair_union_confirmation_data':fake}),\
                 patch.object(runner,'sha256',side_effect=AssertionError('fresh input hashed')) as digest,\
                 patch.object(runner,'execute_study',side_effect=AssertionError('GPU work began')) as execute:
                with self.assertRaisesRegex(RuntimeError,'dual study incomplete'):runner.run(args)
                digest.assert_not_called();execute.assert_not_called()
            self.assertEqual(json.loads((args.output/'FAILURE.json').read_text())['logical_requests'],0)
            self.assertFalse((args.output/'GATE_PASSED.json').exists())

    def test_variable_min_max_and_asymmetric_counts_from_frozen_geometry_only(self):
        low=runner.frozen_choices(selection());high=runner.frozen_choices(selection(True))
        self.assertEqual((runner.validate_choices(low),len(runner.extra_geometries(low)),runner.scientific_requests(low)),(8,0,434))
        self.assertEqual((runner.validate_choices(high),len(runner.extra_geometries(high)),runner.scientific_requests(high)),(16,8,714))
        mixed={'4096':low['4096'],'8192':high['8192']}
        self.assertEqual(runner.scientific_requests(mixed),162+34*12+4)
        self.assertEqual(runner.expected_counts(high),dict(reference_requests=32,own_quality_requests=64,baseline_quality_requests=16,
            matching_quality_requests=8,prime_requests=18,timing_requests=576,owners=18))
        for choices in (low,high,mixed):
            m=runner.validate_choices(choices);r=len(runner.extra_geometries(choices))
            self.assertEqual(runner.scientific_requests(choices),162+34*m+r)

    def test_original_order_cannot_rerank_drop_duplicate_or_invent_contender(self):
        for kind in ('drop','order','duplicate','unregistered','group'):
            value=selection(True);cell=value['per_context']['4096']
            if kind=='drop':cell['union_by_group'].pop('full_tree')
            if kind=='order':cell['union_modes'].reverse()
            if kind=='duplicate':cell['union_by_group']['retrieval/eager'][1]=copy.deepcopy(cell['union_by_group']['retrieval/eager'][0])
            if kind=='unregistered':cell['union_by_group']['full_tree'][0]['arm']['nodes']=999
            if kind=='group':cell['union_by_group']['retrieval/eager']=cell['union_by_group']['retrieval/graph']
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):runner.frozen_choices(value)

    def test_shared32_sequence_all_arms_both_contexts_and_private_owner_blocks(self):
        for maximum in (False,True):
            choices=runner.frozen_choices(selection(maximum));sequence,owners,warm=runner.timing_schedule(choices,list(range(900,916)))
            self.assertEqual(len(sequence),32);self.assertEqual(len(owners),runner.validate_choices(choices)+2)
            self.assertEqual(len(warm),32*len(owners));self.assertEqual((sequence,owners,warm),runner.timing_schedule(choices,list(range(900,916))))
            for index,owner in enumerate(owners):
                block=warm[index*32:index*32+32]
                self.assertEqual([{k:r[k] for k in ('source_index','repeat')} for r in block],sequence)
                self.assertEqual({(r['mode'],r['length']) for r in block},{(owner['mode'],owner['length'])})
        with self.assertRaises(RuntimeError):runner.timing_schedule(choices,list(range(900,915)))

    def test_own_graph_limit_restored_on_failure_no_legacy_globals_changed(self):
        from rescue.draft_graph import ResidentDraftRuntime
        old=ResidentDraftRuntime.MAX_ENTRIES;groups=runner.legacy.GROUPS
        def fail(*a):
            self.assertEqual(ResidentDraftRuntime.MAX_ENTRIES,128);raise RuntimeError('synthetic fail')
        with patch.object(runner,'_execute_study',side_effect=fail),self.assertRaisesRegex(RuntimeError,'synthetic fail'):runner.execute_study(None,None)
        self.assertEqual(ResidentDraftRuntime.MAX_ENTRIES,old);self.assertEqual(runner.legacy.GROUPS,groups)

    def test_eos_and_action_query_native_alignment_checks_remain_strict(self):
        eos=7;ref=[3,eos]
        row=dict(generated=ref.copy(),tokens=2,output_agrees=True,accepted=1,proposed=1,blocks=1,draft_forwards=1,selector_refreshes=0,
            trace=[dict(round_index=0,prefix=4095,cap=2,proposed=1,accepted=1,terminal=True,workset_tokens=512)],
            observed_rounds=[dict(budget_features=[0.]*12)],queries=[dict(query='q')],native_t={'sha':'t'},native_s={'sha':'s'},nonzero_injection=True)
        runner.check_own_output(row,ref,eos);runner.legacy.own_quality_pair(row,copy.deepcopy(row))
        for kind in ('action','query','native','nonzero','eos'):
            bad=copy.deepcopy(row)
            if kind=='action':bad['trace'][0]['cap']=4
            if kind=='query':bad['queries'][0]['query']='changed'
            if kind=='native':bad['native_s']['sha']='changed'
            if kind=='nonzero':bad['nonzero_injection']=False
            if kind=='eos':bad['generated']=[3,4]
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):
                if kind=='eos':runner.check_own_output(bad,bad['generated'],eos)
                else:runner.legacy.own_quality_pair(row,bad)

    @torch.no_grad()
    def test_real_cpu_nonzero_mapper_native_all_query_equivalence_reuses_quality_contract(self):
        from rescue.efficient_attention import install_efficient_attention
        from rescue.engine import RescueEngine
        from rescue.packing import pack_projections
        # Existing tiny-model fixtures use flat sibling imports. Keep that
        # fixture-only search path temporary for both discover/package modes.
        with patch.object(sys,'path',[str(Path(__file__).resolve().parent),*sys.path]):
            from test_rescue_draft_runtime import components,config,policy_for
            from test_rescue_system import COST
        teacher,student,mapper=components()
        for model in (teacher,student):install_efficient_attention(model);pack_projections(model)
        collected=[]
        for mode in ('eager','resident'):
            observer=runner.QualityObserver()
            with RescueEngine(teacher,student,config(mode,limit=9),mapper,policy_for(4),COST,round_observer=observer) as engine:
                observer.engine=engine
                with runner.record_draft_steps() as steps:row=engine.generate(list(range(3,22)))
                row.update(observed_rounds=observer.rounds,queries=steps,nonzero_injection=observer.nonzero_injection,
                    native_t=runner.native_digest(engine.native_t),native_s=runner.native_digest(engine.native_s))
                collected.append(row);runner.detach_quality_observer(engine)
        runner.legacy.own_quality_pair(*collected)
        self.assertTrue(all(row['nonzero_injection'] and len(row['queries'])==row['draft_forwards'] for row in collected))


class CompleteSyntheticUnionLifecycle(unittest.TestCase):
    def exercise(self,output,maximum,bad_first=False):
        """Actual MIN/MAX orchestration, synthetic no-forward engines only."""
        from rescue import baselines,engine,loading,packing,query_mapper,specextend_qwen,step_budget
        state=dict(alive=set(),quality_ids=set(),timing_ids=set(),quality_done=False,queries=None,created=[],maximum_alive=0)
        payload=dict(kind='step_budget',hidden=1,margin=.05,prior_weight=1.,seed=412)
        env=dict(nvidia_smi_gpu='synthetic H100 UUID',torch='synthetic',cuda='synthetic',target_dtype='torch.float32',draft_dtype='torch.bfloat16',prefix_dtype='torch.bfloat16',tf32=False)
        class Frozen:
            def eval(self):return self
            def requires_grad_(self,*a):return self
        class Model(Frozen):
            def __init__(self,dtype):self.dtype=dtype;self.config=SimpleNamespace(num_hidden_layers=1)
            def parameters(self):return [SimpleNamespace(device=torch.device('cuda:0'),dtype=self.dtype)]
        teacher,student=Model(torch.float32),Model(torch.bfloat16)
        class Tokenizer:
            eos_token_id=None
            def __len__(self):return 64
        class Policy(Frozen):margin=.05;prior_weight=1.;seed=412
        class Engine:
            family='retrieval'
            def __init__(self,t,s,c,*a,round_observer=None,top_chunks=None,**kw):
                self.config=c;self.closed=False;self.calls=0;self.identity=uuid.uuid4().hex;self.round_observer=round_observer
                self.native_t=self.native_s=SimpleNamespace();self.draft_runtime=None if c.draft_execution=='eager' else SimpleNamespace(max_entries=128)
                reference=self.family=='retrieval' and top_chunks is None
                self.category='reference' if reference else 'timing' if state['quality_done'] else 'quality'
                state['created'].append(self);state['alive'].add(self.identity);state['maximum_alive']=max(state['maximum_alive'],len(state['alive']))
                if len(state['alive'])!=1:raise AssertionError('Multiple live serving owners')
                if self.category=='timing':state['timing_ids'].add(self.identity)
                if self.category=='quality':state['quality_ids'].add(self.identity)
            def __enter__(self):return self
            def __exit__(self,*a):self.close()
            def reference(self,*a):return dict(generated=[0]*128,tokens=128)
            def generate_mock(self,ids,eos):
                self.calls+=1
                if state['queries'] is not None:state['queries'].append(dict(query={'sha':'q'},logits=True,summary_requested=False,summary_count=0))
                if self.round_observer is not None:self.round_observer.rounds=[dict(budget_features=None)];self.round_observer.nonzero_injection=self.family=='rescue'
                generated=[0]*128
                if bad_first and self.family=='rescue' and len(state['quality_ids'])==1 and self.calls==1:generated[-1]=7
                return dict(request_id=uuid.uuid4().hex,generated=generated,tokens=128,accepted=0,proposed=1,blocks=1,
                    trace=[],draft_forwards=1,selector_refreshes=0,draft_runtime={'request':dict(replays=1,captures=int(self.calls==1),capture_host_seconds=.01 if self.calls==1 else 0.)})
            def close(self):self.closed=True;state['alive'].discard(self.identity)
        class Rescue(Engine):family='rescue'
        class Tree(Engine):
            family='full_tree'
            def __init__(self,template,path,arm):self.arm=arm;super().__init__(teacher,student,runner.legacy.own_config(),top_chunks=1)
            def generate(self,ids):
                self.calls+=1;return dict(generated=[0]*128,tokens=128,upstream=dict(output_token_ids=[0]*128,total_generated=128,raw_total_generated=128,generation_overshoot=0,accept_length_list=[0]*128))
            def signature(self):return {'identity':self.identity}
            def stats(self):return snapshot(self)
        class Group:
            def __init__(self,*a):self.engines=[]
            def idle(self):pass
            def add(self,e):self.engines.append(e)
            def close(self,e):e.close();self.engines.remove(e)
            def generate(self,e,ids,eos):return e.generate_mock(ids,eos)
        def snapshot(e):return dict(entries=0 if e.closed else 1,captures=int(e.calls>0),replays=e.calls,max_graph_entries=128)
        @contextmanager
        def queries():
            values=[];state['queries']=values
            try:yield values
            finally:state['queries']=None
        write=runner.write
        def gate_write(path,value):
            if Path(path).name=='QUALITY_PASSED.json':
                if state['alive']:raise AssertionError('Quality owner survives into timing')
                state['quality_done']=True
            write(path,value)
        canonical=lambda row,ref,*a,**kw:runner.require(row['generated']==ref,'Synthetic canonical mismatch')
        args=SimpleNamespace(output=output,teacher=output/'teacher',student=output/'student',mapper=output/'mapper',policy=output/'policy',cost=output/'cost',prime_file=output/'prime')
        args.cost.write_text('{}');choices=runner.frozen_choices(selection(maximum));expected=runner.expected_counts(choices)
        docs={i:dict(prefix_ids=[1],suffix_ids=[2],document_ids=[3]*8192) for i in range(900,916)}
        prepared=dict(choices=choices,documents=docs,counts=dict.fromkeys(expected,0),seen=set(),sources={},asset_paths=[],assets={},
            baseline_protocol=dict(environment=env,flash_attention={}),horizon_protocol=dict(environment=env,flash_attention={}),baseline_references={},metadata={})
        with ExitStack() as stack:
            for p in [patch.object(baselines,'BaselineEngine',Engine),patch.object(engine,'RescueEngine',Rescue),patch.object(loading,'load_models',return_value=(teacher,student,Tokenizer())),
                patch.object(packing,'share_packed_projections'),patch.object(query_mapper,'mapper_from_payload',return_value=Frozen()),patch.object(step_budget.StepBudget,'from_payload',return_value=Policy()),
                patch.object(specextend_qwen,'build',return_value=SimpleNamespace(base_model=teacher,draft_model=student)),patch.object(runner,'HookOwners',Group),
                patch.object(runner,'gpu_snapshot',return_value=env),patch.object(runner,'source_bindings',return_value={}),patch.object(runner.baseline,'check_full_bindings',return_value={}),
                patch.object(runner.baseline,'flash_receipt',return_value={}),patch.object(runner.baseline,'FullOwner',Tree),patch.object(runner.legacy,'read_prime',return_value=({n:[1]*n for n in runner.LENGTHS},{str(n):[0]*128 for n in runner.LENGTHS})),
                patch.object(runner.torch,'load',return_value=payload),patch.object(runner,'native_digest',return_value={'all_layers':'native'}),patch.object(runner,'runtime_snapshot',side_effect=snapshot),
                patch.object(runner,'owner_signature',side_effect=lambda e:{'identity':e.identity}),patch.object(runner,'record_draft_steps',side_effect=queries),patch.object(runner,'write',side_effect=gate_write),
                patch.object(runner.legacy,'own_quality_pair'),patch.object(runner.legacy,'baseline_quality_pair'),patch.object(runner,'check_pair'),patch.object(runner,'check_own_output',side_effect=canonical),
                patch.object(runner.legacy,'check_quality_row',side_effect=canonical),patch.object(runner.baseline,'check_output',side_effect=canonical),patch.object(runner.baseline,'check_reproducible'),patch.object(runner,'progress')]:stack.enter_context(p)
            try:result=runner.execute_study(args,prepared)
            except BaseException:
                self.assertFalse(state['alive'],'Failure leaked a private serving owner')
                raise
        self.assertEqual(prepared['counts'],expected);self.assertEqual(result['logical_requests'],runner.scientific_requests(choices))
        self.assertEqual(state['maximum_alive'],1);self.assertFalse(state['alive']);self.assertFalse(state['quality_ids']&state['timing_ids'])
        self.assertEqual(len(state['quality_ids']),64+expected['baseline_quality_requests']+expected['matching_quality_requests'])
        self.assertEqual(len(state['timing_ids']),expected['owners'])
        self.assertTrue(all(e.closed for e in state['created']))
        owners=[json.loads(x) for x in (output/'owners.jsonl').read_text().splitlines()]
        runtime=[json.loads(x) for x in (output/'runtime.jsonl').read_text().splitlines()]
        sequence=runner.timing_schedule(choices,sorted(docs))[0]
        for owner in owners:
            rows=[r for r in runtime if r['owner_id']==owner['owner_id']]
            self.assertEqual([{k:r[k] for k in ('source_index','repeat')} for r in rows],sequence)
            self.assertEqual(owner['closed_runtime_stats']['entries'],0)
        self.assertFalse((output/'COMPLETE.json').exists())

    def test_actual_min434_flow_quality_closed_before10fresh_timing_owners(self):
        with tempfile.TemporaryDirectory() as tmp:self.exercise(Path(tmp),False)

    def test_actual_max714_flow_quality_closed_before18fresh_timing_owners(self):
        with tempfile.TemporaryDirectory() as tmp:self.exercise(Path(tmp),True)

    def test_quality_failure_preserves_raw_before_gate_and_closes_private_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'Synthetic canonical mismatch'):self.exercise(output,False,bad_first=True)
            raw=[json.loads(x) for x in (output/'own_quality.jsonl').read_text().splitlines()]
            self.assertEqual(len(raw),1);self.assertEqual(raw[0]['generated'][-1],7)
            self.assertFalse((output/'QUALITY_PASSED.json').exists());self.assertFalse((output/'runtime.jsonl').exists())


if __name__=='__main__':unittest.main()
