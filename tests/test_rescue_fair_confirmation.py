"""Frozen-choice/count/math and safety helpers; no confirmation data access."""
import copy
from contextlib import contextmanager,ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch

from rescue import fair_confirmation as runner
from rescue import baseline_selection as baseline
from rescue.round_overhead_run import sha256
from rescue.efficient_attention import install_efficient_attention
from rescue.engine import RescueEngine
from rescue.packing import pack_projections
from test_rescue_draft_runtime import components,config,policy_for
from test_rescue_system import COST


def selection(distinct=True):
    choices={a['mode']:a for a in baseline.arms()};result={}
    modes=('retrieval16_k4_eager','retrieval32_k8_resident','retrieval32_k12_graph','tree16_n64') if distinct else (
        'retrieval16_k4_eager','retrieval16_k4_resident','retrieval16_k4_graph','tree16_n64')
    for n in runner.LENGTHS:
        result[str(n)]={'groups':{group:dict(mode=mode,arm=choices[mode]) for group,mode in zip(runner.GROUPS,modes)}}
    return dict(per_context=result)


class FairConfirmationHelpers(unittest.TestCase):
    def test_graph_owner_limit_is128_and_restored_even_on_execution_error(self):
        from rescue.draft_graph import ResidentDraftRuntime
        previous=ResidentDraftRuntime.MAX_ENTRIES
        def fail(*args):
            self.assertEqual(ResidentDraftRuntime.MAX_ENTRIES,128)
            raise RuntimeError('synthetic execution failure')
        with patch.object(runner,'_execute_study',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'synthetic execution'):runner.execute_study(None,None)
        self.assertEqual(ResidentDraftRuntime.MAX_ENTRIES,previous)

    def test_failed_baseline_gate_precedes_any_confirmation_hash_open_or_gpu_call(self):
        from rescue import fair_confirmation_data as data
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=SimpleNamespace(**{name:root/name for name in ('teacher','student','mapper','policy','cost',
                'freeze_receipt','dataset_manifest','data_file','prime_file')},output=root/'fresh_output')
            with patch.object(data,'validate_source_freeze',side_effect=RuntimeError('baseline incomplete')) as gate,\
                 patch.object(data,'verify_dataset_manifest',side_effect=AssertionError('new manifest read')) as new_data,\
                 patch.object(runner,'sha256',side_effect=AssertionError('new data hashed')) as digest,\
                 patch.object(runner,'execute_study',side_effect=AssertionError('GPU work began')) as execute:
                with self.assertRaisesRegex(RuntimeError,'baseline incomplete'):runner.run(args)
                gate.assert_called_once_with(args.freeze_receipt);new_data.assert_not_called();digest.assert_not_called();execute.assert_not_called()
            self.assertEqual(json.loads((args.output/'FAILURE.json').read_text())['logical_requests'],0)
            self.assertFalse((args.output/'GATE_PASSED.json').exists())

    def test_source_gate_requires_complete_tree_and_launcher_not_a_declared_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths=['rescue/fair_confirmation.py','exp/attention_horizon/selection.py','exp/selective_kv/bridge.py',
                'tests/test_rescue_fair_confirmation.py','tests/test_rescue_fair_confirmation_data.py',
                'tests/test_rescue_fair_confirmation_report.py','exp/draft_graph/fair_confirmation_20261020_preregistration.json',
                'exp/draft_graph/run_fair_confirmation_20261020_v1.sh']+[
                    f'exp/specextend_bridgekv/specextend/classic/source{i}.py' for i in range(27)]
            for name in paths:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# frozen source fixture\n')
            inventory={name:sha256(root/name) for name in paths}
            runner.validate_minimum_sources(root,inventory)
            missing=dict(inventory);del missing['exp/draft_graph/run_fair_confirmation_20261020_v1.sh']
            with self.assertRaisesRegex(RuntimeError,'omits mandatory'):runner.validate_minimum_sources(root,missing)
            (root/paths[-1]).unlink()
            with self.assertRaisesRegex(RuntimeError,'Complete archived'):runner.validate_minimum_sources(root,inventory)

    def test_eight_choices_and_geometry_dedup_determine434plusr_before_data(self):
        choices=runner.frozen_choices(selection())
        self.assertEqual(len(runner.extra_geometries(choices)),4)
        counts=runner.expected_counts(choices)
        self.assertEqual(sum(v for k,v in counts.items() if k!='owners'),438)
        self.assertEqual(counts['timing_requests'],320)
        self.assertEqual(len(runner.extra_geometries(runner.frozen_choices(selection(False)))),0)
        self.assertEqual(sum(v for k,v in runner.expected_counts(runner.frozen_choices(selection(False))).items() if k!='owners'),434)
        # Resident and Graph can share the same extra eager geometry.
        frozen=selection();frozen['per_context']['4096']['groups']['retrieval/graph']=dict(
            mode='retrieval32_k8_graph',arm=next(a for a in baseline.arms() if a['mode']=='retrieval32_k8_graph'))
        self.assertEqual(len(runner.extra_geometries(runner.frozen_choices(frozen))),3)

    def test_missing_or_misgrouped_frozen_choice_fails_without_reranking(self):
        for kind in ('missing','misgroup','nonmenu'):
            value=selection()
            if kind=='missing':del value['per_context']['4096']['groups']['full_tree']
            if kind=='misgroup':value['per_context']['4096']['groups']['retrieval/graph']=value['per_context']['4096']['groups']['retrieval/eager']
            if kind=='nonmenu':value['per_context']['4096']['groups']['full_tree']['arm']['nodes']=128
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):runner.frozen_choices(value)

    def test_10serialowners_320warm_seeded_full16_documents_no_outcome_selection(self):
        choices=runner.frozen_choices(selection());documents=list(range(900,916))
        owners,warm=runner.timing_schedule(choices,documents)
        self.assertEqual(len(owners),10);self.assertEqual(len(warm),320)
        self.assertEqual((owners,warm),runner.timing_schedule(choices,documents))
        self.assertNotEqual((owners,warm),runner.timing_schedule(choices,documents,seed=9))
        for offset,owner in enumerate(owners):
            block=warm[offset*32:offset*32+32]
            self.assertEqual({(r['mode'],r['length']) for r in block},{(owner['mode'],owner['length'])})
            self.assertEqual({(r['source_index'],r['repeat']) for r in block},{(d,r) for d in documents for r in range(2)})
        with self.assertRaises(RuntimeError):runner.timing_schedule(choices,documents[:-1])

    def test_own_frozen_configuration_has_real_mapper_and_old_policy_execution(self):
        config=runner.own_config()
        self.assertEqual((config.injection_layers,config.inject_tokens,config.injection_strength),((27,),64,.5))
        self.assertEqual(config.length_policy,'risk');self.assertEqual(config.draft_execution,'cuda_graph')
        self.assertEqual(config.coverage_execution,'eager');self.assertEqual(config.prefill_precision,'bfloat16')
        self.assertFalse(config.profile)

    def test_prime_reuses_exact_old271_inputs_and_canonical128_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'prime.jsonl';item=dict(source_index=271,split='train',prefix_ids=[1],suffix_ids=[2],document_ids=[3]*8192)
            path.write_text(json.dumps(item)+'\n')
            original=runner.PRIME_SHA256;runner.PRIME_SHA256=sha256(path)
            try:
                protocol={'input_sha256':{f'271:{n}':hashlib.sha256(json.dumps(runner.context(item,n)).encode()).hexdigest() for n in runner.LENGTHS}}
                refs={f'271:{n}':list(range(128)) for n in runner.LENGTHS}
                inputs,stored=runner.read_prime(path,protocol,refs)
                self.assertEqual(set(inputs),set(runner.LENGTHS));self.assertEqual(stored['4096'],refs['271:4096'])
                bad=copy.deepcopy(protocol);bad['input_sha256']['271:4096']='a'*64
                with self.assertRaisesRegex(RuntimeError,'prime input'):runner.read_prime(path,bad,refs)
                bad=copy.deepcopy(refs);bad['271:4096']=bad['271:4096'][:32]
                with self.assertRaisesRegex(RuntimeError,'canonical128'):runner.read_prime(path,protocol,bad)
            finally:runner.PRIME_SHA256=original


class GenuineMapperQuality(unittest.TestCase):
    @torch.no_grad()
    def test_collection_and_pair_check_use_real_nonzero_v_native_and_all_query_bytes(self):
        teacher,student,mapper=components()
        for model in (teacher,student):install_efficient_attention(model);pack_projections(model)
        collected=[];prompt=list(range(3,22))
        for mode in ('eager','resident'):
            observer=runner.QualityObserver()
            with RescueEngine(teacher,student,config(mode,limit=9),mapper,policy_for(4),COST,round_observer=observer) as engine:
                observer.engine=engine
                with runner.record_draft_steps() as queries:row=engine.generate(prompt)
                row.update(observed_rounds=observer.rounds,queries=queries,nonzero_injection=observer.nonzero_injection,
                    native_t=runner.native_digest(engine.native_t),native_s=runner.native_digest(engine.native_s))
                collected.append(row);runner.detach_quality_observer(engine)
        runner.own_quality_pair(*collected)
        self.assertTrue(all(r['nonzero_injection'] for r in collected))
        self.assertTrue(all(len(r['queries'])==r['draft_forwards'] for r in collected))
        changed=copy.deepcopy(collected[1]);changed['queries'][0]['query']['sha256']='a'*64
        with self.assertRaisesRegex(RuntimeError,'changed queries'):runner.own_quality_pair(collected[0],changed)
        changed=copy.deepcopy(collected[1]);changed['nonzero_injection']=False
        with self.assertRaisesRegex(RuntimeError,'real same-dtype'):runner.own_quality_pair(collected[0],changed)


class MockCompleteExecutionLifecycle(unittest.TestCase):
    def test_actual_execute_control_flow_closes_all_quality_owners_before10_fresh_timing_owners(self):
        """434 synthetic requests exercise runner control flow, NOT model math."""
        from rescue import baselines,engine,loading,packing,query_mapper,specextend_qwen,step_budget
        from rescue.draft_graph import ResidentDraftRuntime
        state=dict(alive=set(),quality_ids=set(),timing_ids=set(),quality_done=False,queries=None,created=[])
        payload=dict(kind='step_budget',hidden=1,margin=.05,prior_weight=1.,seed=412)
        environment=dict(nvidia_smi_gpu='frozen synthetic H100 UUID',torch='synthetic',cuda='synthetic',
            target_dtype='torch.float32',draft_dtype='torch.bfloat16',prefix_dtype='torch.bfloat16',tf32=False)
        class Frozen:
            def eval(self):return self
            def requires_grad_(self,*args):return self
        class Model(Frozen):
            def __init__(self,dtype):self.dtype=dtype;self.config=SimpleNamespace(num_hidden_layers=1)
            def parameters(self):return [SimpleNamespace(device=torch.device('cuda:0'),dtype=self.dtype)]
        teacher,student=Model(torch.float32),Model(torch.bfloat16)
        class Tokenizer:
            eos_token_id=None
            def __len__(self):return 64
        class MockPolicy(Frozen):
            margin=.05;prior_weight=1.;seed=412
        class MockEngine:
            family='retrieval'
            def __init__(self,t,s,c,*unused,round_observer=None,top_chunks=None,**kwargs):
                self.teacher,self.student,self.config=t,s,c;self.round_observer=round_observer
                self.closed=False;self.calls=0;self.native_t=self.native_s=SimpleNamespace()
                self.draft_runtime=None if c.draft_execution=='eager' else SimpleNamespace(max_entries=ResidentDraftRuntime.MAX_ENTRIES)
                self.category='reference' if self.family=='retrieval' and top_chunks is None else 'timing' if state['quality_done'] else 'quality'
                self.identity=uuid.uuid4().hex;state['created'].append(self);state['alive'].add(self.identity)
                if self.category=='timing':
                    self_check=state['alive']-{self.identity}
                    if self_check:raise AssertionError('Quality/cache owner survived into timing')
                    state['timing_ids'].add(self.identity)
                elif self.category=='quality':state['quality_ids'].add(self.identity)
                if len(state['alive'])!=1:raise AssertionError('Multiple private owners alive')
            def __enter__(self):return self
            def __exit__(self,*args):self.close()
            def reference(self,*args):return dict(generated=[0]*128,tokens=128)
            def generate_mock(self,ids,eos):
                self.calls+=1
                if state['queries'] is not None:state['queries'].append(dict(query={'sha256':'q'},logits=True,summary_requested=False,summary_count=0))
                if self.round_observer is not None:
                    self.round_observer.rounds=[dict(budget_features=None)]
                    self.round_observer.nonzero_injection=self.family=='rescue'
                return dict(request_id=uuid.uuid4().hex,generated=[0]*128,tokens=128,accepted=0,proposed=1,blocks=1,
                    trace=[],draft_forwards=1,selector_refreshes=0,draft_runtime={'request':dict(replays=1,captures=int(self.calls==1))})
            def close(self):self.closed=True;state['alive'].discard(self.identity)
        class MockRescue(MockEngine):family='rescue'
        class MockTree(MockEngine):
            family='full_tree'
            def __init__(self,template,path,arm):
                self.arm=arm
                super().__init__(teacher,student,runner.own_config(),top_chunks=1)
            def generate(self,ids):
                self.calls+=1
                return dict(generated=[0]*128,tokens=128,upstream=dict(output_token_ids=[0]*128,total_generated=128,
                    raw_total_generated=128,generation_overshoot=0,accept_length_list=[0]*128))
            def signature(self):return dict(identity=self.identity)
            def stats(self):return snapshot(self)
        class Group:
            def __init__(self,*args):self.engines=[]
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
        original_write=runner.write
        def write(path,value):
            if Path(path).name=='QUALITY_PASSED.json':
                if state['alive']:raise AssertionError('Quality gate reached with live owner')
                state['quality_done']=True
            original_write(path,value)
        canonical=lambda row,ref,*args,**kwargs: self.assertEqual(row['generated'],ref)
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            output=Path(tmp);(output/'cost').write_text('{}')
            args=SimpleNamespace(output=output,teacher=output/'teacher',student=output/'student',mapper=output/'mapper',policy=output/'policy',cost=output/'cost',prime_file=output/'prime')
            choices=runner.frozen_choices(selection(False));docs={i:dict(prefix_ids=[1],suffix_ids=[2],document_ids=[3]*8192) for i in range(900,916)}
            prepared=dict(choices=choices,documents=docs,counts=dict.fromkeys(runner.expected_counts(choices),0),seen=set(),sources={},
                asset_paths=[],assets={},baseline_protocol={'environment':environment,'flash_attention':{}},baseline_references={},metadata={})
            patches=[patch.object(baselines,'BaselineEngine',MockEngine),patch.object(engine,'RescueEngine',MockRescue),
                patch.object(loading,'load_models',return_value=(teacher,student,Tokenizer())),patch.object(packing,'share_packed_projections'),
                patch.object(query_mapper,'mapper_from_payload',return_value=Frozen()),patch.object(step_budget.StepBudget,'from_payload',return_value=MockPolicy()),
                patch.object(specextend_qwen,'build',return_value=SimpleNamespace(prefill_precision='native',base_model=teacher,draft_model=student)),
                patch.object(runner,'HookOwners',Group),patch.object(runner,'gpu_snapshot',return_value=environment),
                patch.object(runner,'source_bindings',return_value={}),patch.object(runner.baseline,'check_full_bindings',return_value={}),
                patch.object(runner.baseline,'flash_receipt',return_value={}),patch.object(runner.baseline,'FullOwner',MockTree),
                patch.object(runner,'read_prime',return_value=({n:[1]*n for n in runner.LENGTHS},{str(n):[0]*128 for n in runner.LENGTHS})),
                patch.object(runner.torch,'load',return_value=payload),patch.object(runner,'native_digest',return_value={'all_layers':'native'}),
                patch.object(runner,'runtime_snapshot',side_effect=snapshot),patch.object(runner,'owner_signature',side_effect=lambda e:{'identity':e.identity}),
                patch.object(runner,'record_draft_steps',side_effect=queries),patch.object(runner,'write',side_effect=write),
                patch.object(runner,'own_quality_pair'),patch.object(runner,'baseline_quality_pair'),patch.object(runner,'check_pair'),
                patch.object(runner,'check_own_output',side_effect=canonical),patch.object(runner.baseline,'check_output',side_effect=canonical),
                patch.object(runner.baseline,'check_reproducible'),patch.object(runner,'progress')]
            for p in patches:stack.enter_context(p)
            result=runner.execute_study(args,prepared)
            self.assertTrue(result['passed']);self.assertEqual(result['logical_requests'],434)
            self.assertEqual(len(state['quality_ids']),72);self.assertEqual(len(state['timing_ids']),10)
            self.assertFalse(state['quality_ids'] & state['timing_ids']);self.assertFalse(state['alive'])
            owners=[json.loads(line) for line in (output/'owners.jsonl').read_text().splitlines()]
            self.assertEqual(len(owners),10);self.assertTrue(all(o['engine_closed'] for o in owners))


if __name__=='__main__':unittest.main()
