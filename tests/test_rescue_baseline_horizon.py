"""Horizon orchestration tests; synthetic clocks/tokens are not GPU results."""
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch

from rescue import baseline_horizon as runner
from rescue import baseline_horizon_data as data
from rescue.round_overhead_run import sha256


DOCUMENTS=list(data.TRAIN_INDICES)+list(data.OPT_INDICES)+[604]


def stage1_selection():
    arms={a['mode']:a for a in runner.base.arms()}
    modes=('retrieval32_k4_eager','retrieval32_k4_resident','retrieval32_k4_graph','tree32_n64')
    return dict(per_context={str(n):dict(groups={g:dict(mode=m,arm=copy.deepcopy(arms[m]),
        warm_requests=6,warm_tps=12.,original_receipt='unchanged') for g,m in zip(runner.GROUPS,modes)}) for n in runner.LENGTHS})


def synthetic_rows():
    sequence,_,_=runner.schedule(DOCUMENTS)
    return [dict(mode=a['mode'],length=n,**item,warm_index=j,phase='warm',performance_eligible=True,
        tokens=128,outer_seconds=2.,request_id=f'{n}:{a["mode"]}:{j}',
        draft_runtime=dict(request=dict(captures=int(j==4),capture_host_seconds=.1 if j==4 else 0.)))
        for n in runner.LENGTHS for a in runner.base.arms() for j,item in enumerate(sequence)]


class HorizonHelpers(unittest.TestCase):
    def test_actual_retrieval_and_working_bind_bound_all_frontier_graph_shapes_below128(self):
        """Only CPU cache/control states: no model forward, graph capture or data.

        T retained chunks give B=32T and W=B-32+r, r=1..32.
        Each round rebinds W (refresh4 only changes *which* chunks). True
        logits steps1..k give31+k distinct lengths; nonterminal full-k
        acceptance alone can reach the false-logits catchup, giving32 more.
        Native position/token values and selected IDs do not enter graph keys.
        """
        from rescue.attention import DraftAttention
        from rescue.baselines import BASELINE_GRAPH_ENTRIES,ChunkRetrieval
        from rescue.cache import NativeCache,WorkingCache
        from rescue.draft_graph import ResidentDraftRuntime
        from rescue.injection import ContentInjector
        for top in (16,32):
            native=NativeCache(1,1,2,5000,torch.float32,'cpu')
            native.key_storage.zero_();native.value_storage.zero_()
            working=WorkingCache(native,top*32,12,direct_gather=True,mode='packed')
            selector=ChunkRetrieval(top);injector=ContentInjector(None,5000,'cpu',tokens=0,strength=0.,optimized=True)
            bindings=ResidentDraftRuntime._bindings(SimpleNamespace(injector=injector))
            self.assertEqual(bindings,((),(),()))
            shapes={k:set() for k in (1,2,4,8,12)}
            one=torch.zeros((1,1,1,2))
            for residue in range(1,33):
                n=4096+residue;native.lengths[:]=[n]
                attention=torch.zeros((1,n)) if residue%4==0 else None
                positions=selector.select(n,'cpu',attention)
                base=32*top-32+residue
                self.assertEqual(len(positions),base)
                self.assertEqual(len(selector.selected),top)
                self.assertEqual(selector.selected[-1],(n-1)//32)
                for k in shapes:
                    working.bind(native,positions)
                    self.assertEqual(working.lengths,[base])
                    for j in range(1,k+2):
                        working.append(0,one,one)
                        self.assertEqual(working.lengths,[base+j])
                        shapes[k].add((working.lengths[0],j<=k,False,bindings))
            for k,keys in shapes.items():
                self.assertEqual(sum(key[1] is True for key in keys),31+k)
                self.assertEqual(sum(key[1] is False for key in keys),32)
                self.assertEqual(len(keys),63+k)
                self.assertLessEqual(len(keys),75)
                self.assertLess(len(keys),BASELINE_GRAPH_ENTRIES)
            # The actual adapter path forwards summary=False for this lean
            # baseline. An inert runtime spy observes flags only, no forward.
            calls=[]
            class RuntimeSpy:
                def __init__(self):self.native=native
                def step(self,token,logits=True,summary=False):
                    calls.append((logits,summary));return None,None,[]
            adapter=DraftAttention(None,working,injector,observe=False,optimized=True,runtime=RuntimeSpy())
            adapter.prefill=False
            adapter.step(1,native,logits=True);adapter.step(1,native,logits=False)
            self.assertEqual(calls,[(True,False),(False,False)])
            self.assertEqual(ResidentDraftRuntime._bindings(SimpleNamespace(injector=injector)),bindings)

    def test_exact2540_serial38_config_horizon_and_same32_order_everywhere(self):
        sequence,owners,warm=runner.schedule(DOCUMENTS)
        self.assertEqual((len(sequence),len(owners),len(warm)),(32,76,2432))
        self.assertEqual(sum(v for k,v in runner.EXPECTED.items() if k!='owners'),2540)
        self.assertEqual({(x['source_index'],x['repeat']) for x in sequence},{(i,r) for i in DOCUMENTS for r in range(2)})
        for index,owner in enumerate(owners):
            block=warm[index*32:(index+1)*32]
            self.assertEqual([{k:r[k] for k in ('source_index','repeat')} for r in block],sequence)
            self.assertEqual({(r['mode'],r['length']) for r in block},{(owner['mode'],owner['length'])})
        self.assertEqual((sequence,owners,warm),runner.schedule(DOCUMENTS))
        self.assertNotEqual((sequence,owners,warm),runner.schedule(DOCUMENTS,9))
        with self.assertRaises(RuntimeError):runner.schedule(DOCUMENTS[:-1])

    def test_primary32_exact_tie_and_original_stage1_retained_not_replaced(self):
        original=stage1_selection();before=copy.deepcopy(original);rows=synthetic_rows()
        selected=runner.select_winners(rows,original)
        for n in runner.LENGTHS:
            cell=selected['per_context'][str(n)]
            self.assertEqual(cell['groups']['retrieval/eager']['mode'],'retrieval16_k12_eager')
            self.assertEqual(cell['groups']['full_tree']['mode'],'tree16_n32')
            self.assertEqual(cell['stage1_groups'],before['per_context'][str(n)]['groups'])
            self.assertEqual(len(cell['union_modes']),8)
            for group in runner.GROUPS:self.assertEqual(cell['union_by_group'][group][0],before['per_context'][str(n)]['groups'][group])
        self.assertEqual(original,before)
        # A genuinely higher32-request score wins even if its first6 are worse.
        for row in rows:
            if row['mode']=='retrieval32_k1_graph':row['outer_seconds']=4. if row['warm_index']<6 else 1.
        best=runner.select_winners(rows,original)['per_context']['4096']['groups']['retrieval/graph']
        self.assertEqual(best['mode'],'retrieval32_k1_graph')
        self.assertLess(best['prefix6']['tps'],64.)
        self.assertEqual(best['warm_tps'],4096/50.)
        for row in rows:
            if row['mode']=='retrieval32_k1_eager':row['outer_seconds']-=1e-10
        self.assertEqual(runner.select_winners(rows,original)['per_context']['4096']['groups']['retrieval/eager']['mode'],'retrieval32_k1_eager')

    def test_missing_duplicate_profile_or_invalid_clock_cannot_freeze(self):
        for corruption in ('missing','duplicate','profile','clock','stage1'):
            rows=synthetic_rows();old=stage1_selection()
            if corruption=='missing':rows.pop()
            if corruption=='duplicate':rows[1]=copy.deepcopy(rows[0])
            if corruption=='profile':rows[0]['performance_eligible']=False
            if corruption=='clock':rows[0]['outer_seconds']=float('nan')
            if corruption=='stage1':old['per_context']['4096']['groups'].pop('full_tree')
            with self.subTest(corruption=corruption),self.assertRaises(RuntimeError):runner.select_winners(rows,old)

    def test_frozen_protocol_source_and_cohort_tampering_rejected(self):
        root=Path(runner.__file__).resolve().parents[1]
        value=json.loads((root/runner.PREREGISTRATION).read_text());runner.validate_preregistration(value)
        for kind in ('requests','cohort','backend','scope','model','grid','anchor'):
            bad=copy.deepcopy(value)
            if kind=='requests':bad['execution']['timing_requests']=2431
            if kind=='cohort':bad['data']['train_indices'][-1]=999
            if kind=='backend':bad['common']['graph_max_entries']=32
            if kind=='scope':bad['no_new_confirmation_access']=False
            if kind=='model':bad['common']['teacher']='different model'
            if kind=='grid':bad['arms']['retrieval']['fixed_k'][-1]=16
            if kind=='anchor':bad['frozen_rescue_anchor_for_later_confirmation']['mapper_sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):runner.validate_preregistration(bad)

    def test_actual_sources_must_include27_tree_and_keep_shared_stage1_bytes(self):
        paths=['rescue/baseline_horizon.py','rescue/baseline_horizon_data.py','rescue/baseline_horizon_report.py',
            'rescue/baselines.py','exp/attention_horizon/selection.py','exp/selective_kv/bridge.py',runner.PREREGISTRATION,runner.PIPELINE,
            'tests/test_rescue_baseline_horizon.py','tests/test_rescue_baseline_horizon_data.py','tests/test_rescue_baseline_horizon_report.py']+[
            f'exp/specextend_bridgekv/specextend/classic/source{i}.py' for i in range(27)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in paths:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# frozen fixture\n')
            previous={name:sha256(root/name) for name in paths if name.startswith('exp/specextend_bridgekv/') or name=='rescue/baselines.py'}
            inventory=runner.source_inventory(root,{'source_sha256':previous})
            self.assertEqual(len([p for p in inventory if p.startswith('exp/specextend_bridgekv/')]),27)
            absent=dict(previous);absent['rescue/missing_runtime.py']='f'*64
            with self.assertRaisesRegex(RuntimeError,'missing or changed'):runner.source_inventory(root,{'source_sha256':absent})
            (root/'rescue/baselines.py').write_text('# changed algorithm\n')
            with self.assertRaisesRegex(RuntimeError,'shared stage1'):runner.source_inventory(root,{'source_sha256':previous})
            (root/'rescue/baselines.py').write_text('# frozen fixture\n');(root/paths[-1]).unlink()
            with self.assertRaisesRegex(RuntimeError,'Complete unchanged'):runner.source_inventory(root,{'source_sha256':previous})

    def test_stage1_gate_precedes_all_data_hash_or_model_work(self):
        from rescue import fair_confirmation_data
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=SimpleNamespace(teacher=root/'models/t',student=root/'models/s',output=root/'fresh',
                stage1_results=root/'evidence/results',stage1_analysis=root/'evidence/results/analysis',
                stage1_verified=root/'evidence/VERIFIED_COMPLETE.json',stage1_manifest=root/'evidence/FINAL_ARTIFACTS_MANIFEST.json',
                **{name:root/'data'/name for name in runner.SOURCE_NAMES})
            with patch.object(fair_confirmation_data,'validate_baseline_freeze',side_effect=RuntimeError('stage1 incomplete')) as gate,\
                 patch.object(data,'load_horizon_documents',side_effect=AssertionError('old data opened')) as read,\
                 patch.object(runner,'sha256',side_effect=AssertionError('data hashed')) as digest,\
                 patch.object(runner,'execute_study',side_effect=AssertionError('GPU work')) as execute:
                with self.assertRaisesRegex(RuntimeError,'stage1 incomplete'):runner.run(args)
                gate.assert_called_once();read.assert_not_called();digest.assert_not_called();execute.assert_not_called()
            self.assertEqual(json.loads((args.output/'FAILURE.json').read_text())['logical_requests'],0)
            self.assertFalse((args.output/'GATE_PASSED.json').exists())

    def test_portable_stage1_bytes_pin_original_paths_and_detect_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';output=root/'new';old.mkdir();output.mkdir()
            (old/'results/analysis').mkdir(parents=True);(old/'results/raw').write_text('raw\n');(old/'verified').write_text('verified\n')
            manifest=old/'manifest';manifest.write_text(json.dumps({'file_sha256':{'results/raw':sha256(old/'results/raw')}}))
            args=SimpleNamespace(stage1_manifest=manifest,stage1_results=old/'results',stage1_analysis=old/'results/analysis',stage1_verified=old/'verified')
            copied=runner.copy_stage1_evidence(args,output)
            self.assertEqual((output/'stage1_evidence/results/raw').read_text(),'raw\n')
            pins=json.loads((output/copied['pin_index']).read_text())['original_to_saved']
            self.assertEqual(pins[str((old/'results/raw').resolve())]['saved_path'],'stage1_evidence/results/raw')
            (old/'results/raw').write_text('tampered\n')
            with self.assertRaisesRegex(RuntimeError,'changed during copy'):runner.copy_stage1_evidence(args,root/'another')


class FullSyntheticExecution(unittest.TestCase):
    def exercise(self,output,bad_first=False):
        """Run all2540 real orchestration iterations; no model/clock performance claims."""
        from rescue import baselines,loading,packing,specextend_qwen
        state=dict(alive=set(),maximum_alive=0,created=[],serving=[])
        environment=dict(nvidia_smi_gpu='synthetic frozen GPU',torch='synthetic',cuda='synthetic',
            target_dtype='torch.float32',draft_dtype='torch.bfloat16',prefix_dtype='torch.bfloat16',tf32=False)
        class Model:
            def __init__(self,dtype):self.dtype=dtype
            def eval(self):return self
            def requires_grad_(self,*a):return self
            def parameters(self):return [SimpleNamespace(device=torch.device('cuda:0'),dtype=self.dtype)]
        teacher,student=Model(torch.float32),Model(torch.bfloat16)
        class Tokenizer:
            eos_token_id=None
            def __len__(self):return 64
        class Engine:
            def __init__(self,t,s,c,top_chunks=None,**kw):
                self.config=c;self.closed=False;self.calls=0;self.identity=uuid.uuid4().hex;self.is_reference=top_chunks is None
                self.draft_runtime=None if c.draft_execution=='eager' else SimpleNamespace(max_entries=128,retain_query=False)
                state['alive'].add(self.identity);state['maximum_alive']=max(state['maximum_alive'],len(state['alive']));state['created'].append(self)
                if len(state['alive'])!=1:raise AssertionError('More than one live private owner')
                if not self.is_reference:state['serving'].append(self)
            def __enter__(self):return self
            def __exit__(self,*a):self.close()
            def reference(self,*a):return dict(generated=[0]*128,tokens=128)
            def generate(self,ids,*a):
                self.calls+=1
                generated=[0]*128
                if bad_first and len(state['serving'])==1 and self.calls==1:generated[-1]=7
                return dict(request_id=uuid.uuid4().hex,generated=generated,tokens=128,accepted=0,proposed=1,blocks=1,
                    trace=[],draft_forwards=1,selector_refreshes=0,draft_runtime={'request':dict(replays=1,captures=int(self.calls==1),capture_host_seconds=.01 if self.calls==1 else 0.)})
            def close(self):self.closed=True;state['alive'].discard(self.identity)
        class Tree(Engine):
            def __init__(self,template,path,arm):
                self.arm=arm;super().__init__(teacher,student,runner.base.linear_config(runner.base.arms()[0]),top_chunks=arm['top_chunks'])
            def generate(self,ids):
                result=super().generate(ids);result['upstream']=dict(output_token_ids=result['generated'].copy(),total_generated=128,
                    raw_total_generated=128,generation_overshoot=0,accept_length_list=[0]*128)
                return result
            def signature(self):return dict(identity=self.identity)
            def stats(self):return snapshot(self)
        class Group:
            def __init__(self,*a):self.engines=[]
            def idle(self):pass
            def add(self,e):self.engines.append(e)
            def close(self,e):e.close();self.engines.remove(e)
            def generate(self,e,ids,eos):return e.generate(ids,eos)
        def snapshot(e):return dict(entries=0 if e.closed else 1,captures=int(e.calls>0),replays=e.calls,max_graph_entries=128)
        args=SimpleNamespace(output=output,teacher=output/'teacher',student=output/'student')
        docs={i:dict(source_index=i,document_sha256=f'{i:064x}',prefix_ids=[1],suffix_ids=[2],document_ids=[3]*8192) for i in DOCUMENTS}
        old_inputs={f'271:{n}':hashlib.sha256(json.dumps(runner.context(docs[271],n)).encode()).hexdigest() for n in runner.LENGTHS}
        prepared=dict(documents=docs,prime=docs[271],sources={},assets={},asset_paths=[],counts=dict.fromkeys(runner.EXPECTED,0),seen=set(),
            stage1_protocol=dict(environment=environment,flash_attention={},input_sha256=old_inputs),stage1_selection=stage1_selection(),
            stage1_references={f'271:{n}':[0]*128 for n in runner.LENGTHS},metadata=dict(stage1_selection_sha256='a'*64))
        def canonical(row,ref,*a):runner.require(row['generated']==ref,'Synthetic canonical mismatch')
        with ExitStack() as stack:
            for p in [patch.object(baselines,'BaselineEngine',Engine),patch.object(loading,'load_models',return_value=(teacher,student,Tokenizer())),
                patch.object(packing,'share_packed_projections'),patch.object(specextend_qwen,'build',return_value=SimpleNamespace(base_model=teacher,draft_model=student)),
                patch.object(runner,'HookOwners',Group),patch.object(runner,'gpu_snapshot',return_value=environment),patch.object(runner,'source_bindings',return_value={}),
                patch.object(runner,'gpu_telemetry',return_value=dict(read_only=True,primary_clock_included=False,values={'uuid':'synthetic frozen GPU'})),
                patch.object(runner.base,'check_full_bindings',return_value={}),patch.object(runner.base,'flash_receipt',return_value={}),
                patch.object(runner.base,'FullOwner',Tree),patch.object(runner,'runtime_snapshot',side_effect=snapshot),
                patch.object(runner,'owner_signature',side_effect=lambda e:{'identity':e.identity}),patch.object(runner.base,'check_output',side_effect=canonical),
                patch.object(runner.base,'check_reproducible'),patch.object(runner,'progress')]:stack.enter_context(p)
            try:return runner.execute_study(args,prepared),state,prepared
            except BaseException:
                self.assertFalse(state['alive'],'Failure leaked an experiment-owned serving engine')
                raise

    def test_actual2540_request_control_flow_has32independent_ar_76_private_closed_owners(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);result,state,prepared=self.exercise(output)
            self.assertTrue(result['passed']);self.assertEqual(result['logical_requests'],2540);self.assertEqual(prepared['counts'],runner.EXPECTED)
            self.assertEqual(state['maximum_alive'],1);self.assertFalse(state['alive']);self.assertEqual(len(state['created']),108)
            self.assertEqual(len(state['serving']),76);self.assertTrue(all(e.closed and e.calls==33 for e in state['serving']))
            owners=[json.loads(x) for x in (output/'owners.jsonl').read_text().splitlines()]
            runtime=[json.loads(x) for x in (output/'runtime.jsonl').read_text().splitlines()]
            self.assertEqual(len(owners),76);self.assertEqual(len(runtime),2432)
            self.assertEqual(len((output/'tree_raw.jsonl').read_text().splitlines()),528)
            self.assertEqual(len((output/'gpu_diagnostics.jsonl').read_text().splitlines()),152)
            sequence=runner.schedule(DOCUMENTS)[0]
            for owner in owners:
                rows=[r for r in runtime if r['owner_id']==owner['owner_id']]
                self.assertEqual([{k:r[k] for k in ('source_index','repeat')} for r in rows],sequence)
                self.assertTrue(owner['engine_closed']);self.assertEqual(owner['closed_runtime_stats']['entries'],0)
            self.assertTrue((output/'FROZEN_SELECTION.json').is_file());self.assertFalse((output/'COMPLETE.json').exists())

    def test_raw_saved_before_first_canonical_failure_and_private_owner_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'Synthetic canonical'):self.exercise(output,bad_first=True)
            primes=[json.loads(x) for x in (output/'prime.jsonl').read_text().splitlines()]
            self.assertEqual(len(primes),1);self.assertEqual(primes[0]['generated'][-1],7)
            self.assertFalse((output/'FROZEN_SELECTION.json').exists());self.assertFalse((output/'COMPLETE.json').exists())


if __name__=='__main__':unittest.main()
