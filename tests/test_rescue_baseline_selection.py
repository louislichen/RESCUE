"""Frozen menu/accounting and genuine tiny-model checks, not speed results."""
import ast
import copy
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rescue import baseline_selection as runner
from rescue.baselines import BaselineEngine
from rescue.draft_graph_benchmark import HookOwners
from rescue.efficient_attention import install_efficient_attention
from rescue.packing import pack_projections
from rescue.round_overhead_run import sha256
from test_rescue_system import models


def synthetic_linear(k=12):
    arm=next(a for a in runner.arms() if a['mode']==f'retrieval16_k{k}_resident')
    trace=[];progress=0
    while progress<128:
        cap=min(k,128-progress);next_progress=progress+cap
        if next_progress<128:next_progress+=1
        trace.append(dict(round_index=len(trace),prefix=4095+progress,cap=cap,proposed=cap,
            accepted=cap,terminal=next_progress>=128,workset_tokens=512))
        progress=next_progress
    generated=list(range(128))
    return arm,dict(generated=generated,tokens=128,output_agrees=True,length=4096,trace=trace,
        blocks=len(trace),accepted=sum(t['accepted'] for t in trace),proposed=sum(t['proposed'] for t in trace),
        projected_positions=0,draft_shadow_probes=0,execution_backend='resident',baseline_algorithm='qwen_chunk_retrieval',
        draft_forwards=sum(t['proposed'] for t in trace)+len(trace)-1,selector_refreshes=3)


def synthetic_tree(nodes=64,eos=None):
    arm=next(a for a in runner.arms() if a['mode']==f'tree16_n{nodes}')
    scored=list(range(128));visible=scored if eos is None else scored[:scored.index(eos)+1]
    # 14 rounds x(9 accepted +1 target) =140 computed tokens, overshoot12.
    raw=dict(generated=visible,tokens=len(visible),total_seconds=2.,total_tps=len(visible)/2.,
        upstream=dict(output_token_ids=scored,total_generated=128,raw_total_generated=140,
            generation_overshoot=12,accept_length_list=[9]*14),
        observed_full_config=dict(target_use_flash_prefill=True,target_use_hybrid_tree_attn=True,
            draft_use_flash_prefill=True,use_retrieval_cache=True,retrieve_top_k=16,
            retrieval_chunk_size=32,retrieve_every_n_steps=4))
    return arm,raw,visible


def selection_rows():
    return [dict(mode=arm['mode'],length=n,source_index=i,repeat=r,phase='warm',performance_eligible=True,
        tokens=128 if i!=271 else 64,outer_seconds=2.+r,request_id=f'{arm["mode"]}:{n}:{i}:{r}')
        for n in runner.LENGTHS for arm in runner.arms() for i in runner.DOCUMENTS for r in range(2)]


class BaselineSelectionHelpers(unittest.TestCase):
    def test_exact38_menu_76private_serial_blocks_456warm_and538requests(self):
        arms=runner.arms();owners,warm=runner.schedule()
        self.assertEqual(len(arms),38);self.assertEqual(len({a['mode'] for a in arms}),38)
        self.assertEqual(len([a for a in arms if a['family']=='retrieval']),30)
        self.assertEqual(len([a for a in arms if a['family']=='full_tree']),8)
        self.assertEqual(len(owners),76);self.assertEqual(len(warm),456)
        self.assertEqual(runner.schedule(),(owners,warm));self.assertNotEqual(runner.schedule(5),(owners,warm))
        self.assertEqual(sum(v for k,v in runner.EXPECTED.items() if k!='owners'),538)
        for offset,owner in enumerate(owners):
            block=warm[offset*6:offset*6+6]
            self.assertEqual({(r['mode'],r['length']) for r in block},{(owner['mode'],owner['length'])})
            self.assertEqual({(r['source_index'],r['repeat']) for r in block},
                             {(i,r) for i in runner.DOCUMENTS for r in range(2)})
        for arm in arms:
            if arm['family']=='retrieval':
                cfg=runner.linear_config(arm)
                self.assertEqual(cfg.fixed_k,arm['fixed_k']);self.assertEqual(cfg.draft_execution,arm['backend'])
                self.assertEqual((cfg.inject_tokens,cfg.injection_strength,cfg.injection_layers),(0,0.,()))
                self.assertEqual(cfg.prefill_precision,'bfloat16');self.assertFalse(cfg.profile)
            else:self.assertEqual((arm['max_depth'],arm['threshold']),(10,.7))

    def test_parameterized_linear_cap12_prefix_and_eos_not_old_cap4(self):
        for k in (1,2,4,8,12):
            arm,row=synthetic_linear(k);runner.check_output(row,row['generated'],arm)
            wrong=copy.deepcopy(row);wrong['trace'][0]['prefix']+=1
            with self.assertRaisesRegex(RuntimeError,'prefix'):runner.check_output(wrong,row['generated'],arm)
        arm,row=synthetic_linear(12);row['trace'][0]['cap']=4
        with self.assertRaisesRegex(RuntimeError,'cap'):runner.check_output(row,row['generated'],arm)

    def test_tree64_preserves_real_overshoot12_unexposed_tail_and_all_cost(self):
        arm,raw,reference=synthetic_tree();row=runner.normalize_tree(raw,arm,reference)
        runner.check_output(row,reference,arm)
        self.assertEqual(row['raw_total_generated'],140);self.assertEqual(row['generation_overshoot'],12)
        self.assertEqual(row['accepted'],126);self.assertEqual(row['blocks'],14)
        self.assertIsNone(row['proposed']);self.assertIsNone(row['acceptance_rate'])
        self.assertFalse(row['raw_tail_ids_available']);self.assertEqual(row['total_seconds'],2.)
        self.assertEqual(row['upstream'],raw['upstream'])
        arm,raw,reference=synthetic_tree(eos=7);row=runner.normalize_tree(raw,arm,reference,eos=7)
        runner.check_output(row,reference,arm,eos=7)
        self.assertEqual(row['scored_generated_before_eos'],list(range(128)))
        self.assertEqual(row['raw_total_generated'],140);self.assertEqual(row['tokens'],8)
        bad=copy.deepcopy(row);bad['generation_overshoot']=3
        with self.assertRaisesRegex(RuntimeError,'overshoot'):runner.check_output(bad,reference,arm,eos=7)
        bad=copy.deepcopy(row);bad['wrapper_generated'][-1]=999
        with self.assertRaisesRegex(RuntimeError,'wrapper'):runner.check_output(bad,reference,arm,eos=7)

    def test_wrong_tree_token_is_preserved_not_clipped_away(self):
        arm,raw,reference=synthetic_tree();reference=reference.copy();raw['upstream']['output_token_ids'][-1]=999
        raw['generated'][-1]=999
        row=runner.normalize_tree(raw,arm,reference)
        self.assertEqual(row['generated'][-1],999);self.assertFalse(row['output_agrees'])
        with self.assertRaisesRegex(RuntimeError,'Client-visible'):runner.check_output(row,reference,arm)

    def test_primary_selection_is_sixwarm_pooled_ratio_with_exact_lexical_ties(self):
        rows=selection_rows();selected=runner.select_winners(rows)
        for n in runner.LENGTHS:
            groups=selected['per_context'][str(n)]['groups']
            self.assertEqual(groups['retrieval/eager']['mode'],'retrieval16_k12_eager')
            self.assertEqual(groups['full_tree']['mode'],'tree16_n32')  # lexical, not numeric node order
            receipt=groups['retrieval/eager'];self.assertEqual(receipt['warm_tps'],640/15.)
        # Tiny real score differences must not be rounded/isclose-tied away.
        for row in rows:
            if row['mode']=='retrieval32_k1_eager':row['outer_seconds']-=1e-10
        self.assertEqual(runner.select_winners(rows)['per_context']['4096']['groups']['retrieval/eager']['mode'],
                         'retrieval32_k1_eager')
        for corruption in ('missing','phase','timing'):
            bad=copy.deepcopy(rows)
            if corruption=='missing':bad.pop()
            if corruption=='phase':bad[0]['phase']='cold_prime'
            if corruption=='timing':bad[0]['outer_seconds']=float('nan')
            with self.subTest(corruption=corruption),self.assertRaises(RuntimeError):runner.select_winners(bad)

    def test_train_only_locked_cohort_and_natural_contexts(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'LENGTHS',(8,16)):
            path=Path(tmp)/'data.jsonl'
            rows=[dict(source_index=i,split='train',document_sha256=f'{i:064x}',prefix_ids=[1],suffix_ids=[2],
                       document_ids=[3]*20) for i in runner.DOCUMENTS]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.assertEqual(set(runner.read_documents(path,sha256(path))),set(runner.DOCUMENTS))
            with self.assertRaisesRegex(RuntimeError,'SHA'):runner.read_documents(path,'a'*64)
            rows[0]['split']='test';path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(RuntimeError,'train'):runner.read_documents(path,sha256(path))

    def test_source_bindings_and_flash_binary_gate_reject_namespace_or_stub(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);modules={}
            def imported(name):
                path=root/Path(*name.split('.')).with_suffix('.py');path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('# frozen\n');module=SimpleNamespace(__file__=str(path));modules[name]=module;return module
            with patch.object(runner.importlib,'import_module',side_effect=imported):
                self.assertEqual(len(runner.source_bindings(root)),11)
            modules['rescue.engine'].__file__=str(root/'foreign/engine.py')
            with patch.object(runner.importlib,'import_module',side_effect=modules.__getitem__):
                with self.assertRaisesRegex(RuntimeError,'foreign'):runner.source_bindings(root)
            files={}
            for name in ('flash_attn','flash_attn.flash_attn_interface','flash_attn_2_cuda'):
                path=root/(name.replace('.','_')+('.so' if name.endswith('_cuda') else '.py'))
                path.write_bytes(b'synthetic fixture, never loaded')
                files[name]=SimpleNamespace(__file__=str(path),__version__='2.8.3')
            with patch.object(runner.importlib,'import_module',side_effect=files.__getitem__):
                receipt=runner.flash_receipt();self.assertEqual(receipt['version'],'2.8.3')
                files['flash_attn_2_cuda'].__file__=files['flash_attn'].__file__
                with self.assertRaisesRegex(RuntimeError,'binary'):runner.flash_receipt()

    def test_no_global_recorder_shadow_and_no_old_hardcoded_checker(self):
        source=inspect.getsource(runner.run);tree=ast.parse(source)
        assigned={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        self.assertFalse(assigned & {'record','context','schedule','progress','write','append','arms','source_bindings'})
        self.assertNotIn('normalize_full',source);self.assertNotIn('mapper_from_payload',source)
        self.assertNotIn('StepBudget.from_payload',source)


class RealLinearK12(unittest.TestCase):
    device='cpu'
    candidate='resident'

    @torch.no_grad()
    def test_real_k12_exact_outputs_work_and_eos_with_lean_private_owner(self):
        teacher,_,_=models();student=copy.deepcopy(teacher)
        # Real attention/KV math with guaranteed acceptance/catchup boundaries.
        teacher.lm_head.weight.zero_();student.lm_head.weight.zero_()
        teacher=teacher.to(self.device);student=student.to(self.device)
        if self.device=='cuda':student.to(torch.bfloat16)
        for model in (teacher,student):
            model.eval().requires_grad_(False);install_efficient_attention(model);pack_projections(model)
        group=HookOwners(teacher,student);previous={};prompt=[3+i%30 for i in range(63)]
        for mode in ('eager',self.candidate):
            arm=next(a for a in runner.arms() if a['mode']==f'retrieval16_k12_{"graph" if mode=="cuda_graph" else mode}')
            owner=BaselineEngine(teacher,student,runner.linear_config(arm),top_chunks=16)
            group.add(owner)
            try:
                for eos in (None,0):
                    reference=owner.reference(prompt,eos)['generated']
                    row=group.generate(owner,prompt,eos)
                    row.update(length=len(prompt),output_agrees=row['generated']==reference)
                    runner.check_output(row,reference,arm,eos)
                    self.assertTrue(all(t['cap']<=12 for t in row['trace']))
                    self.assertTrue(any(t['cap']>4 for t in row['trace']))
                    self.assertEqual(row['draft_shadow_probes'],0);self.assertEqual(row['projected_positions'],0)
                    if eos in previous:runner.check_reproducible(previous[eos],row,arm)
                    previous[eos]=row
                    if mode!='eager':self.assertFalse(owner.draft_runtime.retain_query)
                    if eos is None:self.assertGreater(row['draft_forwards'],row['proposed'])
            finally:group.close(owner)
        group.idle()


@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable; CPU tests are not GPU claims')
class RealGraphK12(RealLinearK12):
    device='cuda'
    candidate='cuda_graph'


if __name__=='__main__':unittest.main()
