"""Fabricated evidence/data tests only; no scientific files or models opened."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rescue import fair_union_confirmation_data as data


PROJECT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('legacy_synthetic_data_fixture',PROJECT/'tests/test_rescue_fair_confirmation_data.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def append(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def seal(root,analysis,stage2):
    results=root/'results';complete=results/'COMPLETE.json';frozen=results/'FROZEN_SELECTION.json'
    figures=data.FIGURES if stage2 else ('quality','tps','captures','capture_time','cold','service','selected_repeat1')
    for name in figures:
        for suffix in ('.png','.svg'):
            path=analysis/(name+suffix)
            if not path.exists():path.write_bytes(b'SYNTHETIC_NOT_RENDERED_SCIENTIFIC_EVIDENCE')
    render=dict(actually_inspected_by_root=True,rendered_pngs_inspected=list(figures),
        figure_sha256={name+suffix:data.sha256(analysis/(name+suffix)) for name in figures for suffix in ('.png','.svg')})
    write(analysis/'RENDER_INSPECTION.json',render)
    source=root/'INDEPENDENT_RAW_AUDIT_SOURCE.py';source.write_text('# SYNTHETIC TEST SOURCE ONLY\n')
    audit=dict(status='INDEPENDENT_RAW_AUDIT_PASSED',all76_owners_verified=True,
        frozen_selection_sha256=data.sha256(frozen),analysis_summary_sha256=data.sha256(analysis/'summary.json'),
        audit_source_file=source.name,audit_source_sha256=data.sha256(source))
    audit.update({'all2540_requests_verified' if stage2 else 'all538_requests_verified':True,
        'retained_union_verified' if stage2 else 'eight_frozen_choices_verified':True,
        'horizon_complete_sha256' if stage2 else 'baseline_complete_sha256':data.sha256(complete),
        'horizon_protocol_sha256' if stage2 else 'baseline_protocol_sha256':data.sha256(results/'protocol.json')})
    if stage2:audit['verified_union_sha256']=data.sha256(analysis/'VERIFIED_UNION.json')
    write(root/'INDEPENDENT_RAW_AUDIT.json',audit)
    if stage2:
        (analysis/'analysis_support.py').write_text('# SYNTHETIC TEST DEPENDENCY ONLY\n')
        write(root/'ANALYSIS_RERUN_VERIFIED.json',dict(standalone_summary_exact_rerun=True,
            summary_sha256=data.sha256(analysis/'summary.json'),saved_analysis_sha256=data.sha256(analysis/'analysis_source.py'),
            dependency_sha256={'analysis_support.py':data.sha256(analysis/'analysis_support.py')}))
    analyzed=dict(status='COMPLETE',artifacts_sha256={p.name:data.sha256(p) for p in analysis.iterdir() if p.is_file() and p.name!='ANALYSIS_COMPLETE.json'})
    write(analysis/'ANALYSIS_COMPLETE.json',analyzed)
    inventory={str(p.relative_to(root)):data.sha256(p) for p in root.rglob('*') if p.is_file() and
        p.name not in ('FINAL_ARTIFACTS_MANIFEST.json','VERIFIED_COMPLETE.json')}
    write(root/'FINAL_ARTIFACTS_MANIFEST.json',dict(file_sha256=inventory))
    verified=dict(status='VERIFIED_COMPLETE',passed=True,raw_independent_audit_passed=True,rendered_figures_inspected=True,
        strict_analysis_complete=True,all76_owners_verified=True,standalone_summary_exact_rerun=True,
        frozen_selection_sha256=data.sha256(frozen),analysis_summary_sha256=data.sha256(analysis/'summary.json'),
        final_artifacts_manifest_sha256=data.sha256(root/'FINAL_ARTIFACTS_MANIFEST.json'),
        independent_raw_audit_sha256=data.sha256(root/'INDEPENDENT_RAW_AUDIT.json'),render_inspection_sha256=data.sha256(analysis/'RENDER_INSPECTION.json'))
    if stage2:verified.update(all2540_requests_verified=True,retained_union_verified=True,horizon_complete_sha256=data.sha256(complete),
        verified_union_sha256=data.sha256(analysis/'VERIFIED_UNION.json'),standalone_rerun_receipt_sha256=data.sha256(root/'ANALYSIS_RERUN_VERIFIED.json'))
    else:verified.update(all538_requests_verified=True,eight_frozen_choices_verified=True,baseline_complete_sha256=data.sha256(complete),
        verified_winners_sha256=data.sha256(analysis/'VERIFIED_WINNERS.json'))
    write(root/'VERIFIED_COMPLETE.json',verified)


def fixture(root,same_winners=False):
    env=legacy.synthetic_environment(root)
    r1,a1=env['results'],env['analysis'];stage1=r1.parent
    p1=json.loads((r1/'protocol.json').read_text())
    environment=dict(nvidia_smi_gpu='SYNTHETIC_H100',torch='synthetic',cuda='synthetic',target_dtype='torch.float32',
        draft_dtype='torch.bfloat16',prefix_dtype='torch.bfloat16',tf32=False)
    inventory=data.old._tokenizer_inventory(env['gate_args']['tokenizer_root'])
    teacher=str(env['gate_args']['tokenizer_root'])
    p1.update(args=dict(teacher=teacher),assets_sha256={teacher+'/'+n:s for n,s in inventory.items()},environment=environment,
        frozen_rescue_anchor_for_future_confirmation={k:data.OWN_METHOD[k] for k in ('mapper_sha256','policy_sha256','cost_sha256')})
    write(r1/'protocol.json',p1)
    first=json.loads((r1/'FROZEN_SELECTION.json').read_text());first['protocol_sha256']=data.sha256(r1/'protocol.json');write(r1/'FROZEN_SELECTION.json',first)
    complete=json.loads((r1/'COMPLETE.json').read_text());complete['frozen_selection_sha256']=data.sha256(r1/'FROZEN_SELECTION.json');write(r1/'COMPLETE.json',complete)
    seal(stage1,a1,False)
    stage1_args=dict(stage1_results=r1,stage1_analysis=a1,stage1_verified=stage1/'VERIFIED_COMPLETE.json',stage1_manifest=stage1/'FINAL_ARTIFACTS_MANIFEST.json')
    proof1=data.old.validate_baseline_freeze(r1,a1,stage1_args['stage1_verified'],stage1_args['stage1_manifest'])
    stage2=root/'horizon';r2=stage2/'results';a2=r2/'analysis_final_v1';a2.mkdir(parents=True)
    write(stage2/'PIPELINE_EXIT.json',dict(exit_code=0))
    for name in p1['source_sha256']:
        target=r2/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((r1/'source'/name).read_bytes())
    documents=list(data.horizon.TRAIN_INDICES+data.horizon.OPT_INDICES+(604,))
    sequence=[dict(source_index=i,repeat=rep) for rep in (0,1) for i in documents]
    menu=data._menu();references=[];primes=[];warm=[];owners=[];tree=[];index=0
    for n in data.LENGTHS:
        for i in documents:
            references.append(dict(request_id=f'{index:032x}',execution_index=index,source_index=i,length=n,
                mode='canonical_ar',phase='reference',generated=[1]*128));index+=1
    for n in data.LENGTHS:
        for j,arm in enumerate(menu):
            owner_id=f'{10000+n+j:032x}';block=[]
            for step in [dict(source_index=271,phase='cold_prime')]+[dict(v,phase='warm') for v in sequence]:
                row=dict(request_id=f'{index:032x}',execution_index=index,owner_id=owner_id,mode=arm['mode'],arm=arm,length=n,
                    generated=[1]*128,tokens=128,outer_seconds=2.,output_agrees=True,performance_eligible=True,
                    observer_enabled=False,profile_enabled=False,mapper_executed=False,controller_executed=False,**step)
                if same_winners and arm['mode'] in {v['mode'] for v in first['per_context'][str(n)]['groups'].values()}:
                    row['outer_seconds']=1.
                index+=1;block.append(row)
                (primes if step['phase']=='cold_prime' else warm).append(row)
                if arm['family']=='full_tree':tree.append(dict(request_id=row['request_id']))
            owners.append(dict(owner_id=owner_id,mode=arm['mode'],length=n,engine_closed=True,hook_restored=True,maximum_live_owners=1,
                closed_runtime_stats=dict(entries=0),prime_request_id=block[0]['request_id'],warm_request_ids=[r['request_id'] for r in block[1:]]))
    for name,rows in dict(reference=references,prime=primes,runtime=warm,owners=owners,tree_raw=tree).items():append(r2/(name+'.jsonl'),rows)
    protocol=dict(source_indices=documents,shared_warm_sequence=sequence,arms=menu,source_sha256=p1['source_sha256'],
        assets_sha256=p1['assets_sha256'],args=p1['args'],environment=environment,tokenizer_eos_token_id=999,
        stage1_freeze=proof1,stage1_selection_sha256=proof1['selection_freeze_sha256'],
        frozen_rescue_anchor_for_later_confirmation=p1['frozen_rescue_anchor_for_future_confirmation'])
    write(r2/'protocol.json',protocol)
    computed=data._recompute(warm,protocol,first)
    frozen=dict(per_context=computed,complete_request_count=2540,no_new_confirmation_opened=True,
        protocol_sha256=data.sha256(r2/'protocol.json'),stage1_selection_sha256=proof1['selection_freeze_sha256'],
        source_sha256=protocol['source_sha256'],assets_sha256=protocol['assets_sha256'])
    write(r2/'FROZEN_SELECTION.json',frozen)
    write(r2/'COMPLETE.json',dict(passed=True,counts=data.STAGE2_COUNTS,logical_requests=2540,owner_closed=76,goal_complete=False,
        sources_unchanged=True,assets_unchanged=True,dependencies_unchanged=True,all_canonical=True,backend_trajectories_equal=True,
        data_unchanged=True,stage1_verified_unchanged=True,stage1_selection_sha256=proof1['selection_freeze_sha256'],frozen_selection_sha256=data.sha256(r2/'FROZEN_SELECTION.json')))
    write(a2/'summary.json',dict(status='COMPLETE',problems=[],observed_counts=data.STAGE2_COUNTS,authoritative_union=True,stage2=computed))
    write(a2/'VERIFIED_UNION.json',dict(status='VERIFIED_FROZEN_UNION',authoritative=True,per_context=computed,
        future_contrasts=sum(len(v['union_modes']) for v in computed.values())))
    (a2/'analysis_source.py').write_text('# SYNTHETIC ANALYSIS SOURCE ONLY\n');(a2/'REPORT.md').write_text('SYNTHETIC TEST ONLY\n')
    seal(stage2,a2,True)
    args=dict(**stage1_args,stage2_results=r2,stage2_analysis=a2,stage2_verified=stage2/'VERIFIED_COMPLETE.json',stage2_manifest=stage2/'FINAL_ARTIFACTS_MANIFEST.json')
    proof=data.validate_union_freeze(**args)
    source=env['source']
    for name in data.REQUIRED_NEW_SOURCE:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        if name.endswith('_preregistration.json'):write(path,data.protocol_template())
        else:path.write_text('# SYNTHETIC FINAL SOURCE ONLY\n')
    template=source/data.REQUIRED_NEW_SOURCE[-2]
    formal=source.parent/'union_final_protocol.json';write(formal,data.make_final_protocol(template,proof))
    gate_args=dict(**args,metadata_paths=env['metadata'],raw_source_manifest=env['gate_args']['raw_source_manifest'],raw_source=env['raw'],
        tokenizer_root=env['gate_args']['tokenizer_root'],source_root=source,final_protocol=formal)
    env.update(stage1=stage1,stage2=stage2,r2=r2,a2=a2,args=args,proof=proof,gate_args=gate_args,
               union_gate=source.parent/'union_gate',formal=formal,union_source=source)
    return env


class Tokenizer:
    def encode(self,text,add_special_tokens=False):
        assert add_special_tokens is False
        if text=='short':return [400]*8191
        if text=='prefixduplicate':return [42]*8192
        if text=='sameprefix':return [1000]*8192
        return [1000+int(text.removeprefix('eligible'))]*8192


def candidates(count=16):
    return [(0,'old_doc0'),(1,'short'),(2,'prefixduplicate')]+[(3+i,'eligible'+str(i)) for i in range(count)]


class UnionFreezeTests(unittest.TestCase):
    def test_retired_rule_or_precision_change_cannot_finalize_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=fixture(Path(tmp));value=data.protocol_template();path=Path(tmp)/'template.json'
            for key in ('rule','precision','seed'):
                bad=copy.deepcopy(value)
                if key=='rule':bad['selection_rule']['minimum_source_index']=900
                if key=='precision':bad['fixed']['target_dtype']='bfloat16'
                if key=='seed':bad['seed']=20261024
                write(path,bad)
                with self.subTest(key=key),self.assertRaises(RuntimeError):data.make_final_protocol(path,env['proof'])

    def test_minimum_union_dedup_and_maximum_matching_reference_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof=fixture(Path(tmp),same_winners=True)['proof']
            self.assertEqual((proof['M'],proof['R'],proof['scientific_requests']),(8,4,438))
            self.assertEqual(proof['statistics']['simultaneous_lower_quantile'],.00625)
            self.assertTrue(all(v['stages']==['stage1','stage2'] for c in proof['provenance'].values() for v in c.values()))
        registered={a['mode']:a for a in data._menu()}
        modes=('retrieval16_k1_eager','retrieval16_k2_eager','retrieval16_k4_resident',
               'retrieval16_k8_resident','retrieval16_k12_graph','retrieval32_k1_graph','tree16_n8','tree32_n8')
        choices={str(n):{m:registered[m] for m in modes} for n in data.LENGTHS}
        self.assertEqual(len(data._extra(choices)),8)

    def test_complete_both_studies_recompute_union_counts_and_simultaneous_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=fixture(Path(tmp));proof=env['proof']
            self.assertEqual((proof['M'],proof['R']),(16,4))
            self.assertEqual(proof['scientific_requests'],162+34*16+4)
            self.assertEqual(proof['statistics']['comparisons'],16)
            self.assertEqual(proof['statistics']['simultaneous_lower_quantile'],.05/16)
            self.assertEqual(sum(v for k,v in proof['expected_counts'].items() if k!='owners'),proof['scientific_requests'])
            for n in data.LENGTHS:
                choices=proof['choices'][str(n)]
                self.assertEqual(len(choices),8)
                self.assertIn('retrieval16_k12_graph',choices)
                self.assertIn('retrieval32_k4_graph',choices)

    def test_raw_clock_or_union_tamper_cannot_be_hidden_by_resigned_receipts(self):
        for corruption in ('clock','union','source','references','render','rerun','rerun_dependency'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as tmp:
                env=fixture(Path(tmp));r2,a2=env['r2'],env['a2']
                if corruption=='clock':
                    rows=[json.loads(v) for v in (r2/'runtime.jsonl').read_text().splitlines()];rows[0]['outer_seconds']=.1;append(r2/'runtime.jsonl',rows)
                if corruption=='union':
                    value=json.loads((r2/'FROZEN_SELECTION.json').read_text());value['per_context']['4096']['union_modes'].pop();write(r2/'FROZEN_SELECTION.json',value)
                    complete=json.loads((r2/'COMPLETE.json').read_text());complete['frozen_selection_sha256']=data.sha256(r2/'FROZEN_SELECTION.json');write(r2/'COMPLETE.json',complete)
                if corruption=='source':
                    p=json.loads((r2/'protocol.json').read_text());key=next(iter(p['source_sha256']));p['source_sha256'][key]='a'*64;write(r2/'protocol.json',p)
                if corruption=='references':
                    rows=[json.loads(v) for v in (r2/'reference.jsonl').read_text().splitlines()];rows[0]['generated'][0]=2;append(r2/'reference.jsonl',rows)
                seal(env['stage2'],a2,True)
                if corruption=='render':
                    value=json.loads((a2/'RENDER_INSPECTION.json').read_text());value['actually_inspected_by_root']=False;write(a2/'RENDER_INSPECTION.json',value)
                if corruption=='rerun':
                    value=json.loads((env['stage2']/'ANALYSIS_RERUN_VERIFIED.json').read_text());value['summary_sha256']='b'*64;write(env['stage2']/'ANALYSIS_RERUN_VERIFIED.json',value)
                if corruption=='rerun_dependency':
                    value=json.loads((env['stage2']/'ANALYSIS_RERUN_VERIFIED.json').read_text());value['dependency_sha256']['analysis_support.py']='b'*64
                    write(env['stage2']/'ANALYSIS_RERUN_VERIFIED.json',value)
                    verified=json.loads((env['stage2']/'VERIFIED_COMPLETE.json').read_text())
                    verified['standalone_rerun_receipt_sha256']=data.sha256(env['stage2']/'ANALYSIS_RERUN_VERIFIED.json')
                    inventory=json.loads((env['stage2']/'FINAL_ARTIFACTS_MANIFEST.json').read_text())
                    inventory['file_sha256']['ANALYSIS_RERUN_VERIFIED.json']=verified['standalone_rerun_receipt_sha256']
                    write(env['stage2']/'FINAL_ARTIFACTS_MANIFEST.json',inventory)
                    verified['final_artifacts_manifest_sha256']=data.sha256(env['stage2']/'FINAL_ARTIFACTS_MANIFEST.json')
                    write(env['stage2']/'VERIFIED_COMPLETE.json',verified)
                with self.assertRaises((RuntimeError,KeyError)):data.validate_union_freeze(**env['args'])

    def test_missing_dual_gate_prevents_raw_hash_open_or_tokenizer_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);gate=root/'GATE_SOURCE_FREEZE.json';write(gate,dict(status='TEMPLATE_NOT_EXECUTED'))
            with (patch.object(data,'sha256',side_effect=AssertionError('candidate hash')),
                  patch.object(data,'_raw_rows',side_effect=AssertionError('raw open')),
                  patch.object(data,'_load_tokenizer',side_effect=AssertionError('tokenizer'))):
                with self.assertRaisesRegex(RuntimeError,'Unfrozen'):data.prepare_confirmation(gate)
            self.assertFalse((root/'data').exists())

    def test_freeze_never_hashes_candidate_and_locks_newrule_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=fixture(Path(tmp));original=data.sha256
            def digest(path):
                self.assertNotEqual(Path(path).resolve(),env['raw'].resolve(),'Candidate opened during freeze')
                return original(path)
            with patch.object(data,'sha256',side_effect=digest):gate=data.freeze_gate(env['union_gate'],**env['gate_args'])
            self.assertEqual(gate['selection_rule']['minimum_source_index'],0)
            self.assertEqual(gate['selection_rule']['minimum_natural_tokens'],8192)
            self.assertEqual(gate['union_freeze']['M'],16)
            self.assertFalse(gate['candidate_content_accessed'])
            self.assertEqual(set(gate['baseline_arguments']),{'baseline_results','analysis_dir','verified_complete','final_manifest'})
            self.assertEqual(set(gate['horizon_arguments']),set(gate['baseline_arguments']))

    def test_prepare_first_eligible_and_portable_proof_after_originals_are_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=fixture(Path(tmp));data.freeze_gate(env['union_gate'],**env['gate_args']);gate=env['union_gate']/'GATE_SOURCE_FREEZE.json'
            with patch.object(data,'_raw_rows',return_value=iter(candidates())),patch.object(data,'_load_tokenizer',return_value=Tokenizer()):
                manifest=data.prepare_confirmation(gate)
            self.assertEqual([v['source_index'] for v in manifest['selected']],list(range(3,19)))
            self.assertEqual(len(manifest['skipped']),3)
            self.assertTrue(all(v['original_tokens']==8192 for v in manifest['selected']))
            with (patch.object(data,'validate_source_freeze',side_effect=AssertionError('Original absolute path accessed')),
                  patch.object(data,'_raw_rows',side_effect=AssertionError('Raw read offline'))):
                proof=data.verify_dataset_manifest(env['union_gate']/'data')
            self.assertEqual(proof['status'],'PREPARED_NOT_EXECUTED')

    def test_first_eligible_holes_fake_short_and_unknown_skip_are_rejected(self):
        for fault in ('hole','short','unknown'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as tmp:
                env=fixture(Path(tmp));data.freeze_gate(env['union_gate'],**env['gate_args']);gate=env['union_gate']/'GATE_SOURCE_FREEZE.json'
                with patch.object(data,'_raw_rows',return_value=iter(candidates())),patch.object(data,'_load_tokenizer',return_value=Tokenizer()):data.prepare_confirmation(gate)
                root=env['union_gate']/'data';m=json.loads((root/'DATASET_MANIFEST.json').read_text())
                if fault=='hole':m['skipped'].pop()
                if fault=='short':m['skipped'][1]['original_tokens']=8192
                if fault=='unknown':m['skipped'][1]['reason']='early_EOS'
                write(root/'DATASET_MANIFEST.json',m)
                with self.assertRaises(RuntimeError):data.verify_dataset_manifest(root)

    def test_short_cohort_failure_is_preserved_and_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=fixture(Path(tmp));data.freeze_gate(env['union_gate'],**env['gate_args']);gate=env['union_gate']/'GATE_SOURCE_FREEZE.json'
            with patch.object(data,'_raw_rows',return_value=iter(candidates(15))),patch.object(data,'_load_tokenizer',return_value=Tokenizer()):
                with self.assertRaisesRegex(RuntimeError,'Fewer than16'):data.prepare_confirmation(gate)
            self.assertTrue((env['union_gate']/'data/DATA_PREP_FAILURE.json').is_file())
            with self.assertRaisesRegex(RuntimeError,'Fresh confirmation'):data.prepare_confirmation(gate)

    def test_gate_source_change_blocks_candidate_before_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=fixture(Path(tmp));data.freeze_gate(env['union_gate'],**env['gate_args']);gate=env['union_gate']/'GATE_SOURCE_FREEZE.json'
            (env['union_source']/data.REQUIRED_NEW_SOURCE[0]).write_text('# CHANGED\n')
            with patch.object(data,'_raw_rows',side_effect=AssertionError('raw accessed')):
                with self.assertRaisesRegex(RuntimeError,'source/tokenizer changed'):data.prepare_confirmation(gate)
            self.assertFalse((env['union_gate']/'data').exists())

    def test_alias_and_formal_multiplicity_changes_block_before_candidate_access(self):
        for fault in ('alias','multiplicity','extra_count'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as tmp:
                env=fixture(Path(tmp));data.freeze_gate(env['union_gate'],**env['gate_args']);gate=env['union_gate']/'GATE_SOURCE_FREEZE.json'
                if fault=='alias':
                    value=json.loads(gate.read_text());value['horizon_arguments']['baseline_results']=str(Path(tmp)/'wrong');write(gate,value)
                else:
                    value=json.loads(env['formal'].read_text())
                    if fault=='multiplicity':value['statistics']['simultaneous_lower_quantile']=.00625
                    else:value['expected_extra_eager_references']=0
                    write(env['formal'],value)
                with patch.object(data,'_raw_rows',side_effect=AssertionError('candidate read')):
                    with self.assertRaises(RuntimeError):data.prepare_confirmation(gate)
                self.assertFalse((env['union_gate']/'data').exists())

    def test_missing_old_cohort_or_wrong_target_tokenizer_never_opens_candidate(self):
        for fault in ('missing_cohort','wrong_tokenizer'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as tmp:
                env=fixture(Path(tmp));args=dict(env['gate_args'])
                if fault=='missing_cohort':args['metadata_paths']=args['metadata_paths'][:-1]
                else:(args['tokenizer_root']/'tokenizer.json').write_text('{"changed":"synthetic"}\n')
                original=data.sha256
                def digest(path):
                    self.assertNotEqual(Path(path).resolve(),env['raw'].resolve())
                    return original(path)
                with patch.object(data,'sha256',side_effect=digest):
                    with self.assertRaises(RuntimeError):data.freeze_gate(env['union_gate'],**args)
                self.assertTrue((env['union_gate']/'GATE_FAILURE.json').is_file())

    def test_within_selected_document_and_prefix_duplicates_are_excluded(self):
        for text,reason in (('eligible0','document_duplicate'),('sameprefix','prefix_duplicate')):
            with self.subTest(reason=reason),tempfile.TemporaryDirectory() as tmp:
                env=fixture(Path(tmp));data.freeze_gate(env['union_gate'],**env['gate_args']);gate=env['union_gate']/'GATE_SOURCE_FREEZE.json'
                rows=[(0,'eligible0'),(1,text)]+[(2+i,'eligible'+str(i+1)) for i in range(15)]
                with patch.object(data,'_raw_rows',return_value=iter(rows)),patch.object(data,'_load_tokenizer',return_value=Tokenizer()):
                    manifest=data.prepare_confirmation(gate)
                if reason=='document_duplicate':
                    self.assertEqual(manifest['skipped'],[dict(source_index=1,
                        document_sha256=hashlib.sha256(text.encode()).hexdigest(),reason=reason)])
                else:
                    self.assertEqual(manifest['skipped'][0]['original_tokens'],8192)
                    self.assertEqual(set(manifest['skipped'][0]['input_sha256']),{'4096','8192'})
                self.assertEqual(manifest['skipped'][0]['reason'],reason)
                self.assertEqual([v['source_index'] for v in manifest['selected']],[0]+list(range(2,17)))


if __name__=='__main__':unittest.main()
