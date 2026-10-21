"""Synthetic CPU contract tests; no scientific models, fresh data or GPUs."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from rescue import baseline_horizon_report as report


def module(name):
    path=Path(__file__).parent/('test_rescue_'+name+'.py')
    spec=importlib.util.spec_from_file_location('fixture_'+name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def rows(path,values):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(v)+'\n' for v in values))


def fixture(root):
    from rescue import baseline_horizon as runner
    from rescue import baseline_horizon_data as data
    base=root/'stage1_evidence';bres=base/'results';old=module('baseline_selection_report').full_grid_fixture(bres)
    dpaths=module('baseline_horizon_data').fixture(root/'data')
    hashes={n:data.sha256(p) for n,p in dpaths.items()}
    documents,prime,manifest=data.load_horizon_documents(dpaths,expected_source_sha256=hashes,tokenizer_sha256={'tokenizer.json':'a'*64})
    dump(root/'data/DATASET_MANIFEST.json',manifest);rows(root/'data/documents.jsonl',documents.values());dump(root/'data/prime.json',prime)
    p=old['protocol'];p.update(input_sha256={f'271:{n}':manifest['input_sha256'][f'271:{n}'] for n in report.LENGTHS})
    p['environment'].update(nvidia_smi_gpu='0, GPU-synthetic, NVIDIA H100, 81920',torch='synthetic',cuda='synthetic')
    for i in range(27):
        name=f'exp/specextend_bridgekv/specextend/fixture_{i}.py';file=bres/'source'/name
        file.parent.mkdir(parents=True,exist_ok=True);file.write_text('# Synthetic program fixture.\n');p['source_sha256'][name]=report.sha(file)
    dump(bres/'protocol.json',p);frozen=json.loads((bres/'FROZEN_SELECTION.json').read_text());frozen['protocol_sha256']=report.sha(bres/'protocol.json');frozen['source_sha256']=p['source_sha256'];dump(bres/'FROZEN_SELECTION.json',frozen)
    complete=json.loads((bres/'COMPLETE.json').read_text());complete['frozen_selection_sha256']=report.sha(bres/'FROZEN_SELECTION.json');dump(bres/'COMPLETE.json',complete)
    ban=bres/'analysis';dump(ban/'summary.json',dict(status='COMPLETE',problems=[],winner_status='VERIFIED_FROZEN',observed_counts=complete['counts']))
    dump(ban/'VERIFIED_WINNERS.json',dict(status='VERIFIED_FROZEN',authoritative=True,winners=dict(selected=old['expected_winners'])))
    for name in ('quality','tps','captures','capture_time','cold','service','selected_repeat1'):
        for ext in ('.png','.svg'):(ban/(name+ext)).write_text('Synthetic program-test artifact, not scientific rendered output.\n')
    dump(ban/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',artifacts_sha256={p.name:report.sha(p) for p in ban.iterdir()}))
    dump(base/'PIPELINE_EXIT.json',dict(exit_code=0))
    dump(base/'FINAL_ARTIFACTS_MANIFEST.json',dict(file_sha256={str(p.relative_to(base)):report.sha(p) for p in base.rglob('*') if p.is_file()}))
    verified=dict(status='VERIFIED_COMPLETE',**{k:True for k in ('passed','raw_independent_audit_passed','rendered_figures_inspected','strict_analysis_complete','all538_requests_verified','all76_owners_verified','eight_frozen_choices_verified')},
        baseline_complete_sha256=report.sha(bres/'COMPLETE.json'),frozen_selection_sha256=report.sha(bres/'FROZEN_SELECTION.json'),analysis_summary_sha256=report.sha(ban/'summary.json'),verified_winners_sha256=report.sha(ban/'VERIFIED_WINNERS.json'),final_artifacts_manifest_sha256=report.sha(base/'FINAL_ARTIFACTS_MANIFEST.json'))
    dump(base/'VERIFIED_COMPLETE.json',verified)
    proof=dict(status='BASELINE_VERIFIED',choices={str(n):{g:frozen['per_context'][str(n)]['groups'][g]['arm'] for g in report.GROUPS} for n in report.LENGTHS},
        selection_freeze_sha256=verified['frozen_selection_sha256'],verified_winners_sha256=verified['verified_winners_sha256'],baseline_complete_sha256=verified['baseline_complete_sha256'],analysis_summary_sha256=verified['analysis_summary_sha256'],verified_complete_sha256=report.sha(base/'VERIFIED_COMPLETE.json'),final_manifest_sha256=verified['final_artifacts_manifest_sha256'],baseline_protocol_sha256=report.sha(bres/'protocol.json'),pipeline_exit_sha256=report.sha(base/'PIPELINE_EXIT.json'))
    pins={str(Path('/synthetic/stage1')/p.relative_to(base)):dict(saved_path=str(p.relative_to(root)),sha256=report.sha(p)) for p in base.rglob('*') if p.is_file()}
    dump(base/'PIN_INDEX.json',dict(original_to_saved=pins))
    for name in p['source_sha256']:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((bres/'source'/name).read_bytes())
    sequence,owners,warm_order=runner.schedule(list(documents))
    protocol=dict(p,seed=report.SEED,source_indices=list(documents),expected_counts=report.EXPECTED,expected_requests=2540,
        stage1_evidence=dict(root='stage1_evidence',baseline_results='results',analysis_dir='results/analysis',verified_complete='VERIFIED_COMPLETE.json',final_manifest='FINAL_ARTIFACTS_MANIFEST.json',pin_index='stage1_evidence/PIN_INDEX.json',pin_index_sha256=report.sha(base/'PIN_INDEX.json')),
        stage1_freeze=proof,shared_warm_sequence=sequence,owner_order=owners,warm_order=warm_order,tokenizer_eos_token_id=100000,vocab_size=151936,
        data_manifest='data/DATASET_MANIFEST.json',data_manifest_sha256=report.sha(root/'data/DATASET_MANIFEST.json'),data_source_paths={n:'data/'+n for n in hashes},selected_documents='data/documents.jsonl',prime_document='data/prime.json',selected_documents_sha256=report.sha(root/'data/documents.jsonl'),prime_document_sha256=report.sha(root/'data/prime.json'),data_receipt_sha256=manifest['receipt_sha256'],tokenizer_sha256=manifest['tokenizer_sha256'],input_sha256=manifest['input_sha256'])
    dump(root/'protocol.json',protocol)
    dump(root/'GATE_PASSED.json',dict(passed=True,stage1_freeze=proof,source_sha256=protocol['source_sha256'],data_manifest_sha256=protocol['data_manifest_sha256'],old_metadata_sources=hashes,expected_counts=report.EXPECTED,expected_requests=2540,no_new_confirmation_opened=True))
    reference=list(range(128));canonical=[];primes=[];warm=[];owner_rows=[];tree_raw=[];telemetry=[];index=0
    for n in report.LENGTHS:
        for d in report.DOCS:
            canonical.append(dict(source_index=d,length=n,mode='canonical_ar',generated=reference,request_id=f'horizon:{index}',execution_index=index));index+=1
    templates={(r['mode'],r['length']):r for r in old['warm']}
    for oi,entry in enumerate(owners):
        mode,n=entry['mode'],entry['length'];arm=next(a for a in report.support.menu() if a['mode']==mode);owner_id=f'owner:{oi}';signature={'identity':oi+10}
        requests=[dict(mode=mode,length=n,source_index=271,phase='cold_prime')]+[dict(mode=mode,length=n,phase='warm',**v) for v in sequence]
        owned=[];capture_total=0;capture_time=0.
        def record_telemetry(phase):
            fields=['uuid','temperature.gpu','clocks.current.sm','clocks.current.memory','power.draw','power.limit','utilization.gpu','memory.used','memory.total']
            vals=dict(zip(fields,['GPU-synthetic','60','1200','1593','200','350','90','40000','81920']))
            telemetry.append(dict(owner_id=owner_id,mode=mode,length=n,phase=phase,execution_index=index,telemetry=dict(query_fields=fields,values=vals,raw_csv='synthetic',wall_time_unix=1000000000.+index,read_only=True,primary_clock_included=False)))
        record_telemetry('owner_start')
        for count,request in enumerate(requests,1):
            cold=count==1;row=copy.deepcopy(templates[mode,n]);backend=arm['backend'];isgraph=backend=='cuda_graph';resident=backend=='resident'
            delta=dict(captures=2 if cold and isgraph else int(isgraph and count==2),replays=128 if isgraph else 0,resident_steps=128 if resident else 0,capture_host_seconds=.2 if cold and isgraph else .1 if isgraph and count==2 else 0.)
            capture_total+=delta['captures'];capture_time+=delta['capture_host_seconds']
            stats=dict(mode=backend,entries=capture_total if isgraph else 0,captures=capture_total,replays=128*count if isgraph else 0,resident_steps=128*count if resident else 0,capture_host_seconds=capture_time,max_graph_entries=128 if isgraph or resident else None)
            factor=.4 if mode in ('retrieval16_k4_eager','retrieval32_k8_resident','retrieval32_k4_graph','tree16_n32') else 1.+oi/100
            seconds=(2.+(.01*report.DOCS.index(request['source_index'])))*factor+delta['capture_host_seconds']
            row.update(request,arm=arm,owner_id=owner_id,owner_signature=signature,runtime_stats=stats,request_id=f'horizon:{index}',execution_index=index,
                outer_seconds=seconds,outer_tps=128/seconds,total_seconds=seconds-.005,prefill_seconds=.1,decode_seconds=seconds-.11,cleanup_seconds=.005)
            if backend in ('resident','cuda_graph'):row['draft_runtime']=dict(mode=backend,request=delta,lifetime=copy.deepcopy(stats),max_graph_entries=128,retain_query=False)
            elif arm['family']=='retrieval':row.pop('draft_runtime',None)
            if cold:row.update(constructor_and_detach_seconds=.02,cold_complete_outer_seconds=.02+seconds)
            else:row['warm_index']=count-2
            (primes if cold else warm).append(row);owned.append(row)
            if arm['family']=='full_tree':tree_raw.append(dict(request_id=row['request_id'],execution_index=index,result=dict(generated=reference,tokens=128,upstream=row['upstream'])))
            index+=1
        total=.02+sum(r['outer_seconds'] for r in owned)+.001;final=owned[-1]['runtime_stats']
        owner_rows.append(dict(mode=mode,length=n,owner_id=owner_id,maximum_live_owners=1,owner_signature=signature,constructor_and_detach_seconds=.02,close_seconds=.001,prime_request_id=owned[0]['request_id'],prime_tokens=128,cold_prime_outer_seconds=owned[0]['outer_seconds'],warm_tokens=4096,warm_request_ids=[r['request_id'] for r in owned[1:]],warm_outer_seconds_sum=sum(r['outer_seconds'] for r in owned[1:]),accounted_service_seconds=total,accounted_service_tokens=4224,accounted_service_tps=4224/total,final_runtime_stats=copy.deepcopy(final),closed_runtime_stats=dict(final,entries=0),engine_closed=True,hook_restored=True))
        record_telemetry('owner_end')
    assert index==2540
    for name,values in (('reference',canonical),('prime',primes),('runtime',warm),('owners',owner_rows),('tree_raw',tree_raw),('gpu_diagnostics',telemetry)):rows(root/(name+'.jsonl'),values)
    dump(root/'references.json',{f'{d}:{n}':reference for n in report.LENGTHS for d in report.DOCS})
    # Crucially use the actual independent runner, not the analyzer's own selector.
    selected=runner.select_winners(warm,frozen)
    dump(root/'FROZEN_SELECTION.json',dict(selected,complete_request_count=2540,no_new_confirmation_opened=True,protocol_sha256=report.sha(root/'protocol.json'),stage1_selection_sha256=proof['selection_freeze_sha256'],source_sha256=protocol['source_sha256'],assets_sha256=protocol['assets_sha256']))
    dump(root/'COMPLETE.json',dict(passed=True,counts=report.EXPECTED,expected_counts=report.EXPECTED,logical_requests=2540,owner_closed=76,goal_complete=False,sources_unchanged=True,assets_unchanged=True,dependencies_unchanged=True,all_canonical=True,backend_trajectories_equal=True,data_unchanged=True,stage1_verified_unchanged=True,analysis_required_locally=True,data_manifest_sha256=protocol['data_manifest_sha256'],stage1_selection_sha256=proof['selection_freeze_sha256'],frozen_selection_sha256=report.sha(root/'FROZEN_SELECTION.json')))
    dump(root.parent/'PIPELINE_EXIT.json',dict(exit_code=0));dump(root.parent/'EVIDENCE_MANIFEST.json',dict(file_sha256={str(p.relative_to(root.parent)):report.sha(p) for p in root.rglob('*') if p.is_file()}))
    return dict(hashes=hashes,warm=warm,owners=owner_rows,stage1=frozen,selected=selected,protocol=protocol)


class HorizonReportTests(unittest.TestCase):
    def test_saved_evidence_paths_cannot_open_outside_the_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader=report.Evidence(tmp)
            with patch.object(Path,'read_text',side_effect=AssertionError('must not read an outside path')):
                self.assertEqual(reader.json('../outside.json'),{})
                self.assertEqual(reader.rows('/outside.jsonl'),[])
            self.assertEqual(len(reader.problems),2)

    def test_whole_document_ratio_shared_resamples_and_prefix_counts(self):
        cells=[dict(tokens=256,seconds=1.+i) for i in range(16)];other=[dict(tokens=256,seconds=2.+2*i) for i in range(16)]
        value=report.paired_interval(cells,other);self.assertEqual(value['point'],100.)
        np.testing.assert_allclose(value['ci95'],[100.,100.],atol=1e-12)
        self.assertNotEqual(sum(c['tokens'] for c in cells)/sum(c['seconds'] for c in cells),np.mean([c['tokens']/c['seconds'] for c in cells]))
        variable=[dict(tokens=256,seconds=2.+i**.7) for i in range(16)]
        independent=np.random.default_rng(20261025).integers(0,16,size=(10000,16))
        actual=report.paired_interval(cells,variable)
        a=np.array([c['seconds'] for c in cells]);b=np.array([c['seconds'] for c in variable])
        sampled=100.*(b[independent].sum(1)/a[independent].sum(1)-1.)
        np.testing.assert_allclose(actual['ci95'],np.percentile(sampled,[2.5,97.5]),rtol=0,atol=1e-12)
        with self.assertRaises(ValueError):report.paired_interval(cells,draws=100000)

    def test_full2540_actual_runner_selection_complete_and_union_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'results';state=fixture(root)
            with patch.dict(report.DATA_SHA,state['hashes'],clear=True):value=report.analyze(root)
            self.assertEqual(value['status'],'COMPLETE',value.get('problems'));self.assertTrue(value['authoritative_union']);self.assertFalse(value['goal_complete'])
            self.assertEqual(value['stage2'],state['selected']['per_context']);self.assertEqual(value['observed_requests'],2540)
            for n in report.LENGTHS:
                entry=value['stage2'][str(n)];self.assertEqual(entry['stage1_groups'],state['stage1']['per_context'][str(n)]['groups'])
                self.assertLessEqual(len(entry['union_modes']),8)
                for v in value['metrics'][str(n)].values():self.assertEqual((v['documents'],v['prefix6']['documents'],v['prefix32']['documents']),(16,6,16))

    def test_missing_warm_tampered_owner_data_and_selection_never_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'results';state=fixture(root)
            for kind in ('missing','owner','data','selection'):
                originals={name:(root/name).read_bytes() for name in ('runtime.jsonl','owners.jsonl','data/prime.json','FROZEN_SELECTION.json')}
                if kind=='missing':rows(root/'runtime.jsonl',state['warm'][:-1])
                if kind=='owner':
                    owners=copy.deepcopy(state['owners']);owners[0]['accounted_service_seconds']-=1;rows(root/'owners.jsonl',owners)
                if kind=='data':
                    data=json.loads((root/'data/prime.json').read_text());data['document_ids'][0]+=1;dump(root/'data/prime.json',data)
                if kind=='selection':
                    data=json.loads((root/'FROZEN_SELECTION.json').read_text());data['per_context']['4096']['stage1_groups']['full_tree']['mode']='tree16_n8';dump(root/'FROZEN_SELECTION.json',data)
                with patch.dict(report.DATA_SHA,state['hashes'],clear=True):value=report.analyze(root)
                self.assertEqual(value['status'],'PARTIAL',kind);self.assertFalse(value['authoritative_union'],kind)
                for name,content in originals.items():(root/name).write_bytes(content)

    def test_render_partial_separate_quality_tps_and_saved_standalone_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary=dict(status='PARTIAL',authoritative_union=False,goal_complete=False,metrics={},observed_counts={},gpu_telemetry=[])
            target=Path(tmp)/'analysis';receipt=report.write_report(summary,Path(tmp),target)
            self.assertIn('quality',receipt['figures']);self.assertIn('tps',receipt['figures']);self.assertIn('gpu_telemetry',receipt['figures'])
            self.assertEqual(report.sha(report.__file__),report.sha(target/'analysis_source.py'))
            self.assertTrue((target/'analysis_support.py').is_file());self.assertFalse((target/'VERIFIED_UNION.json').exists())
            for name in receipt['figures']:self.assertGreater((target/(name+'.png')).stat().st_size,10000)
            result=subprocess.run([sys.executable,'-c',"import runpy,sys;runpy.run_path('analysis_source.py');assert 'torch' not in sys.modules"],cwd=target,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
