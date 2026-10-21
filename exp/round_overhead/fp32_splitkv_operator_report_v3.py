"""Portable saved-data v3 candidate-analytic smoke report, never a system benchmark.

Only stdlib plus Matplotlib/NumPy for figures. No Torch/Triton/model/runtime
imports, inferred tolerance, Token claim or performance extrapolation.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil

LENGTHS=(2048,4096,6144,8192,12288,16384)
MASKS=('bool_broadcast','additive_broadcast','bool_full','additive_full')
FAMILIES=('zero_scores','constant_v','all_masked','one_key')
SOURCES={
 'capsule/exp/round_overhead/fp32_splitkv_candidate_v1.py':'9f133cc824915afd09d9ad25a1b16e6cd721522e1f583cabd5e102235358cfd2',
 'capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v1.py':'62957e77c8d26b61dd89fc7692189ff0f46d61f3d77b9f6131b996eae26de98d',
 'capsule/tests/test_fp32_splitkv_candidate_v1.py':'d05aab7170f75451f1af4e93aac9a895c25e9b3f15ae9c59d11e2b6117bed8c0',
 'capsule/tests/test_fp32_splitkv_cuda_gate_v1.py':'ae1f6665852f99f947c95ade0c7f766186fb80615b1260911a5f8b27aada78ef',
 'capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v2.py':'1d4b340786851eb41c80489fb6c3b775101ba8125e97d50e81903d4536a52549',
 'capsule/tests/test_fp32_splitkv_cuda_gate_v2.py':'525ad7ade08a3fff6c8b593c88d8d32bc4d7f3399de35025f48146b4f1ff02b5',
 'capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v3.py':'77501b123927d4ce0606da9017b14d8adaa84ce90059285b70735d8de1f3d39e',
 'capsule/tests/test_fp32_splitkv_cuda_gate_v3.py':'fd94317f0c1514a459f531c615ea3a39124d12c4e72f2700090c7549c4e336cb',
 'launch.sh':'4ef32e8c396971fb2c138f167f42d69c64f24430fa17d1c71b69d7527a372a2c'}
RAW_PINS={
 'results/PLAN.json':'8e1b78fc295547d0e93d16f0e4f76dfe08ff0d7d52df625976e62dc7af477bf5',
 'results/CASES.jsonl':'060a76ccab52d661debebb1b5eadf5ee2fa7b1ca0904a97672b6643f7f0683d2',
 'results/TERMINAL.json':'184674b6a991601378fb3e10ee4b7b4125477efad1400fd21599a5ac98e8c7b6',
 'PIPELINE_EXIT.json':'f30ae9a4467d8cfc7da59ff1f397b4dd94a4827a1ab1e082b66ff4eeeb24cdbf',
 'PREREGISTRATION.json':'49135c6f083d7809ca166e4443b56a9174e3b3bad9c6c8fe3608008d0165aa79',
 'SOURCE_SHA256SUMS':'1ec04182e28d1bbee0a22da5f2298d421cd984ca48fa259cfb51fa79f8d2f644',
 'COLLECTION_RECEIPT.json':'9460d9a78baf8da858abea150aeafef4e67bada21bbe6a1fa0de294ee2cb8185'}

FIGURES=('random_max_abs','known_output_matrix','jit_entries')
GPU='GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9'
GATES=('quality_gate','performance_gate','model_gate','token_gate')
DYNAMIC=('N','Q0','Q1','Q2','K0','K1','K2','V0','V1','V2','M0','M1','M2','M3')


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def issha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def nonnegative(value):return type(value) in (int,float) and math.isfinite(value) and value>=0
def read(path):return json.loads(Path(path).read_bytes())
def safe(root,name):
    p=PurePosixPath(name)
    require(p.parts and not p.is_absolute() and p.as_posix()==name and '..' not in p.parts and '\\' not in name,'Unsafe evidence path')
    target=Path(root).joinpath(*p.parts)
    require(not any(p.is_symlink() for p in (target,*target.parents)),'Symlink evidence forbidden')
    return target


def expected_cases():
    """Frozen112 scalar descriptors reproduced without importing the gate."""
    result=[]
    for q in range(2,6):
        for n in LENGTHS:
            for mask in MASKS:result.append(dict(kind='random',q=q,n=n,mask=mask,exact=False))
    for q in range(2,6):
        result.extend((dict(kind='zero_scores',q=q,n=16512 if q==2 else 2048,mask='none',exact=True),
            dict(kind='constant_v',q=q,n=2048,mask='none',exact=True),
            dict(kind='all_masked',q=q,n=q,mask='bool_broadcast' if q%2==0 else 'additive_broadcast',exact=True),
            dict(kind='one_key',q=q,n=129 if q<5 else 1025,allowed_key={2:0,3:127,4:128,5:1024}[q],
                mask='bool_broadcast' if q%2==0 else 'additive_broadcast',exact=True)))
    for i,c in enumerate(result):c.update(case_id=f'operator_{i:03d}',seed=20261021+i,query_layout='noncontiguous_folded',kv_head_stride=0)
    return result


def validate_snapshots(row):
    before,after=row.get('inputs_before'),row.get('inputs_after')
    require(isinstance(before,list) and len(before)==4 and before==after,'Input backing, pointer, version or view changed')
    q,n=row['q'],row['n'];full=row['mask'].endswith('full')
    shapes=([4,7,q,128],[4,7,n,128],[4,7,n,128],None if row['mask']=='none' else ([4,7,q,n] if full else [1,1,q,n]))
    for i,(tensor,shape) in enumerate(zip(before,shapes)):
        if shape is None:require(tensor is None,'Unexpected mask allocation');continue
        dtype='torch.bool' if i==3 and row['mask'].startswith('bool') else 'torch.float32'
        require(isinstance(tensor,dict) and tensor.get('shape')==shape and tensor.get('dtype')==dtype and tensor.get('device')=='cuda:0' and
            type(tensor.get('pointer')) is int and tensor['pointer']>0 and type(tensor.get('version')) is int and tensor['version']>=0 and
            tensor.get('storage_offset')==0 and tensor.get('is_conj') is False and tensor.get('is_neg') is False and
            type(tensor.get('backing_valid_bytes')) is int and tensor['backing_valid_bytes']>0 and issha(tensor.get('backing_sha256')),'Invalid recorded input state')
        stride=tensor.get('stride');require(isinstance(stride,list) and len(stride)==4 and all(type(v) is int and v>=0 for v in stride),'Invalid view stride')
        if i==0:require(stride==[7*128,128,28*128,1],'Folded noncontiguous query layout differs')
        if i in (1,2):require(stride==[n*128,0,128,1],'Expanded GQA zero-head-stride view differs')
    if row['kind']=='one_key':
        for tensor in before[:2]:
            require(tensor['backing_valid_bytes']<=64*1024*1024 and tensor['backing_sha256']==hashlib.sha256(bytes(tensor['backing_valid_bytes'])).hexdigest(),
                'V2 one-key recorded Q/K backing is not all-zero bytes')
    runtime=row.get('runtime_specials',{})
    require(runtime.get('N')==n and runtime.get('non_specialized')==list(DYNAMIC),'Dynamic N/stride contract differs')
    for key,i in (('q_strides',0),('k_strides',1),('v_strides',2),('mask_strides',3)):
        require(runtime.get(key)==(None if before[i] is None else before[i]['stride']),'Runtime stride ledger differs')


def validate_rows(rows,plan,terminal):
    require(plan.get('cases')==expected_cases(),'Frozen112 plan descriptor matrix changed')
    require(len(rows)==112,'Saved v3 operator run must contain exactly112 completed cases')
    compile_keys={name:set() for name in ('_stage_one','_stage_two')};jit=[];random=[];known=[]
    for i,row in enumerate(rows):
        descriptor=plan['cases'][i]
        require(all(type(row.get(k)) is type(v) and row.get(k)==v for k,v in descriptor.items()),'Case order/seed/shape is not exact planned prefix')
        require(row.get('finite') is True and row.get('output_contract') is True,'Nonfinite/output-contract failure differs')
        require(all(row.get(k)=='NOT_RUN' for k in GATES) and row.get('random_equivalence_gate')=='NOT_DEFINED','Model/Token/performance or random numerical-pass promotion')
        require(row.get('reference_analytic_gate')=='DIAGNOSTIC_ONLY' and row.get('reference_equivalence_gate')=='NOT_ESTABLISHED_DIAGNOSTIC_ONLY','Reference diagnostic-only contract changed')
        validate_snapshots(row)
        error=row.get('errors_diagnostic_only',{})
        require(all(nonnegative(error.get(k)) for k in ('max_abs','rms','max_relative_floor1e6')) and type(error.get('candidate_reference_bitwise')) is bool,'Missing finite diagnostic error metrics')
        hashes=row.get('output_logical_bytes_sha256',{})
        require(set(hashes)=={'candidate','reference'} and all(issha(v) for v in hashes.values()) and
            (hashes['candidate']==hashes['reference']) is error['candidate_reference_bitwise'],'Output hash/bitwise ledger differs')
        if row['kind']=='random':
            require(row.get('candidate_known_bitwise') is None and row.get('reference_known_bitwise') is None and
                row.get('status')=='DIAGNOSTIC_COMPLETE' and row.get('mandatory_checks_passed') is True,'Random diagnostics treated as known-output/numerical pass')
            random.append({**descriptor,**error})
        else:
            candidate,reference=row.get('candidate_known_bitwise'),row.get('reference_known_bitwise')
            require(type(candidate) is bool and type(reference) is bool,'Known-output boolean observation absent')
            require(candidate is True and row.get('mandatory_checks_passed') is True and row.get('status')=='CANDIDATE_ANALYTIC_CASE_PASSED','Mandatory candidate analytic case did not pass')
            known.append({**descriptor,'candidate_known_bitwise':candidate,'reference_known_bitwise':reference,**error})
        cache=row.get('compiled_cache',[])
        require(len(cache)==2 and {c.get('kernel') for c in cache}==set(compile_keys),'Two actual JIT stage records missing')
        one=dict(case_id=row['case_id'],case_index=i)
        for item in cache:
            name=item['kernel'];keys=item.get('key_sha256s',[])
            require(item.get('device')==0 and type(item.get('compiled_entries')) is int and item['compiled_entries']==len(keys)>0 and
                keys==sorted(set(keys)) and all(issha(key) for key in keys) and compile_keys[name]<=set(keys),'Invalid/decreasing JIT cache inventory')
            compile_keys[name]=set(keys);one[name]=len(keys)
        jit.append(one)
    require(len(random)==96 and len(known)==16 and all(k['candidate_known_bitwise'] for k in known) and
        [k['case_id'] for k in known if not k['reference_known_bitwise']]==['operator_099','operator_103','operator_107','operator_111'],
        'Observed96random/16candidate-known/reference diagnostic scope differs')
    require([k['max_abs'] for k in known if not k['reference_known_bitwise']]==
        [4.76837158203125e-7,9.5367431640625e-7,4.76837158203125e-7,4.76837158203125e-7] and
        all(k['candidate_reference_bitwise'] is k['reference_known_bitwise'] and
            (k['max_abs']==0 if k['reference_known_bitwise'] else k['max_abs']>0) for k in known),
        'Known candidate/reference diagnostic error ledger differs')
    for key in ('completed_cases','attempted_cases','reference_returned_cases','candidate_returned_cases','synchronized_cases'):
        require(type(terminal.get(key)) is int and terminal[key]==len(rows),'Terminal112-case counter mismatch: '+key)
    require(terminal.get('current_case') is None and terminal.get('current_stage')=='complete' and 'error' not in terminal,
        'Different terminal completion stage')
    require(terminal.get('compiled_cache')==rows[-1]['compiled_cache'],'Terminal actual JIT inventory not bound to last case')

    return random,known,jit


def validate_collection(root):
    """Check locally transcribed receipt; not signed remote attestation."""
    receipt=read(safe(root,'COLLECTION_RECEIPT.json'))
    names=set(SOURCES)|{'results/PLAN.json','results/CASES.jsonl','results/TERMINAL.json','PIPELINE_EXIT.json',
        'PREREGISTRATION.json','SOURCE_SHA256SUMS','LAUNCH_STARTED.json','operator.log'}
    require(receipt.get('status')=='TERMINAL_OPERATOR_SMOKE_COLLECTED_READ_ONLY' and receipt.get('scp_exit_codes')==[0,0,0] and
        receipt.get('pipeline_exit_code')==0 and receipt.get('operator_cases_recorded')==receipt.get('operator_cases_planned')==112 and
        receipt.get('passed_scope')=='CANDIDATE_ANALYTIC_ONLY' and receipt.get('model_calls')==0 and
        receipt.get('token_gate')==receipt.get('performance_gate')=='NOT_RUN' and receipt.get('goal_complete') is False and
        receipt.get('prior_v1_v2_failures_preserved') is True and receipt.get('automatic_retry') is False,'Collection scope differs')
    hashes=receipt.get('collected_file_sha256',{})
    require(len(names)==17 and set(hashes)==names and all(issha(h) and sha(safe(root,n))==h for n,h in hashes.items()),
        'Collection seventeen-file local-byte ledger mismatch')
    expected={n:sha(safe(root,n)) for n in ('PIPELINE_EXIT.json','results/TERMINAL.json','results/CASES.jsonl')}
    for key in ('remote_read_only_barrier_before','remote_read_only_barrier_after'):
        barrier=receipt.get(key,{})
        require(barrier.get('command_exit')==0 and barrier.get('file_sha256')==expected,'Read-only collection barrier differs')
    process=receipt.get('process_observations_before_collection',{})
    require(process.get('tmux_has_session_exit')==process.get('ps_pid_and_direct_children_exit')==1 and
        process.get('nvidia_compute_apps_exit')==0 and process.get('nvidia_compute_apps_stdout')=='','Transcribed process absence differs')
    return dict(verified_local_file_count=17,double_barrier_matches_local_bytes=True,three_scp_exit0=True,
        observation_provenance='MAIN_AGENT_TOOL_OBSERVATION_TRANSCRIPTION_NOT_SIGNED_ATTESTATION',
        compiled_cache_inventory_transferred=False)


def analyze(root):
    root=Path(root).absolute();issues=[]
    result=dict(status='PARTIAL_EVIDENCE_ERROR',contract_version=3,fixture_version=2,observed_cases=0,planned_cases=112,random=[],known=[],jit=[],
        known_planned=expected_cases()[96:],not_run_case_ids=[c['case_id'] for c in expected_cases()],
        numerical_pass=False,model='NOT_MEASURED',tokens='NOT_MEASURED',acceptance='NOT_MEASURED',tps='NOT_MEASURED',latency='NOT_MEASURED',
        confidence_intervals=None,goal_complete=False,goal_status='ACTIVE',system_experiment_complete=False,
        candidate_analytic_smoke_complete=False,reference_equivalence='NOT_ESTABLISHED',evidence_issues=issues)
    try:
        pins={**SOURCES,**RAW_PINS}
        for name,pin in pins.items():require(sha(safe(root,name))==pin,'Saved raw/source hash mismatch: '+name)
        p=read(safe(root,'PREREGISTRATION.json'));plan=read(safe(root,'results/PLAN.json'));terminal=read(safe(root,'results/TERMINAL.json'));exitrow=read(safe(root,'PIPELINE_EXIT.json'))
        result['synthetic_program_test_only']=p.get('synthetic_program_test_only') is True
        require(p.get('fixture_version')==plan.get('fixture_version')==exitrow.get('fixture_version')==2 and
            p.get('contract_version')==plan.get('contract_version')==exitrow.get('contract_version')==3,'Explicit fixture2/contract3 missing')
        require(p.get('status')=='FROZEN_BEFORE_OPERATOR_EXECUTION' and p.get('operator_cases')==112 and p.get('exact_known_cases')==16 and
            p.get('random_diagnostic_cases')==96 and p.get('random_equivalence_tolerance') is None and p.get('random_error_metrics_are_diagnostic_only') is True and
            p.get('expected_model_calls')==0 and p.get('reference_analytic_gate')=='DIAGNOSTIC_ONLY' and
            p.get('reference_equivalence_gate')=='NOT_ESTABLISHED' and p.get('passed_scope')=='CANDIDATE_ANALYTIC_ONLY',
            'Preregistered candidate-analytic-only scope differs')
        for key,name in (('candidate_sha256','capsule/exp/round_overhead/fp32_splitkv_candidate_v1.py'),
            ('candidate_cpu_test_sha256','capsule/tests/test_fp32_splitkv_candidate_v1.py'),('gate_sha256','capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v3.py'),
            ('gate_cpu_test_sha256','capsule/tests/test_fp32_splitkv_cuda_gate_v3.py'),('v1_gate_sha256','capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v1.py'),
            ('v1_gate_cpu_test_sha256','capsule/tests/test_fp32_splitkv_cuda_gate_v1.py'),('v2_gate_sha256','capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v2.py'),
            ('v2_gate_cpu_test_sha256','capsule/tests/test_fp32_splitkv_cuda_gate_v2.py'),('launcher_sha256','launch.sh')):
            require(p.get(key)==SOURCES[name],'Preregistered source pin changed: '+key)
        expected_sum={name.removeprefix('capsule/') if name.startswith('capsule/') else '../launch.sh':pin for name,pin in SOURCES.items()}
        sums={line.split('  ',1)[1]:line.split('  ',1)[0] for line in safe(root,'SOURCE_SHA256SUMS').read_text().splitlines()}
        require(sums==expected_sum,'Capsule nine-file SHA inventory differs')
        trees=[ast.parse(safe(root,'capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v'+str(v)+'.py').read_text()) for v in (2,3)]
        for name in ('cases','_inputs'):
            functions=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name) for tree in trees]
            require(ast.dump(functions[0],include_attributes=False)==ast.dump(functions[1],include_attributes=False),
                'V3 changed frozen V2 fixture function: '+name)
        require(plan.get('status')=='PLAN_ONLY' and plan.get('actual_cuda') is False and plan.get('random_tolerance') is None and
            plan.get('random_diagnostic_cases')==96 and plan.get('exact_cases')==16 and plan.get('reference_analytic_gate')=='DIAGNOSTIC_ONLY',
            'Saved PLAN scope changed')
        for obj in (p,plan,terminal):
            require(obj.get('gpu_uuid')==GPU and obj.get('dtype')=='float32' and obj.get('tf32') is False and
                obj.get('torch_version')=='2.7.1+cu128' and obj.get('triton_version')=='3.3.1','GPU/FP32/TF32/runtime pin differs')
        require(plan.get('splits')==p.get('splits')==8 and plan.get('block_keys')==p.get('block_keys')==128,'Split/block geometry differs')
        require(terminal.get('status')=='CANDIDATE_ANALYTIC_SMOKE_PASSED' and terminal.get('passed') is True and
            terminal.get('candidate_analytic_smoke_passed') is True and terminal.get('passed_scope')=='CANDIDATE_ANALYTIC_ONLY' and
            terminal.get('reference_equivalence_gate')=='NOT_ESTABLISHED_DIAGNOSTIC_ONLY' and
            all(terminal.get(k) is False for k in ('numerical_pass','speed_measured','bitwise_equivalence_claim')) and
            terminal.get('actual_cuda') is True and terminal.get('cuda_available') is True and terminal.get('torch_cuda_initialized') is True and
            all(terminal.get(k)=='NOT_RUN' for k in GATES),'Terminal candidate-only completion/no-model scope differs')
        expected_pins=dict(candidate_sha256=SOURCES['capsule/exp/round_overhead/fp32_splitkv_candidate_v1.py'],
            candidate_test_sha256=SOURCES['capsule/tests/test_fp32_splitkv_candidate_v1.py'],
            gate_sha256=SOURCES['capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v3.py'],fixture_version=2,contract_version=3,
            v1_gate_sha256=SOURCES['capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v1.py'],
            v1_gate_test_sha256=SOURCES['capsule/tests/test_fp32_splitkv_cuda_gate_v1.py'],
            v2_gate_sha256=SOURCES['capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v2.py'],
            v2_gate_test_sha256=SOURCES['capsule/tests/test_fp32_splitkv_cuda_gate_v2.py'])
        require(terminal.get('source_pins')==expected_pins,'Terminal v3 source pins differ')
        require(exitrow.get('exit_code')==0 and exitrow.get('terminal_sha256')==sha(safe(root,'results/TERMINAL.json')) and
            exitrow.get('preserve_partial') is True and all(exitrow.get(k) is False for k in ('automatic_retry','performance_measured','goal_complete')) and
            exitrow.get('model_calls')==0,'Pipeline exit0/no-model terminal binding differs')
        require(p.get('prior_failures')==[
            dict(version=1,terminal_sha256='53e400b6539296e3c1d472c3a7f1615fb055764bd162ec6fb1446cbf62dbcf3c',exit_code=1,completed=100,planned=112,preserved=True),
            dict(version=2,terminal_sha256='be66edc3f9f65c44a333bbf1b7d6cc99394732c0f7b44b41bbca4c27dd0dee27',exit_code=1,completed=100,planned=112,preserved=True)],
            'V1/V2 preserved-failure identities differ')
        collection=validate_collection(root)
        data=safe(root,'results/CASES.jsonl').read_bytes();require(data.endswith(b'\n'),'Truncated case JSONL')
        require(terminal.get('cases_sha256')==sha(safe(root,'results/CASES.jsonl')),'Terminal CASES SHA mismatch')
        rows=[json.loads(line) for line in data.splitlines()];random,known,jit=validate_rows(rows,plan,terminal)
        result.update(status='COMPLETE_CANDIDATE_ANALYTIC_SMOKE_ONLY',observed_cases=len(rows),random=random,known=known,jit=jit,not_run_case_ids=[],
            verified_saved_evidence=True,actual_operator_cuda=True,candidate_analytic_smoke_complete=True,candidate_known_true=16,reference_known_true=12,
            reference_known_false=4,random_numerical_pass='NOT_DEFINED',raw_and_source_sha256=pins,
            observed_one_key_qk_zero_backing_verified=True,fixture_v2_cases_and_inputs_ast_unchanged=True,
            prior_v1_v2_failures_preserved=True,collection_verification=collection,
            random_max_abs_by_length={str(n):max(r['max_abs'] for r in random if r['n']==n) for n in LENGTHS})
    except (OSError,ValueError,TypeError,KeyError,IndexError,StopIteration,SyntaxError) as error:
        issues.append(str(error));result['verified_saved_evidence']=False
    return result


def figures(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':13,'xtick.labelsize':12,'ytick.labelsize':12})
    tag=('SYNTHETIC CPU FIXTURE — ' if summary.get('synthetic_program_test_only') else '')+(
        'COMPLETE v3: candidate-analytic smoke only' if summary.get('verified_saved_evidence') else 'PARTIAL: evidence validation failed')
    caption=(f"Observed {summary['observed_cases']}/112 operator pairs; contract 3 / fixture 2. H100; FP32, TF32 off; split 8, tile 128.\n"
        'Candidate analytic equality is mandatory; reference analytic equality is diagnostic only. V1/V2 FAIL remain preserved.\n'
        'No model / Target Token / TPS / latency measurement. Deterministic operator cases, not documents: no document CI.\n'
        'Random errors are diagnostics only; no numerical-equivalence tolerance or pass criterion was defined.')
    def save(fig,name,title):
        fig.suptitle(tag+'\n'+title,fontsize=14);fig.text(.5,.015,caption,ha='center',fontsize=12)
        fig.tight_layout(rect=(0,.21,1,.93));fig.savefig(output/(name+'.png'),dpi=170);fig.savefig(output/(name+'.svg'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,7))
    colors=('#216882','#b66c30','#6b6a9c','#4a785d')
    for q,color in zip(range(2,6),colors):
        rows=[r for r in summary['random'] if r['q']==q]
        x=[LENGTHS.index(r['n'])+((q-2)*4+MASKS.index(r['mask'])-7.5)*.025 for r in rows]
        ax.scatter(x,[r['max_abs'] for r in rows],s=55,color=color,label=f'Q={q} (4 masks / length)',alpha=.8)
    ax.set_xticks(range(6),[f'{n//1024}K\n{n}' for n in LENGTHS]);ax.set_xlabel('Actual standalone-operator key length N (keys; K=1024)')
    ax.set_ylabel('Max absolute candidate − reference output error');ax.set_ylim(bottom=0);ax.set_xlim(-.5,5.5)
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0));ax.grid(axis='y',alpha=.25);ax.legend(ncols=2,fontsize=12)
    save(fig,'random_max_abs',f"{len(summary['random'])} random diagnostic errors; horizontal jitter is display-only")
    fig,ax=plt.subplots(figsize=(13.5,11.5));observed={r['case_id']:r for r in summary['known']};matrix=np.zeros((16,2),dtype=int)
    for i,c in enumerate(summary['known_planned']):
        row=observed.get(c['case_id'])
        for j,key in enumerate(('candidate_known_bitwise','reference_known_bitwise')):
            matrix[i,j]=0 if row is None else (1 if row[key] else 2)
    ax.imshow(matrix,cmap=ListedColormap(('#e7e7e7','#bdd7c5','#d98e66')),vmin=0,vmax=2,aspect='auto')
    ax.set_xticks((0,1),('Candidate == analytic known\nMANDATORY','Original reference == analytic known\nDIAGNOSTIC'));ax.xaxis.tick_top()
    labels=[f"{c['case_id']}  Q={c['q']}  {c['kind']}  N={c['n']}" for c in summary['known_planned']]
    ax.set_yticks(range(16),labels);ax.tick_params(axis='both',length=0,pad=10)
    for i in range(16):
        for j in range(2):ax.text(j,i,('NOT_RUN','TRUE','FALSE')[matrix[i,j]],ha='center',va='center',fontsize=12)
    ax.set_xlabel('Boolean observations, not model/Token gate results',labelpad=16)
    counts=f"candidate {summary.get('candidate_known_true',0)} TRUE, reference {summary.get('reference_known_true',0)} TRUE / {summary.get('reference_known_false',0)} FALSE"
    save(fig,'known_output_matrix',f"Known-output flags: {len(summary['known'])}/16; {counts} — not a model gate")
    fig,ax=plt.subplots(figsize=(13,7));x=[r['case_index'] for r in summary['jit']]
    for stage,color in zip(('_stage_one','_stage_two'),colors):
        ax.step(x,[r[stage] for r in summary['jit']],where='post',color=color,linewidth=2,label=stage.lstrip('_')+' observed cache entries')
    if summary['jit']:
        ax.annotate('112/112 completed; final entries 12 / 1',(111,summary['jit'][-1]['_stage_one']),
            xytext=(-285,15),textcoords='offset points',fontsize=12,arrowprops={'arrowstyle':'->','color':'#555555'})
    ax.set_xticks((0,24,48,72,96,111));ax.set_xlabel(f"Frozen planned case index (000–111); {len(summary['jit'])}/112 observed")
    ax.set_ylabel('Cumulative observed JIT compiled entries (count)');ax.set_xlim(-.5,112.5);ax.set_ylim(bottom=0,top=14)
    ax.set_yticks(range(0,14,2));ax.grid(axis='y',alpha=.25);ax.legend(loc='upper left',fontsize=12)
    save(fig,'jit_entries','Actual two-stage JIT cache inventory, not compile time or latency')


def write_report(summary,root,output):
    root,output=Path(root).absolute(),Path(output).absolute();require(not output.exists(),'Fresh analysis directory required')
    output.mkdir(parents=True);shutil.copyfile(__file__,output/'analysis_entry.py');figures(summary,output)
    (output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
    complete=summary.get('verified_saved_evidence') is True
    lines=['# V3 — candidate-analytic operator smoke only','',
        'Status: '+summary['status']+'. This is an operator gate, not a model/system-performance experiment. '+
        'Model, Target Token equality, acceptance, TPS and latency are NOT_MEASURED. Numerical equivalence is NOT_ESTABLISHED; numerical_pass remains false; the system-performance goal remains active.','']
    if complete:
        lines += [
        'All 112 planned operator pairs returned and synchronized: 96 random diagnostic cases plus 16 candidate-analytic known cases. '+
        'All 16 candidate outputs equal their analytic known outputs bitwise. This completes only the new candidate-analytic smoke contract, not a general numerical, Token or speed gate.','',
        'The 96 random cases have finite outputs and unchanged recorded input backing/view ledgers. Their candidate/reference errors are diagnostics only: '+
        'no numerical-equivalence tolerance was defined. DIAGNOSTIC_COMPLETE is not a numerical pass. No document CI is appropriate for specified operator cases.','',
        'V3 is a separately frozen contract correction: candidate==analytic remains mandatory; original reference==analytic is diagnostic only. '+
        'All 112 fixture descriptors, seeds, input construction (AST-checked against V2), masks and candidate algorithm pins remain unchanged. '+
        'The earlier V1 and V2 failures (100/112, exit1) remain separate preserved failures and are not relabeled as passes.','',
        'Reference diagnostic flags are 12 TRUE / 4 FALSE. The four FALSE observations are the one-key cases 099/103/107/111, with maxabs differences of '+
        '4.76837158203125e-7, 9.5367431640625e-7, 4.76837158203125e-7 and 4.76837158203125e-7 respectively. '+
        'Q/K zero backing hashes are verified. These observations do not establish the internal reference arithmetic cause or downstream Token equality.','',
        'Actual cumulative JIT entries end at stage_one 12 / stage_two 1. These are cache-entry counts, not compile-time, operator latency or throughput measurements. '+
        'FP32 tensors and disabled TF32 flags do not by themselves establish exact compiled instruction arithmetic.','',
        'The COLLECTION_RECEIPT records 17 local whitelisted file hashes, matching before/after terminal/CASES/exit SHA barriers, three SCP exit 0 results and process-absence observations. '+
        'This report verifies that the saved transcribed receipt matches local bytes; it does not independently re-observe the remote machine, '+
        'claim signed attestation, or claim that the complete remote compiled cache was collected.','']
    if summary.get('synthetic_program_test_only'):lines.insert(2,'SYNTHETIC CPU FIXTURE ONLY — not additional measured CUDA data.\n')
    if summary['evidence_issues']:lines.insert(2,'EVIDENCE ERROR: no unverifiable cases are plotted; the intended completion status is not certified.\n')
    lines+=['## Verified scope','',f"Saved-evidence validation: `{summary.get('verified_saved_evidence',False)}`; observed `{summary['observed_cases']}/112`; report status `{summary['status']}`.",'',
        '|Actual random key length N|Diagnostic cases|Largest observed maxabs error|','|---:|---:|---:|']
    for n in LENGTHS:
        rows=[r for r in summary['random'] if r['n']==n]
        lines.append(f"|{n}|{len(rows)}|{max((r['max_abs'] for r in rows),default='NOT_RUN')}|")
    lines+=['','## All planned known cases','',
        '|Case|Q|Actual N|Family|Candidate==known|Reference==known|Candidate/reference maxabs|','|---|---:|---:|---|---|---|---:|']
    known={r['case_id']:r for r in summary['known']}
    for c in summary['known_planned']:
        row=known.get(c['case_id']);value=lambda k:('TRUE' if row[k] else 'FALSE') if row else 'NOT_RUN'
        lines.append(f"|{c['case_id']}|{c['q']}|{c['n']}|{c['kind']}|{value('candidate_known_bitwise')}|{value('reference_known_bitwise')}|{row['max_abs'] if row else 'NOT_RUN'}|")
    lines+=['','## Figures','']
    for name in FIGURES:lines += [f'[{name} PNG]({name}.png) · [SVG]({name}.svg)',f'![{name}]({name}.png)','']
    lines+=['## Evidence issues','']+(['- '+x for x in summary['evidence_issues']] or ['None in saved raw/source/collection-receipt checks. Completion is restricted to the candidate-analytic smoke contract.'])
    lines+=['','## Saved raw and portable source','',
        '[Standalone analysis and figure source](analysis_entry.py). Run: `python analysis_entry.py /absolute/operator_root --output /absolute/fresh/analysis`. '+
        'Only stdlib+NumPy/Matplotlib are required; no repository/runtime imports.','']
    for name in (*RAW_PINS,*SOURCES):lines.append(f'[{name}]({os.path.relpath(root/name,output)})')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    receipt=dict(status=summary['status'],candidate_analytic_smoke_complete=summary['candidate_analytic_smoke_complete'],system_experiment_complete=False,goal_complete=False,render_inspection_required=True,
        file_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=analyze(args.root);write_report(summary,args.root,args.output)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('random','known','jit','known_planned','raw_and_source_sha256')},indent=2))


if __name__=='__main__':main()
