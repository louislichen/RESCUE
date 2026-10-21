"""Portable saved-data v2 PARTIAL operator report, never a system benchmark.

Only stdlib plus Matplotlib/NumPy for figures. No Torch/Triton/model/runtime
imports, inferred tolerance, Token claim or performance extrapolation.
"""
import argparse
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
 'launch.sh':'7d44762a7804f3d093fdc2fdfe2fa708ce8152fec82b90283188c3bfdac85355'}
RAW_PINS={
 'results/PLAN.json':'5e848bf8ccca9f7c4bdd321085c320f7e4240f93656627c9d8b4ed6e41577c77',
 'results/CASES.jsonl':'fec380ec254abbda12f4c54c75fc7fd18b878dbb3e4bde1149236ed23438d85d',
 'results/TERMINAL.json':'be66edc3f9f65c44a333bbf1b7d6cc99394732c0f7b44b41bbca4c27dd0dee27',
 'PIPELINE_EXIT.json':'7896ea7c5df29a32e9336ad9bab326ea5c77d2b2e9760f4219a63e85a2739678',
 'PREREGISTRATION.json':'fb21fdfea5bb92ee1368e9f419fd3709d2ff83a6efeb0b5d4b3ea3e26e1708ae',
 'SOURCE_SHA256SUMS':'2d92456c49434c98173160358333744c2d145aec1dcffb22f95ffe7693c78547'}
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
    require(len(rows)==100,'Saved failed prefix must contain exactly100 completed cases')
    compile_keys={name:set() for name in ('_stage_one','_stage_two')};jit=[];random=[];known=[]
    for i,row in enumerate(rows):
        descriptor=plan['cases'][i]
        require(all(type(row.get(k)) is type(v) and row.get(k)==v for k,v in descriptor.items()),'Case order/seed/shape is not exact planned prefix')
        require(row.get('finite') is True and row.get('output_contract') is True,'Nonfinite/output-contract failure differs')
        require(all(row.get(k)=='NOT_RUN' for k in GATES) and row.get('random_equivalence_gate')=='NOT_DEFINED','Model/Token/performance or random numerical-pass promotion')
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
            passed=candidate and reference
            require(row.get('mandatory_checks_passed') is passed and row.get('status')==('EXACT_SIMPLE_CASE_PASSED' if passed else 'FAIL'),'Known-output gate status differs')
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
    require(len(random)==96 and len(known)==4 and sum(k['candidate_known_bitwise'] for k in known)==4 and
        sum(k['reference_known_bitwise'] for k in known)==3 and known[-1]['case_id']=='operator_099' and
        known[-1]['candidate_known_bitwise'] is True and known[-1]['reference_known_bitwise'] is False and
        known[-1]['max_abs']==4.76837158203125e-7,'Observed96random/4known failure scope differs')
    for key in ('completed_cases','attempted_cases','reference_returned_cases','candidate_returned_cases','synchronized_cases'):
        require(type(terminal.get(key)) is int and terminal[key]==len(rows),'Terminal100-case counter mismatch: '+key)
    require(terminal.get('current_case')=='operator_099' and terminal.get('current_stage')=='diagnose_and_record' and
        terminal.get('error')=='Mandatory case checks failed: operator_099','Different terminal failing case/stage')
    return random,known,jit


def analyze(root):
    root=Path(root).absolute();issues=[]
    result=dict(status='PARTIAL_OPERATOR_FAIL',fixture_version=2,observed_cases=0,planned_cases=112,random=[],known=[],jit=[],
        known_planned=expected_cases()[96:],not_run_case_ids=[c['case_id'] for c in expected_cases()],
        numerical_pass=False,model='NOT_MEASURED',tokens='NOT_MEASURED',tps='NOT_MEASURED',latency='NOT_MEASURED',
        confidence_intervals=None,goal_complete=False,experiment_complete=False,evidence_issues=issues)
    try:
        pins={**SOURCES,**RAW_PINS}
        for name,pin in pins.items():require(sha(safe(root,name))==pin,'Saved raw/source hash mismatch: '+name)
        p=read(safe(root,'PREREGISTRATION.json'));plan=read(safe(root,'results/PLAN.json'));terminal=read(safe(root,'results/TERMINAL.json'));exitrow=read(safe(root,'PIPELINE_EXIT.json'))
        result['synthetic_program_test_only']=p.get('synthetic_program_test_only') is True
        require(p.get('fixture_version')==plan.get('fixture_version')==exitrow.get('fixture_version')==2,'Explicit v2 fixture version missing')
        require(p.get('status')=='FROZEN_BEFORE_OPERATOR_EXECUTION' and p.get('operator_cases')==112 and p.get('exact_known_cases')==16 and
            p.get('random_diagnostic_cases')==96 and p.get('random_equivalence_tolerance') is None and p.get('random_error_metrics_are_diagnostic_only') is True and
            p.get('expected_model_calls')==0,'Preregistered diagnostic-only scope differs')
        for key,name in (('candidate_sha256','capsule/exp/round_overhead/fp32_splitkv_candidate_v1.py'),
            ('candidate_cpu_test_sha256','capsule/tests/test_fp32_splitkv_candidate_v1.py'),('gate_sha256','capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v2.py'),
            ('gate_cpu_test_sha256','capsule/tests/test_fp32_splitkv_cuda_gate_v2.py'),('v1_gate_sha256','capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v1.py'),
            ('v1_gate_cpu_test_sha256','capsule/tests/test_fp32_splitkv_cuda_gate_v1.py'),('launcher_sha256','launch.sh')):
            require(p.get(key)==SOURCES[name],'Preregistered source pin changed: '+key)
        expected_sum={name.removeprefix('capsule/') if name.startswith('capsule/') else '../launch.sh':pin for name,pin in SOURCES.items()}
        sums={line.split('  ',1)[1]:line.split('  ',1)[0] for line in safe(root,'SOURCE_SHA256SUMS').read_text().splitlines()}
        require(sums==expected_sum,'Capsule seven-file SHA inventory differs')
        require(plan.get('status')=='PLAN_ONLY' and plan.get('actual_cuda') is False and plan.get('random_tolerance') is None and
            plan.get('random_diagnostic_cases')==96 and plan.get('exact_cases')==16,'Saved PLAN scope changed')
        for obj in (p,plan,terminal):
            require(obj.get('gpu_uuid')==GPU and obj.get('dtype')=='float32' and obj.get('tf32') is False and
                obj.get('torch_version')=='2.7.1+cu128' and obj.get('triton_version')=='3.3.1','GPU/FP32/TF32/runtime pin differs')
        require(plan.get('splits')==p.get('splits')==8 and plan.get('block_keys')==p.get('block_keys')==128,'Split/block geometry differs')
        require(terminal.get('status')=='FAIL' and all(terminal.get(k) is False for k in ('passed','numerical_pass','operator_smoke_passed','speed_measured','bitwise_equivalence_claim')) and
            terminal.get('actual_cuda') is True and terminal.get('cuda_available') is True and terminal.get('torch_cuda_initialized') is True and
            all(terminal.get(k)=='NOT_RUN' for k in GATES),'Terminal FAIL/actual-CUDA/no-model scope differs')
        require(terminal.get('source_pins')==dict(candidate_sha256=SOURCES['capsule/exp/round_overhead/fp32_splitkv_candidate_v1.py'],
            candidate_test_sha256=SOURCES['capsule/tests/test_fp32_splitkv_candidate_v1.py'],gate_sha256=SOURCES['capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v2.py'],
            fixture_version=2,v1_gate_sha256=SOURCES['capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v1.py'],
            v1_gate_test_sha256=SOURCES['capsule/tests/test_fp32_splitkv_cuda_gate_v1.py']),'Terminal v2 source pins differ')
        require(exitrow.get('exit_code')==1 and exitrow.get('terminal_sha256')==sha(safe(root,'results/TERMINAL.json')) and
            exitrow.get('preserve_partial') is True and all(exitrow.get(k) is False for k in ('automatic_retry','performance_measured','goal_complete')) and
            exitrow.get('model_calls')==0,'Pipeline exit1/no-model terminal binding differs')
        prior=p.get('prior_failure',{})
        require(prior.get('preserved_not_restarted_or_reclassified') is True and prior.get('failed_case')=='operator_099' and
            prior.get('terminal_sha256')=='53e400b6539296e3c1d472c3a7f1615fb055764bd162ec6fb1446cbf62dbcf3c' and
            prior.get('cases_sha256')=='113ae66ca0a539780ce8873e5bc71d25a48369c3639c9380a782a75e26f3870e','V1 preserved-failure identity differs')
        data=safe(root,'results/CASES.jsonl').read_bytes();require(data.endswith(b'\n'),'Truncated case JSONL')
        rows=[json.loads(line) for line in data.splitlines()];random,known,jit=validate_rows(rows,plan,terminal)
        result.update(observed_cases=len(rows),random=random,known=known,jit=jit,not_run_case_ids=[c['case_id'] for c in plan['cases'][len(rows):]],
            verified_saved_evidence=True,actual_operator_cuda=True,failed_case='operator_099',candidate_known_true=4,reference_known_true=3,
            reference_known_false=1,random_numerical_pass='NOT_DEFINED',raw_and_source_sha256=pins,
            observed_one_key_qk_zero_backing_verified=True,zero_qk_fixture_suffices_for_reference_known_bitwise=False,
            collection_verification='SEPARATE_MAIN_AGENT_AUDIT',
            random_max_abs_by_length={str(n):max(r['max_abs'] for r in random if r['n']==n) for n in LENGTHS})
    except (OSError,ValueError,TypeError,KeyError,IndexError) as error:
        issues.append(str(error));result['status']='PARTIAL_EVIDENCE_ERROR';result['verified_saved_evidence']=False
    return result


def figures(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':13,'xtick.labelsize':12,'ytick.labelsize':12})
    tag=('SYNTHETIC CPU FIXTURE — ' if summary.get('synthetic_program_test_only') else '')+'PARTIAL v2'
    caption=(f"Observed {summary['observed_cases']}/112 planned operator pairs; terminal FAIL. H100; FP32, TF32 off; split8, tile128.\n"
        'Fixture v2: one-key Q/K zero; original candidate algorithm, V and strict known-output checks unchanged.\n'
        'No model / Target Token / TPS / latency measurement. Deterministic operator cases, not documents: no document CI.\n'
        'Random errors are diagnostics only; no numerical-equivalence tolerance or pass criterion was defined.')
    def save(fig,name,title):
        fig.suptitle(tag+' — '+title,fontsize=14);fig.text(.5,.015,caption,ha='center',fontsize=12)
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
    ax.set_xticks((0,1),('Candidate == analytic known (bitwise)','Original reference == analytic known (bitwise)'));ax.xaxis.tick_top()
    labels=[f"{c['case_id']}  Q={c['q']}  {c['kind']}  N={c['n']}" for c in summary['known_planned']]
    ax.set_yticks(range(16),labels);ax.tick_params(axis='both',length=0,pad=10)
    for i in range(16):
        for j in range(2):ax.text(j,i,('NOT_RUN','TRUE','FALSE')[matrix[i,j]],ha='center',va='center',fontsize=12)
    ax.set_xlabel('Boolean observations, not model/Token gate results',labelpad=16)
    save(fig,'known_output_matrix',f"Known-output flags: {len(summary['known'])}/16 observed; TRUE / FALSE are not a model gate")
    fig,ax=plt.subplots(figsize=(13,7));x=[r['case_index'] for r in summary['jit']]
    for stage,color in zip(('_stage_one','_stage_two'),colors):
        ax.step(x,[r[stage] for r in summary['jit']],where='post',color=color,linewidth=2,label=stage.lstrip('_')+' observed cache entries')
    count=len(summary['jit']);ax.axvspan(count-.5,111.5,color='#eeeeee')
    if summary['jit']:
        ax.axvline(99,color='#b66c30',linestyle=':',linewidth=1.5)
        ax.annotate('099: terminal FAIL',(99,summary['jit'][-1]['_stage_one']),xytext=(-130,25),textcoords='offset points',fontsize=12,arrowprops={'arrowstyle':'->','color':'#555555'})
    ax.text((count+111)/2,.5,f'{112-count} cases\nNOT_RUN',ha='center',va='bottom',fontsize=12)
    ax.set_xticks((0,24,48,72,96,111));ax.set_xlabel('Frozen planned case index (000–111); only000–099 observed')
    ax.set_ylabel('Cumulative observed JIT compiled entries (count)');ax.set_xlim(-.5,111.5);ax.set_ylim(bottom=0,top=11)
    ax.set_yticks(range(0,11));ax.grid(axis='y',alpha=.25);ax.legend(loc='upper left',fontsize=12)
    save(fig,'jit_entries','Actual two-stage JIT cache inventory, not compile time or latency')


def write_report(summary,root,output):
    root,output=Path(root).absolute(),Path(output).absolute();require(not output.exists(),'Fresh analysis directory required')
    output.mkdir(parents=True);shutil.copyfile(__file__,output/'analysis_entry.py');figures(summary,output)
    (output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
    lines=['# PARTIAL v2 — FP32 split-KV standalone operator gate','',
        'Terminal FAIL; 100/112 planned operator pairs completed and synchronized. The frozen100-case prefix consists of96 random diagnostics and4 known-output cases;12 known cases were NOT_RUN. '+
        'This is an operator gate, not a model/system-performance experiment. Model, Target Token equality, acceptance, TPS and latency are NOT_MEASURED.','',
        'The random96 cases returned finite outputs with unchanged recorded input backing/view metadata, but no random numerical-equivalence tolerance was defined. '+
        'DIAGNOSTIC_COMPLETE is not a numerical pass. No document CI is appropriate: these are specified operator cases, not paired document samples.','',
        'The failing operator_099 is Q=2,N=129,one_key(allowed_key=0),bool_broadcast. Candidate==analytic-known was TRUE; original efficient-attention reference==known was FALSE. '+
        'Candidate/reference maxabs difference was4.76837158203125e-7. Thus the mandatory joint known-output check failed; this does not establish a candidate implementation error, '+
        'does not pass this frozen gate, and does not justify weakening it or claiming downstream Token equivalence.','',
        'Fixture v2 only zeros Q/K for one-key cases after the original random draws. Recorded Q/K backing-byte hashes verify zeros for observed operator_099. '+
        'The original reference still differs from the analytic known value by4.76837158203125e-7 against the candidate; '+
        'therefore the proposed zero-Q/K fixture repair did not remove this strict known-output failure. The underlying reference arithmetic cause is not established here. '+
        'This is not a candidate algorithm change; v1 remains a separate preserved failure, not restarted or reclassified.','',
        'Actual recorded cumulative JIT entries ended at stage_one9 / stage_two1. These are cache-entry counts, not compile-time or latency measurements. '+
        'Source/layout pins and output hashes are checked against the saved data; no Torch/Triton/runtime is imported by this report.','']
    lines+=['This portable report verifies pinned local source/raw bytes and exit/terminal linkage. '+
        'Remote process absence and collection barriers are a separate main-agent audit; no absent collection receipt is invented, '+
        'and no complete remote compiled-cache inventory is claimed.','']
    if summary.get('synthetic_program_test_only'):lines.insert(2,'SYNTHETIC CPU FIXTURE ONLY — not additional measured CUDA data.\n')
    if summary['evidence_issues']:lines.insert(2,'EVIDENCE ERROR: numeric statements below describe the intended frozen failed run; no unverifiable cases are plotted.\n')
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
    lines+=['## Evidence issues','']+(['- '+x for x in summary['evidence_issues']] or ['None in saved raw/source checks. Terminal operator failure is retained, not a report error.'])
    lines+=['','## Saved raw and portable source','',
        '[Standalone analysis and figure source](analysis_entry.py). Run: `python analysis_entry.py /absolute/operator_root --output /absolute/fresh/analysis`. '+
        'Only stdlib+NumPy/Matplotlib are required; no repository/runtime imports.','']
    for name in (*RAW_PINS,*SOURCES):lines.append(f'[{name}]({os.path.relpath(root/name,output)})')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    receipt=dict(status=summary['status'],experiment_complete=False,goal_complete=False,render_inspection_required=True,
        file_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=analyze(args.root);write_report(summary,args.root,args.output)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('random','known','jit','known_planned','raw_and_source_sha256')},indent=2))


if __name__=='__main__':main()

