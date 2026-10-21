"""V2 independent local saved-record audit of the frozen 130-call devcheck.

Schema correction only: V1 SHA 2410756036a3cadfa0243e211a3dca18823df947e28cb3bac03e9d95d3273f95
required an uncollected retrieval counter on the own path. Its original source
is preserved. This version requires all thirteen real own counters and compares
optional counters using explicit presence/value, without imputing zero.

No runner/report/model implementation is imported. Quality/Graph-owner primitives
are shared with the previously independent UNION auditor, NOT the report. Their
two exact source files are exported alongside this audit for standalone replay.
All matrix/count/selection/metric/go-no-go logic below is separate from report
logic. Recorded tensor digests are evidence, not a new forward-pass proof. CIs
are intentionally not independently recomputed. A pass never completes the goal.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import sys

sys.dont_write_bytecode = True
SOURCE_BYTES = Path(__file__).read_bytes()
UNION_PATH = Path(__file__).with_name('audit_fair_union_terminal.py')
if not UNION_PATH.is_file():
    UNION_PATH = Path(__file__).with_name('INDEPENDENT_DEVCHECK_UNION_SUPPORT.py')
BASE_PATH = Path(__file__).with_name('audit_baseline_terminal.py')
if not BASE_PATH.is_file():
    BASE_PATH = Path(__file__).with_name('INDEPENDENT_UNION_RAW_AUDIT_SUPPORT.py')
UNION_BYTES, BASE_BYTES = UNION_PATH.read_bytes(), BASE_PATH.read_bytes()
spec = importlib.util.spec_from_file_location('_independent_devcheck_union_primitives', UNION_PATH)
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
need, finite, near = u.need, u.finite, u.near
COUNTS = dict(own_quality_requests=64, quality_owners=64, prime_requests=2,
              timing_requests=64, owners=2)
DOCS = (211,213,220,226,233,236,246,248,253,271,272,288,303,306,327,604)
LENGTHS = (4096,8192)
GPU = 'GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9'
SESSION = 'rescue_frozen_devcheck_v1'
REMOTE = '/mnt/8T/cl/rescue_experiment_cache/frozen_devcheck_20261020_v1'
OUTPUTS = ('INDEPENDENT_DEVCHECK_RAW_AUDIT_SOURCE.py',
           'INDEPENDENT_DEVCHECK_UNION_SUPPORT.py',
           'INDEPENDENT_UNION_RAW_AUDIT_SUPPORT.py',
           'INDEPENDENT_DEVCHECK_RAW_AUDIT.json')
ASSETS = dict(u.ASSETS)
CONFIG = dict(selector='hierarchical', budget=512, recent=64, coverage_rows=64,
    coverage_candidates=1024, coverage_batch=16, coarse_chunk_size=32,
    refresh_every=4, refresh_policy='periodic', injection_layers=[27],
    inject_tokens=64, injection_strength=.5, draft_execution='cuda_graph',
    coverage_execution='eager', max_new_tokens=128, max_draft=12,
    working_mode='packed', optimized_execution=True, deferred_probes=True,
    attention_backend='sdpa', prefill_precision='bfloat16', schedule='serial',
    profile=False, length_policy='risk')
MANDATORY = {'LAUNCH_MANIFEST.json','STAGE_RECEIPT.json','LAUNCH_ATTEMPT.json',
    'RUN_STARTED.json','PIPELINE_EXIT.json','DEPENDENCY_PREFLIGHT.json','GPU_PREFLIGHT.json',
    'results/protocol.json','results/COMPLETE.json','results/GATE_PASSED.json',
    'results/QUALITY_PASSED.json','results/own_quality.jsonl',
    'results/quality_owners.jsonl','results/runtime.jsonl','results/prime.jsonl',
    'results/owners.jsonl','results/gpu_diagnostics.jsonl','results/references.json',
    'results/reference.jsonl','results/prime_references.json','results/reference_reuse.json',
    'results/data/documents.jsonl','results/data/DATASET_MANIFEST.json','results/data/prime.json',
    'results/FROZEN_DEVCHECK_SOURCE.py','capsule/LAUNCH_MANIFEST.json','COLLECTOR_SOURCE.py'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_sha(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def canonical(tokens,eos,vocab):
    need(isinstance(tokens,list) and 0<len(tokens)<=128 and
         all(type(v) is int and 0<=v<vocab for v in tokens), 'Invalid canonical token IDs')
    need((len(tokens)==128 or tokens[-1]==eos) and
         (eos not in tokens or tokens[-1]==eos and eos not in tokens[:-1]),
         'Output was not capped at128 or first EOS')


class Inputs:
    """Never dereference original model, corpus or remote source paths."""
    def __init__(self,root,analysis):
        original=Path(root).absolute()
        need(not any(p.is_symlink() for p in (original,*original.parents)), 'Symlink export root')
        self.root=original.resolve();self.analysis=Path(analysis).absolute()
        need(self.analysis.is_relative_to(self.root), 'Analysis must be inside the explicit export')
        self.used={};self.extra=set()
        self.collection_raw=self.path('TERMINAL_COLLECTION.json').read_bytes()
        self.collection=json.loads(self.collection_raw);self.inventory=self.collection.get('files',{})
        need(isinstance(self.inventory,dict) and MANDATORY<=set(self.inventory), 'Incomplete terminal inventory')
        for name,info in self.inventory.items():
            raw=self.raw(name)
            need(isinstance(info,dict) and type(info.get('bytes')) is int and
                 info['bytes']==len(raw) and info.get('sha256')==sha(raw), 'Collected SHA/size mismatch: '+name)
        self.extra.add('EVIDENCE_MANIFEST.json');evidence=self.json('EVIDENCE_MANIFEST.json')
        need(evidence.get('file_sha256')==dict({k:v['sha256'] for k,v in self.inventory.items()},
             **{'TERMINAL_COLLECTION.json':sha(self.collection_raw)}) and
             evidence.get('status')==self.collection.get('status') and evidence.get('goal_complete') is False,
             'Exported evidence inventory is not the exact collected file/receipt set')
        name=(self.analysis/'ANALYSIS_COMPLETE.json').relative_to(self.root).as_posix()
        self.extra.add(name);receipt=self.json(name)
        need(receipt.get('status')=='COMPLETE' and receipt.get('independent_audit') is False and
             receipt.get('goal_complete') is False and receipt.get('fresh_confirmation') is False,
             'A complete, honestly scoped local report is required')
        files=receipt.get('file_sha256',{})
        need({'summary.json','analysis_source.py','REPORT.md'}<=set(files), 'Analysis inventory incomplete')
        for name,digest in files.items():
            full=(self.analysis/name).relative_to(self.root).as_posix()
            self.extra.add(full);self.pin(full,digest)
        self.summary=self.json((self.analysis/'summary.json').relative_to(self.root).as_posix())

    def path(self,name):
        need(isinstance(name,str) and name and '\\' not in name and '\x00' not in name,'Invalid artifact path')
        p=PurePosixPath(name)
        need(p.as_posix()==name and not p.is_absolute() and '..' not in p.parts and
             not {'.ssh','.git','.env','__pycache__'}.intersection(p.parts), 'Unsafe artifact path')
        path=self.root/Path(*p.parts)
        need(not any(v.is_symlink() for v in (path,*path.parents)) and path.is_file() and
             path.resolve().is_relative_to(self.root), 'Missing or symlink artifact: '+name)
        return path

    def raw(self,name):
        need(name in self.inventory or name in self.extra,'Consumed bytes not in terminal/analysis inventory: '+name)
        raw=self.path(name).read_bytes();record=dict(bytes=len(raw),sha256=sha(raw))
        need(name not in self.used or self.used[name]==record,'Artifact changed during audit: '+name)
        self.used[name]=record;return raw

    def json(self,name):
        return json.loads(self.raw(name))

    def rows(self,name):
        raw=self.raw(name)
        need(raw and raw.endswith(b'\n') and all(v.strip() for v in raw.splitlines()),'Incomplete JSONL: '+name)
        values=[json.loads(v) for v in raw.splitlines()]
        need(all(isinstance(v,dict) for v in values),'Non-object JSONL record')
        return values

    def pin(self,name,digest):
        need(valid_sha(digest) and sha(self.raw(name))==digest,'Pinned SHA mismatch: '+name)

    def finish(self):
        need(self.path('TERMINAL_COLLECTION.json').read_bytes()==self.collection_raw,'Collection changed during audit')
        for name,record in self.used.items():
            raw=self.path(name).read_bytes()
            need(len(raw)==record['bytes'] and sha(raw)==record['sha256'],'Consumed artifact changed: '+name)
        need(Path(__file__).read_bytes()==SOURCE_BYTES and UNION_PATH.read_bytes()==UNION_BYTES and
             BASE_PATH.read_bytes()==BASE_BYTES and u.SUPPORT_SOURCE_BYTES==BASE_BYTES,
             'Independent audit source/dependency changed')


def terminal_chain(x,p):
    c=x.collection
    need(c.get('status')=='FROZEN_DEVCHECK_TERMINAL_COLLECTED_NOT_YET_AUDITED' and
         c.get('terminal') is True and c.get('counts')==c.get('expected_counts')==COUNTS and
         c.get('logical_requests')==c.get('expected_requests')==130 and
         c.get('fixed_root')==REMOTE and c.get('session')==SESSION and
         c.get('goal_complete') is c.get('fresh_confirmation') is False,'Not the complete130 terminal export')
    for name in ('handles_before','handles_after'):
        h=c[name]
        need(h.get('tmux_present') is False and h.get('actual_python_processes')==[] and
             h.get('actual_execution_processes')==[], 'Terminal execution/tmux handles not absent')
    originals=c.get('source_before',{})
    need(set(originals)==set(x.inventory)-{'COLLECTOR_SOURCE.py'} and
         all(originals[k]==x.inventory[k] for k in originals),'Terminal byte/set barrier differs')
    x.pin('COLLECTOR_SOURCE.py',c.get('collector_sha256'))
    manifest=x.json('LAUNCH_MANIFEST.json')
    need(manifest.get('status')=='FROZEN_DEVCHECK_STAGED_NOT_LAUNCHED' and
         manifest.get('remote_root')==REMOTE and manifest.get('session')==SESSION and
         manifest.get('launch_authorized') is manifest.get('receiver_launches') is False and
         all(manifest.get(k) is True for k in ('no_fresh_data','no_training','no_search')) and
         manifest.get('expected_counts')==COUNTS and manifest.get('new_generation_requests')==130,
         'Launch is not the exact disabled/frozen old-development package')
    pins=manifest.get('file_sha256',{});need(pins and set(pins)<=set(x.inventory),'Launch source inventory missing')
    for name,digest in pins.items():x.pin(name,digest)
    tools=manifest.get('tool_sha256',{})
    need(tools and all(pins.get(k)==v for k,v in tools.items()),'Tools are not launch-pinned')
    need(tools.get('tools/collect_frozen_devcheck.py')==c.get('collector_sha256'),
         'Terminal collector differs from the immutable staged collector')
    cap=x.json('capsule/LAUNCH_MANIFEST.json')
    x.pin('capsule/LAUNCH_MANIFEST.json',manifest['capsule_manifest_sha256'])
    need(cap.get('status')=='STAGED_DISABLED_NOT_LAUNCHED' and cap.get('launcher_disabled') is True and
         cap.get('launch_authorized') is False and cap.get('union_freeze')==manifest.get('union_freeze')==p['union_freeze'],
         'Original disabled capsule/union differs')
    for name,digest in cap['file_sha256'].items():x.pin('capsule/'+name,digest)
    stage=x.json('STAGE_RECEIPT.json');attempt=x.json('LAUNCH_ATTEMPT.json');start=x.json('RUN_STARTED.json');end=x.json('PIPELINE_EXIT.json')
    links=dict(launch_manifest_sha256=sha(x.raw('LAUNCH_MANIFEST.json')),launch_attempt_sha256=sha(x.raw('LAUNCH_ATTEMPT.json')))
    need(stage.get('status')==manifest['status'] and stage.get('launch_authorized') is False and
         stage.get('launch_manifest_sha256')==links['launch_manifest_sha256'],'Receive receipt differs')
    need(attempt.get('status')=='ONCE_ONLY_LAUNCH_ATTEMPT' and attempt.get('authorized_explicitly') is True and
         attempt.get('session')==SESSION and attempt.get('launch_manifest_sha256')==links['launch_manifest_sha256'],
         'Missing exact once-only explicit launch receipt')
    need(start.get('status')=='RUN_STARTED' and start.get('session')==SESSION and all(start.get(k)==v for k,v in links.items()),'Run-start linkage differs')
    need(type(end.get('exit_code')) is int and end['exit_code']==0 and all(end.get(k)==v for k,v in links.items()) and
         end.get('goal_complete') is False and end.get('analysis_required_locally') is True and
         end.get('results_complete_sha256')==sha(x.raw('results/COMPLETE.json')),'Not cleanly exited or completion unbound')
    dep=x.json('DEPENDENCY_PREFLIGHT.json');gpu=x.json('GPU_PREFLIGHT.json')
    need(dep.get('passed') is dep.get('no_installation') is True and all(dep.get(k)==v for k,v in links.items()),'Dependency preflight differs')
    need(gpu.get('passed') is True and gpu.get('tests_run')==8 and gpu.get('skipped')==0 and
         len(gpu.get('actual_gpu_cases',[]))==8 and gpu.get('gpu_uuid')==GPU and all(gpu.get(k)==v for k,v in links.items()),'Eight actual CUDA cases/no-skips gate missing')
    need(not (x.root/'FAILURE.json').exists() and not (x.root/'results/FAILURE.json').exists(),'Failure marker exists')
    return manifest


def study(x,p,stage):
    meta=p['stage'+str(stage)+'_evidence'];prefix='results/'+meta['root'];proof=p['union_freeze']['stage'+str(stage)]
    x.pin('results/'+meta['pin_index'],meta['pin_index_sha256'])
    for record in x.json('results/'+meta['pin_index'])['original_to_saved'].values():x.pin('results/'+record['saved_path'],record['sha256'])
    x.pin(prefix+'/'+meta['final_manifest'],proof['final_manifest_sha256'])
    x.pin(prefix+'/'+meta['verified_complete'],proof['verified_complete_sha256'])
    final=x.json(prefix+'/'+meta['final_manifest']);verified=x.json(prefix+'/'+meta['verified_complete'])
    for name,digest in final['file_sha256'].items():x.pin(prefix+'/'+name,digest)
    need(verified.get('status')=='VERIFIED_COMPLETE' and all(verified.get(k) is True for k in
         ('passed','raw_independent_audit_passed','rendered_figures_inspected','strict_analysis_complete','all76_owners_verified',
          'standalone_summary_exact_rerun','all538_requests_verified' if stage==1 else 'all2540_requests_verified',
          'eight_frozen_choices_verified' if stage==1 else 'retained_union_verified')),
         'Prior baseline study not fully audited/rendered/sealed')
    r=prefix+'/'+meta['baseline_results'];a=prefix+'/'+meta['analysis_dir'];old=x.json(r+'/protocol.json');frozen=x.json(r+'/FROZEN_SELECTION.json')
    key='baseline' if stage==1 else 'horizon'
    x.pin(r+'/protocol.json',proof[key+'_protocol_sha256']);x.pin(r+'/COMPLETE.json',proof[key+'_complete_sha256'])
    x.pin(r+'/FROZEN_SELECTION.json',proof['selection_freeze_sha256' if stage==1 else 'frozen_selection_sha256'])
    x.pin(a+'/summary.json',proof['analysis_summary_sha256'])
    # Prior studies are trusted SEALED inputs, not newly independently audited
    # old executions. Still check their stage-specific attestations/hash links,
    # instead of accepting generic passed=True scaffolding. Stage1 attests its
    # exact rerun with a flag only; it has no separate rerun-receipt file/hash.
    count=538 if stage==1 else 2540
    link={key+'_complete_sha256':r+'/COMPLETE.json',
          'frozen_selection_sha256':r+'/FROZEN_SELECTION.json',
          'analysis_summary_sha256':a+'/summary.json',
          'final_artifacts_manifest_sha256':prefix+'/'+meta['final_manifest'],
          'independent_raw_audit_sha256':prefix+'/INDEPENDENT_RAW_AUDIT.json',
          'render_inspection_sha256':a+'/RENDER_INSPECTION.json',
          'verified_winners_sha256' if stage==1 else 'verified_union_sha256':a+('/VERIFIED_WINNERS.json' if stage==1 else '/VERIFIED_UNION.json')}
    if stage==2:
        link['horizon_protocol_sha256']=r+'/protocol.json'
        link['standalone_rerun_receipt_sha256']=prefix+'/ANALYSIS_RERUN_VERIFIED.json'
    for field,name in link.items():x.pin(name,verified.get(field))
    done=x.json(r+'/COMPLETE.json');end=x.json(prefix+'/PIPELINE_EXIT.json')
    need(done.get('logical_requests')==count and done.get('owner_closed')==76 and done.get('goal_complete') is False and
         all(done.get(k) is True for k in ('passed','sources_unchanged','assets_unchanged','dependencies_unchanged','all_canonical','backend_trajectories_equal')) and
         done.get('frozen_selection_sha256')==sha(x.raw(r+'/FROZEN_SELECTION.json')) and
         type(end.get('exit_code')) is int and end['exit_code']==0,'Prior stage completion/exit receipt differs')
    prior=x.json(prefix+'/INDEPENDENT_RAW_AUDIT.json')
    need(prior.get('status')=='INDEPENDENT_RAW_AUDIT_PASSED' and prior.get('passed') is True and
         prior.get('logical_requests')==count and prior.get('all'+str(count)+'_requests_verified') is True and
         prior.get('all76_owners_verified') is True and
         prior.get('eight_frozen_choices_verified' if stage==1 else 'retained_union_verified') is True,
         'Prior independent raw-audit attestation differs')
    for field,name in {key+'_complete_sha256':r+'/COMPLETE.json',key+'_protocol_sha256':r+'/protocol.json',
                       'frozen_selection_sha256':r+'/FROZEN_SELECTION.json','analysis_summary_sha256':a+'/summary.json'}.items():x.pin(name,prior.get(field))
    x.pin(prefix+'/'+prior['audit_source_file'],prior['audit_source_sha256'])
    if 'dependency_file' in prior:x.pin(prefix+'/'+prior['dependency_file'],prior['dependency_sha256'])
    render=x.json(a+'/RENDER_INSPECTION.json')
    figures=(('quality','tps','captures','capture_time','cold','service','selected_repeat1') if stage==1 else
             ('quality','tps','captures','capture_time','retained_tps','cold_service','cumulative','latency','gpu_telemetry'))
    need(render.get('actually_inspected_by_root') is True and render.get('rendered_pngs_inspected')==list(figures), 'Prior complete rendered-figure inspection differs')
    for name in figures:
        for suffix in ('.png','.svg'):x.pin(a+'/'+name+suffix,render['figure_sha256'].get(name+suffix))
    if stage==2:
        rerun=x.json(prefix+'/ANALYSIS_RERUN_VERIFIED.json')
        need(rerun.get('status')=='STANDALONE_EXACT_SUMMARY_RERUN_VERIFIED' and rerun.get('standalone_summary_exact_rerun') is True and
             rerun.get('summary_sha256')==rerun.get('rerun_summary_sha256')==sha(x.raw(a+'/summary.json')) and
             rerun.get('saved_analysis_sha256')==sha(x.raw(a+'/analysis_source.py')),'Prior saved-source exact rerun link differs')
    raw=x.rows(r+'/runtime.jsonl');count=6 if stage==1 else 32;arms=u.s.menu();winners={}
    need(len(raw)==38*2*count,'Prior baseline full raw matrix incomplete')
    for n in LENGTHS:
        scores=[]
        for arm in arms:
            block=[v for v in raw if v['length']==n and v['mode']==arm['mode']]
            need(len(block)==count and all(type(v['tokens']) is int and v['tokens']>0 and finite(v['outer_seconds'],True) for v in block),'Invalid prior baseline token/time score')
            scores.append((sum(v['tokens'] for v in block)/sum(v['outer_seconds'] for v in block),arm))
        winners[str(n)]={g:min((v for v in scores if u.group(v[1])==g),key=lambda v:(-v[0],v[1]['mode']))[1] for g in u.s.GROUPS}
        need(all(frozen['per_context'][str(n)]['groups'][g]['mode']==arm['mode'] for g,arm in winners[str(n)].items()),'Prior selected group was not pooled-TPS maximum')
    return dict(root=r,protocol=old,frozen=frozen,raw=raw,winners=winners)


def expected_choices(first,second):
    result={}
    for n in LENGTHS:
        out={}
        for group in u.s.GROUPS:
            for study in (first,second):
                arm=study['winners'][str(n)][group]
                if arm['mode'] not in out:out[arm['mode']]=arm
        result[str(n)]=out
    return result


def audit_protocol(x,p,manifest,first,old,choices):
    need(p.get('status')=='FROZEN_OLD_DEVELOPMENT_CHECK' and
         p.get('source_indices')==list(DOCS) and p.get('lengths')==list(LENGTHS) and
         p.get('repeats')==2 and p.get('max_new_tokens')==128 and p.get('seed')==20261025 and
         p.get('expected_counts')==COUNTS and p.get('expected_requests')==130,
         'Frozen cohort/context/repeats/counts differ')
    need(p.get('fresh_confirmation') is p.get('goal_complete') is False and
         all(p.get(k) is True for k in ('no_new_confirmation_access','no_training','no_search','mapper_frozen','policy_frozen')),
         'False development/no-search/frozen-method truth flags')
    seq=[];rng=random.Random(20261025)
    for rep in (0,1):
        docs=list(DOCS);rng.shuffle(docs);seq.extend(dict(source_index=d,repeat=rep) for d in docs)
    need(p['shared_warm_sequence']==old['protocol']['shared_warm_sequence']==seq and
         p['owner_order']==[dict(mode='rescue_graph',length=n) for n in LENGTHS], 'Shared32 sequence or owner order differs')
    proof=p['union_freeze']
    need(proof.get('status')=='UNION_VERIFIED' and proof.get('choices')==choices and
         p.get('frozen_choices')==choices and proof.get('M')==p.get('comparisons')==sum(map(len,choices.values())),
         'Retained stage1+stage2 deduplicated context/mode union differs')
    env=p['environment']
    for key in ('nvidia_smi_gpu','torch','cuda','target_dtype','draft_dtype','prefix_dtype','tf32'):
        need(env[key]==first['protocol']['environment'][key]==old['protocol']['environment'][key],'Frozen environment differs: '+key)
    need(GPU in env['nvidia_smi_gpu'] and env['target_dtype']=='torch.float32' and
         env['draft_dtype']==env['prefix_dtype']=='torch.bfloat16' and env['tf32'] is False and
         p['flash_attention']==first['protocol']['flash_attention']==old['protocol']['flash_attention'], 'GPU/precision/Flash differs')
    need(p.get('graph_max_entries')==128 and p.get('max_live_owners')==1 and
         p.get('template_prefill_precision')=='bfloat16' and p.get('target_layers')==p.get('draft_layers')==28,'Runtime/cache contract differs')
    cfg=p['configs']['rescue_graph']
    need(all(cfg.get(k)==v for k,v in CONFIG.items()) and p['configs']['rescue_eager']==dict(cfg,draft_execution='eager'),'Frozen M1/M2/M3 configuration differs')
    sources=p.get('source_sha256',{})
    need(sources==manifest.get('scientific_source_sha256') and set(u.CORE_SOURCE)<=set(sources),'Scientific runtime/source inventory differs')
    for name,digest in sources.items():
        x.pin('results/source/'+name,digest);x.pin('capsule/after/'+name,digest)
    x.pin('results/'+p['entry_source_file'],p['entry_source_sha256'])
    x.pin('tools/frozen_devcheck.py',p['entry_source_sha256'])
    bindings=p.get('source_bindings',{})
    need({'rescue.engine','rescue.query_mapper','rescue.step_budget','rescue.draft_graph'}<=set(bindings),'Actual runtime bindings missing')
    for name,info in bindings.items():
        path=name.replace('.','/')+'.py'
        need(sources.get(path)==info.get('sha256') and info.get('path')==p['args']['source_root'].rstrip('/')+'/'+path,'Runtime binding escaped frozen source root')
    for key,digest in ASSETS.items():
        role=key.removesuffix('_sha256')
        need(p.get(key)==digest==p['assets_sha256'].get(p['args'][role]),'Mandatory Mapper/policy/oldcost differs')
    for study_ in (first,old):
        op=study_['protocol']
        for role in ('teacher','student'):
            model=p['args'][role]
            need(model==op['args'][role],'Model roots differ')
            pins={name:digest for name,digest in op['assets_sha256'].items() if name.startswith(model.rstrip('/')+'/')}
            need(pins and all(p['assets_sha256'].get(name)==digest for name,digest in pins.items()),'Frozen model/tokenizer asset pins differ')
    need(p['input_sha256']==old['protocol']['input_sha256'],'Own inputs differ from baseline')
    for filename,key in (('documents.jsonl','selected_documents'),('DATASET_MANIFEST.json','data_manifest'),('prime.json','prime_document')):
        name='results/data/'+filename;oldname=old['root']+'/data/'+filename
        x.pin(name,p[key+'_sha256']);need(x.raw(name)==x.raw(oldname),'Old tokenized data snapshot changed')
    docs=x.rows('results/data/documents.jsonl');need([r['source_index'] for r in docs]==list(DOCS),'Old document identities/order differ')
    hashes={}
    for row in docs:
        for n in LENGTHS:
            pre,body,suffix=(row[k] for k in ('prefix_ids','document_ids','suffix_ids'))
            take=n-len(pre)-len(suffix);ids=pre+body[:take]+suffix
            need(take>0 and len(ids)==n and all(type(v) is int and v>=0 for v in ids),'Actual input context differs')
            hashes[f"{row['source_index']}:{n}"]=sha(json.dumps(ids).encode())
    need(hashes==p['input_sha256'],'Exact old4K/8K input SHA differs')
    need(p['data_source_paths']==old['protocol']['data_source_paths'],'Old metadata layout differs')
    for name in p['data_source_paths'].values():need(x.raw('results/'+name)==x.raw(old['root']+'/'+name),'Old source-cohort metadata changed')
    return seq


def reference_audit(x,p,first,old):
    refs=x.json('results/references.json');reference=x.rows('results/reference.jsonl');prime=x.json('results/prime_references.json')
    eos=u.tokenizer_eos(x,p);vocab=p['vocab_size']
    need(p['tokenizer_eos_token_id']==eos and type(vocab) is int and vocab>eos,'Token/EOS vocabulary differs')
    need(x.raw('results/reference.jsonl')==x.raw(old['root']+'/reference.jsonl') and
         refs==x.json(old['root']+'/references.json') and len(reference)==32,'Reused32 AR rows differ')
    need(set(refs)=={f'{d}:{n}' for n in LENGTHS for d in DOCS} and
         len({(r['length'],r['source_index']) for r in reference})==32,'AR reference matrix differs')
    for row in reference:
        canonical(row['generated'],eos,vocab)
        need(refs[f"{row['source_index']}:{row['length']}"]==row['generated'],'Reference row/output mapping differs')
    oldprime=x.json(first['root']+'/references.json')
    need(prime=={str(n):oldprime[f'271:{n}'] for n in LENGTHS},'Two paid old271 reference outputs differ')
    for ids in prime.values():canonical(ids,eos,vocab)
    reuse=x.json('results/reference_reuse.json');x.pin('results/reference_reuse.json',p['reference_reuse_sha256'])
    expected=dict(status='SEALED_OLD_REFERENCE_REUSE',reused_reference_requests=34,generated_reference_requests=0,
        references_sha256=sha(x.raw(old['root']+'/references.json')),reference_rows_sha256=sha(x.raw(old['root']+'/reference.jsonl')),
        stage2_protocol_sha256=sha(x.raw(old['root']+'/protocol.json')),stage1_protocol_sha256=sha(x.raw(first['root']+'/protocol.json')),
        stage1_references_sha256=sha(x.raw(first['root']+'/references.json')),input_sha256=p['input_sha256'],
        prime_input_sha256={str(n):first['protocol']['input_sha256'][f'271:{n}'] for n in LENGTHS},reference_rows_are_reused_not_new_generation_calls=True)
    need(all(reuse.get(k)==v for k,v in expected.items()) and p.get('reused_reference_requests')==34 and
         p.get('generated_reference_requests')==0 and p.get('prime_input_sha256')==expected['prime_input_sha256'],
         '34 reused references not correctly bound; no new AR allowed')
    return refs,prime,eos,vocab


OWN_WORK = ('tokens','accepted','proposed','blocks','draft_forwards','selector_refreshes',
    'working_gather_layers','working_full_gather_layers','working_gathered_position_layers',
    'working_incremental_binds','draft_shadow_probes','projected_positions','rpc_calls')
OPTIONAL_WORK = ('target_probe_rounds',)
PREVIOUS_AUDIT_SOURCE_SHA256 = '2410756036a3cadfa0243e211a3dca18823df947e28cb3bac03e9d95d3273f95'


def trajectory_signature(row):
    """Strict real own-work schema; never fabricate an uncollected counter.

    V1 borrowed the retrieval-only target_probe_rounds requirement. This own
    runner omits it, so absence is now explicitly represented rather than zero.
    Presence/value must still agree for both compared trajectories.
    """
    need(all(k in row and type(row[k]) is int and row[k]>=0 for k in OWN_WORK),
         'Missing/invalid mandatory own-work counter')
    need(all(k not in row or type(row[k]) is int and row[k]>=0 for k in OPTIONAL_WORK),
         'Invalid optional collected own-work counter')
    return [row['generated'], [[t[k] for k in u.s.TRACE] for t in row['trace']],
            *[row[k] for k in OWN_WORK],
            *[(k,k in row,row[k] if k in row else None) for k in OPTIONAL_WORK]]


def check_matrix(q,qo,prime,warm,owners,sequence,refs,prime_refs,eos,vocab):
    """Recompute identities, actual work and all32 eager/Graph byte-digest pairs."""
    need(dict(zip(COUNTS,map(len,(q,qo,prime,warm,owners))))==COUNTS,'Actual130/66-owner matrix incomplete')
    requests=q+prime+warm;ids=[r['request_id'] for r in requests];indices=[r['execution_index'] for r in requests]
    need(len(set(ids))==130 and all(re.fullmatch('[0-9a-f]{32}',v) for v in ids) and
         all(type(v) is int for v in indices) and sorted(indices)==list(range(130)),'130 request/execution bijection differs')
    need(len({r['owner_id'] for r in qo+owners})==66,'Quality/timing owners were reused')
    qowners={r['request_id']:r for r in qo};quality={};pairs=0;nonzero=0
    expected_q=[(n,d,m) for n in LENGTHS for d in DOCS for m in ('rescue_eager','rescue_graph')]
    need([(r['length'],r['source_index'],r['mode']) for r in q]==expected_q and [r['execution_index'] for r in q]==list(range(64)),
         'All64 quality generations must precede timing in frozen order')
    for r in q:
        key=r['length'],r['source_index'],r['mode'];arm=dict(mode=r['mode'],family='rescue',backend='eager' if r['mode']=='rescue_eager' else 'cuda_graph')
        need(r['arm']==arm and r['phase']=='own_quality' and r['performance_eligible'] is False and
             r['generated']==refs[f"{r['source_index']}:{r['length']}"] and r['tokens']==len(r['generated']) and r['output_agrees'] is True,'Quality method/canonical output/timing scope differs')
        canonical(r['generated'],eos,vocab);u.quality(r,arm,eos,vocab);u.runtime(r,arm,True)
        owner=qowners[r['request_id']]
        need(all(owner[k]==r[k] for k in ('owner_id','owner_signature','mode','length','source_index')) and
             owner['engine_closed'] is owner['hook_restored'] is True and owner['closed_runtime_stats']['entries']==0 and
             owner['final_runtime_stats']==r['runtime_stats'] and finite(owner['close_seconds']), 'Quality owner did not close or binding differs')
        quality[key]=r;nonzero+=int(r['nonzero_injection'] is True)
    for n in LENGTHS:
        for d in DOCS:
            a,b=(quality[n,d,m] for m in ('rescue_eager','rescue_graph'));u.paired_quality(a,b)
            need(trajectory_signature(a)==trajectory_signature(b),'Eager/Graph work trajectory differs');pairs+=1
    need(nonzero==64,'Actual nonzero injection absent in a quality cell')
    need([(r['length'],r['mode']) for r in owners]==[(n,'rescue_graph') for n in LENGTHS], 'Timing owner plan differs')
    timing_pairs=0
    for j,n in enumerate(LENGTHS):
        owner=owners[j];paid=prime[j];block=[r for r in warm if r['owner_id']==owner['owner_id']]
        need(paid['owner_id']==owner['owner_id'] and paid['execution_index']==64+j*33 and paid['length']==n and
             paid['source_index']==271 and paid['phase']=='cold_prime' and
             [r['execution_index'] for r in block]==list(range(65+j*33,97+j*33)) and
             [r['warm_index'] for r in block]==list(range(32)),'One paid prime plus32 warm consecutive ledger differs')
        u.owner_audit(owner,[paid,*block],sequence)
        for r in [paid,*block]:
            need(r['mode']=='rescue_graph' and r['arm']==u.OWN and r['length']==n and r['performance_eligible'] is True and
                 all(r.get(k) is False for k in ('observer_enabled','profile_enabled')) and
                 all(r.get(k) is True for k in ('mapper_executed','controller_executed')) and
                 'timeline' not in r and 'cost_samples' not in r and 'observed_rounds' not in r and
                 'queries' not in r,'Primary timing method changed or is instrumented')
            ref=prime_refs[str(n)] if r is paid else refs[f"{r['source_index']}:{n}"]
            need(r['generated']==ref and r['tokens']==len(ref) and r['output_agrees'] is True and
                 finite(r['outer_seconds'],True) and near(r['outer_tps'],r['tokens']/r['outer_seconds']), 'Timing canonical/outer TPS differs')
            canonical(r['generated'],eos,vocab);u.trajectory(r,u.OWN);u.runtime(r,u.OWN)
            need(all(finite(r[k]) for k in ('prefill_seconds','decode_seconds','cleanup_seconds','total_seconds')) and
                 r['outer_seconds']+1e-8>=r['total_seconds'] and
                 r['outer_seconds']+1e-8>=sum(r[k] for k in ('prefill_seconds','decode_seconds','cleanup_seconds')),
                 'Whole-generate outer omits internal stage time')
            if r is not paid:
                need(r['phase']=='warm' and trajectory_signature(r)==trajectory_signature(quality[n,r['source_index'],'rescue_graph']),
                     'Warm quality/work/cap/acceptance trajectory differs')
                timing_pairs+=1
    need(timing_pairs==64,'Incomplete warm-to-quality alignment')
    return dict(quality_pairs=pairs,nonzero_quality_cells=nonzero,warm_quality_pairs=timing_pairs,requests=130,quality_owners=64,timing_owners=2)


def pool(rows):
    tokens=sum(r['tokens'] for r in rows);seconds=sum(r['outer_seconds'] for r in rows)
    accepted=sum(r['accepted'] for r in rows);blocks=sum(r['blocks'] for r in rows)
    need(type(tokens) is int and tokens>0 and finite(seconds,True) and type(accepted) is int and accepted>=0 and type(blocks) is int and blocks>0,'Invalid pooled raw score')
    proposed=None if any(r.get('proposed') is None for r in rows) else sum(r['proposed'] for r in rows)
    return dict(tokens=tokens,seconds=seconds,pooled_tps=tokens/seconds,accepted=accepted,blocks=blocks,
                acceptance_length=accepted/blocks,acceptance_rate=None if proposed is None else accepted/proposed)


def compare_points(warm,baseline,choices,sequence,refs,summary):
    """Only full stage2 same16 clocks: never mix stage1 six-warm measurements."""
    metrics={};contrasts=[]
    for n in LENGTHS:
        modes={'rescue_graph':warm,**{mode:baseline for mode in choices[str(n)]}}
        metrics[str(n)]={}
        for mode,source in modes.items():
            rows=[r for r in source if r['length']==n and r['mode']==mode]
            need(len(rows)==32 and [{k:r[k] for k in ('source_index','repeat')} for r in rows]==sequence and
                 all(r['generated']==refs[f"{r['source_index']}:{n}"] and r['tokens']==len(r['generated']) and r['output_agrees'] is True and
                     finite(r['outer_seconds'],True) for r in rows), 'Retained same16 paired32 baseline/own matrix differs')
            actual=pool(rows);reported=summary['statistics']['contexts'][str(n)][mode]
            need(reported.get('documents')==16 and all(near(reported[k],actual[k]) for k in actual if actual[k] is not None) and
                 all(reported[k] is None for k in actual if actual[k] is None),'Independent pooled token/time/quality point disagrees with report')
            need(reported.get('raw_tree_unit') is (actual['acceptance_rate'] is None), 'Full-tree acceptance denominator was invented')
            metrics[str(n)][mode]=actual
        own=metrics[str(n)]['rescue_graph']['pooled_tps']
        for mode in choices[str(n)]:
            point=100*(own/metrics[str(n)][mode]['pooled_tps']-1)
            contrasts.append(dict(length=n,baseline_mode=mode,documents=16,point_percent=point))
    reported=summary['statistics']['contrasts']
    need(len(reported)==len(contrasts) and {(r['length'],r['baseline_mode']) for r in reported}=={(r['length'],r['baseline_mode']) for r in contrasts}, 'Reported retained contrasts missing/duplicated')
    mapping={(r['length'],r['baseline_mode']):r for r in reported}
    for row in contrasts:
        saved=mapping[row['length'],row['baseline_mode']]
        need(saved.get('documents')==16 and math.isclose(saved['point_percent'],row['point_percent'],rel_tol=1e-12,abs_tol=1e-9), 'Independent contrast point differs')
    feasible=all(r['point_percent']>0 for r in contrasts)
    need(summary.get('development_feasible') is feasible,'Report old-development GO/NO-GO disagrees with all-retained points')
    return dict(metrics=metrics,contrasts=contrasts,development_feasible=feasible,
                rule='All strict raw gates plus positive pooled TPS POINT against every retained arm at both contexts; no CI gate here')


def audit(root,analysis):
    x=Inputs(root,analysis)
    for name in OUTPUTS:need(not (x.root/name).exists(),'Never overwrite an existing independent audit: '+name)
    p=x.json('results/protocol.json');manifest=terminal_chain(x,p)
    first,old=study(x,p,1),study(x,p,2);choices=expected_choices(first,old)
    sequence=audit_protocol(x,p,manifest,first,old,choices)
    refs,prime_refs,eos,vocab=reference_audit(x,p,first,old)
    q,qo,prime,warm,owners=[x.rows('results/'+name+'.jsonl') for name in ('own_quality','quality_owners','prime','runtime','owners')]
    matrix=check_matrix(q,qo,prime,warm,owners,sequence,refs,prime_refs,eos,vocab)
    diag=x.rows('results/gpu_diagnostics.jsonl')
    expected=[(phase,owners[j]['owner_id'],n,64+j*33+(33 if phase=='owner_end' else 0)) for j,n in enumerate(LENGTHS) for phase in ('owner_start','owner_end')]
    need([(r['phase'],r['owner_id'],r['length'],r['execution_index']) for r in diag]==expected and
         all(r['mode']=='rescue_graph' and isinstance(r['telemetry'],dict) and r['telemetry'] for r in diag), 'Four timing-owner boundary telemetry records differ')
    complete=x.json('results/COMPLETE.json');quality=x.json('results/QUALITY_PASSED.json');gate=x.json('results/GATE_PASSED.json')
    need(complete.get('counts')==complete.get('expected_counts')==COUNTS and complete.get('logical_requests')==130 and
         complete.get('goal_complete') is complete.get('fresh_confirmation') is False and
         all(complete.get(k) is True for k in ('passed','sources_unchanged','assets_unchanged','dependencies_unchanged','quality_gate_passed',
         'all_canonical','owners_closed','data_unchanged','dual_study_freeze_unchanged','entry_source_unchanged','no_training','no_search')),
         'Terminal130 completion/source/asset/dependency flags differ')
    need(all(quality.get(k) is True for k in ('passed','own_all32_pairs','actual_same_bfloat16_nonzero_mapper','quality_owners_closed')) and
         gate.get('passed') is True and gate.get('union_freeze')==p['union_freeze'] and gate.get('source_sha256')==p['source_sha256'] and
         gate.get('entry_source_sha256')==p['entry_source_sha256'] and gate.get('expected_counts')==COUNTS and gate.get('expected_requests')==130 and
         gate.get('old_data_only') is gate.get('no_new_confirmation_opened') is True,'Pre-run/quality gates differ')
    summary=x.summary
    need(summary.get('status')=='COMPLETE' and summary.get('problems')==[] and summary.get('goal_complete') is summary.get('fresh_confirmation') is False and
         summary.get('observed_counts')==COUNTS and summary.get('observed_requests')==130 and summary.get('frozen_choices')==choices and
         summary.get('paired_complete_documents_by_context')=={str(n):list(DOCS) for n in LENGTHS},'Strict complete report scope differs')
    points=compare_points(warm,old['raw'],choices,sequence,refs,summary)
    x.finish()
    return dict(schema=1,status='INDEPENDENT_FROZEN_DEVCHECK_RAW_AUDIT_PASSED',passed=True,
        goal_complete=False,fresh_confirmation=False,independent_of_runner_and_report=True,
        synthetic_program_test_only=p.get('synthetic_program_test_only') is True,
        quality_primitives_shared_with_preexisting_independent_union_auditor=True,
        previous_audit_source_sha256=PREVIOUS_AUDIT_SOURCE_SHA256,
        schema_correction='Own thirteen mandatory work counters; target_probe_rounds optional presence/value only, never imputed',
        work_counter_evidence={name:dict(mandatory=list(OWN_WORK),optional={key:dict(present=sum(key in r for r in rows),not_collected=sum(key not in r for r in rows)) for key in OPTIONAL_WORK})
            for name,rows in (('own_quality',q),('cold_prime',prime),('warm',warm))},
        matrix=matrix,counts=COUNTS,reused_ar_references=34,generated_ar_references=0,
        retained_comparisons=sum(map(len,choices.values())),frozen_choices=choices,points=points,
        protocol_sha256=sha(x.raw('results/protocol.json')),
        analysis_summary_sha256=sha(x.raw((x.analysis/'summary.json').relative_to(x.root).as_posix())),
        terminal_collection_sha256=sha(x.collection_raw),audited_input_count=len(x.used),input_sha256={k:v['sha256'] for k,v in x.used.items()},
        audit_source_sha256=sha(SOURCE_BYTES),union_support_sha256=sha(UNION_BYTES),base_support_sha256=sha(BASE_BYTES),
        limitations=['Saved Native-KV/Q and same-BF16 V digests/positions/counts, not newly recomputed model tensors or learned-policy argmax.',
          'Timing Mapper/nonzero and causal actions align with saved quality traces and frozen source/assets; no quality observer in timed requests.',
          'Prior baselines are sealed historical owner blocks; this check cannot remove hardware drift or prove production speed.',
          'Prior baseline seals/attestation links are checked as trusted inputs, not independent reaudits of every old execution/CI; stage1 rerun has an attested flag only.',
          'Point estimates and development GO/NO-GO independently recomputed. Bootstrap confidence intervals are NOT independently recomputed.',
          'Uses already-authored independent raw-auditor primitives; no report/runner validators imported. No fresh data read and no goal-completion claim.'],
        audited_at_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--analysis-dir',type=Path,required=True)
    args=parser.parse_args();receipt=audit(args.root,args.analysis_dir)
    files=dict(zip(OUTPUTS,(SOURCE_BYTES,UNION_BYTES,BASE_BYTES,(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())))
    for name,raw in files.items():
        with (args.root/name).open('xb') as handle:handle.write(raw)
    print(json.dumps({k:receipt[k] for k in ('status','counts','retained_comparisons','points','goal_complete')},indent=2))


if __name__=='__main__':main()

