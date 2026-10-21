"""Independent saved-record audit of a terminal frozen-UNION confirmation.

No runner, report, model, tokenizer implementation or data-preparation helper is
imported. The only support is the separately authored stdlib stage1 auditor.
This verifies recorded tensor digests/trajectory evidence, not a fresh forward
pass or independent learned-policy argmax. Confidence intervals are deliberately
left to a separately sealed exact saved-source rerun; raw points are recomputed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re


SUPPORT_PATH = Path(__file__).with_name('audit_baseline_terminal.py')
if not SUPPORT_PATH.is_file():
    SUPPORT_PATH = Path(__file__).with_name('INDEPENDENT_UNION_RAW_AUDIT_SUPPORT.py')
AUDIT_SOURCE_BYTES = Path(__file__).read_bytes()
SUPPORT_SOURCE_BYTES = SUPPORT_PATH.read_bytes()
spec = importlib.util.spec_from_file_location('independent_union_stage1_support', SUPPORT_PATH)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
if SUPPORT_PATH.read_bytes() != SUPPORT_SOURCE_BYTES:
    raise RuntimeError('Independent support changed during import')
need, finite, near = s.need, s.finite, s.near
SEED, LENGTHS, GROUPS = 20261026, (4096, 8192), s.GROUPS
OWN = dict(mode='rescue_graph', family='rescue', backend='cuda_graph')
ASSETS = dict(mapper_sha256='d08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
    policy_sha256='3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
    cost_sha256='d92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1')
OUTPUTS = ('INDEPENDENT_UNION_RAW_AUDIT.json', 'INDEPENDENT_UNION_RAW_AUDIT_SOURCE.py',
           'INDEPENDENT_UNION_RAW_AUDIT_SUPPORT.py')
REQUIRED_SOURCE = ('rescue/fair_union_confirmation.py', 'rescue/fair_union_confirmation_data.py',
    'rescue/fair_union_confirmation_report.py', 'tests/test_rescue_fair_union_confirmation.py',
    'tests/test_rescue_fair_union_confirmation_data.py', 'tests/test_rescue_fair_union_confirmation_report.py',
    'exp/draft_graph/fair_union_confirmation_20261020_preregistration.json',
    'exp/draft_graph/run_fair_union_confirmation_20261020_v1.sh')
CORE_SOURCE = tuple('rescue/'+n+'.py' for n in ('__init__','engine','config','attention','baselines','draft_graph','selection',
    'injection','query_mapper','step_budget','loading','packing','efficient_attention','attention_summary','features',
    'scheduler','messages','cache','policy','transport','baseline_selection','fair_confirmation_data','baseline_horizon_data'))+(
    'exp/attention_horizon/selection.py','exp/selective_kv/bridge.py')
OLD_FILES = ('exp/long_system/data/train.jsonl','exp/long_system/data/development.jsonl','exp/long_system/data/test.jsonl',
    *('exp/module_'+part+'/data_v1_20261018/'+name+'.jsonl'
      for part in ('opt','arch') for name in ('fit','tune','warmup','confirmation')))
RUNTIME = ('captures','replays','resident_steps','shadow_updates','capture_host_seconds')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


class Inputs:
    """Track every consumed byte; never follow original remote asset paths."""
    def __init__(self, root, analysis):
        self.root = Path(root).resolve(); self.analysis = Path(analysis).resolve()
        need(self.analysis.is_relative_to(self.root), 'Local analysis escapes terminal export')
        self.used = {}; self.allowed_analysis = set()
        self.collection_path = s.local(self.root,'TERMINAL_COLLECTION.json')
        self.collection_bytes = self.collection_path.read_bytes()
        self.collection = json.loads(self.collection_bytes)
        self.inventory = self.collection.get('files', {})
        need(isinstance(self.inventory, dict) and self.inventory, 'Empty terminal collection')
        for name, info in self.inventory.items():
            raw = self.raw(name)
            need(type(info.get('bytes')) is int and info['bytes'] == len(raw) and info.get('sha256') == sha(raw),
                 'Terminal collection SHA/size mismatch: '+name)
        name = str((self.analysis/'ANALYSIS_COMPLETE.json').relative_to(self.root))
        self.allowed_analysis.add(name)
        analyzed = self.json(name)
        need(analyzed.get('status') == 'COMPLETE', 'PARTIAL local analysis cannot pass terminal audit')
        files = analyzed.get('file_sha256', {})
        need(isinstance(files, dict) and files, 'Missing local analysis artifact inventory')
        for relative, expected in files.items():
            full = (self.analysis/relative).relative_to(self.root).as_posix()
            self.allowed_analysis.add(full)
            self.pin(full, expected)
        need({'summary.json','analysis_source.py','REPORT.md'} <= set(files), 'Incomplete saved local analysis')

    def raw(self, name):
        path = s.local(self.root, name)
        need(name in self.inventory or name in self.allowed_analysis,
             'Consumed file is not pinned by collection/local analysis: '+name)
        raw = path.read_bytes()
        current = dict(bytes=len(raw), sha256=sha(raw))
        need(name not in self.used or self.used[name] == current, 'File changed during audit: '+name)
        self.used[name] = current
        return raw

    def json(self, name):
        return json.loads(self.raw(name))

    def rows(self, name, empty=False):
        raw = self.raw(name)
        if empty and not raw: return []
        need(raw and raw.endswith(b'\n') and all(v.strip() for v in raw.splitlines()), 'Incomplete JSONL: '+name)
        return [json.loads(v) for v in raw.splitlines()]

    def pin(self, name, expected):
        need(valid_sha(expected) and sha(self.raw(name)) == expected, 'Saved SHA differs: '+name)

    def finish(self):
        need(s.local(self.root,'TERMINAL_COLLECTION.json').read_bytes() == self.collection_bytes, 'Terminal receipt changed during audit')
        for name, record in self.used.items():
            raw = s.local(self.root, name).read_bytes()
            need(len(raw) == record['bytes'] and sha(raw) == record['sha256'], 'Input changed before audit seal: '+name)
        need(Path(__file__).read_bytes() == AUDIT_SOURCE_BYTES and SUPPORT_PATH.read_bytes() == SUPPORT_SOURCE_BYTES,
             'Auditor or support changed during audit')


def geometry(arm):
    return arm['top_chunks'], arm['fixed_k']


def group(arm):
    return 'full_tree' if arm['family'] == 'full_tree' else 'retrieval/'+('graph' if arm['backend']=='cuda_graph' else arm['backend'])


def choices_plan(choices, documents):
    menu = {a['mode']: a for a in s.menu()}
    need(set(choices) == set(map(str, LENGTHS)), 'Both contexts must retain their union')
    extra = []
    for n in LENGTHS:
        arms = choices[str(n)]; need(isinstance(arms, dict) and 4 <= len(arms) <= 8, 'Invalid retained contrast count')
        need(all(menu.get(mode) == arm for mode, arm in arms.items()), 'Unregistered retained contender')
        groups = [group(a) for a in arms.values()]
        need(groups == sorted(groups, key=GROUPS.index) and all(1 <= groups.count(g) <= 2 for g in GROUPS),
             'Original group/stage union order changed')
        seen = {geometry(a) for a in arms.values() if a['family']=='retrieval' and a['backend']=='eager'}
        for a in arms.values():
            if a['family']=='retrieval' and a['backend']!='eager' and geometry(a) not in seen:
                extra.append(dict(length=n, top_chunks=a['top_chunks'], fixed_k=a['fixed_k'])); seen.add(geometry(a))
    m, r = sum(len(v) for v in choices.values()), len(extra)
    need(8 <= m <= 16 and 0 <= r <= 8 and len(documents)==len(set(documents))==16, 'Invalid M/R/cohort')
    counts = dict(reference_requests=32, own_quality_requests=64, baseline_quality_requests=m,
        matching_quality_requests=r, prime_requests=m+2, timing_requests=32*(m+2), owners=m+2)
    rng=random.Random(SEED); sequence=[]
    for repeat in (0,1):
        order=list(documents); rng.shuffle(order); sequence.extend(dict(source_index=i, repeat=repeat) for i in order)
    owners=[]; requests=[]
    for n in LENGTHS:
        modes=['rescue_graph', *choices[str(n)]]; random.Random(SEED+n).shuffle(modes)
        for mode in modes:
            owners.append(dict(mode=mode,length=n))
            requests.append(dict(mode=mode,length=n,source_index=271,phase='cold_prime'))
            requests.extend(dict(mode=mode,length=n,**v,phase='warm',warm_index=j) for j,v in enumerate(sequence))
    return m,r,extra,counts,sequence,owners,requests


def data_audit(x, protocol):
    manifest=x.json('results/data/DATASET_MANIFEST.json')
    x.pin('results/data/DATASET_MANIFEST.json',protocol['dataset_manifest_sha256'])
    need(manifest.get('status')=='PREPARED_NOT_EXECUTED' and manifest.get('seed')==SEED and all(manifest.get(k) is True
         for k in ('no_outcomes_read','no_model_generation','no_training')), 'Changed pre-access preparation contract')
    times=[datetime.fromisoformat(manifest[k]) for k in ('source_frozen_at_utc','candidate_first_open_not_before_utc','prepared_at_utc')]
    need(all(t.tzinfo is not None for t in times) and times==sorted(times), 'Candidate opened before complete source freeze')
    need(manifest['dataset_file']=='confirmation.jsonl' and manifest['source_snapshot_root']=='source', 'Unexpected prepared-data paths')
    x.pin('results/data/confirmation.jsonl',manifest['dataset_sha256'])
    receipts=manifest['receipt_sha256']; required=('source_freeze','exclusions','final_protocol','old_raw_source_manifest')
    need(all('receipts/'+n+'.json' in receipts for n in required), 'Incomplete pre-data receipt chain')
    for name,digest in receipts.items(): x.pin('results/data/'+name,digest)
    gate=x.json('results/data/receipts/source_freeze.json'); formal=x.json('results/data/receipts/final_protocol.json')
    excluded=x.json('results/data/receipts/exclusions.json'); proof=manifest['union_freeze']
    need(gate['union_freeze']==formal['union_freeze']==protocol['union_freeze']==proof and
         gate['source_sha256']==manifest['source_sha256']==protocol['source_sha256'] and
         gate['tokenizer_sha256']==manifest['tokenizer_sha256'] and gate['template_sha256']==manifest['template_sha256'],
         'Prepared cohort is not bound to the exact dual-study/source/tokenizer freeze')
    for field,name in (('source_freeze_sha256','source_freeze'),('exclusion_receipt_sha256','exclusions'),('final_protocol_sha256','final_protocol')):
        need(manifest[field]==receipts['receipts/'+name+'.json'], 'Prepared receipt link changed')
    need(gate.get('status')==formal.get('status')=='FROZEN_BEFORE_DATA_ACCESS' and gate.get('candidate_content_accessed') is False and
         formal.get('is_template') is False and gate.get('seed')==formal.get('seed')==SEED, 'No final pre-data freeze')
    rule=manifest['selection_rule']
    need(protocol['selection_rule']==gate['selection_rule']==formal['selection_rule']==rule and rule['original_split']=='validation' and
         rule['minimum_source_index']==0 and rule['minimum_natural_tokens']==8192 and rule['documents']==16 and
         rule['context_lengths']==list(LENGTHS) and 'first eligible sixteen' in rule['order'], 'Fresh cohort selection rule changed')
    template=gate['template']; need(json_sha(template)==manifest['template_sha256'], 'Prompt template changed')
    metadata=excluded['metadata_sources']
    need(excluded['status']=='OLD_DATA_METADATA_ONLY' and len(metadata)==11 and {v['relative_name'] for v in metadata}==set(OLD_FILES) and
         all(valid_sha(v['sha256']) and isinstance(v['rows'],list) and v['rows'] for v in metadata), 'Eleven old-cohort exclusions missing')
    old=[r for v in metadata for r in v['rows']]
    deny={r['document_sha256'] for r in old}; prefixes={str(n):{r['input_sha256'][str(n)] for r in old if str(n) in r.get('input_sha256',{})} for n in LENGTHS}
    need(set(excluded['document_sha256'])==deny and all(set(excluded['input_sha256'][n])==v for n,v in prefixes.items()), 'Exclusion denysets differ from saved metadata')
    documents=x.rows('results/data/confirmation.jsonl'); selected=manifest['selected']; inputs={}; last=-1
    need(len(documents)==len(selected)==16, 'Exactly sixteen preselected documents required')
    for row,detail in zip(documents,selected):
        i=row['source_index']; body=row['document_ids']
        need(type(i) is int and i>last and row['split']=='confirmation' and row['original_split']=='validation' and
             type(row['original_tokens']) is int and row['original_tokens']>=8192 and len(body)==8192 and
             all(type(t) is int and t>=0 for t in body) and valid_sha(row['document_sha256']) and
             dict(prefix_ids=row['prefix_ids'],suffix_ids=row['suffix_ids'])==template, 'Prepared document identity/body/template differs')
        last=i; hashes={}
        for n in LENGTHS:
            take=n-len(row['prefix_ids'])-len(row['suffix_ids']); ids=row['prefix_ids']+body[:take]+row['suffix_ids']
            need(take>0 and len(ids)==n and all(type(t) is int and t>=0 for t in ids), 'Actual input is not exact4K/8K')
            hashes[str(n)]=sha(json.dumps(ids).encode()); inputs[f'{i}:{n}']=hashes[str(n)]
        need(detail==dict(source_index=i,original_split='validation',document_sha256=row['document_sha256'],original_tokens=row['original_tokens'],input_sha256=hashes),
             'Selected input fingerprints differ')
    skips=manifest['skipped']; skip={r['source_index']:r for r in skips}; chosen={r['source_index']:r for r in selected}
    need(len(skip)==len(skips) and not set(skip)&set(chosen) and set(skip)|set(chosen)==set(range(last+1)), 'First-eligible scan ledger has gaps/duplicates')
    for i in range(last+1):
        row=chosen.get(i,skip.get(i)); fingerprint=row['document_sha256']; hashes=row.get('input_sha256',{}); reason=row.get('reason')
        need(valid_sha(fingerprint), 'Scan lacks original document fingerprint')
        if i in chosen:
            need(fingerprint not in deny and all(v not in prefixes[n] for n,v in hashes.items()), 'Old/reserved/selected duplicate admitted')
            deny.add(fingerprint)
            for n,v in hashes.items(): prefixes[n].add(v)
        elif reason=='document_duplicate': need(fingerprint in deny,'Unjustified duplicate skip')
        elif reason=='natural_tokens_below8192': need(fingerprint not in deny and type(row['original_tokens']) is int and 0<=row['original_tokens']<8192,'Unjustified short skip')
        elif reason=='prefix_duplicate': need(fingerprint not in deny and row['original_tokens']>=8192 and set(hashes)==set(prefixes) and any(v in prefixes[n] for n,v in hashes.items()),'Unjustified prefix skip')
        else: need(False,'Outcome-dependent/unknown scan skip')
    need(protocol['input_sha256']==inputs and protocol['dataset_sha256']==manifest['dataset_sha256'] and
         protocol['source_freeze_sha256']==manifest['source_freeze_sha256'] and protocol['final_protocol_sha256']==manifest['final_protocol_sha256'], 'Executed input/source differs from prepared data')
    paths=manifest['portable_proof_arguments']
    need(set(paths)=={a+'_'+b for a in ('stage1','stage2') for b in ('results','analysis','verified','manifest')},'Missing preparation-time portable dual proof')
    for stage in ('stage1','stage2'):
        parent=Path(paths[stage+'_manifest']).parent; record=proof[stage]
        x.pin('results/data/'+paths[stage+'_manifest'],record['final_manifest_sha256'])
        x.pin('results/data/'+paths[stage+'_verified'],record['verified_complete_sha256'])
        portable=x.json('results/data/'+paths[stage+'_manifest'])
        for name,digest in portable['file_sha256'].items():x.pin('results/data/'+(parent/name).as_posix(),digest)
    return [r['source_index'] for r in documents],proof,manifest,gate,formal


def study(x,protocol,key,proof,stage):
    meta=protocol[key]; prefix='results/'+meta['root']; index=x.json('results/'+meta['pin_index'])
    x.pin('results/'+meta['pin_index'],meta['pin_index_sha256'])
    for record in index['original_to_saved'].values(): x.pin('results/'+record['saved_path'],record['sha256'])
    manifest_path=prefix+'/'+meta['final_manifest']; verified_path=prefix+'/'+meta['verified_complete']
    x.pin(manifest_path,proof['final_manifest_sha256']); x.pin(verified_path,proof['verified_complete_sha256'])
    manifest=x.json(manifest_path); verified=x.json(verified_path)
    for name,digest in manifest['file_sha256'].items(): x.pin(prefix+'/'+name,digest)
    need(verified['status']=='VERIFIED_COMPLETE' and all(verified.get(k) is True for k in
         ('passed','raw_independent_audit_passed','rendered_figures_inspected','strict_analysis_complete','all76_owners_verified')), 'Unsigned/incomplete development study')
    r=prefix+'/'+meta['baseline_results']; a=prefix+'/'+meta['analysis_dir']; p=x.json(r+'/protocol.json'); f=x.json(r+'/FROZEN_SELECTION.json')
    x.pin(r+'/protocol.json',proof['baseline_protocol_sha256' if stage==1 else 'horizon_protocol_sha256'])
    x.pin(r+'/FROZEN_SELECTION.json',proof['selection_freeze_sha256' if stage==1 else 'frozen_selection_sha256'])
    x.pin(r+'/COMPLETE.json',proof['baseline_complete_sha256' if stage==1 else 'horizon_complete_sha256'])
    x.pin(a+'/summary.json',proof['analysis_summary_sha256'])
    raw=x.rows(r+'/runtime.jsonl'); total=6 if stage==1 else 32; winners={}
    need(len(raw)==38*2*total,'Development selection raw matrix incomplete')
    for n in LENGTHS:
        scores=[]
        for arm in s.menu():
            rows=[row for row in raw if row['length']==n and row['mode']==arm['mode']]
            need(len(rows)==total and all(type(v['tokens']) is int and v['tokens']>0 and finite(v['outer_seconds'],True) for v in rows),'Invalid original development score')
            scores.append((sum(v['tokens'] for v in rows)/sum(v['outer_seconds'] for v in rows),arm))
        winners[str(n)]={g:min((v for v in scores if group(v[1])==g),key=lambda v:(-v[0],v[1]['mode']))[1] for g in GROUPS}
        need(all(f['per_context'][str(n)]['groups'][g]['mode']==arm['mode'] for g,arm in winners[str(n)].items()), 'Original development winner is not exact pooled maximum')
    return p,f,winners,r


def tensor(value,dimensions=None):
    need(isinstance(value,dict) and valid_sha(value.get('sha256')) and isinstance(value.get('dtype'),str) and
         isinstance(value.get('shape'),list) and all(type(d) is int and d>0 for d in value['shape']) and
         (dimensions is None or len(value['shape'])==dimensions),'Invalid recorded tensor digest')


def tokenizer_eos(x,protocol):
    """Use prior independently exported metadata, never reopen model assets."""
    prefix='results/'+protocol['stage1_evidence']['root']+'/asset_metadata/Qwen2-7B-Instruct/'
    values={}
    for name in ('config.json','tokenizer_config.json','tokenizer.json'):
        raw=x.raw(prefix+name);original=protocol['args']['teacher'].rstrip('/')+'/'+name
        need(protocol['assets_sha256'].get(original)==sha(raw),'Old exported tokenizer is not the current frozen Target asset')
        values[name]=json.loads(raw)
    token=values['tokenizer_config.json']['eos_token']
    if isinstance(token,dict):token=token.get('content')
    found=[v['id'] for v in values['tokenizer.json']['added_tokens'] if v.get('content')==token]
    need(isinstance(token,str) and token and len(found)==1 and type(found[0]) is int and found[0]>=0,'EOS does not map to one actual exported added token')
    eos=found[0]; configured=values['config.json']['eos_token_id']
    need(configured==eos or isinstance(configured,list) and eos in configured,'Model/tokenizer EOS differs')
    decoder=values['tokenizer_config.json'].get('added_tokens_decoder',{})
    if str(eos) in decoder:need(decoder[str(eos)]['content']==token,'Tokenizer added-token decoder differs')
    return eos


def trajectory(row,arm):
    trace=row['trace']; need(trace and row['blocks']==len(trace),'Missing actual round trace')
    progress=0; own=arm['family']=='rescue'
    for j,t in enumerate(trace):
        cap,p,a=t['cap'],t['proposed'],t['accepted']; remaining=128-progress
        allowed=(min(2,remaining),min(4,remaining)) if own else (min(arm['fixed_k'],remaining),)
        need(all(type(v) is int for v in (cap,p,a)) and cap in allowed and 1<=p<=cap and 0<=a<=p and
             t['round_index']==j and t['prefix']==row['length']-1+progress and t['terminal'] is (j==len(trace)-1), 'Causal cap/proposal/accept/EOS state differs')
        need(type(t['workset_tokens']) is int and 0<t['workset_tokens']<=(512 if own else arm['top_chunks']*32), 'Actual workset exceeds frozen budget')
        progress+=a+1
    need(row['tokens'] in (progress,progress-1) and progress<=129 and row['accepted']==sum(t['accepted'] for t in trace) and
         row['proposed']==sum(t['proposed'] for t in trace) and near(row['acceptance_rate'],row['accepted']/row['proposed']), 'Accepted commits/aggregate rate cannot explain visible output')
    forwards=sum(t['proposed']+int(t['accepted']==t['proposed'] and not t['terminal']) for t in trace)
    need(row['draft_forwards']==forwards,'Draft forward count omits proposal/catchup work')


def logical(row,arm):
    return s.trace_signature(row,arm) if arm['family']=='full_tree' else [row['generated'],[[t[k] for k in s.TRACE] for t in row['trace']],
        *[row.get(k) for k in s.WORK]]


def quality(row,arm,eos=None,vocab=None):
    trajectory(row,arm); own=arm['family']=='rescue'; rounds=row['observed_rounds']; queries=row['queries']
    need(len(rounds)==row['blocks'] and len(queries)==row['draft_forwards'],'Missing all-round/all-step quality evidence')
    generated=[]; cursor=0; nonzero=False
    for j,(t,o) in enumerate(zip(row['trace'],rounds)):
        selected=o['selected']; p,a=t['proposed'],t['accepted']
        need(all(o.get(k)==t[k] for k in ('round_index','prefix','accepted')) and o['progress']==len(generated) and o['remaining']==128-len(generated) and
             isinstance(selected,list) and len(selected)==len(set(selected))==t['workset_tokens'] and
             selected==sorted(selected) and all(type(v) is int and 0<=v<t['prefix'] for v in selected),'Selection/observed round does not align with actual action')
        if own:need(len(selected)==512 and 0 in selected and set(range(t['prefix']-64,t['prefix']))<=set(selected),'Own512 cache lost mandatory first/recent positions')
        else:need(o.get('budget_features') is None and not o.get('injection_evidence'), 'Retrieval quality executed M2/M3')
        draft,checks=o['draft'],o['checks']; need(len(draft)==p and len(checks)==p+1,'Missing actual draft/Target check tokens')
        need(all(type(v) is int and v>=0 and (vocab is None or v<vocab) for v in draft+checks),'Invalid draft/Target check token ID')
        if eos is not None:
            need((eos not in draft or (draft[-1]==eos and eos not in draft[:-1])) and (p==t['cap'] or draft[-1]==eos),
                 'Proposal stopped before its cap without a unique final Draft EOS')
        accepted=next((k for k in range(p) if draft[k]!=checks[k]),p)
        need(accepted==a,'Actual contiguous token agreement differs from accepted count')
        generated.extend(draft[:a])
        if len(generated)<row['tokens']:generated.append(checks[a])
        if own:
            vector=o.get('budget_features')
            need((p<2 and vector is None) or (p>=2 and isinstance(vector,list) and len(vector)==12 and
                 all(type(v) in (float,int) and math.isfinite(v) for v in vector)), 'Missing real causal twelve-dimensional features')
            if vector is not None:
                close=lambda a,b:math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-6)
                previous=.75 if j==0 else (1. if row['trace'][j-1]['accepted']==row['trace'][j-1]['proposed'] else row['trace'][j-1]['accepted']/(row['trace'][j-1]['accepted']+1))
                need(close(vector[0],math.log(t['prefix'])/10) and close(vector[4],(j%4)/4) and
                     close(vector[5],o['progress']/128) and close(vector[6],o['remaining']/128) and close(vector[7],previous), 'Features do not use causal committed state')
            for v in o.get('injection_evidence',[]):
                w,n=v['working_value'],v['native_value']; tensor(w,2);tensor(n,2); pos=v['positions']; count=v['different_elements']
                need(v['layer']==27 and len(pos)==64 and len(set(pos))==len(pos) and set(pos)<=set(selected) and
                     all(0<position<t['prefix']-64 for position in pos) and
                     w['shape']==n['shape'] and w['shape'][0]==len(pos) and w['dtype']==n['dtype']=='torch.bfloat16' and
                     type(count) is int and 0<=count<=math.prod(w['shape']) and v.get('native_matches_memo') is True and
                     ((count==0)==(w['sha256']==n['sha256'])),'Nonzero injection proof is not same-position same-BF16 V')
                nonzero |= count>0
        for depth in range(p):
            q=queries[cursor]; tensor(q['query'],4)
            need(q['query']['shape']==[1,12,1,128] and q['query']['dtype']=='torch.bfloat16' and
                 q['logits'] is True and q['summary_requested'] is (own and depth<2) and q['summary_count']==int(own and depth<2), 'Proposal Query/summary work altered causal first-two-step schedule')
            cursor+=1
        if a==p and not t['terminal']:
            q=queries[cursor]; tensor(q['query'],4)
            need(q['query']['shape']==[1,12,1,128] and q['query']['dtype']=='torch.bfloat16' and
                 q['logits'] is False and q['summary_requested'] is False and q['summary_count']==0,'Catchup became an extra M3 decision')
            cursor+=1
    need(generated==row['generated'] and cursor==len(queries),'Observed draft/check commits do not reconstruct client output')
    if own:need(nonzero and row['nonzero_injection'] is True,'Missing genuine nonzero Mapper byte evidence')
    else:need(row['nonzero_injection'] is False and row['projected_positions']==row['draft_shadow_probes']==0,'Retrieval quality injected content or extra probes')
    terminal=row['trace'][-1]
    for name,expected,dtype in (('native_t',terminal['prefix']+terminal['proposed']+1,'torch.float32'),
                                ('native_s',terminal['prefix']+terminal['proposed'],'torch.bfloat16')):
        cache=row[name]; need(cache['lengths']==[expected]*28 and len(cache['layers'])==28,'Missing uncropped full28-layer Native valid-span evidence')
        for layer in cache['layers']:
            for kind in ('key','value'):
                value=layer[kind]; tensor(value,4)
                heads=4 if name=='native_t' else 2
                need(value['shape']==[1,heads,expected,128] and value['dtype']==dtype,'Native cache dtype/geometry/valid span differs')


def runtime(row,arm,quality_row=False):
    backend=arm['backend']; signature=row['owner_signature']; own=arm['family']=='rescue'
    for name in ('native_t','native_s','working'):
        cache=signature[name]
        need(all(type(cache.get(k)) is int and cache[k]>0 for k in ('identity','key_ptr','value_ptr','capacity')), 'Missing recorded private cache identity')
    need(all(signature[name]['capacity']>=row['length']+128 for name in ('native_t','native_s')),'Insufficient Native capacity')
    if backend=='eager':need('runtime' not in signature and row['runtime_stats']['entries']==0,'Eager owns hidden Graph runtime');return
    d=row['draft_runtime']; delta=d['request']; stats=row['runtime_stats']
    ptrs=[signature[name][k] for name in ('native_s','working') for k in ('key_ptr','value_ptr')]
    need(signature['runtime']['bound_ptrs']==ptrs and d['mode']==backend and stats['max_graph_entries']==128,
         'Runtime/backend/cache ownership differs')
    # RescueEngine's record has only mode/lifetime/request/scope. The baseline
    # adds these two options explicitly; never invent them on the own path.
    if not own:need(d['max_graph_entries']==128 and d['retain_query'] is quality_row,'Baseline runtime/query ownership differs')
    for k in RUNTIME:need(finite(delta[k]) and (k=='capture_host_seconds' or type(delta[k]) is int),'Invalid runtime delta')
    need(delta['shadow_updates']==row['draft_forwards'],'Runtime omitted per-step binding-copy work')
    need(type(stats['entries']) is int and 0<=stats['entries']<=128 and (delta['captures']>0)==(delta['capture_host_seconds']>0),'Capture count/time/entry ledger differs')
    keys=stats['graph_keys'];need(len(keys)==stats['entries'] and len({json_sha(key) for key in keys})==len(keys),'Graph key/entry ownership ledger differs')
    if backend=='cuda_graph':need(stats['entries']==stats['captures'],'Captured Graph entries differ from lifetime capture count')
    need(all(isinstance(key,list) and len(key)==4 and type(key[0]) is int and key[0]>0 and
         type(key[1]) is type(key[2]) is bool for key in keys),'Malformed actual Graph shape key')
    required=set()
    for t in row['trace']:
        required.update((t['workset_tokens']+j,True,own and j<=2) for j in range(1,t['proposed']+1))
        if t['accepted']==t['proposed'] and not t['terminal']:required.add((t['workset_tokens']+t['proposed']+1,False,False))
    need(required<={tuple(key[:3]) for key in keys},'Actual nonpadded work/logits/summary Graph shapes are missing')
    if not own:need(all(key[2] is False and key[3]==[[],[],[]] for key in keys),'Retrieval Graph silently owns injection/summary bindings')
    if backend=='cuda_graph':need(delta['replays']==row['draft_forwards']>0 and delta['resident_steps']==0,'Graph work silently fell back')
    else:need(delta['resident_steps']==row['draft_forwards']>0 and delta['captures']==delta['replays']==0,'Resident work silently fell back')
    if row['phase']=='cold_prime' or quality_row:
        need(all(near(delta[k],stats[k]) for k in RUNTIME),'Fresh owner borrowed old lifetime work')
        if backend=='cuda_graph':need(delta['captures']>0,'Fresh quality/timing owner borrowed preexisting Graphs')


def paired_quality(a,b):
    need(logical(a,a['arm'])==logical(b,b['arm']) and all(a[k]==b[k] for k in ('observed_rounds','queries','native_t','native_s')),
         'Independent eager/execution Native/Q/selected/features/draft/check evidence differs')


def tree_record(row,raw,arm,eos):
    # Instrumented quality does not expose an outer timing metric. Its raw
    # wrapper clock remains recorded, but it is never primary-performance data.
    checked=dict(row)
    if 'outer_seconds' not in checked:
        need(finite(raw['outer_seconds'],True),'Invalid diagnostic tree clock')
        checked['outer_seconds']=raw['outer_seconds']
    s.tree(checked,raw,arm,eos)


def owner_audit(owner,block,sequence):
    prime,warm=block[0],block[1:]; need(len(warm)==32,'Owner lacks one paid prime and32 warm requests')
    need(owner['prime_request_id']==prime['request_id'] and owner['warm_request_ids']==[r['request_id'] for r in warm] and
         [{k:r[k] for k in ('source_index','repeat')} for r in warm]==sequence,'Owner request order/ledger differs')
    need(all(r['owner_id']==owner['owner_id'] and r['owner_signature']==owner['owner_signature'] for r in block) and
         owner['engine_closed'] is owner['hook_restored'] is True and owner['maximum_live_owners']==1 and
         owner['final_runtime_stats']==warm[-1]['runtime_stats'] and owner['closed_runtime_stats']['entries']==0,'Owner survived close or changed cache identities')
    for k in RUNTIME:
        if k in owner['final_runtime_stats']:need(near(owner['closed_runtime_stats'][k],owner['final_runtime_stats'][k]),'Runtime work changed or vanished during close')
    seconds=sum(r['outer_seconds'] for r in warm); tokens=sum(r['tokens'] for r in warm)
    need(owner['warm_tokens']==tokens and near(owner['warm_outer_seconds_sum'],seconds) and owner['prime_tokens']==prime['tokens'] and
         near(owner['cold_prime_outer_seconds'],prime['outer_seconds']) and near(prime['constructor_and_detach_seconds'],owner['constructor_and_detach_seconds']) and
         near(prime['cold_complete_outer_seconds'],owner['constructor_and_detach_seconds']+prime['outer_seconds']),'Owner raw token/time sums differ')
    service=owner['constructor_and_detach_seconds']+prime['outer_seconds']+seconds+owner['close_seconds']
    need(finite(owner['constructor_and_detach_seconds']) and finite(owner['close_seconds']) and near(owner['accounted_service_seconds'],service) and
         owner['accounted_service_tokens']==tokens+prime['tokens'] and near(owner['accounted_service_tps'],(tokens+prime['tokens'])/service) and
         owner['owner_lifetime_wall_seconds']+1e-8>=service,'Owner constructor/cold/close/capture work omitted')
    if prime['arm']['family']!='full_tree' and prime['arm']['backend']!='eager':
        totals=dict.fromkeys(RUNTIME,0);keys=set()
        for row in block:
            current={json_sha(key) for key in row['runtime_stats']['graph_keys']}
            need(keys<=current,'Persistent Graph entries were silently evicted/reset')
            if prime['arm']['backend']=='cuda_graph':need(len(current-keys)==row['draft_runtime']['request']['captures'],'Late captures differ from new owner shape keys')
            keys=current
            for k in RUNTIME:
                totals[k]+=row['draft_runtime']['request'][k]
                need(near(row['runtime_stats'][k],totals[k]) and near(row['draft_runtime']['lifetime'][k],totals[k]),'Cold/late-capture lifetime ledger differs')


def points(warm,choices,summary):
    stats=summary['statistics']; output={}; contrasts=[]
    need(stats['n']==16 and stats['seed']==SEED and stats['draws']==100000 and stats['comparisons']==sum(map(len,choices.values())) and
         stats['document_index_shared_across_all_contrasts'] is True and near(stats['simultaneous_lower_quantile'],.05/stats['comparisons']), 'Wrong documented bootstrap/multiplicity contract')
    for n in LENGTHS:
        for mode in ('rescue_graph',*choices[str(n)]):
            rows=[r for r in warm if r['length']==n and r['mode']==mode]; sec=sum(r['outer_seconds'] for r in rows); tokens=sum(r['tokens'] for r in rows)
            accepted=sum(r['accepted'] for r in rows); blocks=sum(r['blocks'] for r in rows); proposed=None if any(r['proposed'] is None for r in rows) else sum(r['proposed'] for r in rows)
            metric=stats['contexts'][str(n)][mode]
            need(metric['documents']==16 and metric['tokens']==tokens and near(metric['seconds'],sec) and near(metric['pooled_tps'],tokens/sec) and
                 near(metric['acceptance_length'],accepted/blocks) and (metric['acceptance_rate'] is None if proposed is None else near(metric['acceptance_rate'],accepted/proposed)), 'Primary TPS/acceptance report point differs from raw')
            output[f'{n}:{mode}']=dict(tokens=tokens,seconds=sec,tps=tokens/sec,accepted=accepted,blocks=blocks,proposed=proposed)
    rows=stats['contrasts'];need(len(rows)==stats['comparisons'],'Report omitted a retained contrast')
    expected={(n,mode) for n in LENGTHS for mode in choices[str(n)]};need({(r['length'],r['baseline_mode']) for r in rows}==expected,'Report selected weaker opponents')
    for row in rows:
        n,mode=row['length'],row['baseline_mode']; point=100*(output[f'{n}:rescue_graph']['tps']/output[f'{n}:{mode}']['tps']-1)
        need(row['own_mode']=='rescue_graph' and row['challenger']==mode and row['documents']==16 and math.isclose(row['point_percent'],point,rel_tol=1e-11,abs_tol=1e-9), 'Paired throughput contrast point differs')
        lower=row['simultaneous_one_sided_lower95']; ci=row['pointwise_ci95']
        need(type(lower) in (float,int) and math.isfinite(lower) and len(ci)==2 and all(math.isfinite(v) for v in ci) and ci[0]<=ci[1], 'Missing finite statistical bounds')
        contrasts.append(dict(length=n,baseline_mode=mode,point_percent=point))
    reported_gate=all(r['point_percent']>0 and r['simultaneous_one_sided_lower95']>0 for r in rows)
    need(stats['statistical_gate'] is reported_gate and summary['verified_success'] is reported_gate,'Report success does not obey all-M point/lower-bound gate')
    return output,contrasts


def diagnostics(summary,owners,warm,ownq):
    """Recompute the secondary plotted tables without changing the primary."""
    for owner in owners:
        n,mode=str(owner['length']),owner['mode'];saved=summary['owner_metrics'][n][mode]
        cold=owner['constructor_and_detach_seconds']+owner['cold_prime_outer_seconds']
        need(near(saved['cold_complete_tps'],owner['prime_tokens']/cold) and near(saved['accounted_service_tps'],owner['accounted_service_tps']),
             'Report cold/accounted-service table differs from paid owner ledger')
        rows=[r for r in warm if str(r['length'])==n and r['mode']==mode];reported=summary['diagnostics'][n][mode]
        need(reported['warm_captures']==sum(r.get('draft_runtime',{}).get('request',{}).get('captures',0) for r in rows) and
             near(reported['warm_capture_seconds'],sum(r.get('draft_runtime',{}).get('request',{}).get('capture_host_seconds',0.) for r in rows)),
             'Reported late-capture diagnostic differs from all warm records')
        for stage in ('prefill_seconds','decode_seconds','cleanup_seconds'):
            if all(finite(r.get(stage)) for r in rows):need(near(reported[stage],sum(r[stage] for r in rows)/32),'Stage latency mean differs from raw')
    for n in LENGTHS:
        rows=[r for r in warm if r['length']==n and r['mode']=='rescue_graph'];saved=summary['selection_diagnostics'][str(n)]
        caps={str(c):sum(t['cap']==c for r in rows for t in r['trace']) for c in (1,2,3,4)}
        working=[t['workset_tokens'] for r in rows for t in r['trace']]
        positions=[v/o['prefix'] for r in ownq if r['length']==n and r['mode']=='rescue_graph' for o in r['observed_rounds'] for v in o['selected']]
        need(saved['cap_counts']==caps and saved['working_cache_tokens']==working and saved['selected_relative_positions']==positions,
             'Reported action/cache/position diagnostics differ from raw quality/timing')


def audit(root,analysis):
    root,analysis=Path(root).resolve(),Path(analysis).resolve()
    need(not any((root/n).exists() for n in OUTPUTS),'Never overwrite independent audit evidence')
    x=Inputs(root,analysis); c=x.collection
    need(c.get('status')=='UNION_TERMINAL_COLLECTED_NOT_YET_INDEPENDENTLY_AUDITED' and all(c.get(k) is True for k in
         ('read_only','process_and_tmux_absent','complete_file_set_rechecked','all_file_bytes_rechecked')),'Not a full immutable UNION terminal export')
    exit=x.json('PIPELINE_EXIT.json'); started=x.json('RUN_STARTED.json'); execution=x.json('EXECUTION_MANIFEST.json')
    need(type(exit.get('exit_code')) is int and exit['exit_code']==0 and c.get('pipeline_exit_code')==0 and
         not (root/'results/FAILURE.json').exists() and not (root/'FAILURE.json').exists(),'Failed/PARTIAL scientific execution')
    execution_sha=sha(x.raw('EXECUTION_MANIFEST.json'))
    need(all(v.get('execution_manifest_sha256')==execution_sha and v.get('plan_sha256')==execution['plan_sha256'] for v in (started,exit)), 'Started/exit are not bound to one frozen execution plan')
    x.pin('EXECUTION_PLAN.json',execution['plan_sha256'])
    tool_names=('fair_union_execution.py','run_fair_union_confirmation_20261020_v2.sh')
    helper_names=('rescue/__init__.py','rescue/fair_union_confirmation_data.py','rescue/fair_confirmation_data.py',
        'rescue/baseline_horizon_data.py','exp/draft_graph/receive_fair_union_stage.py')
    expected_execution={'execution_source/tools/'+n for n in tool_names}|{'execution_source/helpers/'+n for n in helper_names}
    need(set(execution['source_sha256'])==set(execution['source_path_map'])==expected_execution and
         all(Path(v).is_absolute() for v in execution['source_path_map'].values()),'Execution source receipt is incomplete/foreign')
    for name,digest in execution['source_sha256'].items():x.pin(name,digest)
    need(set(execution['tools_sha256'])=={'tools/'+n for n in tool_names},'External execution-tool map changed')
    for name,digest in execution['tools_sha256'].items():
        x.pin(name,digest);need(execution['source_sha256']['execution_source/'+name]==digest,'Executed tool differs from saved audit source')
    p=x.json('results/protocol.json'); complete=x.json('results/COMPLETE.json'); summary=x.json(str((analysis/'summary.json').relative_to(root)))
    docs,proof,manifest,gate,formal=data_audit(x,p);choices=proof['choices']
    m,r,extra,expected,sequence,owner_plan,requests=choices_plan(choices,docs); total=162+34*m+r
    need(proof['status']=='UNION_VERIFIED' and proof['M']==m and proof['R']==r and proof['expected_counts']==expected and proof['scientific_requests']==total and
         proof['extra_eager_reference_geometries']==extra and proof['extra_eager_reference_count']==r,'Frozen union M/R/counts changed')
    need(execution['union_freeze']==proof and execution['expected_counts']==expected and execution['expected_requests']==total,'Execution capsule changed the frozen union')
    entry_gate=x.json('results/GATE_PASSED.json')
    need(entry_gate.get('passed') is True and entry_gate.get('confirmation_content_opened') is False and entry_gate.get('union_freeze')==proof and
         entry_gate.get('expected_counts')==expected and entry_gate.get('expected_requests')==total and entry_gate.get('comparisons')==m and
         entry_gate.get('extra_eager_reference_geometries')==extra and entry_gate.get('source_freeze_sha256')==manifest['source_freeze_sha256'],
         'Runner did not close dual-study/source gate before confirmation access')
    need(formal['expected_counts']==expected and formal['expected_scientific_requests']==total and formal['expected_extra_eager_references']==r and
         formal['statistics']['comparisons']==m and near(formal['statistics']['simultaneous_lower_quantile'],.05/m),'Final protocol changed variable-M counts/statistical gate')
    for key,value in ASSETS.items():need(formal['own_method'][key]==value,'Pre-data method did not freeze the current Mapper/policy/cost')
    need(c['counts']==c.get('expected_counts',expected)==expected and c['M']==m and c['R']==r and c['logical_requests']==total,'Collection is partial or from another M/R plan')
    need(complete['counts']==complete['expected_counts']==expected and complete['logical_requests']==total and complete['comparisons']==m and complete['matching_geometry_count']==r and
         complete['goal_complete'] is False and all(complete.get(k) is True for k in ('passed','sources_unchanged','assets_unchanged','dependencies_unchanged',
         'quality_gate_passed','all_canonical','owners_closed','analysis_required_locally','data_unchanged','dual_study_freeze_unchanged')),'Incomplete immutable execution receipt')
    need(p['source_indices']==docs and p['lengths']==list(LENGTHS) and p['repeats']==2 and p['max_new_tokens']==128 and p['seed']==SEED and
         p['frozen_choices']==choices and p['comparisons']==m and p['expected_counts']==expected and p['expected_requests']==total and
         p['extra_eager_reference_geometries']==extra and p['shared_warm_sequence']==sequence and p['owner_order']==owner_plan and
         p['warm_order']==[{k:v for k,v in row.items() if k not in ('phase','warm_index')} for row in requests if row['phase']=='warm'],'Executed frozen cohort/owner/shared32 schedule changed')
    arms={str(n):[dict(OWN),*choices[str(n)].values()] for n in LENGTHS}; maps={n:{a['mode']:a for a in rows} for n,rows in arms.items()}
    need(p['arms']==p['arms_by_context']==arms and p['modes_by_context']=={n:list(values) for n,values in maps.items()},'Context/mode matrix lost a retained opponent')
    source=p['source_sha256'];need(source==manifest['source_sha256'] and set(REQUIRED_SOURCE)|set(CORE_SOURCE)<=set(source),'Incomplete final scientific source freeze')
    need(len([n for n in source if n.startswith('exp/specextend_bridgekv/specextend/') and n.endswith('.py')])==27,'Wrong upstream tree source count')
    for name,digest in source.items():
        for parent in ('results/source','results/data/source'):x.pin(parent+'/'+name,digest)
    for name in helper_names:
        if name.startswith('rescue/'):need(execution['source_sha256']['execution_source/helpers/'+name]==source[name],'Execution gate helper differs from frozen scientific source')
    common=dict(max_new_tokens=128,max_draft=12,refresh_every=4,prefill_precision='bfloat16',schedule='serial',profile=False,
        working_mode='packed',optimized_execution=True,deferred_probes=True,attention_backend='sdpa',coverage_execution='eager')
    for n,values in arms.items():
        for arm in values:
            cfg=p['configs'][n][arm['mode']]
            if arm['family']=='full_tree':
                wanted=dict(nodes=arm['nodes'],top_chunks=arm['top_chunks'],max_depth=10,threshold=.7,chunk_size=32,refresh_every=4,
                    flash_prefill=True,hybrid_tree=True,prefill_precision='bfloat16',mapper=False,controller=False)
            else:
                wanted=dict(common,draft_execution=arm['backend'])
                wanted.update(dict(selector='hierarchical',coverage_rows=64,refresh_policy='periodic',budget=512,recent=64,
                    coverage_candidates=1024,coverage_batch=16,coarse_chunk_size=32,injection_layers=[27],inject_tokens=64,injection_strength=.5,length_policy='risk')
                    if arm['family']=='rescue' else dict(injection_layers=[],inject_tokens=0,injection_strength=0.,length_policy='fixed',fixed_k=arm['fixed_k']))
            need(all(cfg.get(k)==v for k,v in wanted.items()),'Actual frozen configuration changed: '+n+'/'+arm['mode'])
    for section in ('source_bindings','tree_source_bindings'):
        for value in p[section].values():
            if isinstance(value,dict) and 'sha256' in value:
                matching=[n for n in source if value['path']==n or value['path'].endswith('/'+n)]
                need(len(matching)==1 and source[matching[0]]==value['sha256'],'Imported source differs from frozen bytes')
    need({'rescue.fair_union_confirmation','rescue.fair_union_confirmation_data','rescue.fair_union_confirmation_report'}<=set(p['source_bindings']) and
         {'classic.model_classic','shared.modeling_llama_kv_target'}<=set(p['tree_source_bindings']),'Missing actual union/tree import bindings')
    old,f1,w1,oldroot=study(x,p,'stage1_evidence',proof['stage1'],1)
    previous,f2,w2,_=study(x,p,'stage2_evidence',proof['stage2'],2)
    for name,digest in previous['source_sha256'].items():
        if name.startswith(('rescue/','exp/specextend_bridgekv/specextend/')) or name in ('exp/attention_horizon/selection.py','exp/selective_kv/bridge.py'):
            need(source.get(name)==digest,'Confirmation changed a frozen stage2 scientific source: '+name)
    for saved,name in (('analysis_source.py','rescue/fair_union_confirmation_report.py'),
                       ('analysis_fair_support.py','rescue/fair_confirmation_report.py'),
                       ('analysis_horizon_support.py','rescue/baseline_horizon_report.py'),
                       ('analysis_support.py','rescue/baseline_selection_report.py')):
        x.pin(str((analysis/saved).relative_to(root)),source[name])
    retained={}
    for n in LENGTHS:
        retained[str(n)]={}
        for g in GROUPS:
            for arm in (w1[str(n)][g],w2[str(n)][g]):retained[str(n)][arm['mode']]=arm
    need(retained==choices,'Retained choices do not preserve both exact development winners')
    for key,value in ASSETS.items():need(p.get(key)==value,'Changed Mapper/policy/oldcost asset SHA')
    for name,key in (('mapper','mapper_sha256'),('policy','policy_sha256'),('cost','cost_sha256')):
        need(p['assets_sha256'].get(p['args'][name])==ASSETS[key],'Asset path is not the frozen Mapper/policy/cost')
    need(all(p.get(k) is True for k in ('mapper_frozen','policy_frozen','no_training')) and p['max_live_owners']==1 and p['graph_max_entries']==128 and
         p['target_layers']==p['draft_layers']==28 and p['template_prefill_precision']=='bfloat16','Changed mandatory model/execution freeze')
    metadata=p['policy_metadata']
    need(metadata['kind']=='step_budget' and type(metadata['seed']) is int and type(metadata['hidden']) is int and metadata['hidden']>0 and
         metadata['margin']==.05 and finite(metadata['prior_weight']) and metadata['prior_weight']<=1,'Changed recorded current causal-policy metadata')
    for name,digest in old['assets_sha256'].items():need(previous['assets_sha256'].get(name)==p['assets_sha256'].get(name)==digest,'Models/tokenizer/prime differ from development')
    for key in ('nvidia_smi_gpu','torch','cuda','target_dtype','draft_dtype','prefix_dtype','tf32'):
        need(p['environment'][key]==old['environment'][key]==previous['environment'][key],'GPU/runtime/precision differs from frozen studies')
    need(p['environment']['target_dtype']=='torch.float32' and p['environment']['draft_dtype']==p['environment']['prefix_dtype']=='torch.bfloat16' and
         p['environment']['tf32'] is False and p['flash_attention']==old['flash_attention']==previous['flash_attention'] and p['flash_attention']['version']=='2.8.3','Precision/Flash dependency changed')
    eos=tokenizer_eos(x,p);need(p['tokenizer_eos_token_id']==eos and type(p['vocab_size']) is int and eos<p['vocab_size'],'Exported tokenizer/EOS differs')
    files=('reference','own_quality','baseline_quality','matching_quality','prime','runtime','owners','quality_owners','tree_raw')
    data={name:x.rows('results/'+name+'.jsonl',empty=name=='matching_quality') if name!='matching_quality' or r or 'results/matching_quality.jsonl' in x.inventory else [] for name in files}
    refs,ownq,baseq,matchq,prime,warm,owners,quality_owners,tree_raw=[data[k] for k in files]
    observed=dict(reference_requests=len(refs),own_quality_requests=len(ownq),baseline_quality_requests=len(baseq),matching_quality_requests=len(matchq),prime_requests=len(prime),timing_requests=len(warm),owners=len(owners))
    need(observed==expected and len(quality_owners)==64+m+r,'Complete dynamic request/owner matrix missing')
    allrows=refs+ownq+baseq+matchq+prime+warm
    need(len(allrows)==total and len({v['request_id'] for v in allrows})==total and all(re.fullmatch('[0-9a-f]{32}',v['request_id']) for v in allrows) and
         sorted(v['execution_index'] for v in allrows)==list(range(total)),'Scientific UUID/index ledger not bijective')
    references={}
    for j,row in enumerate(refs):
        n,i=row['length'],row['source_index'];need(row['execution_index']==j and row['mode']=='canonical_ar' and row['phase']=='reference' and row['performance_eligible'] is False,'Canonical AR must precede quality/timing')
        s.canonical(row['generated'],eos);need(all(v<p['vocab_size'] for v in row['generated']),'Out-of-vocabulary canonical token')
        need((n,i) not in references,'Duplicate canonical input');references[n,i]=row['generated']
    need(set(references)=={(n,i) for n in LENGTHS for i in docs} and x.json('results/references.json')=={f'{i}:{n}':v for (n,i),v in references.items()},'Canonical32 reference map differs')
    prime_refs=x.json('results/prime_references.json');oldrefs=x.json(oldroot+'/references.json')
    need(prime_refs=={str(n):oldrefs[f'271:{n}'] for n in LENGTHS} and all(p['prime_input_sha256'][str(n)]==old['input_sha256'][f'271:{n}'] for n in LENGTHS),'Old271 paid prime lacks same canonical/input provenance')
    for value in prime_refs.values():s.canonical(value,eos)
    rawmap={v['request_id']:v for v in tree_raw};need(len(rawmap)==len(tree_raw),'Duplicate full-tree raw ledger')
    qrows=ownq+baseq+matchq;qmap={};ownmap={};basemap={};matching={};quality_end=96+m+r
    need(sorted(v['execution_index'] for v in qrows)==list(range(32,quality_end)),'Quality did not finish before all timing owners')
    for row in qrows:
        n,i=row['length'],row['source_index'];arm=row['arm'];need(row['generated']==references[n,i] and row['tokens']==len(row['generated']) and row['output_agrees'] is True and row['performance_eligible'] is False,'Quality output/instrumentation differs from AR')
        if arm['family']=='full_tree':tree_record(row,rawmap[row['request_id']],arm,eos)
        else:quality(row,arm,eos,p['vocab_size']);runtime(row,arm,True)
        need(row['request_id'] not in qmap,'Duplicate quality request');qmap[row['request_id']]=row
    for row in ownq:
        key=row['length'],row['source_index'],row['mode'];backend='eager' if row['mode']=='rescue_eager' else 'cuda_graph'
        need(row['mode'] in ('rescue_eager','rescue_graph') and row['arm']==dict(mode=row['mode'],family='rescue',backend=backend) and row['phase']=='own_quality' and key not in ownmap,'Wrong own eager/Graph quality matrix')
        ownmap[key]=row
    need(set(ownmap)=={(n,i,mode) for n in LENGTHS for i in docs for mode in ('rescue_eager','rescue_graph')},'Missing own Native/Q pair')
    for n in LENGTHS:
        for i in docs:paired_quality(ownmap[n,i,'rescue_eager'],ownmap[n,i,'rescue_graph'])
    for row in baseq:
        key=row['length'],row['mode'];need(row['source_index']==docs[0] and row['phase']=='baseline_sentinel' and key not in basemap and choices[str(row['length'])].get(row['mode'])==row['arm'],'Missing/misregistered baseline sentinel')
        basemap[key]=row
        if row['arm']['family']=='retrieval' and row['arm']['backend']=='eager':matching[row['length'],*geometry(row['arm'])]=row
    need(set(basemap)=={(n,mode) for n in LENGTHS for mode in choices[str(n)]},'Not every retained baseline has a quality sentinel')
    for row,g in zip(matchq,extra):
        arm=dict(mode=f"matching_retrieval{g['top_chunks']}_k{g['fixed_k']}_eager",family='retrieval',backend='eager',top_chunks=g['top_chunks'],fixed_k=g['fixed_k'])
        targets=[a['mode'] for a in choices[str(g['length'])].values() if a['family']=='retrieval' and geometry(a)==geometry(arm)]
        need(row['arm']==arm and row['mode']==arm['mode'] and row['length']==g['length'] and row['source_index']==docs[0] and row['phase']=='matching_geometry_eager' and row['equivalence_targets']==targets,'Matching-eager geometry was selected by outcomes')
        matching[row['length'],*geometry(arm)]=row
    for n in LENGTHS:
        for arm in choices[str(n)].values():
            if arm['family']=='retrieval' and arm['backend']!='eager':paired_quality(matching[n,*geometry(arm)],basemap[n,arm['mode']])
    need(len({o['owner_id'] for o in quality_owners})==len(quality_owners),'Duplicate quality owner ID')
    for owner in quality_owners:
        row=qmap.get(owner['request_id']);need(row is not None and all(owner[k]==row[k] for k in ('owner_id','mode','length','source_index','owner_signature')) and
             owner['engine_closed'] is owner['hook_restored'] is True and finite(owner['close_seconds']) and owner['closed_runtime_stats']['entries']==0 and owner['final_runtime_stats']==row['runtime_stats'],'Quality owner not fully closed before timing')
    quality_gate=x.json('results/QUALITY_PASSED.json')
    need(quality_gate['passed'] is True and quality_gate['comparisons']==m and quality_gate['matching_geometry_count']==r and all(quality_gate[k] is True for k in
         ('own_all32_pairs','baseline_matching_geometry','actual_same_bfloat16_nonzero_mapper','quality_owners_closed')),'Quality gate did not include every required proof')
    sd=sorted(prime+warm,key=lambda v:v['execution_index']);signatures={}
    for j,(row,item) in enumerate(zip(sd,requests)):
        need(row['execution_index']==quality_end+j and all(row.get(k)==v for k,v in item.items()),'Timing owner/shared32 plan changed')
        arm=maps[str(row['length'])][row['mode']];ref=prime_refs[str(row['length'])] if row['phase']=='cold_prime' else references[row['length'],row['source_index']]
        need(row['arm']==arm and row['generated']==ref and row['tokens']==len(ref) and row['performance_eligible'] is row['output_agrees'] is True and
             row['observer_enabled'] is row['profile_enabled'] is False and 'timeline' not in row and 'cost_samples' not in row and
             row['mapper_executed'] is (arm['family']=='rescue') and row['controller_executed'] is (arm['family']=='rescue'),'Timing output/mode/instrumentation changed')
        need(finite(row['outer_seconds'],True) and finite(row['total_seconds'],True) and row['outer_seconds']+1e-6>=row['total_seconds'] and near(row['outer_tps'],row['tokens']/row['outer_seconds']),'Invalid complete-generate primary clock')
        if arm['family']=='full_tree':tree_record(row,rawmap[row['request_id']],arm,eos)
        else:
            trajectory(row,arm);runtime(row,arm)
            if arm['family']=='retrieval':need(row['execution_backend']==arm['backend'] and row['projected_positions']==row['draft_shadow_probes']==0,'Baseline executed Mapper/extra probes')
        if row['phase']=='warm':
            key=(arm['family'],arm.get('top_chunks'),arm.get('fixed_k',arm.get('nodes')),row['length'],row['source_index'])
            value=logical(row,arm);need(key not in signatures or signatures[key]==value,'Repeat/matched-backend trajectory/work differs');signatures[key]=value
            target=ownmap[row['length'],row['source_index'],'rescue_graph'] if arm['family']=='rescue' else basemap[row['length'],row['mode']] if row['source_index']==docs[0] else None
            if target is not None:need(logical(target,arm)==value,'Primary timing differs from instrumented quality action/work')
    need(set(rawmap)=={row['request_id'] for row in qrows+sd if row['arm']['family']=='full_tree'},'Raw full-tree records omit quality or charged overshoot')
    need(len({o['owner_id'] for o in owners})==m+2 and not {o['owner_id'] for o in owners}&{o['owner_id'] for o in quality_owners},'Timing borrowed quality ownership')
    for j,(owner,item) in enumerate(zip(owners,owner_plan)):
        need(all(owner[k]==v for k,v in item.items()) and re.fullmatch('[0-9a-f]{32}',owner['owner_id']),'Timing owner identity/geometry differs')
        owner_audit(owner,sd[33*j:33*(j+1)],sequence)
    need(summary['status']=='COMPLETE' and summary['problems']==[] and summary['observed_counts']==expected and summary['comparisons']==m and summary['expected_requests']==total,'Strict analysis is partial or another protocol')
    metrics,contrasts=points(warm,choices,summary)
    diagnostics(summary,owners,warm,ownq)
    x.finish()
    receipt=dict(schema=1,status='INDEPENDENT_UNION_RAW_AUDIT_PASSED',passed=True,created_at_utc=datetime.now(timezone.utc).isoformat(),
        M=m,R=r,counts=observed,expected_counts=expected,logical_requests=total,all_requests_verified=True,all_owners_verified=True,
        all_quality_owners_verified=True,all_timing_owners_verified=True,all_canonical=True,own_quality_native_query_equal=True,nonzero_mapper=True,
        baseline_quality_checked=True,raw_tree_verified=True,retained_union_verified=True,source_assets_verified=True,report_points_verified=True,
        independent_ci_verified=False,model_forward_reexecuted=False,learned_policy_argmax_reexecuted=False,visual_inspection_performed=False,goal_complete=False,
        protocol_sha256=sha(x.raw('results/protocol.json')),complete_sha256=sha(x.raw('results/COMPLETE.json')),
        analysis_summary_sha256=sha(x.raw(str((analysis/'summary.json').relative_to(root)))),collection_manifest_sha256=sha(x.collection_bytes),
        audit_source_file=OUTPUTS[1],audit_source_sha256=sha(AUDIT_SOURCE_BYTES),dependency_file=OUTPUTS[2],dependency_sha256=sha(SUPPORT_SOURCE_BYTES),
        analysis_input_sha256={n:v['sha256'] for n,v in x.used.items() if n in x.allowed_analysis},consumed_files=len(x.used),
        recorded_quality_scope='All32 own eager/Graph all28-layer Native and every-step Q/selection/features/draft/check digests; baseline first-doc sentinels and missing eager geometries only; all16 timed baseline documents canonical exact',
        statistical_scope='Raw whole-document pooled token/time and acceptance points independently recomputed; CI numerical values require separate exact saved-source rerun, not independent CI verification',
        metrics=metrics,contrast_points=contrasts)
    x.finish()
    for name,raw in ((OUTPUTS[1],AUDIT_SOURCE_BYTES),(OUTPUTS[2],SUPPORT_SOURCE_BYTES),
                     (OUTPUTS[0],(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())):
        with (root/name).open('xb') as handle:handle.write(raw)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--analysis',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.root,args.analysis),indent=2))
