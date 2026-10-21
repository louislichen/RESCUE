"""Two-study freeze gate and fresh union-confirmation data (stdlib import).

No candidate Parquet byte is hashed/opened by importing or freezing this
module. Preparation alone may open it, after both complete evidence chains,
all retained challengers, statistical multiplicity and source are frozen.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

from . import fair_confirmation_data as old
from . import baseline_horizon_data as horizon


SEED = 20261026
LENGTHS = (4096, 8192)
DOCUMENTS = 16
MIN_INDEX = 0
MIN_NATURAL_TOKENS = 8192
GROUPS = old.GROUPS
OLD_COHORT_FILES = old.OLD_COHORT_FILES
FIXED = copy.deepcopy(old.FIXED)
OWN_METHOD = copy.deepcopy(old.OWN_METHOD)
STAGE2_COUNTS = dict(reference_requests=32, prime_requests=76, timing_requests=2432, owners=76)
FIGURES = ('quality','tps','captures','capture_time','retained_tps','cold_service','cumulative','latency','gpu_telemetry')
REQUIRED_NEW_SOURCE = (
    'rescue/fair_union_confirmation_data.py','rescue/fair_union_confirmation.py',
    'rescue/fair_union_confirmation_report.py','tests/test_rescue_fair_union_confirmation_data.py',
    'tests/test_rescue_fair_union_confirmation.py','tests/test_rescue_fair_union_confirmation_report.py',
    'exp/draft_graph/fair_union_confirmation_20261020_preregistration.json',
    'exp/draft_graph/run_fair_union_confirmation_20261020_v1.sh')
RULE = dict(original_split='validation', minimum_source_index=0, minimum_natural_tokens=8192,
    documents=16, order='original source_index ascending; first eligible sixteen',context_lengths=list(LENGTHS),
    duplicate_rule='exclude all eleven old used/reserved cohort document SHA and exact4K/8K input SHA, including within selected cohort',
    replacement='never use generation, EOS, speed or nonzero-injection outcomes to select or replace',
    rationale='Match actual4K/8K inputs using naturally8192-token bodies; the unexecuted>=900/16384 draft was retired before any candidate access')
EXECUTION = dict(primary=old.EXECUTION['primary'],
    owner_trace='At most one live private owner; each context/mode has one separately billed old271 prime then32 warm requests; all legal captures charged',
    first_prime=old.EXECUTION['first_prime'],order=old.EXECUTION['order'],
    secondaries=old.EXECUTION['secondaries'],captures=old.EXECUTION['captures'],
    union='Keep every unique stage1+stage2 challenger at each context; never reselect after confirmation')
COUNTS_RULE = dict(reference_requests=32,own_quality_requests=64,baseline_quality_requests='M',
    matching_quality_requests='R',prime_requests='M+2',timing_requests='32*(M+2)',owners='M+2',
    scientific_requests='162+34*M+R',M='Total retained context/mode contrasts,8..16',R='Missing matching-eager geometries,0..8')
STATS_BASE = dict(bootstrap_seed=SEED,bootstrap_draws=100000,independent_unit='whole document',
    primary='pooled token/time throughput ratio',paired_indices='same document resample for all arms and both contexts',
    plotted_interval='pointwise two-sided95 percentile [0.025,0.975]',
    formal_gate='ALL M effects have positive point estimate AND positive simultaneous one-sided95 lower bound',
    simultaneous_lower_quantile_rule='.05/M',
    filtering='no capture-free filtering, outcome selection, imputation or optional stopping')
require, sha256, json_sha, input_sha = old.require, old.sha256, old.json_sha, old.input_sha
read_json, write_json, utc_now, time_value = old.read_json, old.write_json, old.utc_now, old.time_value
prompt_ids, valid_sha = old.prompt_ids, old.valid_sha


def _path(root,name):
    root=Path(root).resolve();relative=Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe evidence path')
    result=root/relative
    require(result.resolve().is_relative_to(root) and result.is_file() and not result.is_symlink(),'Missing/unsafe evidence file: '+name)
    return result


def _rows(path):
    raw=Path(path).read_bytes();require(raw and raw.endswith(b'\n'),'Incomplete raw JSONL')
    lines=raw.splitlines();require(all(v.strip() for v in lines),'Blank raw JSONL row')
    return [json.loads(v) for v in lines]


def _menu():
    result=[]
    for top in (16,32):
        for k in (1,2,4,8,12):
            for backend in ('eager','resident','graph'):
                result.append(dict(mode=f'retrieval{top}_k{k}_{backend}',family='retrieval',top_chunks=top,
                    fixed_k=k,backend='cuda_graph' if backend=='graph' else backend))
    for top in (16,32):
        for nodes in (8,32,50,64):
            result.append(dict(mode=f'tree{top}_n{nodes}',family='full_tree',top_chunks=top,nodes=nodes,
                backend='upstream_flash_hybrid',max_depth=10,threshold=.7))
    return result


def _group(arm):
    return 'full_tree' if arm['family']=='full_tree' else 'retrieval/'+('graph' if arm['backend']=='cuda_graph' else arm['backend'])


def _manifest(root,path,required):
    manifest=read_json(path);inventory=manifest.get('file_sha256',{})
    require(isinstance(inventory,dict) and inventory,'Empty final artifact manifest')
    for name,digest in inventory.items():
        require(valid_sha(digest) and sha256(_path(root,name))==digest,'Final artifact changed: '+name)
    require(set(required)<=set(inventory),'Final manifest omits a required raw/source/audit/figure artifact')
    return inventory


def _render(analysis,figures):
    receipt=read_json(analysis/'RENDER_INSPECTION.json')
    require(receipt.get('actually_inspected_by_root') is True and
        receipt.get('rendered_pngs_inspected')==list(figures),'Actual complete rendered-figure inspection required')
    for name in figures:
        for suffix in ('.png','.svg'):
            path=analysis/(name+suffix)
            require(path.is_file() and receipt.get('figure_sha256',{}).get(path.name)==sha256(path),'Inspected figure changed')


def _independent(root,results,analysis,verified,stage2):
    audit_path=root/'INDEPENDENT_RAW_AUDIT.json';audit=read_json(audit_path)
    flag='all2540_requests_verified' if stage2 else 'all538_requests_verified'
    choice='retained_union_verified' if stage2 else 'eight_frozen_choices_verified'
    require(audit.get('status')=='INDEPENDENT_RAW_AUDIT_PASSED' and audit.get(flag) is True and
        audit.get('all76_owners_verified') is True and audit.get(choice) is True,'Independent complete raw audit required')
    links=dict(frozen_selection_sha256=results/'FROZEN_SELECTION.json',analysis_summary_sha256=analysis/'summary.json')
    links['horizon_complete_sha256' if stage2 else 'baseline_complete_sha256']=results/'COMPLETE.json'
    links['horizon_protocol_sha256' if stage2 else 'baseline_protocol_sha256']=results/'protocol.json'
    if stage2: links['verified_union_sha256']=analysis/'VERIFIED_UNION.json'
    require(all(audit.get(k)==sha256(v) for k,v in links.items()),'Independent raw audit SHA links differ')
    source=_path(root,audit.get('audit_source_file','INDEPENDENT_RAW_AUDIT_SOURCE.py'))
    require(audit.get('audit_source_sha256')==sha256(source) and
        verified.get('independent_raw_audit_sha256')==sha256(audit_path) and
        verified.get('render_inspection_sha256')==sha256(analysis/'RENDER_INSPECTION.json'),
        'Raw-audit/source/render receipts are not bound by the final seal')
    require(verified.get('standalone_summary_exact_rerun') is True,'Standalone exact summary rerun required')
    if stage2:
        rerun=root/'ANALYSIS_RERUN_VERIFIED.json';value=read_json(rerun)
        require(value.get('standalone_summary_exact_rerun') is True and
            value.get('summary_sha256')==sha256(analysis/'summary.json') and
            value.get('saved_analysis_sha256')==sha256(analysis/'analysis_source.py') and
            isinstance(value.get('dependency_sha256'),dict) and value['dependency_sha256'] and
            all(valid_sha(v) for v in value['dependency_sha256'].values()) and
            verified.get('standalone_rerun_receipt_sha256')==sha256(rerun),'Standalone source/dependency rerun receipt invalid')
        require(all(sha256(_path(analysis,name))==digest for name,digest in value['dependency_sha256'].items()),
            'Standalone saved analysis dependency changed')


def _recompute(warm,protocol,stage1):
    expected=horizon.TRAIN_INDICES+horizon.OPT_INDICES+(604,)
    require(protocol.get('source_indices')==list(expected) and len(warm)==2432,'Locked16-document full warm matrix required')
    sequence=protocol.get('shared_warm_sequence',[])
    require(len(sequence)==32 and {(r.get('source_index'),r.get('repeat')) for r in sequence}==
        {(i,r) for i in expected for r in (0,1)},'Frozen shared32 sequence invalid')
    menu=_menu();require(protocol.get('arms')==menu,'Stage2 finite menu changed')
    computed={}
    for length in LENGTHS:
        candidates=[]
        for arm in menu:
            rows=[r for r in warm if r.get('length')==length and r.get('mode')==arm['mode']]
            require(len(rows)==32 and [{k:r.get(k) for k in ('source_index','repeat')} for r in rows]==sequence and
                all(r.get('arm')==arm and r.get('phase')=='warm' and r.get('performance_eligible') is True and
                    r.get('observer_enabled') is False and r.get('profile_enabled') is False and
                    type(r.get('tokens')) is int and 0<r['tokens']<=128 and
                    type(r.get('outer_seconds')) in (int,float) and math.isfinite(r['outer_seconds']) and r['outer_seconds']>0
                    for r in rows),'Incomplete/profiled/filtered32-request cell')
            def pool(values):
                tokens=sum(r['tokens'] for r in values);seconds=sum(r['outer_seconds'] for r in values)
                return dict(tokens=tokens,seconds=seconds,tps=tokens/seconds,
                    captures=sum(r.get('draft_runtime',{}).get('request',{}).get('captures',0) for r in values),
                    capture_seconds=sum(r.get('draft_runtime',{}).get('request',{}).get('capture_host_seconds',0.) for r in values))
            metric=pool(rows)
            candidates.append(dict(mode=arm['mode'],arm=copy.deepcopy(arm),warm_requests=32,warm_tokens=metric['tokens'],
                warm_seconds=metric['seconds'],warm_tps=metric['tps'],request_ids=[r['request_id'] for r in rows],
                prefix6=pool(rows[:6]),prefix32=metric))
        best=lambda values:min(values,key=lambda r:(-r['warm_tps'],r['mode']))
        groups={g:best([r for r in candidates if _group(r['arm'])==g]) for g in GROUPS}
        original=copy.deepcopy(stage1['per_context'][str(length)]['groups'])
        registered={a['mode']:a for a in menu}
        require(set(original)==set(GROUPS) and all(row.get('arm')==registered.get(row.get('mode')) and
            _group(row['arm'])==group for group,row in original.items()),'Stage1 retained choice left the original38-arm menu')
        union={g:[original[g]]+([] if original[g]['mode']==groups[g]['mode'] else [groups[g]]) for g in GROUPS}
        computed[str(length)]=dict(groups=groups,stage1_groups=original,union_by_group=union,
            union_modes=list(dict.fromkeys(row['mode'] for g in GROUPS for row in union[g])),
            ranking=sorted(candidates,key=lambda r:(-r['warm_tps'],r['mode'])),overall=best(candidates),
            overall_family_winner={f:best([r for r in candidates if r['arm']['family']==f]) for f in ('retrieval','full_tree')})
    return computed


def _extra(choices):
    result=[]
    for n in LENGTHS:
        arms=list(choices[str(n)].values())
        seen={(a['top_chunks'],a['fixed_k']) for a in arms if a['family']=='retrieval' and a['backend']=='eager'}
        for arm in arms:
            if arm['family']!='retrieval' or arm['backend']=='eager':continue
            geometry=(arm['top_chunks'],arm['fixed_k'])
            if geometry not in seen:
                result.append(dict(length=n,top_chunks=geometry[0],fixed_k=geometry[1]));seen.add(geometry)
    return result


def statistics(M=None):
    return dict(copy.deepcopy(STATS_BASE),comparisons='M' if M is None else M,
                simultaneous_lower_quantile=None if M is None else .05/M)


def validate_union_freeze(stage1_results,stage1_analysis,stage1_verified,stage1_manifest,
                          stage2_results,stage2_analysis,stage2_verified,stage2_manifest):
    """Offline dual proof; recompute all76 exact32-warm scores and union."""
    first=old.validate_baseline_freeze(stage1_results,stage1_analysis,stage1_verified,stage1_manifest)
    r1,a1=Path(stage1_results).resolve(),Path(stage1_analysis).resolve();v1=read_json(stage1_verified)
    _render(a1,('quality','tps','captures','capture_time','cold','service','selected_repeat1'))
    _independent(Path(stage1_manifest).resolve().parent,r1,a1,v1,False)
    results,analysis=Path(stage2_results).resolve(),Path(stage2_analysis).resolve()
    verified_path,manifest_path=Path(stage2_verified).resolve(),Path(stage2_manifest).resolve();root=manifest_path.parent
    require(all(p.is_relative_to(root) for p in (results,analysis,verified_path)),'Stage2 evidence escapes root')
    require(read_json(root/'PIPELINE_EXIT.json').get('exit_code')==0 and not (results/'FAILURE.json').exists(),'Stage2 has not exited successfully')
    complete=read_json(results/'COMPLETE.json');protocol=read_json(results/'protocol.json')
    frozen=read_json(results/'FROZEN_SELECTION.json');summary=read_json(analysis/'summary.json')
    union=read_json(analysis/'VERIFIED_UNION.json');analyzed=read_json(analysis/'ANALYSIS_COMPLETE.json');verified=read_json(verified_path)
    require(complete.get('passed') is True and complete.get('counts')==STAGE2_COUNTS and
        complete.get('logical_requests')==2540 and complete.get('owner_closed')==76 and
        complete.get('goal_complete') is False and all(complete.get(k) is True for k in
        ('sources_unchanged','assets_unchanged','dependencies_unchanged','all_canonical','backend_trajectories_equal','data_unchanged','stage1_verified_unchanged')),
        'Complete2540/76 exact immutable horizon run required')
    require(summary.get('status')=='COMPLETE' and summary.get('problems')==[] and
        summary.get('observed_counts')==STAGE2_COUNTS and summary.get('authoritative_union') is True and
        union.get('status')=='VERIFIED_FROZEN_UNION' and union.get('authoritative') is True and
        analyzed.get('status')=='COMPLETE','Stage2 strict authoritative analysis required')
    flags=('passed','raw_independent_audit_passed','rendered_figures_inspected','strict_analysis_complete',
           'all2540_requests_verified','all76_owners_verified','retained_union_verified','standalone_summary_exact_rerun')
    require(verified.get('status')=='VERIFIED_COMPLETE' and all(verified.get(k) is True for k in flags),'Independent full stage2 seal required')
    links=dict(horizon_complete_sha256=results/'COMPLETE.json',frozen_selection_sha256=results/'FROZEN_SELECTION.json',
        analysis_summary_sha256=analysis/'summary.json',verified_union_sha256=analysis/'VERIFIED_UNION.json')
    require(all(verified.get(k)==sha256(v) for k,v in links.items()) and
        verified.get('final_artifacts_manifest_sha256')==sha256(manifest_path),'Stage2 seal SHA links differ')
    required=[]
    for directory in (results,analysis):
        required += [str(p.relative_to(root)) for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    required += ['PIPELINE_EXIT.json','INDEPENDENT_RAW_AUDIT.json','INDEPENDENT_RAW_AUDIT_SOURCE.py','ANALYSIS_RERUN_VERIFIED.json']
    inventory=_manifest(root,manifest_path,required)
    _render(analysis,FIGURES);_independent(root,results,analysis,verified,True)
    for name,digest in analyzed.get('artifacts_sha256',{}).items():
        require(valid_sha(digest) and sha256(_path(analysis,name))==digest,'Stage2 analyzed artifact changed')
    require(analyzed.get('artifacts_sha256'),'Missing stage2 artifact inventory')
    require(protocol.get('stage1_freeze')==first and protocol.get('stage1_selection_sha256')==first['selection_freeze_sha256'] and
        complete.get('stage1_selection_sha256')==first['selection_freeze_sha256'] and
        frozen.get('stage1_selection_sha256')==first['selection_freeze_sha256'],'Stage2 is not bound to external verified stage1')
    require(frozen.get('complete_request_count')==2540 and frozen.get('no_new_confirmation_opened') is True and
        frozen.get('protocol_sha256')==sha256(results/'protocol.json') and complete.get('frozen_selection_sha256')==sha256(results/'FROZEN_SELECTION.json') and
        frozen.get('source_sha256')==protocol.get('source_sha256') and frozen.get('assets_sha256')==protocol.get('assets_sha256'),
        'Stage2 choices are not bound to protocol/source/assets')
    for name,digest in protocol.get('source_sha256',{}).items():
        require(sha256(_path(results,'source/'+name))==digest,'Stage2 executed source changed')
    p1=read_json(r1/'protocol.json')
    require(p1.get('assets_sha256') and all(protocol.get('assets_sha256',{}).get(name)==digest
        for name,digest in p1['assets_sha256'].items()),'Stage2 changed frozen model/tokenizer/prime assets')
    for field in ('mapper_sha256','policy_sha256','cost_sha256'):
        require(p1.get('frozen_rescue_anchor_for_future_confirmation',{}).get(field)==OWN_METHOD[field] and
            protocol.get('frozen_rescue_anchor_for_later_confirmation',{}).get(field)==OWN_METHOD[field],
            'Frozen nonzero Mapper/old policy/cost anchor changed')
    for field in ('nvidia_smi_gpu','torch','cuda','target_dtype','draft_dtype','prefix_dtype','tf32'):
        require(field in p1.get('environment',{}) and protocol.get('environment',{}).get(field)==p1['environment'][field],
            'Stage2 hardware/precision/runtime differs from stage1')
    for name,digest in p1['source_sha256'].items():
        if name.startswith('rescue/') or name.startswith(old.TREE_PREFIX) or name in ('exp/attention_horizon/selection.py','exp/selective_kv/bridge.py'):
            require(protocol['source_sha256'].get(name)==digest,'Stage2 changed shared scientific code')
    references=_rows(results/'reference.jsonl');prime=_rows(results/'prime.jsonl');warm=_rows(results/'runtime.jsonl');owners=_rows(results/'owners.jsonl')
    raw_tree=_rows(results/'tree_raw.jsonl')
    require((len(references),len(prime),len(warm),len(owners))==(32,76,2432,76),'Incomplete stage2 raw/owner records')
    all_requests=references+prime+warm
    require(len({r['request_id'] for r in all_requests})==2540 and sorted(r['execution_index'] for r in all_requests)==list(range(2540)),
        'Stage2 UUID/execution ledger not bijective')
    refmap={(r['source_index'],r['length']):r['generated'] for r in references}
    eos=protocol.get('tokenizer_eos_token_id')
    require(type(eos) is int and eos>=0 and all(r.get('execution_index')==j and r.get('phase')=='reference' and
        isinstance(r.get('generated'),list) and 0<len(r['generated'])<=128 and
        all(type(t) is int and t>=0 for t in r['generated']) and
        (len(r['generated'])==128 or r['generated'][-1]==eos) and
        (eos not in r['generated'] or (r['generated'][-1]==eos and eos not in r['generated'][:-1]))
        for j,r in enumerate(references)),'Stage2 canonical reference/EOS ledger invalid')
    require(len(refmap)==32 and all(r.get('generated')==refmap[(r['source_index'],r['length'])] and
        r.get('tokens')==len(r['generated']) and r.get('output_agrees') is True for r in prime+warm),'Stage2 SD outputs differ from canonical AR')
    require(all(o.get('engine_closed') is True and o.get('hook_restored') is True and o.get('maximum_live_owners')==1 and
        o.get('closed_runtime_stats',{}).get('entries')==0 for o in owners),'Stage2 owner not fully closed/private')
    require(len({o['owner_id'] for o in owners})==76,'Duplicate stage2 owner UUID')
    for owner in owners:
        block=sorted((r for r in prime+warm if r.get('owner_id')==owner['owner_id']),key=lambda r:r['execution_index'])
        require(len(block)==33 and block[0].get('phase')=='cold_prime' and block[0].get('source_index')==271 and
            all(r.get('mode')==owner['mode'] and r.get('length')==owner['length'] for r in block) and
            owner.get('prime_request_id')==block[0]['request_id'] and owner.get('warm_request_ids')==[r['request_id'] for r in block[1:]],
            'Stage2 private owner request ledger differs')
    tree_ids={r['request_id'] for r in prime+warm if r.get('arm',{}).get('family')=='full_tree'}
    require(len(raw_tree)==528 and len({r['request_id'] for r in raw_tree})==528 and
        {r['request_id'] for r in raw_tree}==tree_ids,'Stage2 full original tree ledger incomplete')
    recomputed=_recompute(warm,protocol,read_json(r1/'FROZEN_SELECTION.json'))
    require(frozen.get('per_context')==recomputed and summary.get('stage2')==recomputed and union.get('per_context')==recomputed,
        'Frozen/analysis union differs from independently recomputed exact32-request winners')
    choices,provenance={},{}
    for n in LENGTHS:
        choices[str(n)],provenance[str(n)]={},{}
        for group in GROUPS:
            for j,row in enumerate(recomputed[str(n)]['union_by_group'][group]):
                mode=row['mode'];choices[str(n)][mode]=copy.deepcopy(row['arm'])
                provenance[str(n)].setdefault(mode,dict(group=group,stages=[]))['stages'].append('stage1' if j==0 else 'stage2')
            if recomputed[str(n)]['stage1_groups'][group]['mode']==recomputed[str(n)]['groups'][group]['mode']:
                provenance[str(n)][recomputed[str(n)]['groups'][group]['mode']]['stages'].append('stage2')
    M=sum(len(v) for v in choices.values());extra=_extra(choices);R=len(extra)
    require(8<=M<=16 and 0<=R<=8 and union.get('future_contrasts')==M,'Invalid retained union multiplicity')
    counts=dict(reference_requests=32,own_quality_requests=64,baseline_quality_requests=M,
        matching_quality_requests=R,prime_requests=M+2,timing_requests=32*(M+2),owners=M+2)
    second=dict(horizon_complete_sha256=sha256(results/'COMPLETE.json'),horizon_protocol_sha256=sha256(results/'protocol.json'),
        frozen_selection_sha256=sha256(results/'FROZEN_SELECTION.json'),analysis_summary_sha256=sha256(analysis/'summary.json'),
        verified_union_sha256=sha256(analysis/'VERIFIED_UNION.json'),verified_complete_sha256=sha256(verified_path),
        final_manifest_sha256=sha256(manifest_path),independent_raw_audit_sha256=sha256(root/'INDEPENDENT_RAW_AUDIT.json'),
        render_inspection_sha256=sha256(analysis/'RENDER_INSPECTION.json'),standalone_rerun_receipt_sha256=sha256(root/'ANALYSIS_RERUN_VERIFIED.json'))
    return dict(schema=1,status='UNION_VERIFIED',choices=choices,provenance=provenance,M=M,R=R,
        extra_eager_reference_geometries=extra,extra_eager_reference_count=R,expected_counts=counts,
        scientific_requests=162+34*M+R,statistics=statistics(M),stage1=first,stage2=second,
        union_selection_sha256=second['frozen_selection_sha256'],verified_union_sha256=second['verified_union_sha256'])


def protocol_template():
    return dict(schema=1,status='TEMPLATE_NOT_EXECUTED',is_template=True,seed=SEED,
        selection_rule=copy.deepcopy(RULE),fixed=copy.deepcopy(FIXED),own_method=copy.deepcopy(OWN_METHOD),
        statistics=statistics(),execution=copy.deepcopy(EXECUTION),counts_rule=copy.deepcopy(COUNTS_RULE),
        required_source_files=list(REQUIRED_NEW_SOURCE),
        scope='Unexecuted union-confirmation template; no candidate content opened; not a random official test sample')


def _protocol(value,proof,template=False):
    expected=protocol_template()
    for key in ('schema','seed','selection_rule','fixed','own_method','execution','counts_rule','required_source_files'):
        require(value.get(key)==expected[key],'Changed final method/selection/execution/source contract: '+key)
    require(value.get('statistics')==(statistics() if template else statistics(proof['M'])),'Changed multiplicity/statistical gate')
    if not template:
        require(value.get('status')=='FROZEN_BEFORE_DATA_ACCESS' and value.get('is_template') is False and
            value.get('union_freeze')==proof and value.get('expected_counts')==proof['expected_counts'] and
            value.get('expected_scientific_requests')==proof['scientific_requests'] and
            value.get('expected_extra_eager_references')==proof['R'],'Formal protocol is not bound to complete union/counts')
        require(time_value(value['finalized_at_utc'])<=time_value(utc_now()),'Formal protocol timestamp is in future')


def make_final_protocol(template_path,proof):
    value=read_json(template_path)
    require(value.get('status')=='TEMPLATE_NOT_EXECUTED' and value.get('is_template') is True and
        proof.get('status')=='UNION_VERIFIED','Unexecuted template and verified union required')
    _protocol(value,proof,True)
    value.update(status='FROZEN_BEFORE_DATA_ACCESS',is_template=False,finalized_at_utc=utc_now(),
        union_freeze=copy.deepcopy(proof),statistics=statistics(proof['M']),expected_counts=copy.deepcopy(proof['expected_counts']),
        expected_scientific_requests=proof['scientific_requests'],expected_extra_eager_references=proof['R'])
    return value


def _sources(root,stage2_sources):
    root=Path(root).resolve();require(all((root/n).is_file() for n in REQUIRED_NEW_SOURCE),'Missing final union runner/report/tests/launcher/template')
    tree={n:s for n,s in stage2_sources.items() if n.startswith(old.TREE_PREFIX)}
    actual={str(p.relative_to(root)):sha256(p) for p in (root/old.TREE_PREFIX).rglob('*.py')}
    require(len(tree)==27 and actual==tree,'Complete unchanged27-file upstream tree required')
    paths=list((root/'rescue').glob('*.py'))+list((root/'tests').glob('test_rescue*.py'))
    paths += [root/n for n in tree]+[root/n for n in REQUIRED_NEW_SOURCE]
    paths += [root/'exp/attention_horizon/selection.py',root/'exp/selective_kv/bridge.py']
    require(all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) for p in paths),'Missing/unsafe frozen union source')
    inventory={str(p.relative_to(root)):sha256(p) for p in sorted(set(paths))}
    for name,digest in stage2_sources.items():
        if name.startswith('rescue/') or name.startswith(old.TREE_PREFIX) or name in ('exp/attention_horizon/selection.py','exp/selective_kv/bridge.py'):
            require(inventory.get(name)==digest,'Final confirmation changed scientific execution source: '+name)
    return inventory


def _args(values):
    return {k:str(Path(v).resolve()) for k,v in values.items()}


def _alias(arguments,stage):
    return dict(baseline_results=arguments[stage+'_results'],analysis_dir=arguments[stage+'_analysis'],
        verified_complete=arguments[stage+'_verified'],final_manifest=arguments[stage+'_manifest'])


def freeze_gate(output,*,metadata_paths,raw_source_manifest,raw_source,tokenizer_root,source_root,final_protocol,**arguments):
    output=Path(output).resolve();require(not output.exists(),'Fresh union gate root required');output.mkdir()
    try:
        arguments=_args(arguments);proof=validate_union_freeze(**arguments)
        formal=Path(final_protocol).resolve();_protocol(read_json(formal),proof)
        exclusions=old.exclusion_receipt(metadata_paths)
        raw_manifest=Path(raw_source_manifest).resolve()
        require(raw_manifest.as_posix().endswith('/exp/module_arch/data_v1_20261018/manifest.json'),'Old raw-source manifest required')
        expected=read_json(raw_manifest).get('source_sha256');require(valid_sha(expected),'Missing inherited validation Parquet SHA')
        raw=Path(raw_source);require(raw.is_absolute() and raw.name=='validation-00000-of-00001.parquet','Original validation Parquet path required')
        p1=read_json(Path(arguments['stage1_results'])/'protocol.json');p2=read_json(Path(arguments['stage2_results'])/'protocol.json')
        template_sources=[s for s in exclusions['metadata_sources'] if s['sha256']==p1['data_file_sha256']]
        require(template_sources,'Frozen old271 prompt-template source missing')
        first=json.loads(Path(template_sources[0]['path']).read_text().splitlines()[0])
        template=dict(prefix_ids=first['prefix_ids'],suffix_ids=first['suffix_ids']);prompt_ids(dict(document_ids=[0]*8192,**template),8192)
        source_root=Path(source_root).resolve();tokenizer_root=Path(tokenizer_root).resolve()
        inventory=_sources(source_root,p2['source_sha256']);tokenizer=old._tokenizer_inventory(tokenizer_root)
        require(all(p2.get('assets_sha256',{}).get(p2['args']['teacher'].rstrip('/')+'/'+name)==digest
            for name,digest in tokenizer.items()),'Candidate tokenizer differs from the frozen Target tokenizer/config')
        write_json(output/'EXCLUSIONS.json',exclusions)
        gate=dict(schema=1,status='FROZEN_BEFORE_DATA_ACCESS',frozen_at_utc=utc_now(),seed=SEED,candidate_content_accessed=False,
            selection_rule=copy.deepcopy(RULE),union_arguments=arguments,union_freeze=proof,
            baseline_arguments=_alias(arguments,'stage1'),horizon_arguments=_alias(arguments,'stage2'),
            final_protocol=dict(path=str(formal),sha256=sha256(formal)),source_root=str(source_root),source_sha256=inventory,
            stage2_source_sha256=p2['source_sha256'],tokenizer_root=str(tokenizer_root),tokenizer_sha256=tokenizer,
            template=template,template_sha256=json_sha(template),template_source_sha256=template_sources[0]['sha256'],
            exclusion_receipt=dict(path=str(output/'EXCLUSIONS.json'),sha256=sha256(output/'EXCLUSIONS.json')),
            raw_source=dict(path=str(raw),split='validation',expected_sha256=expected,old_manifest_path=str(raw_manifest),old_manifest_sha256=sha256(raw_manifest)),
            no_training=True,scope='First eligible sixteen unseen validation reports; new>=0/8192 rule frozen before FIRST raw hash/open; not random official test')
        write_json(output/'GATE_SOURCE_FREEZE.json',gate);return gate
    except BaseException as error:
        write_json(output/'GATE_FAILURE.json',dict(status='FAILED_BEFORE_DATA_ACCESS',error_type=type(error).__name__,error=str(error)));raise


def validate_source_freeze(gate_path):
    gate=read_json(gate_path)
    require(gate.get('status')=='FROZEN_BEFORE_DATA_ACCESS' and gate.get('candidate_content_accessed') is False and
        gate.get('seed')==SEED and gate.get('selection_rule')==RULE,'Unfrozen/changed union data gate')
    require(time_value(gate['frozen_at_utc'])<=time_value(utc_now()),'Gate timestamp is in future')
    require(gate.get('baseline_arguments')==_alias(gate['union_arguments'],'stage1') and
        gate.get('horizon_arguments')==_alias(gate['union_arguments'],'stage2'),'Legacy evidence aliases differ from verified dual arguments')
    require(validate_union_freeze(**gate['union_arguments'])==gate['union_freeze'],'Complete union chain changed')
    require(sha256(gate['final_protocol']['path'])==gate['final_protocol']['sha256'],'Formal union protocol changed')
    _protocol(read_json(gate['final_protocol']['path']),gate['union_freeze'])
    require(_sources(gate['source_root'],gate['stage2_source_sha256'])==gate['source_sha256'] and
        old._tokenizer_inventory(gate['tokenizer_root'])==gate['tokenizer_sha256'],'Frozen union source/tokenizer changed')
    require(json_sha(gate['template'])==gate['template_sha256'],'Template changed')
    path=Path(gate['exclusion_receipt']['path']);require(sha256(path)==gate['exclusion_receipt']['sha256'],'Exclusions changed')
    exclusions=read_json(path)
    require(old.exclusion_receipt([s['path'] for s in exclusions['metadata_sources']])==exclusions,'Old metadata changed')
    raw=gate['raw_source'];require(sha256(raw['old_manifest_path'])==raw['old_manifest_sha256'] and
        read_json(raw['old_manifest_path']).get('source_sha256')==raw['expected_sha256'],'Old raw manifest changed')
    return gate,exclusions


def _raw_rows(path):
    import pyarrow.parquet as parquet
    with parquet.ParquetFile(path) as source:
        index=0
        for batch in source.iter_batches(columns=['report']):
            for row in batch.to_pylist():yield index,row['report'];index+=1


def _load_tokenizer(root):
    return old._load_tokenizer(root)


def _copy_proofs(output,arguments):
    result={}
    for stage in ('stage1','stage2'):
        manifest=Path(arguments[stage+'_manifest']);root=manifest.parent;dest=output/'receipts'/stage
        entries=dict(read_json(manifest)['file_sha256'])
        entries[str(manifest.relative_to(root))]=sha256(manifest)
        verified=Path(arguments[stage+'_verified']);entries[str(verified.relative_to(root))]=sha256(verified)
        for name,digest in entries.items():
            source=_path(root,name);require(sha256(source)==digest,'Proof changed during portable copy')
            target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            require(sha256(target)==digest,'Portable proof differs')
        for field in ('results','analysis','verified','manifest'):
            result[stage+'_'+field]=str((dest/Path(arguments[stage+'_'+field]).relative_to(root)).relative_to(output))
    return result


def prepare_confirmation(gate_path):
    gate_path=Path(gate_path).resolve();gate,exclusions=validate_source_freeze(gate_path)
    output=gate_path.parent/'data';require(not output.exists(),'Fresh confirmation data root; never resume or replace');output.mkdir()
    started=utc_now();write_json(output/'DATA_PREP_STARTED.json',dict(started_at_utc=started,source_freeze_sha256=sha256(gate_path)))
    try:
        raw=Path(gate['raw_source']['path']);before=sha256(raw)  # FIRST candidate byte access.
        require(before==gate['raw_source']['expected_sha256'],'Candidate validation Parquet SHA differs')
        tokenizer=_load_tokenizer(gate['tokenizer_root']);docs=set(exclusions['document_sha256'])
        prefixes={n:set(v) for n,v in exclusions['input_sha256'].items()};selected=[];details=[];skipped=[];previous=-1
        iterator=_raw_rows(raw)
        try:
            for index,text in iterator:
                require(type(index) is int and index==previous+1 and isinstance(text,str) and text,'Scan must cover every validation row from0')
                previous=index;digest=hashlib.sha256(text.encode()).hexdigest()
                if digest in docs:skipped.append(dict(source_index=index,document_sha256=digest,reason='document_duplicate'));continue
                ids=tokenizer.encode(text,add_special_tokens=False)
                require(isinstance(ids,list) and all(type(t) is int and t>=0 for t in ids),'Invalid tokenizer IDs')
                if len(ids)<8192:skipped.append(dict(source_index=index,document_sha256=digest,original_tokens=len(ids),reason='natural_tokens_below8192'));continue
                row=dict(split='confirmation',original_split='validation',source_index=index,document_sha256=digest,
                    original_tokens=len(ids),document_ids=ids[:8192],**gate['template'])
                hashes={str(n):input_sha(prompt_ids(row,n)) for n in LENGTHS}
                if any(hashes[n] in prefixes[n] for n in hashes):
                    skipped.append(dict(source_index=index,document_sha256=digest,original_tokens=len(ids),input_sha256=hashes,reason='prefix_duplicate'));continue
                selected.append(row);details.append(dict(source_index=index,original_split='validation',document_sha256=digest,original_tokens=len(ids),input_sha256=hashes))
                docs.add(digest)
                for n,v in hashes.items():prefixes[n].add(v)
                if len(selected)==16:break
        finally:
            if hasattr(iterator,'close'):iterator.close()
        require(len(selected)==16,'Fewer than16 eligible documents; no replacement/resampling')
        require(sha256(raw)==before,'Candidate raw changed');validate_source_freeze(gate_path)
        with (output/'confirmation.jsonl').open('x') as handle:
            for row in selected:handle.write(json.dumps(row)+'\n')
        receipt_paths={}
        for name,source in dict(source_freeze=gate_path,exclusions=Path(gate['exclusion_receipt']['path']),
            final_protocol=Path(gate['final_protocol']['path']),old_raw_source_manifest=Path(gate['raw_source']['old_manifest_path'])).items():
            target=output/'receipts'/(name+'.json');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            receipt_paths[str(target.relative_to(output))]=sha256(target)
        proof_paths=_copy_proofs(output,gate['union_arguments'])
        for name,digest in gate['source_sha256'].items():
            source=_path(Path(gate['source_root']),name);require(sha256(source)==digest,'Source changed before portable copy')
            target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        manifest=dict(schema=1,status='PREPARED_NOT_EXECUTED',seed=SEED,selection_rule=copy.deepcopy(RULE),
            union_freeze=copy.deepcopy(gate['union_freeze']),source_freeze_sha256=sha256(gate_path),final_protocol_sha256=gate['final_protocol']['sha256'],
            source_frozen_at_utc=gate['frozen_at_utc'],candidate_first_open_not_before_utc=started,prepared_at_utc=utc_now(),
            source=dict(path=str(raw),sha256=before,original_split='validation'),source_sha256=gate['source_sha256'],
            tokenizer_sha256=gate['tokenizer_sha256'],template_sha256=gate['template_sha256'],exclusion_receipt_sha256=gate['exclusion_receipt']['sha256'],
            selected=details,skipped=skipped,dataset_file='confirmation.jsonl',dataset_sha256=sha256(output/'confirmation.jsonl'),
            receipt_sha256=receipt_paths,portable_proof_arguments=proof_paths,source_snapshot_root='source',
            no_outcomes_read=True,no_model_generation=True,no_training=True,
            scope='Deterministic first eligible16 validation reports, natural8192 body,4K/8K inputs; not random official test',
            offline_limit='Portable receipts verify saved first-eligible scan, not independent raw-text retokenization')
        write_json(output/'DATASET_MANIFEST.json',manifest);verify_dataset_manifest(output);return manifest
    except BaseException as error:
        write_json(output/'DATA_PREP_FAILURE.json',dict(status='FAILED_PRESERVE_PARTIAL',error_type=type(error).__name__,error=str(error)));raise


def verify_dataset_manifest(data_root):
    root=Path(data_root).resolve();manifest=read_json(root/'DATASET_MANIFEST.json')
    require(manifest.get('status')=='PREPARED_NOT_EXECUTED' and manifest.get('seed')==SEED and
        manifest.get('selection_rule')==RULE and all(manifest.get(k) is True for k in ('no_outcomes_read','no_model_generation','no_training')),
        'Invalid prepared union dataset contract')
    require(not (root/'DATA_PREP_FAILURE.json').exists(),'Failed preparation cannot authorize inference')
    times=[time_value(manifest[k]) for k in ('source_frozen_at_utc','candidate_first_open_not_before_utc','prepared_at_utc')]
    require(times==sorted(times),'Raw access preceded source freeze')
    require(manifest.get('dataset_file')=='confirmation.jsonl' and sha256(root/'confirmation.jsonl')==manifest.get('dataset_sha256'),'Prepared IDs changed')
    for name,digest in manifest.get('receipt_sha256',{}).items():require(sha256(_path(root,name))==digest,'Portable receipt changed')
    gate=read_json(root/'receipts/source_freeze.json');formal=read_json(root/'receipts/final_protocol.json');exclusions=read_json(root/'receipts/exclusions.json')
    require(sha256(root/'receipts/source_freeze.json')==manifest.get('source_freeze_sha256') and
        sha256(root/'receipts/final_protocol.json')==manifest.get('final_protocol_sha256') and
        sha256(root/'receipts/exclusions.json')==manifest.get('exclusion_receipt_sha256'),'Portable gate linkage changed')
    arguments={k:_path(root,v) if k.endswith(('_manifest','_verified')) else (root/v).resolve()
        for k,v in manifest['portable_proof_arguments'].items()}
    require(all(Path(p).is_relative_to(root) for p in arguments.values()),'Portable proof directory escapes data root')
    proof=validate_union_freeze(**arguments);require(proof==manifest.get('union_freeze')==gate.get('union_freeze'),'Portable dual proof differs')
    _protocol(formal,proof)
    require(gate.get('selection_rule')==RULE and gate.get('seed')==SEED and gate.get('candidate_content_accessed') is False and
        gate.get('status')=='FROZEN_BEFORE_DATA_ACCESS' and gate.get('source_sha256')==manifest.get('source_sha256') and
        gate.get('tokenizer_sha256')==manifest.get('tokenizer_sha256') and gate.get('template_sha256')==manifest.get('template_sha256') and
        gate.get('raw_source',{}).get('expected_sha256')==manifest.get('source',{}).get('sha256'),'Portable source/selection chain differs')
    for name,digest in manifest['source_sha256'].items():require(sha256(_path(root,'source/'+name))==digest,'Portable final source changed')
    rows=_rows(root/'confirmation.jsonl');details=manifest.get('selected',[])
    require(len(rows)==len(details)==16,'Exactly sixteen prepared documents required')
    selected={};previous=-1
    for row,detail in zip(rows,details):
        index=row.get('source_index');require(type(index) is int and index>previous and row.get('split')=='confirmation' and
            row.get('original_split')=='validation' and type(row.get('original_tokens')) is int and row['original_tokens']>=8192 and
            len(row.get('document_ids',[]))==8192 and valid_sha(row.get('document_sha256')),'Invalid natural first-eligible selection')
        previous=index;hashes={str(n):input_sha(prompt_ids(row,n)) for n in LENGTHS}
        require(detail==dict(source_index=index,original_split='validation',document_sha256=row['document_sha256'],original_tokens=row['original_tokens'],input_sha256=hashes) and
            dict(prefix_ids=row['prefix_ids'],suffix_ids=row['suffix_ids'])==gate['template'],'Prepared token/template receipt differs')
        selected[index]=detail
    skips=manifest.get('skipped',[]);require(all(isinstance(v,dict) and type(v.get('source_index')) is int for v in skips),'Malformed scan skips')
    skipped={v['source_index']:v for v in skips}
    require(len(skipped)==len(skips) and not set(skipped)&set(selected) and
        set(skipped)|set(selected)==set(range(previous+1)),'First-eligible scan has holes/duplicates/extra rows')
    docs=set(exclusions['document_sha256']);prefixes={n:set(v) for n,v in exclusions['input_sha256'].items()}
    for index in range(previous+1):
        row=selected.get(index) or skipped[index];digest=row.get('document_sha256');require(valid_sha(digest),'Missing scan document hash')
        if index in selected:
            require(digest not in docs and all(v not in prefixes[n] for n,v in row['input_sha256'].items()),'Old/reserved/within-cohort duplicate')
            docs.add(digest)
            for n,v in row['input_sha256'].items():prefixes[n].add(v)
        elif row.get('reason')=='document_duplicate':require(digest in docs,'Unjustified document duplicate skip')
        elif row.get('reason')=='natural_tokens_below8192':require(digest not in docs and type(row.get('original_tokens')) is int and 0<=row['original_tokens']<8192,'Unjustified short skip')
        elif row.get('reason')=='prefix_duplicate':
            hashes=row.get('input_sha256',{});require(digest not in docs and type(row.get('original_tokens')) is int and row['original_tokens']>=8192 and
                set(hashes)==set(map(str,LENGTHS)) and all(valid_sha(v) for v in hashes.values()) and
                any(v in prefixes[n] for n,v in hashes.items()),'Unjustified prefix skip')
        else:require(False,'Unknown/outcome-dependent scan skip')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    freeze=sub.add_parser('freeze')
    for name in ('output','stage1-results','stage1-analysis','stage1-verified','stage1-manifest','stage2-results','stage2-analysis','stage2-verified','stage2-manifest',
                 'raw-source-manifest','raw-source','tokenizer-root','source-root','final-protocol'):
        freeze.add_argument('--'+name,type=Path,required=True)
    freeze.add_argument('--metadata-file',type=Path,action='append',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--gate',type=Path,required=True)
    verify=sub.add_parser('verify');verify.add_argument('--data-root',type=Path,required=True)
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='freeze':args['metadata_paths']=args.pop('metadata_file');result=freeze_gate(**args)
    elif command=='prepare':result=prepare_confirmation(args['gate'])
    else:result=verify_dataset_manifest(args['data_root'])
    print(json.dumps(dict(status=result['status'],selected_documents=len(result.get('selected',[]))),indent=2))


if __name__=='__main__':main()
