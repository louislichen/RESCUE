"""Portable CPU audit of the pre-frozen stage1+stage2 challenger union.

No torch, models, raw corpus or remote path reads. Repeats are pooled within
whole documents; one shared resample covers every arm and both input lengths.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

if __package__:
    from . import fair_confirmation_report as support
    from . import baseline_horizon_report as horizon
else:
    import analysis_fair_support as support
    import analysis_horizon_support as horizon

LENGTHS = (4096, 8192)
GROUPS = ('retrieval/eager', 'retrieval/resident', 'retrieval/graph', 'full_tree')
SEED, DRAWS, DOCUMENTS, REPEATS, OUTPUT_LIMIT = 20261026, 100000, 16, 2, 128
ASSETS = dict(support.ASSETS)
positive, nonnegative, near = support.positive, support.nonnegative, support.near
sha = support.sha
pool_document = support.pool_document
OWN = dict(mode='rescue_graph', family='rescue', backend='cuda_graph')
REQUIRED_SOURCE = ('rescue/fair_union_confirmation_data.py', 'rescue/fair_union_confirmation.py',
    'rescue/fair_union_confirmation_report.py', 'tests/test_rescue_fair_union_confirmation_data.py',
    'tests/test_rescue_fair_union_confirmation.py', 'tests/test_rescue_fair_union_confirmation_report.py',
    'exp/draft_graph/fair_union_confirmation_20261020_preregistration.json',
    'exp/draft_graph/run_fair_union_confirmation_20261020_v1.sh')
SELECTION_RULE = dict(original_split='validation', minimum_source_index=0,
    minimum_natural_tokens=8192, documents=16,
    order='original source_index ascending; first eligible sixteen',
    context_lengths=[4096, 8192],
    duplicate_rule='exclude all eleven old used/reserved cohort document SHA and exact4K/8K input SHA, including within selected cohort',
    replacement='never use generation, EOS, speed or nonzero-injection outcomes to select or replace',
    rationale='Match actual4K/8K inputs using naturally8192-token bodies; the unexecuted>=900/16384 draft was retired before any candidate access')
STATS_BASE=dict(bootstrap_seed=SEED,bootstrap_draws=DRAWS,independent_unit='whole document',
    primary='pooled token/time throughput ratio',paired_indices='same document resample for all arms and both contexts',
    plotted_interval='pointwise two-sided95 percentile [0.025,0.975]',
    formal_gate='ALL M effects have positive point estimate AND positive simultaneous one-sided95 lower bound',
    simultaneous_lower_quantile_rule='.05/M',
    filtering='no capture-free filtering, outcome selection, imputation or optional stopping')
COUNTS_RULE=dict(reference_requests=32,own_quality_requests=64,baseline_quality_requests='M',
    matching_quality_requests='R',prime_requests='M+2',timing_requests='32*(M+2)',owners='M+2',
    scientific_requests='162+34*M+R',M='Total retained context/mode contrasts,8..16',R='Missing matching-eager geometries,0..8')
EXECUTION=dict(primary=support.EXECUTION_CONTRACT['primary'],
    owner_trace='At most one live private owner; each context/mode has one separately billed old271 prime then32 warm requests; all legal captures charged',
    first_prime=support.EXECUTION_CONTRACT['first_prime'],order=support.EXECUTION_CONTRACT['order'],
    secondaries=support.EXECUTION_CONTRACT['secondaries'],captures=support.EXECUTION_CONTRACT['captures'],
    union='Keep every unique stage1+stage2 challenger at each context; never reselect after confirmation')
OLD_COHORT_FILES=('exp/long_system/data/train.jsonl','exp/long_system/data/development.jsonl','exp/long_system/data/test.jsonl',
    'exp/module_opt/data_v1_20261018/fit.jsonl','exp/module_opt/data_v1_20261018/tune.jsonl','exp/module_opt/data_v1_20261018/warmup.jsonl','exp/module_opt/data_v1_20261018/confirmation.jsonl',
    'exp/module_arch/data_v1_20261018/fit.jsonl','exp/module_arch/data_v1_20261018/tune.jsonl','exp/module_arch/data_v1_20261018/warmup.jsonl','exp/module_arch/data_v1_20261018/confirmation.jsonl')


class Evidence(horizon.Evidence):
    """All saved paths must stay inside the archive, even before JSON reads."""


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def statistics_contract(m):
    return dict(STATS_BASE,comparisons=m,simultaneous_lower_quantile=.05/m)


def prepared_dataset(evidence):
    """Reconstruct stored input prefixes and justify every first-eligible skip."""
    data=Evidence(evidence.root/'data');manifest=data.json('DATASET_MANIFEST.json')
    if (manifest.get('status')!='PREPARED_NOT_EXECUTED' or manifest.get('selection_rule')!=SELECTION_RULE or manifest.get('seed')!=SEED or
        any(manifest.get(k) is not True for k in ('no_outcomes_read','no_model_generation','no_training'))):
        data.problem('Unused validation0/8192 preparation contract changed')
    if (data.root/'DATA_PREP_FAILURE.json').exists():data.problem('Failed partial preparation must never authorize confirmation')
    try:
        times=[datetime.fromisoformat(manifest[k]) for k in ('source_frozen_at_utc','candidate_first_open_not_before_utc','prepared_at_utc')]
        if any(t.tzinfo is None for t in times) or times!=sorted(times):raise ValueError('Freeze ordering')
    except (TypeError,KeyError,ValueError):data.problem('Fresh candidate access did not follow source freeze')
    if manifest.get('dataset_file')!='confirmation.jsonl' or manifest.get('source_snapshot_root')!='source':data.problem('Unexpected prepared data/source filenames')
    data.pin('confirmation.jsonl',manifest.get('dataset_sha256'))
    pins=manifest.get('receipt_sha256',{})
    required=('source_freeze','exclusions','final_protocol','old_raw_source_manifest')
    if not isinstance(pins,dict) or not all('receipts/'+n+'.json' in pins for n in required):
        data.problem('Portable pre-data freeze receipt pins incomplete');pins=pins if isinstance(pins,dict) else {}
    for name,digest in pins.items():data.pin(name,digest)
    for field,name in (('source_freeze_sha256','source_freeze'),('final_protocol_sha256','final_protocol'),('exclusion_receipt_sha256','exclusions')):
        if manifest.get(field)!=pins.get('receipts/'+name+'.json'):data.problem('Prepared receipt-chain link differs: '+field)
    gate=data.json('receipts/source_freeze.json');formal=data.json('receipts/final_protocol.json');excluded=data.json('receipts/exclusions.json')
    proof=manifest.get('union_freeze',{});choices=proof.get('choices',{})
    try:choices=normalized_choices(choices);m=sum(len(v) for v in choices.values())
    except (ValueError,KeyError,TypeError):choices={};m=0;data.problem('No verified complete dual-stage union before data access')
    expected=expected_counts(choices) if choices else {};extra=extra_geometries(choices) if choices else []
    if (proof.get('status')!='UNION_VERIFIED' or proof.get('M')!=m or proof.get('R')!=len(extra) or proof.get('expected_counts')!=expected or
        proof.get('scientific_requests')!=162+34*m+len(extra) or proof.get('statistics')!=(statistics_contract(m) if m else None) or
        proof.get('extra_eager_reference_geometries')!=extra or proof.get('extra_eager_reference_count')!=len(extra)):
        data.problem('Frozen union multiplicity/count/statistics differ')
    if (gate.get('status')!='FROZEN_BEFORE_DATA_ACCESS' or gate.get('candidate_content_accessed') is not False or gate.get('seed')!=SEED or
        gate.get('selection_rule')!=SELECTION_RULE or gate.get('union_freeze')!=proof or
        gate.get('source_sha256')!=manifest.get('source_sha256') or gate.get('tokenizer_sha256')!=manifest.get('tokenizer_sha256') or
        gate.get('template_sha256')!=manifest.get('template_sha256') or
        gate.get('raw_source',{}).get('expected_sha256')!=manifest.get('source',{}).get('sha256')):
        data.problem('Source/tokenizer/template/dual-study pre-access freeze changed')
    wanted=dict(schema=1,seed=SEED,selection_rule=SELECTION_RULE,fixed=support.FIXED_CONTRACT,own_method=support.OWN_CONTRACT,
        execution=EXECUTION,counts_rule=COUNTS_RULE,required_source_files=list(REQUIRED_SOURCE))
    if (formal.get('status')!='FROZEN_BEFORE_DATA_ACCESS' or formal.get('is_template') is not False or formal.get('union_freeze')!=proof or
        formal.get('statistics')!=(statistics_contract(m) if m else None) or formal.get('expected_counts')!=expected or
        formal.get('expected_scientific_requests')!=proof.get('scientific_requests') or any(formal.get(k)!=v for k,v in wanted.items())):
        data.problem('Formal frozen union method/execution/statistics/count/source contract differs')
    template=gate.get('template',{})
    if json_sha(template)!=manifest.get('template_sha256'):data.problem('Frozen prompt-template SHA changed')
    sources=manifest.get('source_sha256',{})
    if not sources:data.problem('Prepared source snapshot inventory missing')
    for name,digest in sources.items():data.pin('source/'+name,digest)
    metadata=excluded.get('metadata_sources',[])
    if (len(metadata)!=11 or {s.get('relative_name') for s in metadata}!=set(OLD_COHORT_FILES) or
        excluded.get('status')!='OLD_DATA_METADATA_ONLY' or any(not isinstance(s.get('sha256'),str) or len(s['sha256'])!=64 or
        not isinstance(s.get('rows'),list) or not s['rows'] for s in metadata)):
        data.problem('All eleven exact old used/reserved metadata cohorts must be excluded')
    saved_old_rows=[r for s in metadata for r in s.get('rows',[]) if isinstance(r,dict)]
    if (set(excluded.get('document_sha256',[]))!={r.get('document_sha256') for r in saved_old_rows} or any(
        set(excluded.get('input_sha256',{}).get(str(n),[]))!={r['input_sha256'][str(n)] for r in saved_old_rows if str(n) in r.get('input_sha256',{})} for n in LENGTHS)):
        data.problem('Eleven-cohort saved metadata rows do not reconstruct the exclusion SHA sets')
    deny=set(excluded.get('document_sha256',[]));prefixes={str(n):set(excluded.get('input_sha256',{}).get(str(n),[])) for n in LENGTHS}
    rows=data.rows('confirmation.jsonl');details=manifest.get('selected',[]);docs=[];inputs={};selected={};previous=-1
    if len(rows)!=16 or len(details)!=16:data.problem('Exactly16 first-eligible unused validation documents required')
    for row,detail in zip(rows,details):
        index=row.get('source_index');digest=row.get('document_sha256');body=row.get('document_ids',[])
        valid_sha=lambda value:isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
        if (type(index) is not int or index<=previous or row.get('split')!='confirmation' or row.get('original_split')!='validation' or
            type(row.get('original_tokens')) is not int or row['original_tokens']<8192 or not valid_sha(digest) or
            not isinstance(body,list) or len(body)!=8192 or any(type(t) is not int or t<0 for t in body) or
            dict(prefix_ids=row.get('prefix_ids'),suffix_ids=row.get('suffix_ids'))!=template):
            data.problem('Prepared first-eligible natural8192 body/template differs');continue
        previous=index;docs.append(index);hashes={}
        for n in LENGTHS:
            left,right=template.get('prefix_ids',[]),template.get('suffix_ids',[]);take=n-len(left)-len(right)
            if not 0<take<=len(body) or any(type(t) is not int or t<0 for t in left+right):data.problem('Exact input prefix cannot be reconstructed');continue
            hashes[str(n)]=hashlib.sha256(json.dumps(left+body[:take]+right).encode()).hexdigest();inputs[index,n]=hashes[str(n)]
        expected_detail=dict(source_index=index,original_split='validation',document_sha256=digest,original_tokens=row['original_tokens'],input_sha256=hashes)
        if detail!=expected_detail:data.problem('Prepared selected input/document fingerprints changed')
        selected[index]=expected_detail
    skipped=manifest.get('skipped',[])
    if not isinstance(skipped,list) or any(not isinstance(r,dict) or type(r.get('source_index')) is not int for r in skipped):
        data.problem('Malformed full first-eligible scan receipt');skipped=[]
    skipped_map={r['source_index']:r for r in skipped}
    if len(skipped_map)!=len(skipped) or set(skipped_map)&set(selected) or set(skipped_map)|set(selected)!=set(range(previous+1)):
        data.problem('Validation0 first-eligible scan has omitted/duplicate/extra indices')
    for index in range(previous+1):
        row=selected.get(index) or skipped_map.get(index,{})
        digest=row.get('document_sha256');reason=row.get('reason');hashes=row.get('input_sha256',{})
        if not valid_sha(digest):data.problem('Scan row lacks valid document SHA');continue
        if index in selected:
            if digest in deny or any(v in prefixes[n] for n,v in hashes.items()):data.problem('Old/reserved/selected duplicate passed the frozen scan')
            deny.add(digest)
            for n,v in hashes.items():prefixes[n].add(v)
        elif reason=='document_duplicate':
            if digest not in deny:data.problem('Unjustified document-duplicate scan skip')
        elif reason=='natural_tokens_below8192':
            if digest in deny or type(row.get('original_tokens')) is not int or not 0<=row['original_tokens']<8192:data.problem('Unjustified short-document scan skip')
        elif reason=='prefix_duplicate':
            if (digest in deny or type(row.get('original_tokens')) is not int or row['original_tokens']<8192 or set(hashes)!=set(prefixes) or
                any(not valid_sha(v) for v in hashes.values()) or not any(v in prefixes[n] for n,v in hashes.items())):
                data.problem('Unjustified prefix-duplicate scan skip')
        else:data.problem('Unknown/outcome-dependent selection skip reason')
    # The data helper saves complete portable study trees too; bind all eight
    # proof paths safely, never follow the original arguments in the gate.
    proof_paths=manifest.get('portable_proof_arguments',{})
    if set(proof_paths)!={stage+'_'+field for stage in ('stage1','stage2') for field in ('results','analysis','verified','manifest')}:
        data.problem('Portable dual-study data-freeze proof paths incomplete')
    for key,name in proof_paths.items():
        try:
            path=data.path(name)
            if key.endswith(('_verified','_manifest')) and (path is None or not path.is_file()):data.problem('Portable study receipt missing: '+key)
            elif path is None or not path.is_dir():
                if not key.endswith(('_verified','_manifest')):data.problem('Portable study directory missing: '+key)
        except (TypeError,ValueError):data.problem('Unsafe portable study proof path')
    # Rehash the preparation-time study copies too. The main runner's copied
    # trees are checked independently below; neither copy may merely assert
    # that the other one passed a gate.
    for stage in ('stage1','stage2'):
        if not all(stage+'_'+f in proof_paths for f in ('results','analysis','verified','manifest')):continue
        saved=proof.get(stage,{})
        manifest_name=proof_paths[stage+'_manifest'];signed=data.json(manifest_name);parent=Path(manifest_name).parent
        inventory=signed.get('file_sha256',{})
        if not inventory:data.problem('Empty preparation-time study artifact manifest')
        for name,digest in inventory.items():data.pin(str(parent/name),digest)
        data.pin(manifest_name,saved.get('final_manifest_sha256'));data.pin(proof_paths[stage+'_verified'],saved.get('verified_complete_sha256'))
        r,a=proof_paths[stage+'_results'],proof_paths[stage+'_analysis']
        fields=((('baseline_complete_sha256','COMPLETE.json'),('baseline_protocol_sha256','protocol.json'),('selection_freeze_sha256','FROZEN_SELECTION.json')) if stage=='stage1' else
            (('horizon_complete_sha256','COMPLETE.json'),('horizon_protocol_sha256','protocol.json'),('frozen_selection_sha256','FROZEN_SELECTION.json')))
        for field,name in fields:data.pin(r+'/'+name,saved.get(field))
        data.pin(a+'/summary.json',saved.get('analysis_summary_sha256'))
        data.pin(a+('/VERIFIED_WINNERS.json' if stage=='stage1' else '/VERIFIED_UNION.json'),saved.get('verified_winners_sha256' if stage=='stage1' else 'verified_union_sha256'))
    evidence.problems.extend('Dataset: '+p for p in data.problems)
    return dict(documents=docs,input_sha256=inputs,manifest=manifest,gate=gate,formal=formal,proof=proof,choices=choices)


def geometry(arm):
    return arm['top_chunks'], arm['fixed_k']


def normalized_choices(choices):
    if not isinstance(choices, dict) or set(choices) != set(map(str, LENGTHS)):
        raise ValueError('Both contexts of the frozen union required')
    result = {}
    for n in map(str, LENGTHS):
        values = choices[n]
        if not isinstance(values, dict) or not 4 <= len(values) <= 8:
            raise ValueError('Four to eight unique challengers per context required')
        if any(not isinstance(a, dict) or mode != a.get('mode') or mode == OWN['mode'] for mode, a in values.items()):
            raise ValueError('Union must be keyed by exact unique challenger mode')
        menu = {a['mode']: a for a in horizon.support.menu()}
        if any(menu.get(mode) != arm for mode, arm in values.items()):
            raise ValueError('Challenger is not one of the fixed38 baseline configurations')
        groups=[horizon.group(a) for a in values.values()]
        if (any(not 1<=groups.count(g)<=2 for g in GROUPS) or
            groups!=sorted(groups,key=GROUPS.index)):
            raise ValueError('No baseline family/backend may be omitted')
        result[n] = dict(values)
    return result


def extra_geometries(choices):
    result = []
    for length in LENGTHS:
        values = choices[str(length)].values()
        seen = {geometry(a) for a in values if a.get('family') == 'retrieval' and a['backend'] == 'eager'}
        for arm in values:
            if arm.get('family') != 'retrieval' or arm['backend'] == 'eager':
                continue
            key = geometry(arm)
            if key not in seen:
                result.append(dict(length=length, top_chunks=key[0], fixed_k=key[1]))
                seen.add(key)
    return result


def expected_counts(choices):
    m = sum(len(values) for values in normalized_choices(choices).values())
    r, owners = len(extra_geometries(choices)), m + 2
    return dict(reference_requests=32, own_quality_requests=64, baseline_quality_requests=m,
        matching_quality_requests=r, prime_requests=owners, timing_requests=32 * owners, owners=owners)


def synchronized_statistics(cells, own_modes, challengers, *, seed=SEED, draws=DRAWS, formal=True):
    """Pooled token/time ratios; .05/M lower bound with one shared doc draw."""
    import numpy as np
    if seed != SEED or draws != DRAWS:
        raise ValueError('Frozen seed20261026 and100000 draws required')
    if set(cells) != set(map(str, LENGTHS)) or set(challengers) != set(cells):
        raise ValueError('Both contexts and every frozen challenger required')
    sizes = {len(v) for ctx in cells.values() for v in ctx.values()}
    if len(sizes) != 1 or not sizes or not 0 < next(iter(sizes)) <= DOCUMENTS:
        raise ValueError('Common whole-document alignment required')
    count, m = sizes.pop(), sum(len(v) for v in challengers.values())
    if not 8 <= m <= 16:
        raise ValueError('Frozen union has between8 and16 contrasts')
    index = np.random.default_rng(seed).integers(0, count, (draws, count)) if count > 1 else None
    result = dict(n=count, seed=seed, draws=draws, comparisons=m,
        document_index_shared_across_all_contrasts=True,
        simultaneous_method='Bonferroni M one-sided percentile-bootstrap lower bounds',
        simultaneous_lower_quantile=.05 / m, contexts={}, contrasts=[], statistical_gate=False)
    distributions = {}
    for n in map(str, LENGTHS):
        own, modes = own_modes[n], challengers[n]
        if set(cells[n]) != {own, *modes.values()} or len(set(modes.values())) != len(modes):
            raise ValueError('No missing, duplicate or extra comparison arm allowed')
        result['contexts'][n], distributions[n] = {}, {}
        for mode, documents in cells[n].items():
            if any(not positive(d.get('tokens')) or not positive(d.get('seconds')) for d in documents):
                raise ValueError('Positive actual saved token/time pairs required')
            tokens = np.array([d['tokens'] for d in documents], float)
            seconds = np.array([d['seconds'] for d in documents], float)
            sample = tokens[index].sum(1) / seconds[index].sum(1) if index is not None else None
            distributions[n][mode] = sample
            metric = dict(documents=count, tokens=int(tokens.sum()), seconds=float(seconds.sum()),
                pooled_tps=float(tokens.sum()/seconds.sum()),
                tps_ci95=None if sample is None else np.percentile(sample, [2.5, 97.5]).tolist())
            for label, denominator in (('acceptance_length', 'blocks'), ('acceptance_rate', 'proposed')):
                good = all(nonnegative(d.get('accepted')) and positive(d.get(denominator)) for d in documents)
                num = np.array([d['accepted'] for d in documents], float) if good else None
                den = np.array([d[denominator] for d in documents], float) if good else None
                qs = num[index].sum(1)/den[index].sum(1) if good and index is not None else None
                metric[label] = float(num.sum()/den.sum()) if good else None
                metric[label+'_ci95'] = None if qs is None else np.percentile(qs, [2.5,97.5]).tolist()
            result['contexts'][n][mode] = metric
        for key, mode in modes.items():
            point = 100.*(result['contexts'][n][own]['pooled_tps']/result['contexts'][n][mode]['pooled_tps']-1.)
            sample = 100.*(distributions[n][own]/distributions[n][mode]-1.) if index is not None else None
            result['contrasts'].append(dict(length=int(n), challenger=key, own_mode=own, baseline_mode=mode,
                documents=count, point_percent=point,
                pointwise_ci95=None if sample is None else np.percentile(sample,[2.5,97.5]).tolist(),
                simultaneous_one_sided_lower95=None if sample is None or not formal else float(np.quantile(sample,.05/m))))
    result['statistical_gate'] = bool(formal and count == DOCUMENTS and len(result['contrasts']) == m and all(
        c['point_percent'] > 0 and c['simultaneous_one_sided_lower95'] is not None and
        c['simultaneous_one_sided_lower95'] > 0 for c in result['contrasts']))
    return result


def partial_context_statistics(cells, own, challengers, length):
    """Per-context completed pairs can be drawn, never grant formal success."""
    import numpy as np
    sizes = {len(v) for v in cells.values()}
    if len(sizes) != 1 or not sizes or not next(iter(sizes)):
        return {}, []
    count = sizes.pop()
    index = np.random.default_rng(SEED).integers(0,count,(DRAWS,count)) if count > 1 else None
    values, samples = {}, {}
    for mode, docs in cells.items():
        t, s = np.array([d['tokens'] for d in docs],float), np.array([d['seconds'] for d in docs],float)
        if not np.isfinite(t).all() or not np.isfinite(s).all() or (t<=0).any() or (s<=0).any():
            raise ValueError('Invalid partial document data')
        x = t[index].sum(1)/s[index].sum(1) if index is not None else None
        samples[mode] = x
        row = dict(documents=count,tokens=int(t.sum()),seconds=float(s.sum()),pooled_tps=float(t.sum()/s.sum()),
            tps_ci95=None if x is None else np.percentile(x,[2.5,97.5]).tolist())
        for label, den in (('acceptance_length','blocks'),('acceptance_rate','proposed')):
            good=all(nonnegative(d.get('accepted')) and positive(d.get(den)) for d in docs)
            a=np.array([d['accepted'] for d in docs],float) if good else None
            b=np.array([d[den] for d in docs],float) if good else None
            dist=a[index].sum(1)/b[index].sum(1) if good and index is not None else None
            row[label]=float(a.sum()/b.sum()) if good else None
            row[label+'_ci95']=None if dist is None else np.percentile(dist,[2.5,97.5]).tolist()
        values[mode]=row
    contrasts=[]
    for key,mode in challengers.items():
        dist=100.*(samples[own]/samples[mode]-1.) if index is not None else None
        contrasts.append(dict(length=length,challenger=key,own_mode=own,baseline_mode=mode,documents=count,
            point_percent=100.*(values[own]['pooled_tps']/values[mode]['pooled_tps']-1.),
            pointwise_ci95=None if dist is None else np.percentile(dist,[2.5,97.5]).tolist(),
            simultaneous_one_sided_lower95=None,diagnostic_partial_only=True))
    return values,contrasts


def registered_schedule(choices, documents):
    rng=random.Random(SEED)
    sequence=[]
    for repeat in (0,1):
        order=list(documents);rng.shuffle(order)
        sequence.extend(dict(source_index=d,repeat=repeat) for d in order)
    owners=[]
    for n in LENGTHS:
        modes=[OWN['mode'],*choices[str(n)]]
        random.Random(SEED+n).shuffle(modes)
        owners.extend(dict(mode=mode,length=n) for mode in modes)
    warm=[dict(mode=o['mode'],length=o['length'],**s) for o in owners for s in sequence]
    return owners,warm,sequence


def _portable_study(evidence,protocol,key,kind):
    """Rehash a complete saved study tree, not its original remote paths."""
    entry=protocol.get(key,{})
    expected_root='stage1_evidence' if kind==1 else 'stage2_evidence'
    if entry.get('root')!=expected_root:
        evidence.problem('Missing portable '+expected_root);return {}
    base=Evidence(evidence.root/expected_root)
    def file_digest(name):
        path=base.path(name)
        if not path.is_file():base.problem('Missing signed study file '+name);return None
        return sha(path)
    def rel(field):
        name=entry.get(field)
        if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts:
            base.problem('Unsafe/missing study path '+field);return '__missing__'
        return name
    result,analysis,verified_path,manifest_path=map(rel,('baseline_results','analysis_dir','verified_complete','final_manifest'))
    final=base.json(manifest_path);inventory=final.get('file_sha256',{})
    if not inventory:base.problem('Empty final artifact manifest')
    for name,digest in inventory.items():base.pin(name,digest)
    evidence.pin(entry.get('pin_index'),entry.get('pin_index_sha256'))
    pins=evidence.json(entry.get('pin_index','__missing__')).get('original_to_saved',{})
    if not pins:base.problem('Missing original-to-portable index')
    for value in pins.values():evidence.pin(value.get('saved_path'),value.get('sha256'))
    complete=base.json(result+'/COMPLETE.json');frozen=base.json(result+'/FROZEN_SELECTION.json')
    old=base.json(result+'/protocol.json');summary=base.json(analysis+'/summary.json')
    analyzed=base.json(analysis+'/ANALYSIS_COMPLETE.json');verified=base.json(verified_path)
    pipeline=base.json('PIPELINE_EXIT.json')
    if type(pipeline.get('exit_code')) is not int or pipeline.get('exit_code')!=0 or 'PIPELINE_EXIT.json' not in inventory:
        base.problem('Signed pipeline exit0 absent')
    expected=dict(reference_requests=6 if kind==1 else 32,prime_requests=76,timing_requests=456 if kind==1 else 2432,owners=76)
    total=538 if kind==1 else 2540
    if (complete.get('passed') is not True or complete.get('counts')!=expected or complete.get('logical_requests')!=total or
        complete.get('owner_closed')!=76 or complete.get('goal_complete') is not False or any(complete.get(k) is not True for k in
        ('sources_unchanged','assets_unchanged','dependencies_unchanged','all_canonical','backend_trajectories_equal'))):
        base.problem('Complete study execution/exactness/immutability matrix differs')
    if (summary.get('status')!='COMPLETE' or summary.get('problems')!=[] or analyzed.get('status')!='COMPLETE'):
        base.problem('Strict study analysis incomplete')
    artifacts=analyzed.get('artifacts_sha256',analyzed.get('file_sha256',{}))
    if not artifacts:base.problem('No rendered analysis artifact pins')
    for name,digest in artifacts.items():base.pin(analysis+'/'+name,digest)
    figures=('quality','tps','captures','capture_time','cold','service','selected_repeat1') if kind==1 else (
        'quality','tps','captures','capture_time','retained_tps','cold_service','cumulative','latency','gpu_telemetry')
    for name in figures:
        for suffix in ('.png','.svg'):
            if analysis+'/'+name+suffix not in inventory:base.problem('Missing signed rendered study figure '+name+suffix)
    flags=('passed','raw_independent_audit_passed','rendered_figures_inspected','strict_analysis_complete',
        'all538_requests_verified' if kind==1 else 'all2540_requests_verified','all76_owners_verified',
        'eight_frozen_choices_verified' if kind==1 else 'retained_union_verified')
    if verified.get('status')!='VERIFIED_COMPLETE' or any(verified.get(k) is not True for k in flags):
        base.problem('Independent study raw/render verification receipt missing')
    independent=base.json('INDEPENDENT_RAW_AUDIT.json')
    if (independent.get('status')!='INDEPENDENT_RAW_AUDIT_PASSED' or independent.get(flags[-3]) is not True or
        independent.get('all76_owners_verified') is not True or independent.get(flags[-1]) is not True):
        base.problem('Independent raw audit lacks full request/owner/selection verification')
    audit_source=independent.get('audit_source_file','INDEPENDENT_RAW_AUDIT_SOURCE.py')
    base.pin(audit_source,independent.get('audit_source_sha256'))
    render=base.json(analysis+'/RENDER_INSPECTION.json')
    if render.get('actually_inspected_by_root') is not True or render.get('rendered_pngs_inspected')!=list(figures):
        base.problem('Actual complete figure inspection receipt differs')
    for name in figures:
        for suffix in ('.png','.svg'):base.pin(analysis+'/'+name+suffix,render.get('figure_sha256',{}).get(name+suffix))
    if (verified.get('independent_raw_audit_sha256')!=file_digest('INDEPENDENT_RAW_AUDIT.json') or
        verified.get('render_inspection_sha256')!=file_digest(analysis+'/RENDER_INSPECTION.json') or
        verified.get('standalone_summary_exact_rerun') is not True):base.problem('Final seal does not bind independent audit/render/exact rerun')
    complete_key='baseline_complete_sha256' if kind==1 else 'horizon_complete_sha256'
    protocol_key='baseline_protocol_sha256' if kind==1 else 'horizon_protocol_sha256'
    for field,name in ((complete_key,result+'/COMPLETE.json'),(protocol_key,result+'/protocol.json'),
        ('frozen_selection_sha256',result+'/FROZEN_SELECTION.json'),('analysis_summary_sha256',analysis+'/summary.json')):
        base.pin(name,independent.get(field))
    for field,name in ((complete_key,result+'/COMPLETE.json'),('frozen_selection_sha256',result+'/FROZEN_SELECTION.json'),
        ('analysis_summary_sha256',analysis+'/summary.json'),('final_artifacts_manifest_sha256',manifest_path)):
        base.pin(name,verified.get(field))
    if kind==2:
        rerun=base.json('ANALYSIS_RERUN_VERIFIED.json')
        base.pin('ANALYSIS_RERUN_VERIFIED.json',verified.get('standalone_rerun_receipt_sha256'))
        if rerun.get('standalone_summary_exact_rerun') is not True:base.problem('Stage2 standalone exact rerun is absent')
        base.pin(analysis+'/summary.json',rerun.get('summary_sha256'));base.pin(analysis+'/analysis_source.py',rerun.get('saved_analysis_sha256'))
        dependencies=rerun.get('dependency_sha256',{})
        if not dependencies:base.problem('Stage2 saved dependency rerun pins absent')
        for name,digest in dependencies.items():base.pin(analysis+'/'+name,digest)
    winnerfile='VERIFIED_WINNERS.json' if kind==1 else 'VERIFIED_UNION.json'
    winners=base.json(analysis+'/'+winnerfile)
    status='VERIFIED_FROZEN' if kind==1 else 'VERIFIED_FROZEN_UNION'
    if winners.get('status')!=status or winners.get('authoritative') is not True:
        base.problem('Authoritative frozen study choices missing')
    base.pin(analysis+'/'+winnerfile,verified.get('verified_winners_sha256' if kind==1 else 'verified_union_sha256'))
    if kind==2:base.pin(analysis+'/'+winnerfile,independent.get('verified_union_sha256'))
    if frozen.get('protocol_sha256')!=file_digest(result+'/protocol.json') or complete.get('frozen_selection_sha256')!=file_digest(result+'/FROZEN_SELECTION.json'):
        base.problem('Frozen selection not bound to complete execution/protocol')
    raw=base.rows(result+'/runtime.jsonl')
    if len(raw)!=expected['timing_requests']:base.problem('Complete primary study raw ledger missing')
    if protocol.get('flash_attention')!=old.get('flash_attention'):
        base.problem('Actual Flash Python/extension receipt differs')
    assets=protocol.get('assets_sha256',{})
    if not old.get('assets_sha256') or any(assets.get(name)!=digest for name,digest in old.get('assets_sha256',{}).items()):
        base.problem('Model/tokenizer/old-prime assets differ from study')
    current=protocol.get('source_sha256',{});sources=old.get('source_sha256',{})
    for name,value in sources.items():base.pin(result+'/source/'+name,value)
    shared=('exp/attention_horizon/selection.py','exp/selective_kv/bridge.py')
    if not sources or any(current.get(name)!=digest for name,digest in sources.items() if name.startswith('rescue/') or name in shared):
        base.problem('Shared algorithm/runtime missing or changed after study')
    tree=lambda d:{k:v for k,v in d.items() if k.startswith('exp/specextend_bridgekv/specextend/')}
    if tree(current)!=tree(sources):base.problem('Complete full-tree source changed')
    for field in ('nvidia_smi_gpu','torch','cuda','target_dtype','draft_dtype','prefix_dtype','tf32'):
        if protocol.get('environment',{}).get(field)!=old.get('environment',{}).get(field):base.problem('Study GPU/runtime/precision differs: '+field)
    evidence.problems.extend(('Stage1: ' if kind==1 else 'Stage2: ')+p for p in base.problems)
    return dict(root=base.root,result=result,protocol=old,selection=frozen,summary=summary,winners=winners,
        complete=complete,raw=raw,verified=verified,manifest=final,entry=entry,
        selection_sha256=file_digest(result+'/FROZEN_SELECTION.json'),protocol_sha256=file_digest(result+'/protocol.json'))


def audit_union_chain(evidence,protocol,choices):
    first=_portable_study(evidence,protocol,'stage1_evidence',1)
    second=_portable_study(evidence,protocol,'stage2_evidence',2)
    if not first or not second:return {}
    original=first['selection'];later=second['selection']
    proof=protocol.get('union_freeze',{})
    if proof.get('stage1')!=second['protocol'].get('stage1_freeze'):
        evidence.problem('Union pre-data proof does not preserve the complete external stage1 gate')
    for stage,study in (('stage1',first),('stage2',second)):
        pinned=proof.get(stage,{})
        b=Evidence(study['root']);r=study['result'];a=study['entry']['analysis_dir']
        paths=dict(frozen_selection_sha256=r+'/FROZEN_SELECTION.json',analysis_summary_sha256=a+'/summary.json',
            verified_complete_sha256=study['entry']['verified_complete'],final_manifest_sha256=study['entry']['final_manifest'],
            independent_raw_audit_sha256='INDEPENDENT_RAW_AUDIT.json',render_inspection_sha256=a+'/RENDER_INSPECTION.json')
        if stage=='stage1':
            paths.update(selection_freeze_sha256=r+'/FROZEN_SELECTION.json',baseline_complete_sha256=r+'/COMPLETE.json',
                baseline_protocol_sha256=r+'/protocol.json',verified_winners_sha256=a+'/VERIFIED_WINNERS.json',pipeline_exit_sha256='PIPELINE_EXIT.json')
            # The legacy proof predates added render/raw pins; those are audited
            # through VERIFIED_COMPLETE above, not invented as legacy fields.
            paths.pop('frozen_selection_sha256');paths.pop('independent_raw_audit_sha256');paths.pop('render_inspection_sha256')
        else:
            paths.update(horizon_complete_sha256=r+'/COMPLETE.json',horizon_protocol_sha256=r+'/protocol.json',
                verified_union_sha256=a+'/VERIFIED_UNION.json',standalone_rerun_receipt_sha256='ANALYSIS_RERUN_VERIFIED.json')
        for field,path in paths.items():b.pin(path,pinned.get(field))
        evidence.problems.extend(stage+' proof: '+p for p in b.problems)
    if proof.get('union_selection_sha256')!=second['selection_sha256'] or proof.get('verified_union_sha256')!=proof.get('stage2',{}).get('verified_union_sha256'):
        evidence.problem('Pre-data proof union-selection SHA differs')
    # Independently recover stage1 exact pooled6 choices from every candidate.
    for n in LENGTHS:
        for g in GROUPS:
            values=[]
            for arm in first['protocol'].get('arms',[]):
                if horizon.group(arm)!=g:continue
                rows=[r for r in first['raw'] if r.get('length')==n and r.get('mode')==arm['mode']]
                if len(rows)==6 and all(positive(r.get('tokens')) and positive(r.get('outer_seconds')) for r in rows):
                    values.append((sum(r['tokens'] for r in rows)/sum(r['outer_seconds'] for r in rows),arm['mode']))
            receipt=original.get('per_context',{}).get(str(n),{}).get('groups',{}).get(g,{})
            if not values or receipt.get('mode')!=min(values,key=lambda x:(-x[0],x[1]))[1]:evidence.problem('Stage1 exact6warm original winner changed')
    docs=second['protocol'].get('source_indices',[])
    if docs!=list(horizon.DOCS):evidence.problem('Stage2 reused16-document cohort changed')
    computed=horizon.select_union(second['raw'],docs,original,second['protocol'].get('arms'))
    if later.get('per_context')!=computed:evidence.problem('Stage2 exact32warm winners/original-retaining union differ')
    if later.get('stage1_selection_sha256')!=first['selection_sha256']:
        evidence.problem('Stage2 not bound to immutable stage1 selection')
    for n in map(str,LENGTHS):
        cell=later.get('per_context',{}).get(n,{})
        independent={}
        for g in GROUPS:
            old=original.get('per_context',{}).get(n,{}).get('groups',{}).get(g)
            entries=cell.get('union_by_group',{}).get(g,[])
            if not entries or entries[0]!=old or cell.get('stage1_groups',{}).get(g)!=old:
                evidence.problem('Original stage1 challenger was removed/weakened/replaced')
            for receipt in entries:independent[receipt['mode']]=receipt['arm']
        if choices.get(n)!=independent or cell.get('union_modes')!=list(independent):
            evidence.problem('Executed union omits/adds/reorders a frozen challenger')
    prime=evidence.json('prime_references.json')
    refs=Evidence(first['root']).json(first['result']+'/references.json')
    if any(prime.get(str(n))!=refs.get(f'271:{n}') or len(prime.get(str(n),[]))!=128 or
        protocol.get('prime_input_sha256',{}).get(str(n))!=first['protocol'].get('input_sha256',{}).get(f'271:{n}') for n in LENGTHS):
        evidence.problem('Old271 paid prime differs from verified stage1 canonical input/output')
    return dict(stage1=first,stage2=second)


def audit_metadata(evidence,protocol,dataset,choices):
    m=sum(len(v) for v in choices.values());extra=extra_geometries(choices);expected=expected_counts(choices)
    arms={str(n):[dict(OWN),*choices[str(n)].values()] for n in LENGTHS};manifest=dataset['manifest'];proof=dataset['proof']
    total=162+34*m+len(extra)
    if (protocol.get('arms')!=arms or protocol.get('arms_by_context')!=arms or
        protocol.get('modes_by_context')!={n:[a['mode'] for a in v] for n,v in arms.items()} or
        protocol.get('frozen_choices')!=choices or protocol.get('comparisons')!=m or protocol.get('source_indices')!=dataset['documents'] or
        protocol.get('lengths')!=list(LENGTHS) or protocol.get('repeats')!=2 or protocol.get('max_new_tokens')!=128 or protocol.get('seed')!=SEED or
        protocol.get('expected_counts')!=expected or protocol.get('expected_requests')!=total or protocol.get('extra_eager_reference_geometries')!=extra):
        evidence.problem('Frozen union arm/context/cohort/count plan differs')
    if (protocol.get('union_freeze')!=proof or protocol.get('source_freeze_sha256')!=manifest.get('source_freeze_sha256') or
        protocol.get('final_protocol_sha256')!=manifest.get('final_protocol_sha256') or protocol.get('dataset_sha256')!=manifest.get('dataset_sha256') or
        protocol.get('dataset_manifest')!='data/DATASET_MANIFEST.json' or protocol.get('portable_data_root')!='data' or protocol.get('selection_rule')!=SELECTION_RULE):
        evidence.problem('Executed cohort/source is not bound to the pre-data dual-study freeze')
    evidence.pin('data/DATASET_MANIFEST.json',protocol.get('dataset_manifest_sha256'))
    gate=evidence.json('GATE_PASSED.json')
    if (gate.get('passed') is not True or gate.get('confirmation_content_opened') is not False or gate.get('union_freeze')!=proof or
        gate.get('expected_counts')!=expected or gate.get('expected_requests')!=total or gate.get('comparisons')!=m or
        gate.get('extra_eager_reference_geometries')!=extra or gate.get('source_freeze_sha256')!=manifest.get('source_freeze_sha256')):
        evidence.problem('Runner did not verify complete union/source gate before reading fresh data')
    assets=protocol.get('assets_sha256',{});args=protocol.get('args',{})
    for name,digest in (('mapper',ASSETS['mapper_sha256']),('policy',ASSETS['policy_sha256']),('cost',ASSETS['cost_sha256']),
        ('freeze_receipt',manifest.get('source_freeze_sha256')),('dataset_manifest',protocol.get('dataset_manifest_sha256')),('data_file',manifest.get('dataset_sha256'))):
        if assets.get(args.get(name))!=digest:evidence.problem('Executed immutable asset-path SHA differs: '+name)
    if (any(protocol.get(k)!=v for k,v in ASSETS.items()) or any(protocol.get(k) is not True for k in ('mapper_frozen','policy_frozen','no_training')) or
        protocol.get('max_live_owners')!=1 or protocol.get('template_prefill_precision')!='bfloat16'):
        evidence.problem('Mandatory genuine Mapper/oldcost/policy/execution freeze differs')
    common=dict(max_new_tokens=128,max_draft=12,refresh_every=4,prefill_precision='bfloat16',schedule='serial',profile=False,
        working_mode='packed',optimized_execution=True,deferred_probes=True,attention_backend='sdpa',coverage_execution='eager')
    for n,values in arms.items():
        for arm in values:
            cfg=protocol.get('configs',{}).get(n,{}).get(arm['mode'],{})
            if arm['family']=='full_tree':
                wanted=dict(nodes=arm['nodes'],top_chunks=arm['top_chunks'],max_depth=10,threshold=.7,chunk_size=32,refresh_every=4,
                    flash_prefill=True,hybrid_tree=True,prefill_precision='bfloat16',mapper=False,controller=False)
            else:
                wanted=dict(common,draft_execution=arm['backend'])
                wanted.update(dict(selector='hierarchical',coverage_rows=64,refresh_policy='periodic',budget=512,recent=64,
                    coverage_candidates=1024,coverage_batch=16,coarse_chunk_size=32,injection_layers=[27],inject_tokens=64,injection_strength=.5,length_policy='risk')
                    if arm['family']=='rescue' else dict(injection_layers=[],inject_tokens=0,injection_strength=0.,length_policy='fixed',fixed_k=arm['fixed_k']))
            if any(cfg.get(k)!=v for k,v in wanted.items()):evidence.problem('Actual frozen configuration differs: '+n+'/'+arm['mode'])
    env=protocol.get('environment',{})
    if ('H100' not in env.get('gpu_name','') or env.get('total_memory',0)<80_000_000_000 or
        env.get('target_dtype')!='torch.float32' or env.get('draft_dtype')!='torch.bfloat16' or env.get('prefix_dtype')!='torch.bfloat16' or env.get('tf32') is not False):
        evidence.problem('Single H10080GB/FP32Target/BF16Draft+prefix/TF32off environment differs')
    inventory=protocol.get('source_sha256',{})
    if not inventory or inventory!=manifest.get('source_sha256') or not set(REQUIRED_SOURCE)<=set(inventory):evidence.problem('Pre-data frozen final source inventory differs/incomplete')
    if len([n for n in inventory if n.startswith('exp/specextend_bridgekv/specextend/') and n.endswith('.py')])!=27:
        evidence.problem('Exactly27 unchanged upstream full-tree source files required')
    for name,digest in inventory.items():evidence.pin('source/'+name,digest)
    bindings=protocol.get('source_bindings',{})
    if not {'rescue.fair_union_confirmation','rescue.fair_union_confirmation_data','rescue.fair_union_confirmation_report'}<=set(bindings):
        evidence.problem('Actual final runner/data/report imported source bindings missing')
    for name,value in bindings.items():
        relative=name.replace('.','/')+'.py'
        if not str(value.get('path','')).endswith('/'+relative) or inventory.get(relative)!=value.get('sha256'):evidence.problem('Actual imported source differs: '+name)
    tree=protocol.get('tree_source_bindings',{})
    if not {'classic.model_classic','shared.modeling_llama_kv_target'}<=set(tree):evidence.problem('Actual full-tree model/attention source bindings missing')
    for value in tree.values():
        if 'sha256' in value and inventory.get(value.get('path'))!=value['sha256']:evidence.problem('Actual full-tree source SHA differs')
        if 'namespace_paths' in value and (not value['namespace_paths'] or any(not n.startswith('exp/specextend_bridgekv/specextend/') or '..' in Path(n).parts for n in value['namespace_paths'])):
            evidence.problem('Actual full-tree import namespace escaped frozen source')
    flash=protocol.get('flash_attention',{})
    if flash.get('version')!='2.8.3' or not str(flash.get('files',{}).get('flash_attn_2_cuda',{}).get('path','')).endswith('.so'):
        evidence.problem('Official Flash2.8.3 compiled-extension receipt missing')


def _analyze(directory):
    evidence=Evidence(directory);protocol=evidence.json('protocol.json');dataset=prepared_dataset(evidence)
    choices=dataset['choices']
    if not choices:
        return dict(schema=1,status='PARTIAL',verified_success=False,problems=evidence.problems,statistics={},observed_counts={},comparisons=0)
    audit_metadata(evidence,protocol,dataset,choices)
    audit_union_chain(evidence,protocol,choices)
    outer=Evidence(Path(directory).parent);manifest=outer.json('EVIDENCE_MANIFEST.json').get('file_sha256',{})
    if not manifest:outer.problem('Complete exported raw/source SHA manifest missing')
    for name,digest in manifest.items():outer.pin(name,digest)
    evidence.problems.extend('Export: '+p for p in outer.problems)
    result=audit_execution(evidence,protocol,dataset,choices)
    receipt=evidence.json('COMPLETE.json')
    if (receipt.get('passed') is not True or receipt.get('counts')!=result['expected_counts'] or receipt.get('expected_counts')!=result['expected_counts'] or
        receipt.get('logical_requests')!=result['expected_requests'] or receipt.get('comparisons')!=result['comparisons'] or receipt.get('matching_geometry_count')!=len(extra_geometries(choices)) or
        any(receipt.get(k) is not True for k in ('sources_unchanged','assets_unchanged','dependencies_unchanged','quality_gate_passed','all_canonical','owners_closed',
            'analysis_required_locally','data_unchanged','dual_study_freeze_unchanged')) or receipt.get('goal_complete') is not False):
        evidence.problem('Terminal completion/immutable data/source/dual gate receipt invalid')
    if (evidence.root/'FAILURE.json').exists():evidence.problem('Failure marker exists; this root remains PARTIAL')
    status='COMPLETE' if not evidence.problems else 'PARTIAL'
    common=result['paired_complete_documents'];wmap=result.pop('_wmap');stats={}
    if status=='COMPLETE' and len(common)==16:
        cells={str(n):{a['mode']:[pool_document([wmap[a['mode'],n,d,r] for r in (0,1)]) for d in common] for a in result['arms'][str(n)]} for n in LENGTHS}
        stats=synchronized_statistics(cells,result['own_modes'],result['challengers'])
    else:
        stats=dict(n=len(common),seed=SEED,draws=DRAWS,comparisons=result['comparisons'],contexts={},contrasts=[],statistical_gate=False,
            partial_context_diagnostics=True,formal_intervals_available=False,document_index_shared_across_all_contrasts=False)
        for n in LENGTHS:
            docs=result['paired_complete_documents_by_context'][str(n)]
            if not docs:continue
            cells={a['mode']:[pool_document([wmap[a['mode'],n,d,r] for r in (0,1)]) for d in docs] for a in result['arms'][str(n)]}
            values,contrasts=partial_context_statistics(cells,result['own_modes'][str(n)],result['challengers'][str(n)],n)
            stats['contexts'][str(n)]=values;stats['contrasts'].extend(contrasts)
    stats['per_context_n']={n:len(v) for n,v in result['paired_complete_documents_by_context'].items()}
    return dict(schema=1,status=status,verified_success=status=='COMPLETE' and stats.get('statistical_gate') is True,
        problems=evidence.problems,statistics=stats,dataset_sha256=dataset['manifest'].get('dataset_sha256'),
        union_selection_sha256=dataset['proof'].get('union_selection_sha256'),
        evidence_scope='Portable raw/source/receipt audit only; no model or original corpus reread',**result)


def analyze(directory):
    try:return _analyze(directory)
    except (OSError,KeyError,TypeError,ValueError,IndexError,AttributeError,OverflowError) as error:
        return dict(schema=1,status='PARTIAL',verified_success=False,problems=['Malformed saved evidence; no inference: '+type(error).__name__+': '+str(error)],
            statistics={},observed_counts={},comparisons=0)


def audit_execution(evidence,protocol,dataset,choices):
    """Check every saved output, trajectory, query/cache, owner and interval."""
    docs=dataset['documents'];extra=extra_geometries(choices);expected=expected_counts(choices)
    m=sum(len(v) for v in choices.values());total=162+34*m+len(extra)
    arms={str(n):[dict(OWN),*choices[str(n)].values()] for n in LENGTHS}
    maps={n:{a['mode']:a for a in values} for n,values in arms.items()}
    own_modes={str(n):'rescue_graph' for n in LENGTHS}
    challengers={str(n):{mode:mode for mode in choices[str(n)]} for n in LENGTHS}
    canonical=evidence.rows('reference.jsonl');ownq=evidence.rows('own_quality.jsonl')
    baseq=evidence.rows('baseline_quality.jsonl');matchq=evidence.rows('matching_quality.jsonl') if extra or (evidence.root/'matching_quality.jsonl').exists() else []
    prime=evidence.rows('prime.jsonl');warm=evidence.rows('runtime.jsonl');owners=evidence.rows('owners.jsonl');quality_owners=evidence.rows('quality_owners.jsonl')
    observed=dict(reference_requests=len(canonical),own_quality_requests=len(ownq),baseline_quality_requests=len(baseq),
        matching_quality_requests=len(matchq),prime_requests=len(prime),timing_requests=len(warm),owners=len(owners))
    if observed!=expected:evidence.problem('Complete variable-M request/owner matrix missing or duplicated')
    requests=canonical+ownq+baseq+matchq+prime+warm;ids=[r.get('request_id') for r in requests];indices=[r.get('execution_index') for r in requests]
    valid_indices=all(type(i) is int for i in indices) and len(set(indices))==len(indices)
    if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):evidence.problem('Scientific UUID missing/reused')
    if not valid_indices or sorted(indices)!=list(range(total)):evidence.problem('Execution-index ledger incomplete/reused')
    qrows=ownq+baseq+matchq
    if valid_indices and ((canonical and max(r['execution_index'] for r in canonical)>=32) or
        (qrows and min(r['execution_index'] for r in qrows)<32) or
        (qrows and prime and max(r['execution_index'] for r in qrows)>=min(r['execution_index'] for r in prime))):
        evidence.problem('All canonical/quality gates must finish before timing owners')
    gate=evidence.json('QUALITY_PASSED.json')
    if (gate.get('passed') is not True or gate.get('comparisons')!=m or gate.get('matching_geometry_count')!=len(extra) or any(
        gate.get(k) is not True for k in ('own_all32_pairs','baseline_matching_geometry','actual_same_bfloat16_nonzero_mapper','quality_owners_closed'))):
        evidence.problem('Complete Native/Q/nonzero/geometry quality gate absent')
    refs=evidence.json('references.json');prime_refs=evidence.json('prime_references.json');reference={}
    eos=protocol.get('tokenizer_eos_token_id');vocab=protocol.get('vocab_size')
    if (type(eos) is not int or type(vocab) is not int or not 0<=eos<vocab or
        protocol.get('target_layers')!=28 or protocol.get('draft_layers')!=28 or protocol.get('graph_max_entries')!=128):
        evidence.problem('Frozen EOS/vocabulary/all28layers/Graph128 metadata invalid')
    for row in canonical:
        key=row.get('source_index'),row.get('length');tokens=row.get('generated')
        if (key not in dataset['input_sha256'] or key in reference or not isinstance(tokens,list) or
            not 0<len(tokens)<=128 or not all(type(t) is int and 0<=t<(vocab if type(vocab) is int else 0) for t in tokens) or
            refs.get(f'{key[0]}:{key[1]}')!=tokens):evidence.problem('Independent canonical32 cells differ')
        if isinstance(tokens,list) and ((len(tokens)<128 and (not tokens or tokens[-1]!=eos)) or
            (eos in tokens and (tokens[-1]!=eos or eos in tokens[:-1]))):evidence.problem('Canonical EOS boundary is not exact')
        reference[key]=tokens
    if len(reference)!=32:evidence.problem('Independent canonical32 cells incomplete')
    if protocol.get('input_sha256')!={f'{d}:{n}':v for (d,n),v in dataset['input_sha256'].items()}:
        evidence.problem('Executed context inputs differ from frozen token prefixes')
    oq,bq,mq={},{},{}
    for row in qrows:
        key=row.get('source_index'),row.get('length');ref=reference.get(key)
        if row.get('generated')!=ref or row.get('tokens')!=len(ref or []) or row.get('output_agrees') is not True:
            evidence.problem('Quality output differs from canonical Target AR')
        if row.get('performance_eligible') is not False:evidence.problem('Instrumented quality marked primary timing')
        if row.get('arm',{}).get('family')!='full_tree':
            for p in support.observation_errors(row)+support.quality_pair_errors(row,row):evidence.problem('Quality: '+p)
            if row.get('arm',{}).get('backend') in ('resident','cuda_graph') and row.get('runtime_stats',{}).get('max_graph_entries')!=128:
                evidence.problem('Quality actual Graph entry limit differs')
    for row in ownq:
        key=row.get('source_index'),row.get('length'),row.get('mode');backend='eager' if key[2]=='rescue_eager' else 'cuda_graph'
        if key in oq or key[0] not in docs or key[1] not in LENGTHS or key[2] not in ('rescue_eager','rescue_graph') or row.get('phase')!='own_quality':
            evidence.problem('Own64 quality matrix duplicated/unregistered')
        if row.get('arm')!=dict(mode=key[2],family='rescue',backend=backend):evidence.problem('Own quality declared mode differs')
        oq[key]=row
        for p in support.linear_trace_errors(row,own=True)+support.injection_errors(row):evidence.problem('Own quality: '+p)
    for n in LENGTHS:
        for doc in docs:
            a,b=oq.get((doc,n,'rescue_eager')),oq.get((doc,n,'rescue_graph'))
            if a is None or b is None:evidence.problem('All32 own eager/Graph quality pairs missing');continue
            for p in support.quality_pair_errors(a,b):evidence.problem('Own equality: '+p)
    for row in baseq:
        n,mode=row.get('length'),row.get('mode');arm=maps.get(str(n),{}).get(mode)
        if not arm or arm['family']=='rescue' or (n,mode) in bq or row.get('source_index')!=(docs[0] if docs else None) or row.get('phase')!='baseline_sentinel':
            evidence.problem('Every retained baseline sentinel is required exactly once');continue
        bq[n,mode]=row
        if row.get('arm')!=arm:evidence.problem('Sentinel configuration differs from retained union')
        if arm['family']=='full_tree':
            for p in support.tree_errors(row,arm):evidence.problem('Tree quality: '+p)
        else:
            for p in support.linear_trace_errors(row,fixed_k=arm['fixed_k']):evidence.problem('Retrieval quality: '+p)
            if row.get('nonzero_injection') is not False or any(q.get('summary_requested') is not False or q.get('summary_count')!=0 for q in row.get('queries',[])):
                evidence.problem('Baseline sentinel executed Mapper/M3')
            if arm['backend']=='eager':mq[n,*geometry(arm)]=row
    if set(bq)!={(n,mode) for n in LENGTHS for mode in choices[str(n)]}:evidence.problem('A retained union opponent lacks its sentinel')
    for row in matchq:
        arm=row.get('arm',{});key=row.get('length'),arm.get('top_chunks'),arm.get('fixed_k')
        descriptor=dict(length=key[0],top_chunks=key[1],fixed_k=key[2])
        if descriptor not in extra or key in mq or row.get('phase')!='matching_geometry_eager' or row.get('source_index')!=(docs[0] if docs else None) or arm.get('backend')!='eager':
            evidence.problem('Matching eager geometry differs from pre-data freeze')
        targets=[a['mode'] for a in choices.get(str(key[0]),{}).values() if a['family']=='retrieval' and geometry(a)==key[1:]]
        if row.get('equivalence_targets')!=targets:evidence.problem('Matching eager geometry targets changed')
        mq[key]=row
    for n in LENGTHS:
        for arm in choices[str(n)].values():
            if arm['family']!='retrieval' or arm['backend']=='eager':continue
            a,b=mq.get((n,*geometry(arm))),bq.get((n,arm['mode']))
            if a is None or b is None:evidence.problem('Retained backend lacks exact eager-geometry reference');continue
            for p in support.quality_pair_errors(a,b):evidence.problem('Baseline equality: '+p)
    qo={r.get('request_id'):r for r in quality_owners}
    if len(qo)!=64+m+len(extra) or len(qo)!=len(quality_owners):evidence.problem('All quality owner-close ledgers required')
    for row in qrows:
        owner=qo.get(row.get('request_id'),{})
        if (owner.get('owner_id')!=row.get('owner_id') or owner.get('engine_closed') is not True or owner.get('hook_restored') is not True or
            owner.get('closed_runtime_stats',{}).get('entries')!=0 or owner.get('final_runtime_stats')!=row.get('runtime_stats')):
            evidence.problem('Quality owner/cache/graphs survived into timing')
    pmap,wmap,bad={},{},set()
    for phase,values in (('cold_prime',prime),('warm',warm)):
        for row in values:
            mode,n,doc=row.get('mode'),row.get('length'),row.get('source_index');arm=maps.get(str(n),{}).get(mode)
            if arm is None:evidence.problem('Unknown timed union arm/context');continue
            ref=prime_refs.get(str(n)) if phase=='cold_prime' else reference.get((doc,n))
            errors=support.timing_errors(row,ref,arm,'rescue_graph')
            if row.get('arm')!=arm or row.get('phase')!=phase or any(row.get(k) is not False for k in ('observer_enabled','profile_enabled')):
                errors.append('Unfrozen or instrumented primary request')
            if any(row.get(k) is not (arm['family']=='rescue') for k in ('mapper_executed','controller_executed')):
                errors.append('Actual Mapper/controller execution differs')
            if phase=='cold_prime':
                if doc!=271 or (mode,n) in pmap:errors.append('Unique old271 paid prime required')
                pmap[mode,n]=row
                if arm.get('backend')=='cuda_graph':
                    delta,life=row.get('draft_runtime',{}).get('request',{}),row.get('runtime_stats',{})
                    if not positive(delta.get('captures')) or any(not near(delta.get(k),life.get(k)) for k in ('captures','replays','resident_steps','capture_host_seconds')):
                        errors.append('Cold timing owner borrowed prior quality graphs')
            else:
                key=mode,n,doc,row.get('repeat')
                if doc not in docs or row.get('repeat') not in (0,1) or key in wmap:errors.append('Warm document/repeat matrix duplicated/unregistered')
                wmap[key]=row
                target=oq.get((doc,n,'rescue_graph')) if arm['family']=='rescue' else bq.get((n,mode)) if docs and doc==docs[0] else None
                if target is not None and (support.trace_signature(target)!=support.trace_signature(row) or any(target.get(k)!=row.get(k) for k in
                    ('accepted','proposed','blocks','draft_forwards','selector_refreshes','working_gather_layers','projected_positions'))):
                    errors.append('Timed trajectory differs from its instrumented quality')
            for p in errors:evidence.problem(f'{phase} {mode}/{doc}/{n}: '+p)
            if errors and phase=='warm':bad.add((mode,n,doc))
    keys={(a['mode'],n) for n in LENGTHS for a in arms[str(n)]}
    if set(pmap)!=keys or set(wmap)!={(mode,n,d,r) for mode,n in keys for d in docs for r in (0,1)}:
        evidence.problem('Complete union cold/warm matrix absent')
    owner_plan,warm_plan,sequence=registered_schedule(choices,docs)
    if protocol.get('owner_order')!=owner_plan or protocol.get('warm_order')!=warm_plan or protocol.get('shared_warm_sequence')!=sequence:
        evidence.problem('Shared32 order or seeded private owner plan changed')
    if ([{k:r.get(k) for k in ('mode','length')} for r in prime]!=owner_plan or
        [{k:r.get(k) for k in ('mode','length','source_index','repeat')} for r in warm]!=warm_plan):
        evidence.problem('Actual owner/warm sequence differs from frozen shared plan')
    if valid_indices:
        start=96+m+len(extra)
        for index,row in enumerate(prime):
            pos=start+33*index;block=[r for r in warm if r.get('owner_id')==row.get('owner_id')]
            if row.get('execution_index')!=pos or [r.get('execution_index') for r in block]!=list(range(pos+1,pos+33)) or [r.get('warm_index') for r in block]!=list(range(32)):
                evidence.problem('Owner execution was not one paid cold +32 consecutive shared warm requests')
    omap,owner_metrics,diagnostics,selection={},{},{},{}
    for owner in owners:
        key=owner.get('mode'),owner.get('length');rows=sorted([r for r in prime+warm if r.get('owner_id')==owner.get('owner_id')],key=lambda r:r.get('execution_index',-1))
        if key in omap or key not in keys:evidence.problem('Unknown/duplicate timing owner')
        omap[key]=owner
        if owner.get('maximum_live_owners')!=1:evidence.problem('Single-private-owner bound changed')
        for p in support.lifecycle_errors(owner,rows):evidence.problem('Owner '+str(key)+': '+p)
        if rows:
            cold=owner.get('constructor_and_detach_seconds',0)+rows[0].get('outer_seconds',0)
            owner_metrics.setdefault(str(key[1]),{})[key[0]]=dict(cold_complete_tps=rows[0].get('tokens',0)/cold if positive(cold) else None,
                accounted_service_tps=owner.get('accounted_service_tps'))
    if set(omap)!=keys:evidence.problem('Not every timing private owner was closed')
    if {o.get('owner_id') for o in owners}&{o.get('owner_id') for o in quality_owners}:evidence.problem('Timing borrowed quality owner state')
    # All repeats and matching retrieval backends must do identical logical work.
    logical={}
    for row in warm:
        arm=maps.get(str(row.get('length')),{}).get(row.get('mode'))
        if arm is None:continue
        key=arm['family'],arm.get('top_chunks'),arm.get('fixed_k',arm.get('nodes')),row.get('source_index'),row.get('length')
        if arm['family']=='rescue':key=('rescue',row.get('source_index'),row.get('length'))
        signature=(support.trace_signature(row),row.get('upstream',{}).get('accept_length_list'),*[row.get(k) for k in
            ('accepted','proposed','blocks','draft_forwards','selector_refreshes','working_gather_layers','projected_positions','draft_shadow_probes')])
        if key in logical and logical[key]!=signature:evidence.problem('Repeated/matched-backend logical trajectory changed');bad.add((row['mode'],row['length'],row['source_index']))
        else:logical[key]=signature
    for n in LENGTHS:
        for arm in arms[str(n)]:
            rows=[r for r in warm if r.get('length')==n and r.get('mode')==arm['mode']]
            d=dict(warm_captures=sum(r.get('draft_runtime',{}).get('request',{}).get('captures',0) for r in rows),
                warm_capture_seconds=sum(r.get('draft_runtime',{}).get('request',{}).get('capture_host_seconds',0.) for r in rows)) if rows else {}
            for stage in ('prefill_seconds','decode_seconds','cleanup_seconds'):
                if rows and all(nonnegative(r.get(stage)) for r in rows):d[stage]=sum(r[stage] for r in rows)/len(rows)
            diagnostics.setdefault(str(n),{})[arm['mode']]=d
        own=[r for r in warm if r.get('length')==n and r.get('mode')=='rescue_graph']
        selection[str(n)]=dict(cap_counts={str(c):sum(t.get('cap')==c for r in own for t in r.get('trace',[])) for c in (1,2,3,4)},
            working_cache_tokens=[t['workset_tokens'] for r in own for t in r.get('trace',[]) if type(t.get('workset_tokens')) is int and t['workset_tokens']>=0],
            selected_relative_positions=[p/o['prefix'] for r in ownq if r.get('length')==n and r.get('mode')=='rescue_graph' for o in r.get('observed_rounds',[]) if positive(o.get('prefix')) for p in o.get('selected',[])])
    trees=[r for r in requests if r.get('arm',{}).get('family')=='full_tree'];tree_raw=evidence.rows('tree_raw.jsonl');rawmap={r.get('request_id'):r for r in tree_raw}
    if len(rawmap)!=len(trees) or len(rawmap)!=len(tree_raw):evidence.problem('Every raw full-tree wrapper result must be preserved before checking')
    for row in trees:
        raw=rawmap.get(row.get('request_id'),{});result=raw.get('result',{})
        if raw.get('execution_index')!=row.get('execution_index') or any(result.get(k)!=row.get(field) for k,field in (('generated','wrapper_generated'),('tokens','wrapper_tokens'),('upstream','upstream'))):
            evidence.problem('Tree normalized row differs from pre-check raw wrapper result')
    perctx={str(n):[d for d in docs if all((a['mode'],n,d) not in bad and all((a['mode'],n,d,r) in wmap for r in (0,1)) for a in arms[str(n)])] for n in LENGTHS}
    common=[d for d in docs if all(d in perctx[str(n)] for n in LENGTHS)]
    return dict(observed_counts=observed,expected_counts=expected,expected_requests=total,observed_requests=len(requests),
        documents=docs,comparisons=m,arms=arms,own_modes=own_modes,challengers=challengers,owner_metrics=owner_metrics,
        diagnostics=diagnostics,selection_diagnostics=selection,paired_complete_documents=common,
        paired_complete_documents_by_context=perctx,_wmap=wmap)


def render(summary, output):
    """Draw actual saved completed subsets only, separating quality and TPS."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    stats=summary.get('statistics',{});contexts=stats.get('contexts',{})
    status=summary.get('status','PARTIAL');m=summary.get('comparisons',stats.get('comparisons',0))
    n=stats.get('per_context_n',{str(k):stats.get('n',0) for k in LENGTHS})
    caption=(f'{status}; H10080GB, Qwen2 FP32 Target/BF16 Draft+prefix, TF32 off; exact4K/8K; '
        f'max128 output,2 repeats/doc, n4K={n.get("4096",0)}/16,n8K={n.get("8192",0)}/16; '
        f'{m} pre-frozen union contrasts; all lazy captures charged. Finite owner blocks; not production/Vicuna.')
    def modes(length):
        return [summary.get('own_modes',{}).get(str(length),'rescue_graph'),
            *summary.get('challengers',{}).get(str(length),{}).values()]
    def color(mode):
        arm=next((a for values in summary.get('arms',{}).values() for a in values if a.get('mode')==mode),{})
        return '#2563eb' if mode=='rescue_graph' else '#10b981' if arm.get('family')=='full_tree' else {
            'eager':'#94a3b8','resident':'#64748b','cuda_graph':'#f59e0b'}.get(arm.get('backend'),'#6b7280')
    def label(mode):
        return 'RESCUE' if mode=='rescue_graph' else str(mode).replace('retrieval','R').replace('tree','T').replace('_',' ')
    def point(ax,x,value,limits,c):
        if value is None:return
        ax.scatter([x],[value],color=c,zorder=4)
        if limits is not None:
            ax.vlines(x,limits[0],limits[1],color=c,linewidth=1.6)
            ax.hlines(limits,x-.065,x+.065,color=c)
    names=[]
    def finish(fig,name):
        fig.text(.01,.012,caption,fontsize=8,va='bottom',wrap=True)
        fig.tight_layout(rect=(0,.085,1,.955));fig.savefig(output/(name+'.png'),dpi=160)
        fig.savefig(output/(name+'.svg'));plt.close(fig);names.append(name)
    def ticks(ax,items):
        ax.set_xticks(range(len(items)),[label(v) for v in items],rotation=35,ha='right',fontsize=8)
        ax.grid(axis='y',alpha=.2)
    fig,axes=plt.subplots(3,2,figsize=(15,11))
    for col,length in enumerate(LENGTHS):
        values=contexts.get(str(length),{});allm=modes(length)
        trees=[a['mode'] for a in summary.get('arms',{}).get(str(length),[]) if a.get('family')=='full_tree']
        linear=[a for a in allm if a not in trees]
        for row,metric in enumerate(('acceptance_length','acceptance_rate')):
            for x,mode in enumerate(linear):
                v=values.get(mode,{});point(axes[row,col],x,v.get(metric),v.get(metric+'_ci95'),color(mode))
            ticks(axes[row,col],linear);axes[row,col].set_title(f'{length}: linear quality')
            axes[row,col].set_ylabel('Accepted Draft tokens / round' if row==0 else 'Accepted / proposed Draft tokens')
        for x,mode in enumerate(trees):
            v=values.get(mode,{});point(axes[2,col],x,v.get('acceptance_length'),v.get('acceptance_length_ci95'),color(mode))
        ticks(axes[2,col],trees);axes[2,col].set_title(f'{length}: full-tree raw work (includes overshoot)')
        axes[2,col].set_ylabel('Raw accepted tokens / raw round')
    fig.suptitle('Quality only — document-paired pointwise95% intervals; no invented tree acceptance rate')
    finish(fig,'acceptance')
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for col,length in enumerate(LENGTHS):
        values=contexts.get(str(length),{});items=modes(length)
        for x,mode in enumerate(items):
            v=values.get(mode,{});point(axes[0,col],x,v.get('pooled_tps'),v.get('tps_ci95'),color(mode))
        ticks(axes[0,col],items);axes[0,col].set_title(f'{length}: whole-generate outer');axes[0,col].set_ylabel('Client tokens / second')
        contrasts=[c for c in stats.get('contrasts',[]) if c['length']==length]
        for x,c in enumerate(contrasts):
            point(axes[1,col],x,c['point_percent'],c.get('pointwise_ci95'),color(c['baseline_mode']))
            lower=c.get('simultaneous_one_sided_lower95')
            if lower is not None:
                axes[1,col].scatter([x],[lower],marker='D',facecolors='none',edgecolors='#dc2626',s=35,
                    label=f'One-sided simultaneous95% lower (q=.05/{m})' if x==0 else None)
        ticks(axes[1,col],[c['baseline_mode'] for c in contrasts]);axes[1,col].axhline(0,color='black',linewidth=.8)
        axes[1,col].set_ylabel('RESCUE pooled TPS change (%)')
        axes[1,col].set_title('PARTIAL — no formal simultaneous gate' if status!='COMPLETE' else f'All{m} contrasts use the same16-document resamples')
        if contrasts and status=='COMPLETE':axes[1,col].legend(fontsize=7)
    fig.suptitle('Performance only — frozen union;100000 whole-document bootstrap draws')
    finish(fig,'tps')
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for col,length in enumerate(LENGTHS):
        ledger=summary.get('owner_metrics',{}).get(str(length),{});items=modes(length)
        for row,key in enumerate(('cold_complete_tps','accounted_service_tps')):
            for x,mode in enumerate(items):
                v=ledger.get(mode,{}).get(key)
                if positive(v):axes[row,col].bar(x,v,color=color(mode))
            ticks(axes[row,col],items);axes[row,col].set_ylabel('Client tokens / charged second')
            axes[row,col].set_title(f'{length}: '+('constructor + old271 cold request' if row==0 else 'constructor + cold +32warm + close'))
    fig.suptitle('Cold/service diagnostics — one private owner/arm/context, descriptive; no document CI')
    finish(fig,'cold_service')
    fig,axes=plt.subplots(3,2,figsize=(15,11))
    for col,length in enumerate(LENGTHS):
        values=summary.get('diagnostics',{}).get(str(length),{});items=modes(length)
        for row,key in enumerate(('warm_captures','warm_capture_seconds')):
            for x,mode in enumerate(items):
                if key in values.get(mode,{}):axes[row,col].bar(x,values[mode][key],color=color(mode))
            ticks(axes[row,col],items);axes[row,col].set_ylim(bottom=0)
            axes[row,col].set_ylabel('New captures across32warm' if row==0 else 'Capture host seconds across32warm')
            axes[row,col].set_title(f'{length}: no capture-free filtering')
        for offset,(key,c) in enumerate((('prefill_seconds','#60a5fa'),('decode_seconds','#fbbf24'),('cleanup_seconds','#c4b5fd'))):
            for x,mode in enumerate(items):
                v=values.get(mode,{}).get(key)
                if nonnegative(v):axes[2,col].bar(x+(offset-1)*.22,v,width=.22,color=c,label=key.replace('_seconds','') if x==0 else None)
        ticks(axes[2,col],items);axes[2,col].set_ylabel('Saved inner seconds / warm request');axes[2,col].set_title('Available inner stages — not additive outer attribution')
        if axes[2,col].patches:axes[2,col].legend(fontsize=7)
    fig.suptitle('Measured execution mechanisms — unavailable diagnostics omitted, not imputed')
    finish(fig,'latency_captures')
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for col,length in enumerate(LENGTHS):
        d=summary.get('selection_diagnostics',{}).get(str(length),{});caps=d.get('cap_counts',{})
        if any(caps.values()):axes[0,col].bar(['1','2','3','4'],[caps.get(str(v),0) for v in (1,2,3,4)],color='#2563eb')
        axes[0,col].set_title(f'{length}: frozen own causal2/4 decisions (tail caps1/3 allowed)');axes[0,col].set_ylabel('Actual rounds')
        selected=d.get('selected_relative_positions',[])
        if selected:axes[1,col].hist(selected,bins=20,range=(0,1),color='#2563eb')
        axes[1,col].set_title('Saved quality selected positions only');axes[1,col].set_xlabel('Selected absolute position / committed prefix');axes[1,col].set_ylabel('Observed selected positions')
        cache=d.get('working_cache_tokens',[])
        if cache:axes[2,col].hist(cache,bins=min(20,len(set(cache))),color='#2563eb')
        axes[2,col].set_title('Actual own warm round working-cache size');axes[2,col].set_xlabel('Saved trace workset_tokens');axes[2,col].set_ylabel('Observed verification rounds')
    fig.suptitle('Own-method mechanisms — nonzero Mapper and attention selection remain mandatory')
    finish(fig,'actions_selection')
    return names


def write_report(summary,directory,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    figures=render(summary,output)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    shutil.copyfile(Path(__file__),output/'analysis_source.py')
    shutil.copyfile(Path(support.__file__),output/'analysis_fair_support.py')
    shutil.copyfile(Path(horizon.__file__),output/'analysis_horizon_support.py')
    shutil.copyfile(Path(horizon.support.__file__),output/'analysis_support.py')
    m=summary.get('comparisons',0);quantile=.05/m if m else None
    lines=[f"# {summary.get('status','PARTIAL')}: frozen-union fair confirmation",'',
        f"Evidence audit: {summary.get('status','PARTIAL')}. Formal speed/exactness success: {summary.get('verified_success',False)}.",'',
        f'The union contains {m} contrasts across4K/8K, each retained from the immutable original6-warm and matched32-warm development choices. '
        'No original winner is weakened or replaced. Two repeats are pooled within each of16 whole documents; primary TPS is the ratio of summed client tokens to summed whole-generate outer seconds. '
        'One common100000-draw document-index bootstrap (seed20261026) covers every arm and both contexts. '
        f'Pointwise central95% intervals are separate from the formal Bonferroni one-sided95% lower bounds (quantile .05/M={quantile}). '
        'ALL contrasts require positive points AND positive simultaneous lower bounds, together with all source/data/quality/owner/canonical gates. COMPLETE is not the same as acceleration success.','',
        'Scope: same single H10080GB/Qwen2-7B and1.5B, FP32 Target decode/BF16 Draft and prefix, TF32 off; exact4096/8192 inputs,max128 client outputs,2 repeats. '
        'Sixteen deterministic previously unused validation documents are not a random official test. New selection starts at validation index0, excludes all eleven old used/reserved metadata cohorts and duplicate document/4K/8K-prefix SHA, and requires natural body>=8192. '
        'The unused old>=900/16K template was never executed and is not used here. Original paused71/320 remains paused. No outcome/EOS/speed replacement, optional stopping, capture-free filtering or imputation.','',
        'The attention selection, genuine nonzero V-only Mapper and frozen old-cost causal2/4 policy are unchanged. '
        'Every private owner pays construction, an old271 cold prime,32 warm requests and close; primary uses all32 warm generates including every legal lazy capture. '
        'All arms and contexts share exactly the same frozen32-document/repeat order; owners execute random blocks, not interleaved, so block clock drift remains a limitation. '
        'Cold/service are secondary measured interval sums, not production/steady-state/session-wall claims. Full-tree raw acceptance includes charged overshoot work; no tree proposal rate or unobserved tail IDs are invented.','',
        '| Input | Retained challenger | TPS change | Pointwise95% CI | Simultaneous one-sided95% lower |','|---|---|---:|---|---:|']
    for c in summary.get('statistics',{}).get('contrasts',[]):
        ci=c.get('pointwise_ci95');lo=c.get('simultaneous_one_sided_lower95')
        lines.append(f"| {c['length']} | {c['baseline_mode']} | {c['point_percent']:+.5f}% | "+
            ('unavailable' if ci is None else f'[{ci[0]:+.5f}%, {ci[1]:+.5f}%]')+' | '+('unavailable' if lo is None else f'{lo:+.5f}%')+' |')
    lines+=['','The frozen percentile-bootstrap/Bonferroni gate is an empirical decision rule, not an exact finite-sample coverage theorem. '
        'Local auditing rehashes saved source, raw arrays, manifests, exclusion/scan receipts and frozen tokenizer/template bindings. '
        'It does not re-open or re-tokenize original Parquet/full text and cannot reconstruct full-text SHA from stored tokens.','',
        '## Audit problems','']
    lines+=['- '+p for p in summary.get('problems',[])] or ['None.']
    lines+=['','## Saved figures, analysis source and raw results','']
    for name in figures:lines += [f'[{name} PNG]({name}.png) · [SVG]({name}.svg)','',f'![{name}]({name}.png)','']
    lines+=['[Standalone analysis](analysis_source.py) · [Support](analysis_fair_support.py) · [Horizon support](analysis_horizon_support.py) · [Baseline support](analysis_support.py) · [Summary](summary.json)','']
    for name in ('protocol.json','GATE_PASSED.json','QUALITY_PASSED.json','reference.jsonl','own_quality.jsonl','baseline_quality.jsonl',
        'matching_quality.jsonl','prime.jsonl','runtime.jsonl','owners.jsonl','quality_owners.jsonl','tree_raw.jsonl','COMPLETE.json',
        'data/DATASET_MANIFEST.json','data/confirmation.jsonl'):
        if (Path(directory)/name).is_file():lines.append(f'[{name}](../{name})')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    receipt=dict(schema=1,status=summary.get('status','PARTIAL'),verified_success=summary.get('verified_success',False),
        render_required=True,rendered_figures=figures,file_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        note='Human render inspection and a separate independent raw audit are still required before final handoff.')
    (output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    if summary.get('verified_success'):
        (output/'VERIFIED_SUCCESS.json').write_text(json.dumps(dict(schema=1,status='VERIFIED_SUCCESS',
            evidence_audit_passed=True,statistical_gate_passed=True,comparisons=m,
            summary_sha256=sha(output/'summary.json'),analysis_receipt_sha256=sha(output/'ANALYSIS_COMPLETE.json'),
            scope='Frozen-union same-Qwen single-H100 deterministic16-document confirmation only; human artifact review required'),indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,help='Saved results only, never models or raw corpus')
    parser.add_argument('--output',type=Path,required=True,help='Fresh analysis directory; no overwriting')
    args=parser.parse_args();value=analyze(args.directory);write_report(value,args.directory,args.output)
    print(json.dumps(dict(status=value['status'],verified_success=value.get('verified_success',False),comparisons=value.get('comparisons'),problems=value.get('problems')),indent=2))


if __name__=='__main__':main()
