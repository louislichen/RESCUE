"""Independent stdlib-only audit of a locally exported COMPLETE stage1 run.

No report/runner helper is imported. All scientific files are resolved inside
the explicit export root; original remote paths are never opened. This audits
recorded canonical tokens, trajectories, ownership and hashes, not uncollected
Native-KV/Q tensor hashes or visual appearance. A pass is not goal completion.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re


COUNTS = dict(reference_requests=6, prime_requests=76, timing_requests=456, owners=76)
DOCS = (271, 272, 288)
LENGTHS = (4096, 8192)
GROUPS = ('retrieval/eager', 'retrieval/resident', 'retrieval/graph', 'full_tree')
TRACE = ('round_index', 'prefix', 'cap', 'proposed', 'accepted', 'terminal', 'workset_tokens')
WORK = ('tokens', 'accepted', 'proposed', 'blocks', 'draft_forwards', 'selector_refreshes',
        'working_gather_layers', 'working_full_gather_layers', 'working_gathered_position_layers',
        'working_incremental_binds', 'target_probe_rounds', 'draft_shadow_probes', 'projected_positions', 'rpc_calls')


def need(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    need(not p.is_absolute() and '..' not in p.parts, 'Unsafe relative artifact path: ' + str(relative))
    result = root / p
    need(result.resolve().is_relative_to(root) and result.is_file() and not result.is_symlink(),
         'Missing/unsafe local artifact: ' + str(relative))
    return result


def read(root, relative):
    return json.loads(local(root, relative).read_text())


def lines(root, relative):
    raw = local(root, relative).read_bytes()
    need(raw.endswith(b'\n'), 'Incomplete JSONL tail: ' + relative)
    values = raw.splitlines()
    need(values and all(v.strip() for v in values), 'Empty/blank JSONL: ' + relative)
    return [json.loads(v) for v in values]


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def near(left, right):
    return finite(left) and finite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def menu():
    result = []
    for chunks in (16, 32):
        for width in (1, 2, 4, 8, 12):
            for backend in ('eager', 'resident', 'graph'):
                result.append(dict(mode=f'retrieval{chunks}_k{width}_{backend}', family='retrieval',
                    top_chunks=chunks, fixed_k=width, backend='cuda_graph' if backend == 'graph' else backend))
    for chunks in (16, 32):
        for nodes in (8, 32, 50, 64):
            result.append(dict(mode=f'tree{chunks}_n{nodes}', family='full_tree', top_chunks=chunks,
                nodes=nodes, backend='upstream_flash_hybrid', max_depth=10, threshold=.7))
    return result


def planned(descriptors):
    owners, requests = [], []
    for n in LENGTHS:
        rng = random.Random(20261023 + n)
        modes = [a['mode'] for a in descriptors]
        rng.shuffle(modes)
        for mode in modes:
            owners.append(dict(mode=mode, length=n))
            requests.append(dict(mode=mode, length=n, source_index=271, phase='cold_prime'))
            for repeat in (0, 1):
                documents = list(DOCS)
                rng.shuffle(documents)
                requests.extend(dict(mode=mode, length=n, source_index=i, repeat=repeat, phase='warm')
                                for i in documents)
    return owners, requests


def canonical(ids, eos):
    need(isinstance(ids, list) and 0 < len(ids) <= 128 and
         all(type(t) is int and t >= 0 for t in ids), 'Invalid canonical token array')
    need(len(ids) == 128 or ids[-1] == eos, 'Short canonical output without EOS')
    if eos in ids:
        need(ids[-1] == eos and eos not in ids[:-1], 'Canonical EOS boundary is not first EOS')


def tokenizer_eos(root, protocol, collection):
    """Derive EOS from the actual exported, frozen tokenizer JSON bytes."""
    prefix = 'asset_metadata/Qwen2-7B-Instruct/'
    values, hashes = {}, {}
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        relative = prefix+name; path = local(root,relative)
        source = protocol['args']['teacher'].rstrip('/')+'/'+name
        sha = digest(path)
        need(collection['files'].get(relative,{}).get('read_only_external_source') == source and
             protocol['assets_sha256'].get(source) == sha, 'Exported tokenizer JSON is not a frozen Target asset')
        values[name], hashes[name] = read(root,relative), sha
    eos = values['tokenizer_config.json'].get('eos_token')
    if isinstance(eos,dict): eos = eos.get('content')
    need(isinstance(eos,str) and eos, 'Tokenizer config has no EOS token content')
    matches = [t['id'] for t in values['tokenizer.json']['added_tokens'] if t.get('content') == eos]
    need(len(matches) == 1 and type(matches[0]) is int and matches[0] >= 0, 'Tokenizer EOS does not map to one added-token ID')
    eos_id = matches[0]; configured = values['config.json'].get('eos_token_id')
    need(configured == eos_id or (isinstance(configured,list) and eos_id in configured and
         all(type(t) is int for t in configured)), 'Model config and tokenizer EOS IDs disagree')
    decoder = values['tokenizer_config.json'].get('added_tokens_decoder',{})
    if str(eos_id) in decoder:
        need(decoder[str(eos_id)].get('content') == eos, 'Tokenizer added-token decoder disagrees on EOS')
    return dict(eos_token_id=eos_id,eos_token_content=eos,config_eos_token_id=configured,asset_metadata_sha256=hashes)


def trace_signature(row, arm):
    if arm['family'] == 'full_tree':
        return [row.get(k) for k in ('generated', 'tokens', 'raw_total_generated', 'generation_overshoot',
                'accepted', 'blocks', 'scored_generated_before_eos')] + [row['upstream']['accept_length_list']]
    return [row['generated'], [[t[k] for k in TRACE] for t in row['trace']]] + [row.get(k) for k in WORK]


def linear(row, arm):
    trace = row.get('trace')
    need(isinstance(trace, list) and trace and row['blocks'] == len(trace), 'Missing linear round trace')
    need(row['accepted'] == sum(t['accepted'] for t in trace) and
         row['proposed'] == sum(t['proposed'] for t in trace), 'Linear aggregate acceptance counters differ')
    progress = 0
    for j, step in enumerate(trace):
        cap, proposed, accepted = (step[k] for k in ('cap', 'proposed', 'accepted'))
        need(all(type(v) is int for v in (cap, proposed, accepted)) and
             cap == min(arm['fixed_k'], 128-progress) and 1 <= proposed <= cap and 0 <= accepted <= proposed,
             'Wrong linear cap/proposed/accepted count')
        need(step['round_index'] == j and step['terminal'] is (j == len(trace)-1) and
             step['prefix'] == row['length']-1+progress, 'Linear committed position/termination mismatch')
        need(type(step['workset_tokens']) is int and 0 < step['workset_tokens'] <= arm['top_chunks']*32,
             'Working retrieval cache exceeds configured budget')
        progress += accepted+1
    need(row['tokens'] <= progress <= 129, 'Linear raw committed count inconsistent with visible output')
    need(near(row['acceptance_rate'], row['accepted']/row['proposed']), 'Linear acceptance rate is inconsistent')
    need(row['execution_backend'] == arm['backend'] and row['baseline_algorithm'] == 'qwen_chunk_retrieval' and
         row['projected_positions'] == row['draft_shadow_probes'] == 0, 'Baseline executed a wrong backend/extra Mapper work')
    signature = row['owner_signature']
    for name in ('native_t', 'native_s', 'working'):
        cache = signature[name]
        need(all(type(cache.get(k)) is int and cache[k] > 0 for k in ('identity', 'key_ptr', 'value_ptr', 'capacity')),
             'Missing recorded physical cache identity')
    need(signature['native_t']['capacity'] >= row['length']+128 and
         signature['native_s']['capacity'] >= row['length']+128, 'Native cache capacity is insufficient')
    if arm['backend'] == 'eager':
        need('runtime' not in signature and row['runtime_stats']['entries'] == 0, 'Eager silently owns a Graph runtime')
    else:
        runtime = row['draft_runtime']
        expected_ptrs = [signature[name][key] for name in ('native_s', 'working') for key in ('key_ptr', 'value_ptr')]
        need(signature['runtime']['bound_ptrs'] == expected_ptrs and runtime['mode'] == arm['backend'] and
             runtime['max_graph_entries'] == 128 and runtime['retain_query'] is False, 'Runtime/cache/query ownership mismatch')
        delta = runtime['request']
        if arm['backend'] == 'cuda_graph':
            need(delta['replays'] == row['draft_forwards'] > 0 and
                 finite(delta['captures']) and finite(delta['capture_host_seconds']) and
                 row['runtime_stats']['entries'] <= 128, 'Wrong actual Graph replay/capture accounting')
            if row['phase'] == 'cold_prime':
                need(delta['captures'] > 0 and delta['capture_host_seconds'] > 0, 'Cold owner borrowed preexisting Graphs')
        else:
            need(delta['captures'] == delta['replays'] == 0 and
                 delta['resident_steps'] == row['draft_forwards'] > 0, 'Resident execution silently fell back')


def tree(row, raw, arm, eos):
    result = raw['result']
    need(all(raw[k] == row[k] for k in ('request_id', 'execution_index', 'owner_id', 'mode',
         'length', 'source_index', 'phase', 'outer_seconds')), 'Original tree raw ledger does not match normalized row')
    if row['phase'] == 'warm': need(raw['repeat'] == row['repeat'], 'Tree raw repeat differs')
    for k in ('upstream', 'nodes', 'top_chunks', 'max_depth', 'observed_full_config', 'total_seconds', 'request_id'):
        need(result[k] == row[k], 'Original tree result changed during normalization: ' + k)
    upstream = result['upstream']; scored = upstream['output_token_ids']; accepted = upstream['accept_length_list']
    need(isinstance(scored, list) and 0 < len(scored) <= 128 and upstream['total_generated'] == len(scored),
         'Tree scored output count differs from saved upstream output')
    need(accepted and all(type(v) is int and 0 <= v <= 10 for v in accepted), 'Wrong raw tree accept length')
    raw_count = sum(accepted)+len(accepted)
    need(raw_count == upstream['raw_total_generated'] == row['raw_total_generated'] and
         raw_count >= len(scored) and row['generation_overshoot'] == upstream['generation_overshoot'] == raw_count-len(scored),
         'Raw tree committed/overshoot counters differ')
    visible = scored[:128]
    if eos in visible: visible = visible[:visible.index(eos)+1]
    need(row['generated'] == result['generated'] == row['wrapper_generated'] == visible and
         row['tokens'] == result['tokens'] == row['wrapper_tokens'] == len(visible) and
         row['scored_generated_before_eos'] == scored, 'Tree client-visible EOS/normalization differs')
    need(row['accepted'] == sum(accepted) and row['blocks'] == len(accepted) and
         row['proposed'] is None and row['acceptance_rate'] is None and row['raw_tail_ids_available'] is False,
         'Invented/mislabeled raw tree acceptance statistics')
    policy = dict(nodes=arm['nodes'], top_chunks=arm['top_chunks'], max_depth=10, threshold=.7,
                  chunk_size=32, refresh_every=4, flash_prefill=True, hybrid_tree=True,
                  prefill_precision='bfloat16', mapper=False, controller=False)
    need(row['tree_policy'] == policy, 'Tree policy changed')
    need(row['observed_full_config'] == dict(target_use_flash_prefill=True, target_use_hybrid_tree_attn=True,
         draft_use_flash_prefill=True, use_retrieval_cache=True, retrieve_top_k=arm['top_chunks'],
         retrieval_chunk_size=32, retrieve_every_n_steps=4), 'Observed tree Flash/hybrid/retrieval flags changed')


def audit(root, analysis):
    root, analysis = Path(root).resolve(), Path(analysis).resolve()
    need(root.is_dir() and analysis.is_relative_to(root),
         'All audit inputs must be inside the explicit terminal export')
    for name in ('INDEPENDENT_RAW_AUDIT.json', 'INDEPENDENT_RAW_AUDIT_SOURCE.py'):
        need(not (root/name).exists(), 'Do not overwrite an independent audit artifact')
    collection = read(root, 'TERMINAL_COLLECTION.json')
    need(collection.get('status') == 'TERMINAL_COLLECTED_NOT_YET_INDEPENDENTLY_AUDITED' and
         collection.get('read_only') is True and collection.get('counts') == COUNTS and
         collection.get('pipeline_exit_code') == 0 and all(collection.get(k) is True for k in
         ('process_and_tmux_absent', 'complete_file_set_rechecked', 'all_file_bytes_rechecked')),
         'Collection is not a complete, byte-rechecked terminal export')
    need(collection.get('files'), 'Empty terminal file inventory')
    mandatory = {'PIPELINE_EXIT.json','RUN_STARTED.json','LAUNCH_MANIFEST.json','DEPENDENCY_RECEIPT.json',
                 'results/protocol.json','results/COMPLETE.json','results/FROZEN_SELECTION.json',
                 'results/reference.jsonl','results/references.json','results/prime.jsonl',
                 'results/runtime.jsonl','results/owners.jsonl','results/contexts.jsonl','results/tree_raw.jsonl'}
    need(mandatory.issubset(collection['files']), 'Terminal inventory omits a mandatory raw/proof artifact')
    for name, info in collection['files'].items():
        p = local(root, name)
        need(p.stat().st_size == info['bytes'] and digest(p) == info['sha256'], 'Terminal collection SHA/size mismatch: ' + name)
    need(not (root/'results/FAILURE.json').exists(), 'Failure marker exists')
    exit_record = read(root, 'PIPELINE_EXIT.json')
    need(type(exit_record.get('exit_code')) is int and exit_record['exit_code'] == 0, 'Nonzero/missing pipeline exit')
    protocol = read(root, 'results/protocol.json'); complete = read(root, 'results/COMPLETE.json')
    frozen = read(root, 'results/FROZEN_SELECTION.json')
    need(complete.get('counts') == COUNTS and complete.get('expected_counts') == COUNTS and
         complete.get('logical_requests') == 538 and complete.get('owner_closed') == 76 and
         all(complete.get(k) is True for k in ('passed', 'sources_unchanged', 'assets_unchanged',
         'dependencies_unchanged', 'all_canonical', 'backend_trajectories_equal')) and complete.get('goal_complete') is False,
         'Incomplete execution receipt')
    descriptors = menu(); arms = {a['mode']: a for a in descriptors}; owner_plan, requests = planned(descriptors)
    need(protocol['arms'] == descriptors and protocol['source_indices'] == list(DOCS) and
         protocol['lengths'] == list(LENGTHS) and protocol['repeats'] == 2 and protocol['seed'] == 20261023 and
         protocol['max_new_tokens'] == 128 and protocol['expected_counts'] == COUNTS and
         protocol['expected_requests'] == 538 and protocol['owner_order'] == owner_plan and
         protocol['warm_order'] == [{k:v for k,v in r.items() if k != 'phase'} for r in requests if r['phase']=='warm'],
         'Protocol menu/cohort/request order changed')
    need(protocol['max_live_owners'] == 1 and protocol['graph_max_entries'] == 128 and
         protocol['no_training'] is True and all(protocol[k] is False for k in
         ('mapper_executed', 'controller_executed', 'observer', 'profile', 'retrieval_retain_query')) and
         protocol['template_prefill_precision'] == 'bfloat16', 'Wrong execution/observer/precision contract')
    need(frozen['protocol_sha256'] == digest(root/'results/protocol.json') and
         complete['frozen_selection_sha256'] == digest(root/'results/FROZEN_SELECTION.json') and
         frozen['complete_request_count'] == 538 and frozen['no_new_confirmation_opened'] is True and
         frozen['source_sha256'] == protocol['source_sha256'] and frozen['assets_sha256'] == protocol['assets_sha256'],
         'Frozen choice is not bound to the complete protocol/source/assets')
    for name, sha in protocol['source_sha256'].items():
        for parent in ('after', 'results/source'):
            need(parent+'/'+name in collection['files'] and digest(local(root, parent+'/'+name)) == sha,
                 'Source copy missing from terminal inventory or different: ' + parent+'/'+name)
    launch = read(root,'LAUNCH_MANIFEST.json')
    need(all(launch['source_sha256'].get(name) == sha for name,sha in protocol['source_sha256'].items()),
         'Executed source differs from immutable staged source manifest')
    env = protocol['environment']
    need('GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9' in env['nvidia_smi_gpu'] and
         env['target_dtype']=='torch.float32' and env['draft_dtype']=='torch.bfloat16' and
         env['prefix_dtype']=='torch.bfloat16' and env['tf32'] is False and
         protocol['flash_attention']['version']=='2.8.3', 'Wrong registered hardware/precision/dependency identity')
    for section in ('source_bindings', 'tree_source_bindings'):
        for record in protocol[section].values():
            if isinstance(record, dict) and 'path' in record and 'sha256' in record:
                match = [name for name in protocol['source_sha256']
                         if record['path'] == name or record['path'].endswith('/'+name)]
                need(len(match) == 1 and protocol['source_sha256'][match[0]] == record['sha256'], 'Imported source binding differs')
    eos = tokenizer_eos(root,protocol,collection); eos_id = eos['eos_token_id']
    refs = lines(root, 'results/reference.jsonl'); prime = lines(root, 'results/prime.jsonl')
    warm = lines(root, 'results/runtime.jsonl'); owners = lines(root, 'results/owners.jsonl')
    tree_raw = lines(root, 'results/tree_raw.jsonl')
    need((len(refs), len(prime), len(warm), len(owners), len(tree_raw)) == (6,76,456,76,112), 'Wrong raw ledger counts')
    all_rows = refs+prime+warm
    need(len({r['request_id'] for r in all_rows}) == 538 and
         all(re.fullmatch('[0-9a-f]{32}', r['request_id']) for r in all_rows) and
         sorted(r['execution_index'] for r in all_rows) == list(range(538)), 'UUID/execution index ledger is not bijective')
    references = {}
    for j, row in enumerate(refs):
        need(row['execution_index'] == j and row['phase'] == 'reference' and row['mode'] == 'canonical_ar' and
             row['performance_eligible'] is False, 'Independent AR references do not precede timing')
        key = (row['length'], row['source_index']); need(key not in references, 'Duplicate canonical input')
        canonical(row['generated'], eos_id); references[key] = row['generated']
    need(set(references) == {(n,i) for n in LENGTHS for i in DOCS} and read(root, 'results/references.json') ==
         {f'{i}:{n}': ids for (n,i),ids in references.items()}, 'Canonical reference map differs')
    sd = sorted(prime+warm, key=lambda r:r['execution_index'])
    raw_map = {r['request_id']: r for r in tree_raw}; need(len(raw_map) == 112, 'Duplicate raw tree UUID')
    trajectories, computed = {}, {}
    for row, expected in zip(sd, requests):
        need(all(row.get(k) == v for k,v in expected.items()), 'Request differs from preregistered owner sequence')
        arm = arms[row['mode']]; reference = references[row['length'],row['source_index']]
        need(row['arm'] == arm and row['generated'] == reference and row['tokens'] == len(reference) and
             row['output_agrees'] is True and row['performance_eligible'] is True and
             all(row[k] is False for k in ('observer_enabled','profile_enabled','mapper_executed','controller_executed')) and
             'timeline' not in row and 'cost_samples' not in row, 'Canonical/unprofiled request gate failed')
        need(finite(row['outer_seconds'], True) and near(row['outer_tps'], row['tokens']/row['outer_seconds']) and
             finite(row['total_seconds'], True) and row['outer_seconds']+1e-8 >= row['total_seconds'], 'Invalid full-request clock')
        if arm['family'] == 'retrieval': linear(row, arm)
        else: tree(row, raw_map[row['request_id']], arm, eos_id)
        geometry = (arm['family'],arm['top_chunks'],arm.get('fixed_k',arm.get('nodes')),row['length'],row['source_index'])
        signature = trace_signature(row, arm)
        need(geometry not in trajectories or trajectories[geometry] == signature, 'Cold/repeat/backend trajectory/work mismatch')
        trajectories[geometry] = signature
    need(set(raw_map) == {r['request_id'] for r in sd if arms[r['mode']]['family']=='full_tree'}, 'Tree raw UUID coverage differs')
    owner_ids = set()
    for j, (owner, expected) in enumerate(zip(owners,owner_plan)):
        block = sd[j*7:(j+1)*7]; first, rest = block[0], block[1:]
        need(all(owner.get(k) == v for k,v in expected.items()) and re.fullmatch('[0-9a-f]{32}',owner['owner_id']) and
             owner['owner_id'] not in owner_ids, 'Wrong/duplicated owner identity')
        owner_ids.add(owner['owner_id'])
        need(all(r['owner_id'] == owner['owner_id'] and r['owner_signature'] == owner['owner_signature'] for r in block) and
             owner['prime_request_id'] == first['request_id'] and owner['warm_request_ids'] == [r['request_id'] for r in rest],
             'Contiguous private-owner signature/request ledger changed')
        need({(r['source_index'],r['repeat']) for r in rest} == {(i,k) for i in DOCS for k in (0,1)} and
             owner['maximum_live_owners'] == 1 and owner['engine_closed'] is owner['hook_restored'] is True and
             owner['closed_runtime_stats']['entries'] == 0 and owner['final_runtime_stats'] == rest[-1]['runtime_stats'],
             'Owner is incomplete, leaked, or changed its final runtime')
        warm_tokens = sum(r['tokens'] for r in rest); warm_seconds = sum(r['outer_seconds'] for r in rest)
        need(owner['warm_tokens'] == warm_tokens and near(owner['warm_outer_seconds_sum'],warm_seconds) and
             owner['prime_tokens'] == first['tokens'] and near(owner['cold_prime_outer_seconds'],first['outer_seconds']) and
             near(owner['constructor_and_detach_seconds'],first['constructor_and_detach_seconds']) and
             near(first['cold_complete_outer_seconds'],owner['constructor_and_detach_seconds']+first['outer_seconds']),
             'Owner token/time ledgers differ from raw requests')
        service = owner['constructor_and_detach_seconds']+first['outer_seconds']+warm_seconds+owner['close_seconds']
        need(finite(owner['close_seconds']) and near(owner['accounted_service_seconds'],service) and
             owner['accounted_service_tokens'] == warm_tokens+first['tokens'] and
             near(owner['accounted_service_tps'],(warm_tokens+first['tokens'])/service) and
             owner['owner_lifetime_wall_seconds']+1e-8 >= service, 'Owner cold/service sum double-counts or omits time')
        arm = arms[expected['mode']]
        if arm['family']=='retrieval' and arm['backend']!='eager':
            cumulative = dict.fromkeys(('captures','replays','resident_steps','shadow_updates','capture_host_seconds'),0)
            for record in block:
                for key in cumulative:
                    value = record['draft_runtime']['request'][key]
                    need(finite(value), 'Invalid runtime delta counter')
                    cumulative[key] += value
                    need(near(record['runtime_stats'][key],cumulative[key]) and
                         near(record['draft_runtime']['lifetime'][key],cumulative[key]), 'Runtime cumulative/delta ledger differs')
        computed[expected['length'],expected['mode']] = dict(tokens=warm_tokens,seconds=warm_seconds,
            tps=warm_tokens/warm_seconds,request_ids=[r['request_id'] for r in rest],
            accepted=sum(r['accepted'] for r in rest),blocks=sum(r['blocks'] for r in rest),
            proposed=None if arm['family']=='full_tree' else sum(r['proposed'] for r in rest),
            captures=sum(r.get('draft_runtime',{}).get('request',{}).get('captures',0) for r in rest),
            capture_seconds=sum(r.get('draft_runtime',{}).get('request',{}).get('capture_host_seconds',0.) for r in rest),
            cold_tps=first['tokens']/first['cold_complete_outer_seconds'],service_tps=owner['accounted_service_tps'])
    contexts = lines(root,'results/contexts.jsonl')
    need(len(contexts) == 2 and {r['length'] for r in contexts} == set(LENGTHS), 'Context ledger incomplete')
    for record in contexts:
        need(record['tokens'] == sum(r['tokens'] for r in sd if r['length']==record['length']) and
             finite(record['context_wall_seconds'],True), 'Context token/wall ledger differs')
    relative_analysis = str(analysis.relative_to(root))
    summary = read(root,relative_analysis+'/summary.json'); verified = read(root,relative_analysis+'/VERIFIED_WINNERS.json')
    need(summary['status'] == 'COMPLETE' and summary['problems'] == [] and summary['observed_counts'] == COUNTS and
         summary['winner_status'] == verified['status'] == 'VERIFIED_FROZEN' and verified['authoritative'] is True,
         'Analysis is not strictly complete/authoritative')
    rankings, winners = {}, {}
    for n in LENGTHS:
        ranking = sorted(arms, key=lambda m:(-computed[n,m]['tps'],m)); rankings[str(n)] = ranking; winners[str(n)] = {}
        cell = frozen['per_context'][str(n)]
        need(summary['rankings'][str(n)] == ranking == [r['mode'] for r in cell['ranking']] and
             cell['overall']['mode'] == ranking[0], 'Complete raw primary ranking differs from frozen/analysis')
        for mode in arms:
            metric, actual = summary['metrics'][str(n)][mode],computed[n,mode]
            need(near(metric['pooled_tps'],actual['tps']) and near(metric['warm_seconds'],actual['seconds']) and
                 metric['warm_tokens'] == actual['tokens'] and metric['documents']==3,
                 'Raw primary ratio/sample count differs from analysis')
            need(near(metric['acceptance_length'],actual['accepted']/actual['blocks']) and
                 (metric['acceptance_rate'] is None if actual['proposed'] is None else
                  near(metric['acceptance_rate'],actual['accepted']/actual['proposed'])) and
                 metric['warm_captures']==actual['captures'] and near(metric['warm_capture_seconds'],actual['capture_seconds']) and
                 near(metric['cold_tps'],actual['cold_tps']) and near(metric['accounted_tps'],actual['service_tps']),
                 'Quality/capture/cold/service metrics differ from independent raw records')
        for group in GROUPS:
            eligible = [m for m in ranking if ('full_tree' if arms[m]['family']=='full_tree' else
                'retrieval/'+('graph' if arms[m]['backend']=='cuda_graph' else arms[m]['backend'])) == group]
            winner = eligible[0]; winners[str(n)][group] = winner; record = cell['groups'][group]
            need(record['mode'] == summary['winners']['selected'][str(n)][group] ==
                 verified['winners']['selected'][str(n)][group] == winner and record['arm'] == arms[winner] and
                 record['warm_tps'] == computed[n,winner]['tps'] and record['request_ids'] == computed[n,winner]['request_ids'] and
                 record['warm_tokens'] == computed[n,winner]['tokens'] and near(record['warm_seconds'],computed[n,winner]['seconds']),
                 'One of the eight frozen primary winners differs from independent raw exact-ratio selection')
    analysis_receipt = read(root,relative_analysis+'/ANALYSIS_COMPLETE.json')
    need(analysis_receipt['status'] == 'COMPLETE', 'Analysis artifact receipt is not complete')
    for name, sha in analysis_receipt['artifacts_sha256'].items():
        need(digest(local(root,relative_analysis+'/'+name)) == sha, 'Analysis artifact SHA differs: '+name)
    for name, sha in summary['raw_sha256'].items():
        need(digest(local(root,'results/'+name)) == sha, 'Summary raw SHA differs: '+name)
    for name, info in collection['files'].items():
        need(digest(local(root,name)) == info['sha256'], 'Terminal artifact changed during independent audit: '+name)
    audit_bytes = Path(__file__).read_bytes()
    receipt = dict(status='INDEPENDENT_RAW_AUDIT_PASSED', passed=True,
        audited_at_utc=datetime.now(timezone.utc).isoformat(),counts=COUNTS,logical_requests=538,
        all538_requests_verified=True,all76_owners_verified=True,eight_frozen_choices_verified=True,
        canonical_sd_requests=532,tree_raw_records=112,independent_ar_requests=6,
        terminal_files_verified=len(collection['files']),source_files_verified=len(protocol['source_sha256']),
        baseline_complete_sha256=digest(root/'results/COMPLETE.json'),
        baseline_protocol_sha256=digest(root/'results/protocol.json'),
        frozen_selection_sha256=digest(root/'results/FROZEN_SELECTION.json'),
        analysis_summary_sha256=digest(analysis/'summary.json'),verified_winners_sha256=digest(analysis/'VERIFIED_WINNERS.json'),
        terminal_collection_sha256=digest(root/'TERMINAL_COLLECTION.json'),
        independently_verified_tokenizer_eos=eos,eos_token_id=eos_id,
        independently_recomputed_winners=winners,independently_recomputed_rankings=rankings,
        audit_source_file='INDEPENDENT_RAW_AUDIT_SOURCE.py',audit_source_sha256=hashlib.sha256(audit_bytes).hexdigest(),
        checks=dict(all_terminal_sha_and_sizes=True,uuid_execution_bijection=True,
            all_canonical_eos_exact=True,private_owner_contiguous_ledger=True,recorded_cache_pointers_stable=True,
            cold_repeat_backend_trajectories_equal=True,all_tree_raw_scored_overshoot_verified=True,
            cold_and_service_sums_verified=True,all_primary_exact_ratios_and_rankings_verified=True,
            all_eight_winners_verified=True,all_analysis_artifact_hashes_verified=True),
        scope='Independent saved-record development audit only; no remote queries, original model/raw paths, report logic imports, CI rerun or figure appearance inspection. NativeKV/Q tensor hashes were not collected in this study; pointer stability is not tensor equality. No research-goal completion claim.')
    # All scientific checks precede BOTH exclusive outputs. Failures preserve
    # evidence and do not create a misleading passed receipt.
    with (root/'INDEPENDENT_RAW_AUDIT_SOURCE.py').open('xb') as handle:
        handle.write(audit_bytes)
    with (root/'INDEPENDENT_RAW_AUDIT.json').open('x') as handle:
        json.dump(receipt,handle,indent=2,allow_nan=False);handle.write('\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--analysis-dir',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.root,args.analysis_dir)
    print(json.dumps(dict(status=result['status'],counts=result['counts'],
                         winners=result['independently_recomputed_winners']),indent=2))
