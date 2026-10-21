"""Independent, standard-library-only terminal audit of the six-context curve.

No runner, report, model or shared validator is imported. This is an audit of
recorded evidence, not another GPU forward pass. Bootstrap confidence intervals
are NOT independently recomputed. A pass does not satisfy final confirmation.
External launch/entry/preregistration SHA arguments must come from the saved
pre-launch transport receipt, not be derived from an untrusted export.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).read_bytes()
LENGTHS = (2048, 4096, 6144, 8192, 12288, 16384)
OLD = (4096, 8192)
DOCS = (211, 213, 220, 226, 233, 236, 246, 248, 253, 271, 272, 288, 303, 306, 327, 604)
MODES = ('rescue_graph', 'retrieval32_k4_graph', 'tree32_n8')
ORDER = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))
REMOTE = '/mnt/8T/cl/rescue_experiment_cache/context_curve_20261020_v2'
SESSION = 'rescue_context_curve_v2'
PRIOR_MANIFEST = '0f9e19fddd4cc57a6813c795b9ccdfc6f707bf816ec1323f94e4d45039db1c79'
PRIOR_VERIFIED = 'ba1ca6bfcb320664020969a66ca772f043b5689dcd8b1c41e7196174d6c9cded'
ASSETS = dict(mapper='d08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
    policy='3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
    cost='d92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1')
EXPECTED = dict(reference_requests=64, own_quality_requests=128, baseline_quality_requests=12,
                quality_owners=140, prime_requests=18, timing_requests=576, owners=18)
RUNTIME = ('captures', 'replays', 'resident_steps', 'shadow_updates', 'capture_host_seconds')
TRACE = ('round_index', 'prefix', 'cap', 'proposed', 'accepted', 'terminal', 'workset_tokens')
WORK = ('tokens', 'accepted', 'proposed', 'blocks', 'draft_forwards', 'selector_refreshes',
        'working_gather_layers', 'working_full_gather_layers', 'working_gathered_position_layers',
        'working_incremental_binds', 'draft_shadow_probes', 'projected_positions', 'rpc_calls')


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def digest(value): return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None
def number(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)
def near(a, b): return number(a) and number(b) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)
def ids(value, size=32): return isinstance(value, str) and re.fullmatch('[0-9a-f]{'+str(size)+'}', value) is not None


class Export:
    def __init__(self, root):
        self.root = Path(root).absolute();self.used = {}
        need(not any(p.is_symlink() for p in (self.root, *self.root.parents)), 'Symlink export root')

    def raw(self, name):
        need(isinstance(name, str) and name and '\\' not in name and '\x00' not in name, 'Unsafe artifact path')
        p = PurePosixPath(name)
        need(not p.is_absolute() and '..' not in p.parts and p.as_posix() == name, 'Unsafe relative artifact path')
        path = self.root/name
        need(path.is_file() and not any(v.is_symlink() for v in (path, *path.parents)), 'Missing/symlink artifact: '+name)
        value = path.read_bytes();h = sha(value)
        need(self.used.get(name, h) == h, 'Artifact changed during audit: '+name);self.used[name] = h
        return value

    def json(self, name): return json.loads(self.raw(name))
    def pin(self, name, expected):
        need(digest(expected) and sha(self.raw(name)) == expected, 'External/saved SHA mismatch: '+name)
    def rows(self, name):
        value = self.raw(name)
        need(value and value.endswith(b'\n') and all(r.strip() for r in value.splitlines()), 'Truncated JSONL: '+name)
        result = [json.loads(r) for r in value.splitlines()]
        need(all(isinstance(r, dict) for r in result), 'Non-object JSONL');return result
    def finish(self):
        for name, expected in list(self.used.items()): self.pin(name, expected)
        need(Path(__file__).read_bytes() == SOURCE, 'Auditor source changed')


def canonical(tokens, eos, vocab):
    need(isinstance(tokens, list) and 0 < len(tokens) <= 128 and
         all(type(v) is int and 0 <= v < vocab for v in tokens), 'Invalid canonical tokens')
    need((len(tokens) == 128 or tokens[-1] == eos) and
         (eos not in tokens or tokens[-1] == eos and eos not in tokens[:-1]), 'Noncanonical cap/first EOS')


def tensor(value, shape=None, dtype=None):
    need(isinstance(value, dict) and digest(value.get('sha256')) and isinstance(value.get('shape'), list) and
         all(type(v) is int and v > 0 for v in value['shape']) and isinstance(value.get('dtype'), str) and
         (shape is None or value['shape'] == shape) and (dtype is None or value['dtype'] == dtype), 'Invalid tensor digest/geometry')


def logical(row):
    if row['mode'] == MODES[2]:
        return [row[k] for k in ('generated', 'tokens', 'accepted', 'blocks', 'raw_total_generated',
                'generation_overshoot', 'scored_generated_before_eos')]+[row['upstream']['accept_length_list']]
    need(all(k in row for k in WORK), 'Missing mandatory measured work counter')
    return [row['generated'], [[t[k] for k in TRACE] for t in row['trace']], *[row[k] for k in WORK],
            ('target_probe_rounds' in row, row.get('target_probe_rounds'))]


def linear(row):
    own = row['mode'].startswith('rescue_');trace = row['trace'];progress = 0
    need(trace and row['blocks'] == len(trace), 'Round count/trace mismatch')
    for j, t in enumerate(trace):
        cap, proposed, accepted = (t[k] for k in ('cap', 'proposed', 'accepted'))
        choices = {min(k, 128-progress) for k in ((2, 4) if own else (4,))}
        need(all(type(v) is int for v in (cap, proposed, accepted)) and cap in choices and
             0 <= accepted <= proposed <= cap and proposed > 0 and t['round_index'] == j and
             t['prefix'] == row['length']-1+progress and t['terminal'] is (j == len(trace)-1) and
             type(t['workset_tokens']) is int and 0 < t['workset_tokens'] <= (512 if own else 1024), 'Causal round/action/work mismatch')
        progress += accepted+1
    need(row['accepted'] == sum(t['accepted'] for t in trace) and row['proposed'] == sum(t['proposed'] for t in trace)
         and row['tokens'] in (progress, progress-1) and progress <= 129 and
         near(row['acceptance_rate'], row['accepted']/row['proposed']), 'Aggregate acceptance/commit mismatch')
    need(row['draft_forwards'] == sum(t['proposed']+int(t['accepted'] == t['proposed'] and not t['terminal']) for t in trace),
         'Draft forward/catchup count mismatch')


def quality(row, eos, vocab):
    linear(row);own = row['mode'].startswith('rescue_');generated = [];cursor = 0;nonzero = False
    need(len(row['observed_rounds']) == row['blocks'] and len(row['queries']) == row['draft_forwards'], 'Incomplete quality rounds/queries')
    for t, observed in zip(row['trace'], row['observed_rounds']):
        selected = observed['selected'];p, a = t['proposed'], t['accepted'];j = t['round_index']
        need(all(observed[k] == t[k] for k in ('round_index', 'prefix', 'accepted')) and
             observed['progress'] == len(generated) and observed['remaining'] == 128-len(generated) and
             selected == sorted(set(selected)) and len(selected) == t['workset_tokens'] and
             all(type(v) is int and 0 <= v < t['prefix'] for v in selected), 'Selection/action alignment mismatch')
        if own: need(len(selected) == 512 and 0 in selected and set(range(t['prefix']-64, t['prefix'])) <= set(selected), 'Mandatory first/recent Native context missing')
        draft, checks = observed['draft'], observed['checks']
        need(len(draft) == p and len(checks) == p+1 and all(type(v) is int and 0 <= v < vocab for v in draft+checks) and
             (eos not in draft or draft[-1] == eos and eos not in draft[:-1]) and (p == t['cap'] or draft[-1] == eos), 'Draft/Target checks or EOS stopping invalid')
        need(next((k for k in range(p) if draft[k] != checks[k]), p) == a, 'Accepted prefix differs from recorded actual agreement')
        generated.extend(draft[:a])
        if len(generated) < row['tokens']: generated.append(checks[a])
        vector = observed.get('budget_features')
        if own:
            need((p < 2 and vector is None) or (p >= 2 and isinstance(vector, list) and len(vector) == 12 and
                 all(type(v) in (int, float) and math.isfinite(v) for v in vector)), 'Missing causal controller features')
            if vector is not None:
                previous = .75 if j == 0 else 1. if row['trace'][j-1]['accepted'] == row['trace'][j-1]['proposed'] else row['trace'][j-1]['accepted']/(row['trace'][j-1]['accepted']+1)
                for k, expected in ((0, math.log(t['prefix'])/10), (4, (j % 4)/4),
                                    (5, observed['progress']/128), (6, observed['remaining']/128), (7, previous)):
                    need(math.isclose(vector[k], expected, rel_tol=2e-6, abs_tol=2e-6), 'Controller feature is not causal committed state')
            for evidence in observed['injection_evidence']:
                w, native, pos, count = (evidence[k] for k in ('working_value', 'native_value', 'positions', 'different_elements'))
                tensor(w, dtype='torch.bfloat16');tensor(native, shape=w['shape'], dtype='torch.bfloat16')
                need(evidence['layer'] == 27 and len(w['shape']) == 2 and w['shape'][0] == 64 and
                     len(pos) == len(set(pos)) == 64 and set(pos) <= set(selected) and
                     all(0 < v < t['prefix']-64 for v in pos) and evidence['native_matches_memo'] is True and
                     type(count) is int and 0 <= count <= math.prod(w['shape']) and
                     ((count == 0) == (w['sha256'] == native['sha256'])), 'Invalid same-position BF16 V injection evidence')
                nonzero |= count > 0
        else: need(vector is None and not observed.get('injection_evidence'), 'Retrieval ran Mapper/controller')
        for depth in range(p+int(a == p and not t['terminal'])):
            q = row['queries'][cursor];cursor += 1;proposal = depth < p
            tensor(q['query'], [1, 12, 1, 128], 'torch.bfloat16')
            summary = own and proposal and depth < 2
            need(q['logits'] is proposal and q['summary_requested'] is summary and q['summary_count'] == int(summary), 'Query/summary schedule differs')
    need(generated == row['generated'] and cursor == len(row['queries']), 'Quality checks do not reconstruct canonical output')
    need(row['nonzero_injection'] is own and (nonzero if own else row['projected_positions'] == row['draft_shadow_probes'] == 0), 'No genuine nonzero Mapper, or retrieval injected')
    terminal = row['trace'][-1]
    for role, heads, valid, dtype in (('native_t', 4, terminal['prefix']+terminal['proposed']+1, 'torch.float32'),
                                     ('native_s', 2, terminal['prefix']+terminal['proposed'], 'torch.bfloat16')):
        cache = row[role]
        need(cache['lengths'] == [valid]*28 and len(cache['layers']) == 28, 'Uncropped 28-layer Native span missing')
        for layer in cache['layers']:
            for kind in ('key', 'value'): tensor(layer[kind], [1, heads, valid, 128], dtype)


def graph(row, fresh=False):
    signature, stats = row['owner_signature'], row['runtime_stats']
    for role in ('native_t', 'native_s', 'working'):
        need(all(type(signature[role].get(k)) is int and signature[role][k] > 0 for k in ('identity', 'key_ptr', 'value_ptr', 'capacity')), 'Private cache identity missing')
        if role != 'working':need(signature[role]['capacity'] >= row['length']+128, 'Native capacity insufficient')
    if row['mode'].endswith('_eager'):
        need(stats['entries'] == 0 and 'runtime' not in signature, 'Eager quality hid a Graph runtime');return
    d = row['draft_runtime'];delta = d['request'];keys = stats['graph_keys'];own = row['mode'] == MODES[0]
    need(d['mode'] == 'cuda_graph' and stats['max_graph_entries'] == 128 and signature['runtime']['bound_ptrs'] ==
         [signature[role][k] for role in ('native_s', 'working') for k in ('key_ptr', 'value_ptr')], 'Graph/pointer bindings changed')
    need(all(number(delta[k]) and (k == 'capture_host_seconds' or type(delta[k]) is int) for k in RUNTIME) and
         delta['replays'] == delta['shadow_updates'] == row['draft_forwards'] > 0 and delta['resident_steps'] == 0 and
         (delta['captures'] > 0) is (delta['capture_host_seconds'] > 0), 'Capture/replay/copy work omitted')
    need(len(keys) == len({json.dumps(k, sort_keys=True) for k in keys}) == stats['entries'] == stats['captures'] <= 128 and
         all(isinstance(k, list) and len(k) == 4 and type(k[0]) is int and k[0] > 0 and type(k[1]) is type(k[2]) is bool for k in keys), 'Graph keys/entry ledger mismatch')
    required = set()
    for t in row['trace']:
        required.update((t['workset_tokens']+j, True, own and j <= 2) for j in range(1, t['proposed']+1))
        if t['accepted'] == t['proposed'] and not t['terminal']:required.add((t['workset_tokens']+t['proposed']+1, False, False))
    need(required <= {tuple(k[:3]) for k in keys}, 'Actual nonpadded Graph shapes absent')
    if not own:need(d['retain_query'] is (row['phase'] == 'baseline_quality') and d['max_graph_entries'] == 128 and
                     all(k[2] is False and k[3] == [[], [], []] for k in keys), 'Retrieval Graph owns extra Mapper/query work')
    if fresh:need(delta['captures'] > 0 and all(near(delta[k], stats[k]) for k in RUNTIME), 'Fresh owner borrowed quality/old Graph work')


def audit(root, analysis, launch_sha, entry_sha, prereg_sha):
    x = Export(root);x.pin('LAUNCH_MANIFEST.json', launch_sha)
    collection = x.json('COLLECTION.json');inventory = collection['original_file_sha256']
    need(inventory and collection['remote_writes'] is False, 'No read-only terminal collection')
    for boundary in ('handles_before', 'handles_after'):
        h = collection[boundary]
        need(h['tmux_present'] is False and h['execution_processes'] == [] and h['exit']['exit_code'] == 0, 'Experiment not terminal')
    for name, expected in inventory.items():x.pin(name, expected)
    evidence = x.json('results/EVIDENCE_MANIFEST.json')['file_sha256']
    need(evidence == {n[8:]: h for n, h in inventory.items() if n.startswith('results/') and n != 'results/EVIDENCE_MANIFEST.json'}, 'Raw export inventory mismatch')
    for name, expected in evidence.items():x.pin('results/'+name, expected)
    manifest = x.json('LAUNCH_MANIFEST.json')
    need(manifest['root'] == REMOTE and manifest['session'] == SESSION and manifest['new_generation_requests'] == 798 and
         manifest['launches_on_receive'] is False, 'Wrong isolated launch scope')
    for name, expected in manifest['file_sha256'].items():x.pin(name, expected)
    for name, expected in (('tools/run_context_curve.py', entry_sha), ('results/CURVE_SOURCE.py', entry_sha),
            ('tools/preregistration.json', prereg_sha), ('results/PREREGISTRATION.json', prereg_sha),
            ('results/prior_local_seal/FINAL_ARTIFACTS_MANIFEST.json', PRIOR_MANIFEST),
            ('results/prior_local_seal/VERIFIED_COMPLETE.json', PRIOR_VERIFIED)):x.pin(name, expected)
    for name in ('LAUNCH_ATTEMPT.json', 'RUN_STARTED.json'):
        r = x.json(name);need(r['session'] == SESSION and r['manifest_sha256'] == launch_sha and type(r['pid']) is int, 'Attempt/start not launch-linked')
    end = x.json('PIPELINE_EXIT.json');need(end['exit_code'] == 0 and end['complete_sha256'] == sha(x.raw('results/COMPLETE.json')), 'Pipeline terminal linkage invalid')
    preflight = x.json('GPU_PREFLIGHT.json')
    need(preflight['passed'] is True and preflight['tests_run'] == len(preflight['cases']) == 8 and preflight['skipped'] == 0 and
         x.json('DEPENDENCY_PREFLIGHT.json')['passed'] is True, 'Actual preflight did not pass')
    p = x.json('results/protocol.json');prereg = x.json('results/PREREGISTRATION.json')
    need(p['status'] == 'FROZEN_CONTEXT_CURVE' and prereg['status'] == 'FROZEN_BEFORE_EXECUTION', 'Missing pre-execution protocol freeze')
    for key, expected in dict(lengths=list(LENGTHS), source_indices=list(DOCS), modes=list(MODES), seed=20261028,
                              repeats=2, max_new_tokens=128, graph_max_entries=128).items():
        need(p[key] == prereg[key] == expected, 'Frozen matrix/config mismatch: '+key)
    need(p['expected_requests'] == 798 and p['expected_counts'] == EXPECTED and p['entry_source_sha256'] == entry_sha and
         p['preregistration_sha256'] == prereg_sha and p['max_live_owners'] == 1 and p['goal_complete'] is p['fresh_confirmation'] is False and
         all(p[k] is True for k in ('no_training', 'no_search', 'no_new_confirmation_access', 'paused_test_untouched')), 'Frozen execution scope changed')
    need(prereg['output_root'] == REMOTE and p['args']['output'] == REMOTE+'/results', 'Wrong isolated execution root')
    for role, expected in ASSETS.items():need(p[role+'_sha256'] == expected and p['assets_sha256'][p['args'][role]] == expected, 'Frozen asset changed: '+role)
    reuse = x.json('results/REUSE_RECEIPT.json');x.pin('results/REUSE_RECEIPT.json', p['reuse_receipt_sha256'])
    need(reuse['prior_manifest_sha256'] == p['args']['prior_manifest_sha256'] == PRIOR_MANIFEST and
         reuse['prior_verified_sha256'] == p['args']['prior_verified_sha256'] == PRIOR_VERIFIED, 'External prior seal changed')
    seal = x.json('results/prior_local_seal/VERIFIED_COMPLETE.json');prior = x.json('results/prior_local_seal/FINAL_ARTIFACTS_MANIFEST.json')['file_sha256']
    need(seal['status'] == 'VERIFIED_COMPLETE' and seal['passed'] is True and seal['final_artifacts_manifest_sha256'] == PRIOR_MANIFEST, 'Prior incomplete')
    remap = {'results/reference.jsonl':'reused_reference.jsonl', 'results/own_quality.jsonl':'reused_own_quality.jsonl',
             'results/data/documents.jsonl':'documents.jsonl', 'results/data/prime.json':'prime.json'}
    for name, expected in reuse['old_artifact_pins'].items():
        need(prior[name] == expected, 'Prior consumed bytes not authenticated');x.pin('results/'+remap.get(name, 'prior_consumed/'+name), expected)
    old = x.json('results/prior_consumed/results/protocol.json')
    need(all(p['source_sha256'].get(k) == v for k, v in old['source_sha256'].items()) and p['assets_sha256'] ==
         {k:v for k, v in old['assets_sha256'].items() if k in p['assets_sha256']} and
         all(p['environment'][k] == old['environment'][k] for k in ('nvidia_smi_gpu', 'torch', 'cuda', 'target_dtype', 'draft_dtype', 'prefix_dtype', 'tf32')),
         'Inherited source/assets/environment differs')
    need(p['configs'][MODES[0]] == old['configs'][MODES[0]], 'Frozen Mapper/controller execution configuration changed')
    retrieval = p['configs'][MODES[1]]
    need(all(retrieval[k] == v for k, v in dict(length_policy='fixed', fixed_k=4, draft_execution='cuda_graph',
        coverage_execution='eager', inject_tokens=0, injection_strength=0., injection_layers=[], max_new_tokens=128,
        prefill_precision='bfloat16', schedule='serial', profile=False).items()), 'Retrieval execution configuration changed')
    need(p['configs'][MODES[2]] == dict(nodes=8, top_chunks=32, max_depth=10, threshold=.7, chunk_size=32, refresh_every=4,
        flash_prefill=True, hybrid_tree=True, prefill_precision='bfloat16', mapper=False, controller=False), 'Tree execution configuration changed')
    for name, expected in p['source_sha256'].items():x.pin('results/source/'+name, expected)
    seq = p['shared_warm_sequence'];need(seq == old['shared_warm_sequence'] and len(seq) == 32 and
        {(r['source_index'], r['repeat']) for r in seq} == {(d, r) for d in DOCS for r in (0, 1)}, 'Original shared paired warm sequence changed')
    docs = x.rows('results/documents.jsonl');need([r['source_index'] for r in docs] == list(DOCS) and
        x.json('results/prime.json') == next(r for r in docs if r['source_index'] == 271), 'Old natural documents/prime identity changed')
    inputs = {}
    for r in docs:
        prefix, suffix, body = (r[k] for k in ('prefix_ids', 'suffix_ids', 'document_ids'))
        need(len(prefix)+len(suffix) == 30 and len(body) == 16354 and all(type(t) is int and t >= 0 for t in prefix+body+suffix), 'Natural body/prompt invalid')
        for n in LENGTHS:inputs[f"{r['source_index']}:{n}"] = sha(json.dumps(prefix+body[:n-30]+suffix).encode())
    need(inputs == p['input_sha256'] == reuse['canonical_inputs_all96'] and all(inputs[k] == v for k, v in old['input_sha256'].items()), 'Exact 96 inputs changed')
    records = {name:x.rows('results/'+name+'.jsonl') for name in ('reference', 'reused_reference', 'own_quality', 'reused_own_quality', 'baseline_quality', 'quality_owners', 'prime', 'runtime', 'owners', 'tree_raw', 'gpu_diagnostics')}
    counts = dict(zip(EXPECTED, (len(records[k]) for k in ('reference', 'own_quality', 'baseline_quality', 'quality_owners', 'prime', 'runtime', 'owners'))))
    need(counts == EXPECTED and len(records['reused_reference']) == 32 and len(records['reused_own_quality']) == 64 and
         len(records['tree_raw']) == 198 and len(records['gpu_diagnostics']) == 36, 'Incomplete terminal request/owner/raw matrix')
    need(not (x.root/'results/FAILURE.json').exists(), 'Failed run cannot pass a complete terminal audit')
    done = x.json('results/COMPLETE.json');need(done['passed'] is True and done['counts'] == done['expected_counts'] == EXPECTED and
         done['logical_requests'] == 798 and all(done[k] is True for k in ('all_canonical', 'quality_gate_passed', 'owners_closed', 'sources_unchanged', 'assets_unchanged', 'dependencies_unchanged')), 'Invalid COMPLETE receipt')
    new = sum((records[k] for k in ('reference', 'own_quality', 'baseline_quality', 'prime', 'runtime')), [])
    need(len(new) == 798 and len({r['request_id'] for r in new}) == 798 and all(ids(r['request_id']) for r in new) and
         all(type(r['execution_index']) is int for r in new) and sorted(r['execution_index'] for r in new) == list(range(798)) and
         not {r['request_id'] for r in new} & {r['request_id'] for r in records['reused_reference']+records['reused_own_quality']}, 'UUID/index/new-vs-reused bijection invalid')
    for r in new:need(r['gpu_snapshot_before']['nvidia_smi_gpu'] == p['environment']['nvidia_smi_gpu'], 'Generation GPU identity changed')
    refs = x.json('results/references.json');eos, vocab = p['tokenizer_eos_token_id'], p['vocab_size']
    need(set(refs) == set(inputs), '96 canonical reference cells missing')
    for tokens in refs.values():canonical(tokens, eos, vocab)
    for name, lengths in (('reference', tuple(n for n in LENGTHS if n not in OLD)), ('reused_reference', OLD)):
        rr = records[name];need({(r['source_index'], r['length']) for r in rr} == {(d, n) for d in DOCS for n in lengths}, 'Canonical reference matrix differs')
        for r in rr:
            need(r['generated'] == refs[f"{r['source_index']}:{r['length']}"], 'Canonical reference row differs')
            if name == 'reference':need(r['phase'] == 'reference' and r['mode'] == 'canonical_ar' and r['performance_eligible'] is False, 'AR row changed phase/timing eligibility')
    old_refs = x.json('results/prior_consumed/results/references.json')
    need(all(refs[k] == v for k, v in old_refs.items()) and len(old_refs) == 32, 'Prior canonical references changed')
    provenance = x.json('results/reference_provenance.json')
    need(provenance['new_reference_requests'] == 64 and provenance['reused_reference_requests'] == 32 and provenance['no_new_prime_ar'] is True and
         provenance['reused_rows_sha256'] == sha(x.raw('results/reused_reference.jsonl')) and provenance['refs'] ==
         {k:dict(origin='reused' if int(k.split(':')[1]) in OLD else 'new', input_sha256=v) for k, v in inputs.items()}, 'New-vs-reused AR provenance changed')
    quality_rows = records['own_quality']+records['baseline_quality'];qmap = {};qowners = records['quality_owners']+x.rows('results/prior_consumed/results/quality_owners.jsonl')
    qo = {r['request_id']:r for r in qowners};need(len(qo) == 204, 'Quality owner/request IDs not bijective')
    need(sorted(r['execution_index'] for r in records['reference']) == list(range(64)) and
         sorted(r['execution_index'] for r in quality_rows) == list(range(64, 204)), 'Reference/quality/timing phase order invalid')
    for r in quality_rows+records['reused_own_quality']:
        key = (r['source_index'], r['length'], r['mode']);need(key not in qmap, 'Duplicate quality cell');qmap[key] = r
        own = r['mode'].startswith('rescue_');need(r['performance_eligible'] is False and r['phase'] == ('own_quality' if own else 'baseline_quality') and
             r['generated'] == refs[f"{r['source_index']}:{r['length']}"] and r['output_agrees'] is True and r['tokens'] == len(r['generated']), 'Quality not untimed/canonical')
        quality(r, eos, vocab);graph(r, fresh=True);o = qo[r['request_id']]
        need(o['owner_id'] == r['owner_id'] and o['engine_closed'] is o['hook_restored'] is True and o['closed_runtime_stats']['entries'] == 0 and
             o['final_runtime_stats'] == r['runtime_stats'], 'Quality owner not closed with matching final state')
    expected_quality = {(d, n, m) for d in DOCS for n in LENGTHS for m in ('rescue_eager', MODES[0])} | {(211, n, m) for n in LENGTHS for m in ('retrieval32_k4_eager', MODES[1])}
    need(set(qmap) == expected_quality, 'All96 own /6 retrieval backend pairs required')
    for d, n, mode in [(d, n, MODES[0]) for d in DOCS for n in LENGTHS]+[(211, n, MODES[1]) for n in LENGTHS]:
        a, b = qmap[d, n, mode.replace('_graph', '_eager')], qmap[d, n, mode]
        need(logical(a) == logical(b) and all(a[k] == b[k] for k in ('observed_rounds', 'queries', 'native_t', 'native_s')), 'Eager/Graph Native/Q/selected/features/output evidence differs')
    gate = x.json('results/QUALITY_PASSED.json');need(gate['passed'] is gate['all96_own_pairs'] is gate['quality_owners_closed'] is gate['nonzero_mapper'] is True and
         gate['new_own_pairs'] == 64 and gate['reused_own_pairs'] == 32 and gate['counts'] == dict(EXPECTED, prime_requests=0, timing_requests=0, owners=0), 'Pre-timing quality gate invalid')
    points = timing(records, refs, qmap, seq, p)
    a = Export(analysis);summary = a.json('summary.json');receipt = a.json('ANALYSIS_COMPLETE.json')
    need(summary['status'] == receipt['status'] == 'COMPLETE' and summary['problems'] == [] and summary['observed_counts'] == EXPECTED and
         summary['statistics']['document_order'] == list(DOCS) and summary['goal_complete'] is summary['fresh_confirmation'] is False, 'Strict completed development report required')
    for name, expected in receipt['file_sha256'].items():a.pin(name, expected)
    for n in LENGTHS:
        for mode in MODES:
            actual, saved = points[str(n)][mode], summary['statistics']['contexts'][str(n)][mode]
            for key, value in actual.items():need(saved.get(key) == value if isinstance(value, int) else near(saved.get(key), value), 'Independent pooled point differs: '+str((n, mode, key)))
    contrasts = {(v['length'], v['baseline_mode']):v for v in summary['statistics']['contrasts']}
    need(set(contrasts) == {(n, m) for n in LENGTHS for m in MODES[1:]}, 'Report contrast matrix differs')
    for n, mode in contrasts:
        expected = 100*(points[str(n)][MODES[0]]['pooled_tps']/points[str(n)][mode]['pooled_tps']-1)
        need(math.isclose(contrasts[n, mode]['point_percent'], expected, rel_tol=1e-6, abs_tol=1e-8), 'Independent signed TPS contrast differs')
    x.finish();a.finish()
    return dict(status='INDEPENDENT_CONTEXT_CURVE_RAW_AUDIT_PASSED', passed=True, goal_complete=False, fresh_confirmation=False,
        audited_at_utc=datetime.now(timezone.utc).isoformat(), external_sha256=dict(launch=launch_sha, entry=entry_sha, preregistration=prereg_sha,
        prior_manifest=PRIOR_MANIFEST, prior_verified=PRIOR_VERIFIED), counts=counts, logical_requests=798,
        quality_owners=140, timing_owners=18, canonical_cells=96, reused_reference_requests=32, reused_quality_requests=64,
        document_paired_repeats=2, independent_points=points, bootstrap_independently_recomputed=False,
        recorded_tensor_evidence_only=True, used_file_sha256=x.used, analysis_sha256=a.used,
        limitations=['Recorded Native/Q/injection tensor hashes, not a new GPU forward proof.',
            'Bootstrap confidence intervals are not independently implemented or recomputed.',
            'Reused16 development docs; private owner blocks, finite capture history and hardware-drift limitations unchanged.',
            'Inherited seal is externally pinned, but only saved consumed subset is rehashed; no fresh confirmation or production-speed claim.'])


def timing(records, refs, qmap, seq, p):
    prime, warm, owners = (records[k] for k in ('prime', 'runtime', 'owners'))
    plan = [(n, MODES[i]) for n, order in zip(LENGTHS, ORDER) for i in order]
    need(p['owner_order'] == [dict(length=n, mode=m) for n, m in plan] and
         [(r['length'], r['mode']) for r in prime] == [(r['length'], r['mode']) for r in owners] == plan, 'Owner block order changed')
    owner_ids = [r['owner_id'] for r in owners+records['quality_owners']]
    need(all(ids(v) for v in owner_ids) and len(set(owner_ids)) == 158, 'Private quality/timing owner IDs reused')
    rawtree = {r['request_id']:r for r in records['tree_raw']};need(len(rawtree) == 198, 'Duplicate raw tree UUID')
    repeated = {};points = {str(n):{} for n in LENGTHS}
    for j, (cold, owner) in enumerate(zip(prime, owners)):
        block = [r for r in warm if r['owner_id'] == cold['owner_id']];allrows = [cold]+block;n, mode = plan[j]
        need(len(block) == 32 and [{k:r[k] for k in ('source_index', 'repeat')} for r in block] == seq and
             [r['warm_index'] for r in block] == list(range(32)) and cold['source_index'] == 271 and
             [r['execution_index'] for r in allrows] == list(range(204+33*j, 204+33*(j+1))), 'Prime32 ordered paired block incomplete')
        need(owner['maximum_live_owners'] == 1 and owner['owner_id'] == cold['owner_id'] and owner['prime_request_id'] == cold['request_id'] and
             owner['warm_request_ids'] == [r['request_id'] for r in block] and owner['engine_closed'] is owner['hook_restored'] is True and
             owner['closed_runtime_stats']['entries'] == 0 and owner['final_runtime_stats'] == block[-1]['runtime_stats'], 'Owner lifecycle/closure mismatch')
        for k in RUNTIME:
            if k in owner['final_runtime_stats']:need(near(owner['final_runtime_stats'][k], owner['closed_runtime_stats'][k]), 'Runtime work vanished during owner close')
        totals = dict.fromkeys(RUNTIME, 0);last_keys = set()
        for r in allrows:
            own = mode == MODES[0];need(r['mode'] == mode and r['length'] == n and r['owner_signature'] == owner['owner_signature'] and
                r['phase'] == ('cold_prime' if r is cold else 'warm') and r['performance_eligible'] is True and
                r['observer_enabled'] is r['profile_enabled'] is False and r['mapper_executed'] is r['controller_executed'] is own and
                'timeline' not in r and 'cost_samples' not in r, 'Timed owner/method/instrumentation differs')
            need(r['generated'] == refs[f"{r['source_index']}:{n}"] and r['output_agrees'] is True and r['tokens'] == len(r['generated']) and
                 number(r['outer_seconds'], True) and near(r['outer_tps'], r['tokens']/r['outer_seconds']) and
                 number(r['total_seconds'], True) and r['outer_seconds']+1e-6 >= r['total_seconds'], 'Full generate outer/output ledger invalid')
            key = (n, mode, r['source_index']);signature = logical(r)
            if key in repeated:need(repeated[key] == signature, 'Same-input prime/repeat trajectory changed')
            repeated[key] = signature
            if mode != MODES[2]:
                linear(r);graph(r, fresh=r is cold)
                if own or r['source_index'] == 211:need(logical(qmap[r['source_index'], n, mode]) == signature, 'Untimed quality/timing trajectory changed')
                d = r['draft_runtime']['request'];stats = r['runtime_stats'];keys = {json.dumps(k, sort_keys=True) for k in stats['graph_keys']}
                need(last_keys <= keys and len(keys-last_keys) == d['captures'], 'Late Graph capture omitted/evicted');last_keys = keys
                for k in RUNTIME:
                    totals[k] += d[k];need(near(stats[k], totals[k]) and near(r['draft_runtime']['lifetime'][k], totals[k]), 'Persistent paid capture/work ledger mismatch')
            else:
                raw = rawtree[r['request_id']];result = raw['result'];up = r['upstream'];accept = up['accept_length_list'];rawtotal = up['raw_total_generated']
                need(raw['execution_index'] == r['execution_index'] and raw['outer_seconds'] == r['outer_seconds'] and
                     all(raw[k] == r[k] for k in ('phase', 'mode', 'length', 'source_index', 'owner_id')) and
                     result['upstream'] == up and result['generated'] == r['wrapper_generated'] and result['tokens'] == r['wrapper_tokens'], 'Tree raw wrapper/outer binding mismatch')
                need(r['tree_policy'] == p['configs'][MODES[2]] and r['observed_full_config'] == dict(target_use_flash_prefill=True, target_use_hybrid_tree_attn=True,
                     draft_use_flash_prefill=True, use_retrieval_cache=True, retrieve_top_k=32, retrieval_chunk_size=32, retrieve_every_n_steps=4), 'Actual tree/Flash/hybrid configuration changed')
                need(isinstance(accept, list) and len(accept) == r['blocks'] and all(type(v) is int and 0 <= v <= 10 for v in accept) and
                     r['accepted'] == sum(accept) and rawtotal == sum(v+1 for v in accept) and
                     r['raw_total_generated'] == rawtotal and r['generation_overshoot'] == up['generation_overshoot'] == rawtotal-len(r['scored_generated_before_eos']) and
                     up['output_token_ids'] == r['scored_generated_before_eos'] and up['total_generated'] == len(r['scored_generated_before_eos']) and
                     r['wrapper_generated'] == r['generated'] and r['wrapper_tokens'] == r['tokens'] and r['raw_tail_ids_available'] is False and
                     rawtotal >= r['tokens'] and r['proposed'] is r['acceptance_rate'] is None, 'Tree path/overshoot ledger mismatch')
        seconds, tokens = sum(r['outer_seconds'] for r in block), sum(r['tokens'] for r in block)
        service = owner['constructor_and_detach_seconds']+cold['outer_seconds']+seconds+owner['close_seconds']
        need(near(owner['warm_outer_seconds_sum'], seconds) and owner['warm_tokens'] == tokens and owner['prime_tokens'] == cold['tokens'] and
             near(owner['cold_prime_outer_seconds'], cold['outer_seconds']) and near(owner['constructor_and_detach_seconds'], cold['constructor_and_detach_seconds']) and
             near(cold['cold_complete_outer_seconds'], cold['constructor_and_detach_seconds']+cold['outer_seconds']) and
             near(owner['accounted_service_seconds'], service) and owner['accounted_service_tokens'] == tokens+cold['tokens'] and
             near(owner['accounted_service_tps'], (tokens+cold['tokens'])/service) and owner['owner_lifetime_wall_seconds']+1e-8 >= service, 'Cold/warm/close service cost omitted')
        for phase, boundary, index in (('owner_start', records['gpu_diagnostics'][2*j], 204+33*j), ('owner_end', records['gpu_diagnostics'][2*j+1], 204+33*(j+1))):
            need(boundary['phase'] == phase and boundary['execution_index'] == index and boundary['owner_id'] == owner['owner_id'] and
                 boundary['mode'] == mode and boundary['length'] == n, 'Outside-clock owner telemetry order mismatch')
        accepted, blocks = sum(r['accepted'] for r in block), sum(r['blocks'] for r in block)
        points[str(n)][mode] = dict(documents=16, tokens=tokens, seconds=seconds, pooled_tps=tokens/seconds,
            accepted=accepted, blocks=blocks, acceptance_length=accepted/blocks)
    need(set(rawtree) == {r['request_id'] for r in prime+warm if r['mode'] == MODES[2]}, 'Extra/missing tree raw records')
    return points


def save(value, output):
    output = Path(output).absolute();need(not output.exists() and not any(v.is_symlink() for v in (output, *output.parents)), 'Audit output must be fresh')
    output.mkdir(parents=True)
    with (output/'audit_source.py').open('xb') as handle:handle.write(SOURCE)
    value = dict(value, audit_source_sha256=sha(SOURCE))
    with (output/'AUDIT.json').open('x') as handle:json.dump(value, handle, indent=2, allow_nan=False);handle.write('\n')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path);parser.add_argument('--analysis', type=Path, required=True);parser.add_argument('--output', type=Path, required=True)
    for name in ('launch-sha256', 'entry-sha256', 'preregistration-sha256'):parser.add_argument('--'+name, required=True)
    args = parser.parse_args();value = audit(args.root, args.analysis, args.launch_sha256, args.entry_sha256, args.preregistration_sha256)
    save(value, args.output);print(value['status'])


if __name__ == '__main__':main()
