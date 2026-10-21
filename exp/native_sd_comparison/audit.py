"""Independent saved-record audit; stdlib + NumPy only, no experiment imports.

This inspects saved digests/receipts, not GPU tensors or model weights. It never
opens a saved remote absolute path and creates only a new audit output folder.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

LENGTHS = (2048, 4096, 6144, 8192, 12288, 16384)
DOCS = (211, 213, 220, 226, 233, 236, 246, 248, 253, 271, 272, 288, 303, 306, 327, 604)
MODES = ('native_eager', 'rescue_eager', 'rescue_graph')
ORDER = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))
TRACE = ('round_index', 'prefix', 'cap', 'proposed', 'accepted', 'terminal', 'workset_tokens')
COUNTERS = ('tokens', 'accepted', 'proposed', 'blocks', 'draft_forwards', 'selector_refreshes',
            'working_gather_layers', 'working_full_gather_layers', 'working_gathered_position_layers',
            'working_incremental_binds', 'draft_shadow_probes', 'projected_positions', 'rpc_calls')
OPTIONAL = ('target_probe_rounds', 'selection_similarity_elements', 'injection_condition_steps',
            'static_affine_positions', 'working_restored_value_positions', 'rpc_bytes_sent')
ZERO = ('selector_refreshes', 'working_gather_layers', 'working_full_gather_layers',
        'working_gathered_position_layers', 'working_incremental_binds', 'target_probe_rounds',
        'draft_shadow_probes', 'projected_positions', 'rpc_calls')
RUNTIME = ('captures', 'replays', 'resident_steps', 'shadow_updates', 'capture_host_seconds')
PINS = {
    'manifest': '03431cb5772847fd6931394419a987d8da62421c3ccc4e9d1b7a7d417285c7c9',
    'entry': '693a621903dab568a2b100f96ee5cf8c733df70c9990478cefe2aaed24b01e03',
    'prereg': '429b1caabfa055c0260bcbdf94a206ddf655cc018a220b742190f2a4c82387fc',
    'prior_manifest': '70e46d52c3f3f57b1efa8d1633b665f345d54d7656529c9b09e2d25e185bf41e',
    'prior_verified': 'c5bf8390ec5349c6d7ce29a688d3b9b13d95b6fd11f006e46a7522b9be42dffc',
    'old_manifest': '002545801b8a15fdb94afb083c61dd76633e423ec828f7a3d7e2c1fc6fbaf40a',
}
ASSETS = {
    'mapper_sha256': 'd08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
    'policy_sha256': '3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
    'cost_sha256': 'd92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1',
}
CONTRASTS = (('matched_eager', 'rescue_eager', 'native_eager'),
             ('deployment_combination', 'rescue_graph', 'native_eager'),
             ('draft_graph_execution', 'rescue_graph', 'rescue_eager'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def close(a, b):
    return (type(a) in (int, float) and type(b) in (int, float) and
            math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8))


def digest(d):
    return (isinstance(d, dict) and re.fullmatch('[0-9a-f]{64}', str(d.get('sha256'))) is not None and
            isinstance(d.get('shape'), list) and bool(d['shape']) and
            all(type(v) is int and v > 0 for v in d['shape']) and d.get('dtype') in ('torch.float32', 'torch.bfloat16'))


def trajectory(r):
    return [[t[k] for k in TRACE] for t in r['trace']]


def equal_work(a, b):
    return (a['generated'] == b['generated'] and trajectory(a) == trajectory(b) and
            all(k in a and k in b and a[k] == b[k] for k in COUNTERS) and
            all((k in a) == (k in b) and (k not in a or a[k] == b[k]) for k in OPTIONAL))


class Inspector:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.issues = []
        self.checks = 0
        self.pinned_files = 0

    def check(self, condition, message):
        self.checks += 1
        if not condition and message not in self.issues:
            self.issues.append(message)

    def path(self, name, base=None):
        part = PurePosixPath(name)
        if (part.is_absolute() or '..' in part.parts or not part.parts or '\\' in name or
                '\x00' in name or part.as_posix() != name):
            raise ValueError('Unsafe inventory path: ' + str(name))
        path = (self.root if base is None else Path(base).absolute()).joinpath(*part.parts)
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Symlink evidence path: ' + str(path))
        return path

    def obj(self, name):
        value = json.loads(self.path(name).read_bytes())
        if not isinstance(value, dict):
            raise ValueError('Not a JSON object: ' + name)
        return value

    def rows(self, name):
        raw = self.path(name).read_bytes()
        if not raw.endswith(b'\n'):
            raise ValueError('Truncated JSONL: ' + name)
        rows = [json.loads(v) for v in raw.splitlines()]
        if not all(isinstance(v, dict) for v in rows):
            raise ValueError('Not JSONL objects: ' + name)
        return rows

    def pin(self, name, value, base=None):
        self.check(re.fullmatch('[0-9a-f]{64}', str(value)) is not None and sha(self.path(name, base)) == value,
                   'SHA mismatch: ' + str(name))
        self.pinned_files += 1

    def inventories(self, p):
        terminal = self.root.parent
        exported = self.obj('EVIDENCE_MANIFEST.json')['file_sha256']
        for name, value in exported.items():
            self.pin(name, value)
        collected = json.loads(self.path('COLLECTION.json', terminal).read_bytes())
        stage = json.loads(self.path('PIPELINE_EXIT.json', terminal).read_bytes())
        self.check(stage['exit_code'] == 0 and stage['complete_sha256'] == sha(self.root/'COMPLETE.json'), 'Outer stage did not bind normal exit to COMPLETE')
        self.check(collected['remote_writes'] is False, 'Collection was not read-only')
        for boundary in ('handles_before', 'handles_after'):
            handles = collected[boundary]
            self.check(handles['tmux_present'] is False and handles['execution_processes'] == [] and handles['exit'] == stage,
                       'Actual experiment handles not absent at ' + boundary)
        for name, value in collected['original_file_sha256'].items():
            self.pin(name, value, terminal)
        self.pin('LAUNCH_MANIFEST.json', PINS['manifest'], terminal)
        launch = json.loads(self.path('LAUNCH_MANIFEST.json', terminal).read_bytes())
        for name, value in launch['file_sha256'].items():
            self.pin(name, value, terminal)
        for name in ('LAUNCH_ATTEMPT.json', 'RUN_STARTED.json'):
            handle = json.loads(self.path(name, terminal).read_bytes())
            self.check(handle['manifest_sha256'] == PINS['manifest'] and handle['session'] == 'rescue_native_sd_v1', 'Single launch handle binding '+name)
        gpu = json.loads(self.path('GPU_PREFLIGHT.json', terminal).read_bytes())
        deps = json.loads(self.path('DEPENDENCY_PREFLIGHT.json', terminal).read_bytes())
        self.check(gpu['passed'] is True and gpu['tests_run'] == len(gpu['cases']) == 8 and gpu['skipped'] == 0, 'Actual8 CUDA preflight tests not passed')
        expected_deps = dict(python='3.10.20', torch='2.7.1+cu128', cuda='12.8', transformers='4.40.2', pyarrow='24.0.0', flash_attn='2.8.3')
        self.check(deps['passed'] is True and deps['no_installation'] is True and
                   all(deps['values'][k] == v for k, v in expected_deps.items()) and
                   deps['values']['flash_receipt'] == p['flash_attention'], 'Actual frozen dependency preflight changed')
        self.pin('NATIVE_SD_SOURCE.py', PINS['entry'])
        self.pin('PREREGISTRATION.json', PINS['prereg'])
        self.check(p['entry_source_sha256'] == PINS['entry'] and p['preregistration_sha256'] == PINS['prereg'], 'Protocol entry/prereg changed')
        self.check(not self.path('FAILURE.json').exists(), 'Preserved FAILURE present')

    def reuse(self, p, refs, own, reference_rows):
        receipt = self.obj('REUSE_RECEIPT.json')
        self.pin('REUSE_RECEIPT.json', p['reuse_receipt_sha256'])
        for name, key in (('FINAL_ARTIFACTS_MANIFEST.json', 'prior_manifest'), ('VERIFIED_COMPLETE.json', 'prior_verified')):
            self.pin('prior_local_seal/'+name, PINS[key])
            self.check(receipt[key+'_sha256'] == PINS[key] and p['args'][key+'_sha256'] == PINS[key], 'Prior external seal identity changed: ' + key)
        seal = self.obj('prior_local_seal/VERIFIED_COMPLETE.json')
        inventory = self.obj('prior_local_seal/FINAL_ARTIFACTS_MANIFEST.json')['file_sha256']
        self.check(seal['passed'] is True and seal['six_lengths_complete'] is True and
                   seal['final_artifacts_manifest_sha256'] == PINS['prior_manifest'], 'Prior complete seal invalid')
        for name, value in receipt['curve_artifact_pins'].items():
            self.check(inventory.get('terminal/'+name) == value, 'Curve consumed proof not in sealed inventory: '+name)
            self.pin('prior_consumed/curve/'+name, value)
        old_reuse = self.obj('prior_consumed/curve/results/REUSE_RECEIPT.json')
        for name, value in receipt['old_artifact_pins'].items():
            self.check(old_reuse['old_artifact_pins'].get(name) == value, 'Old consumed proof unbound: '+name)
            self.pin('prior_consumed/old/'+name, value)
        self.check(receipt['old_artifact_pins']['LAUNCH_MANIFEST.json'] == PINS['old_manifest'], 'Original runtime envelope changed')
        old = self.obj('prior_consumed/curve/results/protocol.json')
        old_done = self.obj('prior_consumed/curve/results/COMPLETE.json')
        self.check(old_done['passed'] is True and old_done['logical_requests'] == 798 and
                   self.obj('prior_consumed/curve/PIPELINE_EXIT.json')['exit_code'] == 0 and
                   self.obj('prior_consumed/curve/results/QUALITY_PASSED.json')['passed'] is True, 'Prior execution not complete')
        self.check(refs == self.obj('prior_consumed/curve/results/references.json') and
                   reference_rows == self.rows('prior_consumed/curve/results/reference.jsonl') + self.rows('prior_consumed/curve/results/reused_reference.jsonl') and
                   own == self.rows('prior_consumed/curve/results/own_quality.jsonl') + self.rows('prior_consumed/curve/results/reused_own_quality.jsonl'), 'Prior reference or quality bytes changed')
        self.check(p['shared_warm_sequence'] == old['shared_warm_sequence'] and
                   p['assets_sha256'] == old['assets_sha256'] and p['configs']['rescue_graph'] == old['configs']['rescue_graph'], 'Prior assets/config/request order differs')
        source = self.obj('prior_consumed/old/capsule/LAUNCH_MANIFEST.json')['source_sha256']
        self.check(len(source) == 167 and p['source_sha256'] == source, 'Original frozen runtime inventory differs')
        for name, value in source.items():
            self.pin('source/'+name, value)
        for binding in p['source_bindings'].values():
            relative = PurePosixPath(binding['path']).relative_to(PurePosixPath(p['args']['source_root'])).as_posix()
            self.check(source.get(relative) == binding['sha256'], 'Imported module outside frozen source inventory')
        for role, value in ASSETS.items():
            self.check(p[role] == value and p['assets_sha256'][p['args'][role[:-7]]] == value, 'Mandatory asset bytes changed: '+role)
        documents = self.rows('documents.jsonl')
        self.check([r['source_index'] for r in documents] == list(DOCS) and
                   self.obj('prime.json') == next(r for r in documents if r['source_index'] == 271), 'Fixed 16-doc natural cohort or identical prime changed')
        inputs = {}
        for r in documents:
            left, right, body = (r[k] for k in ('prefix_ids', 'suffix_ids', 'document_ids'))
            self.check(len(left)+len(right) == 30 and len(body) == 16354, 'Natural prompt/body size changed')
            for n in LENGTHS:
                ids = left + body[:n-30] + right
                self.check(len(ids) == n and all(type(t) is int and t >= 0 for t in ids), 'Natural input invalid')
                inputs[f"{r['source_index']}:{n}"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        self.check(inputs == p['input_sha256'] == old['input_sha256'] == receipt['canonical_inputs_all96'] and
                   receipt['no_old_timing_reused'] is True, 'Exact 96 inputs changed or old timing reused')

    def trace(self, r, quality, refs, eos, vocab):
        tag = f"{r['mode']}/{r['length']}/{r['source_index']}/{r.get('repeat', 'q')}"
        native = r['mode'] == MODES[0]
        generated = r['generated']
        self.check(generated == refs[f"{r['source_index']}:{r['length']}"] and r['tokens'] == len(generated) and
                   r['output_agrees'] is True and r['performance_eligible'] is (not quality), 'Canonical/clock eligibility '+tag)
        rounds = r['trace']
        self.check(len(rounds) == r['blocks'] > 0 and r['accepted'] == sum(t['accepted'] for t in rounds) and
                   r['proposed'] == sum(t['proposed'] for t in rounds), 'Round totals '+tag)
        progress = 0
        for i, t in enumerate(rounds):
            cap = t['cap'];a = t['accepted'];k = t['proposed'];rem = 128-progress
            self.check(t['prefix'] == r['length']-1+progress and t['round_index'] == i and
                       t['terminal'] is (i == len(rounds)-1) and 0 <= a <= k <= cap and k >= 1 and
                       (cap == min(4, rem) if native else cap in (min(2, rem), min(4, rem))), 'Causal cap/prefix '+tag)
            if native:
                self.check(t['workset_tokens'] == t['prefix'], 'Native full prefix read '+tag)
            progress += a + (0 if t['terminal'] and progress+a == len(generated) else 1)
        self.check(progress == len(generated), 'Committed terminal count '+tag)
        forwards = r['proposed'] + sum(t['accepted'] == t['proposed'] and not t['terminal'] for t in rounds)
        self.check(r['draft_forwards'] == forwards and close(r['acceptance_rate'], r['accepted']/r['proposed']), 'Catchup/rate '+tag)
        if native:
            state = dict(top_chunks=None, working=False, injector=False, draft_runtime=False, target_hooks=0, round_observer_enabled=quality)
            self.check(r['native_state'] == state and r['baseline_algorithm'] == 'native_full_context' and
                       r['execution_backend'] == 'eager' and all(r[k] == 0 for k in ZERO) and
                       all(r.get(k, 0) == 0 for k in OPTIONAL) and 'draft_runtime' not in r and
                       all(r['owner_signature'][k] is None for k in ('working', 'injector', 'runtime')),
                       'Native hidden intervention '+tag)
        if not quality:
            self.check(r['outer_seconds'] > 0 and close(r['outer_tps'], r['tokens']/r['outer_seconds']) and
                       r['outer_seconds']+1e-6 >= r['total_seconds'] > 0 and
                       all(r[k] >= 0 for k in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds')) and
                       close(r['total_seconds'], sum(r[k] for k in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds'))) and
                       r['observer_enabled'] is False and r['profile_enabled'] is False and
                       r['mapper_executed'] is (not native) and r['controller_executed'] is (not native) and
                       not {'timeline', 'cost_samples'} & set(r), 'Timed full-clock/no instrumentation '+tag)
        else:
            self.quality(r, native, tag, eos, vocab)

    def quality(self, r, native, tag, eos, vocab):
        for role, dtype, heads in (('native_t', 'torch.float32', 4), ('native_s', 'torch.bfloat16', 2)):
            cache = r[role]
            self.check(len(cache['lengths']) == len(cache['layers']) == 28, 'All 28 native layers '+tag)
            for length, layer in zip(cache['lengths'], cache['layers']):
                for kind in ('key', 'value'):
                    d = layer[kind]
                    self.check(digest(d) and d['dtype'] == dtype and d['shape'] == [1, heads, length, 128] and
                               length >= r['length']-1, 'Valid-span native digest '+tag)
        queries = r['queries']
        self.check(len(queries) == r['draft_forwards'] and all(digest(q['query']) for q in queries), 'Every-step Q digest '+tag)
        observed = r['observed_rounds']
        self.check(len(observed) == len(r['trace']), 'Every observed round '+tag)
        output = [];nonzero = False;cursor = 0
        for o, t in zip(observed, r['trace']):
            positions = o['selected'];draft = o['draft'];checks = o['checks']
            self.check(o['progress'] == len(output) and o['remaining'] == 128-len(output) and
                       all(o[k] == t[k] for k in ('round_index', 'prefix', 'accepted')) and
                       len(positions) == len(set(positions)) > 0 and all(type(p) is int and 0 <= p < t['prefix'] for p in positions) and
                       len(draft) == t['proposed'] and len(checks) == len(draft)+1 and
                       all(type(v) is int and 0 <= v < vocab for v in draft+checks), 'Observed state/positions/draft checks '+tag)
            a = 0
            for x, y in zip(draft, checks):
                if x != y:
                    break
                a += 1
                if x == eos:
                    break
            self.check(a == t['accepted'], 'Contiguous observed agreement '+tag)
            output += draft[:a]
            if len(output) < len(r['generated']):
                output.append(checks[a])
            if native:
                self.check(positions == list(range(t['prefix'])) and o['budget_features'] is None and
                           o['summary_count'] == 0 and o['injection_evidence'] == [], 'Native quality nonintervention '+tag)
            else:
                vector = o['budget_features']
                self.check(vector is None or len(vector) == 12 and all(math.isfinite(v) for v in vector), 'Causal 12-feature vector '+tag)
                for ev in o['injection_evidence']:
                    w, n, count = ev['working_value'], ev['native_value'], ev['different_elements']
                    self.check(ev['layer'] == 27 and 0 < len(ev['positions']) <= 64 and
                               len(ev['positions']) == len(set(ev['positions'])) and set(ev['positions']) <= set(positions) and
                               digest(w) and digest(n) and w['shape'] == n['shape'] and len(w['shape']) == 2 and
                               w['shape'][0] == len(ev['positions']) and w['dtype'] == n['dtype'] == 'torch.bfloat16' and
                               ev['native_matches_memo'] is True and 0 <= count <= math.prod(w['shape']) and
                               ((count == 0) == (w['sha256'] == n['sha256'])), 'Real same-BF16 V difference '+tag)
                    nonzero |= count > 0
            for depth in range(t['proposed']):
                q = queries[cursor];cursor += 1
                self.check(q['logits'] is True and q['summary_requested'] is (not native and depth < 2) and
                           (q['summary_count'] == 0 if native else type(q['summary_count']) is int and q['summary_count'] >= 0), 'Per-step summary schedule '+tag)
            if t['accepted'] == t['proposed'] and not t['terminal']:
                q = queries[cursor];cursor += 1
                self.check(q['logits'] is False and q['summary_requested'] is False, 'Catchup summary/logits disabled '+tag)
        self.check(output == r['generated'] and cursor == len(queries), 'Observed token reconstruction '+tag)
        self.check(r['nonzero_injection'] is (not native) and (native or nonzero), 'Mandatory nonzero Mapper '+tag)

    def runtime(self, r):
        tag = f"{r['mode']}/{r['length']}/{r['source_index']}"
        s = r['runtime_stats'];sig = r['owner_signature']
        for name in ('native_t', 'native_s'):
            self.check(all(type(sig[name][k]) is int and sig[name][k] > 0 for k in ('identity', 'key_ptr', 'value_ptr', 'capacity')) and
                       sig[name]['capacity'] >= r['length']+128, 'Private cache pointer/capacity '+tag)
        if r['mode'] != 'rescue_graph':
            self.check(s['mode'] == 'eager' and s['entries'] == 0 and s['graph_keys'] == [] and
                       all(s[k] == 0 for k in RUNTIME) and 'draft_runtime' not in r, 'Eager Graph work absent '+tag)
            return
        d = r['draft_runtime']['request'];keys = s['graph_keys']
        self.check(s['mode'] == 'cuda_graph' and s['max_graph_entries'] == 128 and
                   s['entries'] == s['captures'] == len(keys) <= 128 and len({json.dumps(v) for v in keys}) == len(keys) and
                   d['replays'] == d['shadow_updates'] == r['draft_forwards'] and d['resident_steps'] == 0 and
                   all(type(d[k]) is int and d[k] >= 0 for k in RUNTIME[:-1]) and
                   (d['captures'] > 0) == (d['capture_host_seconds'] > 0), 'Graph shape/count work ledger '+tag)
        pointers = [sig[name][kind] for name in ('native_s', 'working') for kind in ('key_ptr', 'value_ptr')]
        self.check(sig['runtime']['bound_ptrs'] == pointers, 'Graph bound native/working pointers '+tag)
        needed = set()
        for t in r['trace']:
            needed.update((t['workset_tokens']+j, True, j <= 2) for j in range(1, t['proposed']+1))
            if t['accepted'] == t['proposed'] and not t['terminal']:
                needed.add((t['workset_tokens']+t['proposed']+1, False, False))
        self.check(needed <= {tuple(v[:3]) for v in keys}, 'Graph actual step geometry missing '+tag)


def statistics(rows):
    """Independent NumPy pooled estimators and paired document bootstrap."""
    import numpy as np
    index = np.random.default_rng(20261028).integers(0, len(DOCS), size=(10000, len(DOCS)), dtype=np.int64)
    result = dict(document_order=list(DOCS), bootstrap_index_sha256=hashlib.sha256(index.astype('<i8').tobytes()).hexdigest(), contexts={}, contrasts=[])
    cells = {(n, m, d): [] for n in LENGTHS for m in MODES for d in DOCS}
    for r in rows:
        cells[r['length'], r['mode'], r['source_index']].append(r)
    for n in LENGTHS:
        result['contexts'][str(n)] = {};samples = {}
        for m in MODES:
            totals = []
            for d in DOCS:
                group = cells[n, m, d]
                if len(group) != 2 or {r['repeat'] for r in group} != {0, 1}:
                    raise ValueError('Exactly two repeats for each document required')
                totals.append([sum(r[k] for r in group) for k in ('tokens', 'outer_seconds', 'accepted', 'blocks', 'proposed')])
            matrix = np.asarray(totals, dtype=np.float64)
            sums = matrix.sum(axis=0);bs = matrix[index].sum(axis=1)
            v = dict(documents=16, tokens=int(sums[0]), seconds=float(sums[1]), accepted=int(sums[2]), blocks=int(sums[3]), proposed=int(sums[4]))
            for key, numerator, denominator in (('pooled_tps', 0, 1), ('acceptance_length', 2, 3), ('acceptance_rate', 2, 4)):
                sample = bs[:, numerator]/bs[:, denominator]
                v[key] = float(sums[numerator]/sums[denominator])
                v['tps_ci95' if key == 'pooled_tps' else key+'_ci95'] = np.quantile(sample, [.025, .975]).tolist()
                if key == 'pooled_tps':
                    samples[m] = sample
            result['contexts'][str(n)][m] = v
        for name, a, b in CONTRASTS:
            sample = 100*(samples[a]/samples[b]-1)
            result['contrasts'].append(dict(name=name, length=n, own_mode=a, baseline_mode=b, documents=16,
                point_percent=100*(result['contexts'][str(n)][a]['pooled_tps']/result['contexts'][str(n)][b]['pooled_tps']-1),
                ci95=np.quantile(sample, [.025, .975]).tolist()))
    return result


def compare_stats(actual, expected, check):
    if isinstance(actual, dict):
        check(isinstance(expected, dict) and set(actual) <= set(expected), 'Reported statistic fields missing')
        for k, v in actual.items():
            compare_stats(v, expected[k], check)
    elif isinstance(actual, list):
        check(isinstance(expected, list) and len(actual) == len(expected), 'Reported statistic list length differs')
        for a, b in zip(actual, expected):
            compare_stats(a, b, check)
    elif type(actual) is float:
        check(close(actual, expected), 'Independent statistic differs: '+str((actual, expected)))
    else:
        check(actual == expected, 'Independent statistic differs: '+str((actual, expected)))


def audit(root, summary=None):
    i = Inspector(root);p = i.obj('protocol.json');pre = i.obj('PREREGISTRATION.json')
    i.inventories(p)
    for key, value in dict(lengths=list(LENGTHS), source_indices=list(DOCS), modes=list(MODES), repeats=2, max_new_tokens=128, seed=20261028).items():
        i.check(p[key] == pre[key] == value, 'Protocol prereg field differs: '+key)
    configs = p['configs'];native_cfg = configs['native_eager'];own_cfg = configs['rescue_eager'];graph_cfg = configs['rescue_graph']
    i.check(all(native_cfg[k] == v for k,v in dict(selector='full', fixed_k=4, length_policy='fixed', draft_execution='eager', inject_tokens=0, injection_strength=0., injection_layers=[]).items()), 'Native full-context fixed4 no-injection config')
    i.check({k:v for k,v in own_cfg.items() if k != 'draft_execution'} == {k:v for k,v in graph_cfg.items() if k != 'draft_execution'} and
            own_cfg['draft_execution'] == 'eager' and graph_cfg['draft_execution'] == 'cuda_graph', 'Own backends differ beyond execution')
    i.check(all(own_cfg[k] == v for k,v in dict(selector='hierarchical',budget=512,recent=64,coverage_rows=64,coverage_candidates=1024,coverage_batch=16,refresh_every=4,inject_tokens=64,injection_strength=.5,injection_layers=[27],length_policy='risk',coverage_execution='eager').items()), 'Frozen M1/M2/M3 settings changed')
    i.check(all(c[k] == v for c in configs.values() for k,v in dict(attention_backend='sdpa',prefill_precision='bfloat16',max_new_tokens=128,profile=False,schedule='serial').items()), 'Common precision/backend/clock settings differ')
    refs = i.obj('references.json');rr = i.rows('reused_reference.jsonl');own = i.rows('reused_own_quality.jsonl')
    i.check(len(refs) == len(rr) == 96 and len(own) == 192 and
            {f"{r['source_index']}:{r['length']}": r['generated'] for r in rr} == refs, 'Canonical reference matrix')
    eos, vocab = p['tokenizer_eos_token_id'], p['vocab_size']
    i.check(all(0 < len(ids) <= 128 and all(type(v) is int and 0 <= v < vocab for v in ids) and
                (len(ids) == 128 or ids[-1] == eos) and (eos not in ids or ids[-1] == eos and eos not in ids[:-1]) for ids in refs.values()), 'Canonical EOS/vocabulary')
    i.reuse(p, refs, own, rr)
    native = i.rows('native_quality.jsonl');qowners = i.rows('quality_owners.jsonl');cold = i.rows('prime.jsonl');warm = i.rows('runtime.jsonl');owners = i.rows('owners.jsonl')
    counts = dict(native_quality_requests=len(native), quality_owners=len(qowners), prime_requests=len(cold), timing_requests=len(warm), owners=len(owners))
    i.check(counts == dict(native_quality_requests=96, quality_owners=96, prime_requests=18, timing_requests=576, owners=18), 'Complete new690 matrix')
    new = native+cold+warm;ids = [r['request_id'] for r in new]
    i.check(len(new) == len(set(ids)) == 690 and all(re.fullmatch('[0-9a-f]{32}', v) for v in ids) and
            sorted(r['execution_index'] for r in new) == list(range(690)) and not set(ids) & {r['request_id'] for r in own+rr}, 'New request UUID/execution bijection')
    i.check([r['execution_index'] for r in native] == list(range(96)), 'All quality before timing')
    i.check(len({r['owner_id'] for r in qowners+owners}) == 114, '114 distinct private owners')
    env = p['environment']
    i.check(env['gpu_name'] == 'NVIDIA H100' and env['target_dtype'] == 'torch.float32' and
            env['draft_dtype'] == env['prefix_dtype'] == 'torch.bfloat16' and env['tf32'] is False and env['visible_devices'] == '0', 'Frozen actual hardware/precision')
    for r in new:
        i.check(r['gpu_snapshot_before']['nvidia_smi_gpu'] == env['nvidia_smi_gpu'], 'Same physical GPU before every call')
    qmap = {(r['length'], r['source_index'], r['mode']): r for r in own+native}
    i.check(set(qmap) == {(n, d, m) for n in LENGTHS for d in DOCS for m in MODES}, '288 total quality cells')
    for r in own+native:
        i.trace(r, True, refs, eos, vocab);i.runtime(r)
    for n in LENGTHS:
        for d in DOCS:
            a, b = (qmap[n, d, m] for m in MODES[1:])
            i.check(equal_work(a, b) and all(a[k] == b[k] for k in ('native_t', 'native_s', 'observed_rounds', 'queries')),
                    f'Own eager/Graph all-layer/Q/selection/features proof differs: {n}/{d}')
    qowner = {r['request_id']: r for r in qowners}
    for r in native:
        o = qowner[r['request_id']]
        i.check(o['owner_id'] == r['owner_id'] and o['owner_signature'] == r['owner_signature'] and
                o['engine_closed'] is True and o['hook_restored'] is True and o['closed_runtime_stats']['entries'] == 0 and
                o['final_runtime_stats'] == r['runtime_stats'], 'Native quality owner not closed')
    gate = i.obj('QUALITY_PASSED.json')
    i.check(gate['passed'] is True and gate['quality_owners_closed'] is True and gate['native_quality_requests'] == 96, 'Quality gate incomplete')
    repeated = {}
    for r in cold+warm:
        i.trace(r, False, refs, eos, vocab);i.runtime(r)
        key = (r['length'], r['source_index'], r['mode'])
        i.check(equal_work(r, qmap[key]), 'Timing/quality work mismatch '+str(key))
        if key in repeated:
            i.check(equal_work(r, repeated[key]), 'Timing repeat/prime trajectory mismatch '+str(key))
        repeated[key] = r
    expected_order = [(n, MODES[k]) for n, order in zip(LENGTHS, ORDER) for k in order]
    i.check([(r['length'], r['mode']) for r in cold] == expected_order == [(r['length'], r['mode']) for r in owners], 'Counterbalanced owner order')
    for j, (prime, owner) in enumerate(zip(cold, owners)):
        block = [r for r in warm if r['owner_id'] == owner['owner_id']]
        i.check(prime['source_index'] == 271 and len(block) == 32 and prime['execution_index'] == 96+33*j and
                [r['execution_index'] for r in block] == list(range(97+33*j, 129+33*j)) and
                [{k:r[k] for k in ('source_index', 'repeat')} for r in block] == p['shared_warm_sequence'] and
                [r['warm_index'] for r in block] == list(range(32)), 'Exact cold + 32 warm owner block')
        requests = [prime]+block
        i.check(owner['owner_id'] == prime['owner_id'] and owner['maximum_live_owners'] == 1 and
                owner['prime_request_id'] == prime['request_id'] and owner['warm_request_ids'] == [r['request_id'] for r in block] and
                all(r['owner_signature'] == owner['owner_signature'] for r in requests) and owner['engine_closed'] is True and
                owner['hook_restored'] is True and owner['closed_runtime_stats']['entries'] == 0 and
                owner['final_runtime_stats'] == block[-1]['runtime_stats'], 'Private persistent owner closure/pointers')
        seconds = owner['constructor_and_detach_seconds'] + sum(r['outer_seconds'] for r in requests) + owner['close_seconds']
        tokens = sum(r['tokens'] for r in requests)
        i.check(close(owner['warm_outer_seconds_sum'], sum(r['outer_seconds'] for r in block)) and
                owner['warm_tokens'] == sum(r['tokens'] for r in block) and close(seconds, owner['accounted_service_seconds']) and
                tokens == owner['accounted_service_tokens'] and close(tokens/seconds, owner['accounted_service_tps']) and
                owner['owner_lifetime_wall_seconds']+1e-8 >= seconds, 'Finite service all costs accounted')
        if prime['mode'] == 'rescue_graph':
            previous = set();total = {k:0 for k in RUNTIME}
            for r in requests:
                keys = {json.dumps(k) for k in r['runtime_stats']['graph_keys']};delta = r['draft_runtime']['request']
                i.check(previous <= keys and len(keys-previous) == delta['captures'], 'Graph no eviction/borrowed late capture')
                previous = keys
                for k in RUNTIME:
                    total[k] += delta[k]
                    i.check(close(total[k], r['runtime_stats'][k]) and close(total[k], r['draft_runtime']['lifetime'][k]), 'Graph cumulative request/lifetime ledger '+k)
    matrix = {(r['length'], r['source_index'], r['repeat'], r['mode']): r for r in warm}
    i.check(len(matrix) == 576, 'Warm cell uniqueness')
    for n in LENGTHS:
        for d in DOCS:
            for rep in (0, 1):
                i.check(equal_work(matrix[n,d,rep,'rescue_eager'], matrix[n,d,rep,'rescue_graph']), 'Own timing eager/Graph trajectory')
    telemetry = i.rows('gpu_diagnostics.jsonl')
    wanted_tel = [(phase, n, m, 96+33*j+(33 if phase == 'owner_end' else 0)) for j, (n,m) in enumerate(expected_order) for phase in ('owner_start', 'owner_end')]
    i.check([(r['phase'], r['length'], r['mode'], r['execution_index']) for r in telemetry] == wanted_tel, '36 complete owner boundary diagnostics')
    complete = i.obj('COMPLETE.json')
    i.check(complete['logical_requests'] == 690 and complete['counts'] == counts and
            all(complete[k] is True for k in ('passed','all_canonical','quality_gate_passed','owners_closed','sources_unchanged','assets_unchanged','dependencies_unchanged')) and
            complete['goal_complete'] is False and complete['fresh_confirmation'] is False, 'Raw terminal full completion')
    stats = statistics(warm)
    if summary:
        reported = json.loads(Path(summary).read_bytes())
        i.check(reported['status'] == 'COMPLETE', 'Main analysis did not complete')
        compare_stats(stats, reported['statistics'], i.check)
    return dict(status='PASSED' if not i.issues else 'FAILED', passed=not i.issues, checks=i.checks,
                pinned_file_checks=i.pinned_files, issues=i.issues, independent=True, goal_complete=False,
                fresh_confirmation=False, statistics=stats, raw_counts=counts, new_calls=690, owners=114,
                summary_compared=bool(summary), summary_sha256=sha(summary) if summary else None,
                limitations=['Independent saved-record/byte-hash/statistical audit, not independent GPU tensor or model execution.',
                             'Reused16 development documents, fixed native k4 not tuned optimal, finite cold+32warm owner blocks.',
                             'Native eager-only; Graph/native combines algorithm and backend. Pointwise CIs do not cover owner/hardware drift.',
                             'No fresh confirmation, stochastic-distribution, production or SOTA claim.'])


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--summary', type=Path);parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Exclusive new audit output required')
    result = audit(args.root, args.summary)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output/'audit_source.py')
    result['audit_source_sha256'] = sha(__file__)
    (args.output/'AUDIT.json').write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'statistics'}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
