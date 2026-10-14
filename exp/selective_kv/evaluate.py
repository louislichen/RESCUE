"""Untouched held-out output metrics and actual greedy speculative decoding."""
import json
import random
import time

import torch

from run import append, core_forward, legacy_cache, metrics, rows, tensor_ids, trim, write
from runtime import sync, teacher_reference
from selection import mandatory, retrieve


def configuration(e):
    chosen = json.loads((e.out / 'selected.json').read_text())
    for name in chosen:
        e.load_network(name)
    g = chosen['performance_gated']['forced']
    methods = dict(native_full=dict(name=None, full=True),
                   teacher_token=dict(name=None, full=False))
    for name, mode in [('reconstruction', 'fixed'), ('performance_fixed', 'fixed'),
                       ('performance_gated', 'gated')]:
        methods[name] = dict(name=name, mode=mode, alpha=chosen[name]['forced']['alpha'], full=False)
    methods['gated_shuffled'] = dict(name='performance_gated', mode='gated', alpha=g['alpha'], shuffle=True)
    methods['gated_constant'] = dict(name='performance_gated', mode='fixed', alpha=g['gate_mean'])
    write(e.out / 'test_methods.json', methods)
    return methods


@torch.no_grad()
def evaluate(e):
    methods = configuration(e)
    path = e.out / 'offline_test.jsonl'
    done = {(r['source_index'], r['step'], r['method']) for r in rows(path)}
    for i, item in enumerate(e.data['test']):
        bundle, base, source = e.load_bundle(item)
        for state in bundle['states']:
            inputs = e.state_inputs(base, source, state)
            for method, config in methods.items():
                key = (item['source_index'], state['step'], method)
                if key in done:
                    continue
                network = e.networks.get(config['name'])
                logits = e.predict(state, inputs, network, config.get('mode', 'gated'),
                                   config.get('alpha', 0), config.get('shuffle', False),
                                   config.get('full', False))
                result = metrics(logits, state['teacher'].to(e.a.device))
                gates = e.inject.stats()
                append(path, dict(source_index=item['source_index'], step=state['step'], method=method,
                                  prefix_tokens=state['length'], budget=0 if config.get('full') else e.a.budget,
                                  gate=gates, **result))
                done.add(key)
        e.inject.clear()
        print(f'offline fresh test {i+1}/{len(e.data["test"])}', flush=True)


@torch.no_grad()
def speculative(e, item, method, config):
    e.inject.clear()
    e.sc.enabled = False
    e.sc.capture = False
    e.tc.capture = True
    sync()
    start = time.perf_counter()
    ids = item['ids']
    tout = core_forward(e.teacher, tensor_ids(ids[:-1], e.a.device))
    tc = legacy_cache(tout.past_key_values)
    prior, _, _ = e.tc.snapshot()
    sout = core_forward(e.student, tensor_ids(ids[:-1], e.a.device))
    sc = legacy_cache(sout.past_key_values)
    e.tc.capture = False
    sync()
    prefill = time.perf_counter()-start
    timings = dict(selection=0.0, source_pack=0.0, draft=0.0, verify=0.0, cache_update=0.0)
    generated, blocks = [], []
    proposed = accepted = 0
    current = ids[-1]
    gates = []
    sync()
    started = time.perf_counter()
    while len(generated) < e.a.max_new_tokens:
        tick = time.perf_counter()
        if config.get('full', False):
            selection = None
        else:
            selection = retrieve('teacher_token', prior, prior, e.a.budget, e.a.recent)
        e.sc.configure(selection)
        sync()
        timings['selection'] += time.perf_counter()-tick
        tick = time.perf_counter()
        if config['name'] is not None:
            remote = selection[~torch.isin(selection, mandatory(prior.shape[-1], e.a.recent, prior.device))]
            ordered = torch.argsort(prior[-1, remote], descending=True, stable=True)
            positions = remote[ordered[:e.a.inject_tokens]].sort().values
            source = torch.stack([tc[l][1][0, :, positions].transpose(0, 1).reshape(len(positions), 512)
                                  for l in e.teacher_layers]).to(torch.bfloat16).float()
            if config.get('source_policy') == 'mean':
                source = source.mean(1, keepdim=True).expand_as(source)
            elif config.get('source_policy') == 'zero':
                source = torch.zeros_like(source)
            e.inject.bind(e.networks[config['name']], positions, source, prior[-1],
                          config['mode'], config['alpha'], config.get('shuffle', False))
        else:
            e.inject.clear()
        sync()
        timings['source_pack'] += time.perf_counter()-tick
        k = min(e.a.draft_k, e.a.max_new_tokens-len(generated))
        draft = []
        next_input = current
        tick = time.perf_counter()
        for _ in range(k):
            sout = e.forward(e.student, next_input, sc)
            sc = legacy_cache(sout.past_key_values)
            next_input = int(e.logit(e.student, sout).argmax())
            draft.append(next_input)
            if next_input == e.st.eos_token_id:
                break
        sync()
        timings['draft'] += time.perf_counter()-tick
        tick = time.perf_counter()
        tout = core_forward(e.teacher, tensor_ids([current]+draft, e.a.device), tc, attentions=True)
        checks = e.teacher.lm_head(tout.last_hidden_state)[0, :, :e.vocab].float().argmax(-1).tolist()
        sync()
        timings['verify'] += time.perf_counter()-tick
        count = 0
        for j, token in enumerate(draft):
            if token != checks[j]:
                break
            count += 1
            if token == e.st.eos_token_id:
                break
        proposed += len(draft)
        accepted += count
        context = len(ids)+len(generated)
        generated.extend(draft[:count])
        stop = bool(generated and generated[-1] == e.st.eos_token_id) or len(generated) >= e.a.max_new_tokens
        if not stop:
            current = int(checks[count])
            generated.append(current)
            stop = current == e.st.eos_token_id or len(generated) >= e.a.max_new_tokens
        committed = context+count
        blocks.append(dict(proposed=len(draft), accepted=count, committed_prefix=committed,
                           verification_query_index=count))
        tick = time.perf_counter()
        if not stop:
            if count == len(draft):
                sout = e.forward(e.student, draft[-1], sc)
                sc = legacy_cache(sout.past_key_values)
            prior = torch.stack([r[0, :, count, :committed].float().mean(0) for r in tout.attentions])
            tc, sc = trim(tout.past_key_values, committed), trim(sc, committed)
            assert all(key.shape[-2] == committed for key, value in tc)
            assert all(key.shape[-2] == committed for key, value in sc)
            assert prior.shape[-1] == committed
        gates.extend(e.inject.gate_stats)
        sync()
        timings['cache_update'] += time.perf_counter()-tick
        if stop:
            break
    sync()
    decode = time.perf_counter()-started
    gate_mean = float(torch.cat(gates).mean()) if gates else 0.0
    e.inject.clear()
    e.sc.enabled = False
    return dict(source_index=item['source_index'], method=method, budget=0 if config.get('full') else e.a.budget,
                generated=generated, tokens=len(generated), proposed=proposed, accepted=accepted,
                blocks=len(blocks), mean_accepted=accepted/len(blocks), acceptance_rate=accepted/max(1, proposed),
                prefill_seconds=prefill, decode_seconds=decode, total_seconds=prefill+decode,
                decode_tps=len(generated)/decode, total_tps=len(generated)/(prefill+decode),
                stage_seconds=timings, gate_mean=gate_mean, trace=blocks,
                teacher_dtype='float32', student_dtype='bfloat16', extra_teacher_query_forwards=0,
                mapped_values_only_in_working_reads=True, full_backing_cache_retained=True)


@torch.no_grad()
def run_runtime(e):
    methods = configuration(e)
    # Warm-up on training data only, including kernels used by the injection.
    for method, config in methods.items():
        speculative(e, e.data['train'][0], method, config)
    path = e.out / 'runtime_test.jsonl'
    previous = rows(path)
    done = {(r['source_index'], r['method']) for r in previous}
    refs = {r['source_index']: r for r in previous if r['method'] == 'teacher_greedy'}
    for i, item in enumerate(e.data['test']):
        if item['source_index'] not in refs:
            result = teacher_reference(e, item)
            bundle = torch.load(e.doc_path(item), weights_only=False)
            assert result['generated'] == bundle['reference']
            append(path, result)
            refs[item['source_index']] = result
        reference = refs[item['source_index']]
        order = list(methods.items())
        random.Random(e.a.seed+item['source_index']).shuffle(order)
        for method, config in order:
            if (item['source_index'], method) in done:
                continue
            result = speculative(e, item, method, config)
            result['output_agrees'] = result['generated'] == reference['generated']
            append(path, result)
            done.add((item['source_index'], method))
        print(f'runtime fresh test {i+1}/{len(e.data["test"])} agrees=' +
              str(all(r.get('output_agrees', True) for r in rows(path) if r['source_index'] == item['source_index'])), flush=True)
