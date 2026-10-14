#!/usr/bin/env python3
"""E1--E9: frozen native-KV selection, layer matching and utility learning.

The offline experiments use full common-history attention. Runtime retrieval
only uses attention from completed prefill/verification and visible student
keys. Tensor caches are kept outside the repository; JSONL results are resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'exp/attn_adapter_transfer'))
from common import load_model_and_tokenizer, tensor_ids
from run_attention_oracle import legacy_cache, core_forward, logits_of
from attention import AttentionController
from selection import UtilitySelector, js_matrix, mandatory, normalize, retrieve, token_features


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def rows(path):
    path = Path(path)
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


def cpu_cache(cache):
    return tuple(tuple(x.detach().cpu() for x in kv) for kv in legacy_cache(cache))


def device_cache(cache, device):
    return tuple(tuple(x.to(device) for x in kv) for kv in cache)


def trim(cache, length):
    return tuple((k[:, :, :length], v[:, :, :length]) for k, v in legacy_cache(cache))


def metrics(logits, target):
    lp, lq = F.log_softmax(target, -1), F.log_softmax(logits, -1)
    kl = float((lp.exp() * (lp-lq)).sum())
    label = int(target.argmax())
    return dict(kl=kl, teacher_ce=float(-lq[label]), match=int(int(logits.argmax()) == label),
                prediction=int(logits.argmax()), label=label)


def utility_features(t, s, query, anchor):
    f = token_features(t, s, query)
    return torch.cat((f, f - f[anchor]), -1)


class Experiment:
    def __init__(self, a, *, load=True):
        self.a = a
        self.out = Path(a.output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = Path(a.cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.data = {k: rows(Path(a.data) / f'{k}.jsonl') for k in ('train', 'validation', 'test')}
        self.network = None
        if load:
            self.tt, self.teacher = load_model_and_tokenizer(a.teacher, torch.bfloat16, a.device)
            self.st, self.student = load_model_and_tokenizer(a.student, torch.bfloat16, a.device)
            for model in (self.teacher, self.student):
                model.requires_grad_(False)
            self.tc = AttentionController(self.teacher).install()
            self.sc = AttentionController(self.student).install()
            self.vocab = min(len(self.tt), len(self.st), self.teacher.config.vocab_size, self.student.config.vocab_size)
        self.selected_path = self.out / 'selected.json'

    def doc_path(self, item):
        return self.cache / f"{item['split']}_{item['source_index']}.pt"

    def forward(self, model, token, cache):
        return core_forward(model, tensor_ids([token], self.a.device), cache)

    def logit(self, model, out):
        return logits_of(model, out)[0, :self.vocab]

    def state_cache(self, base, state):
        return tuple((torch.cat((k, tk.to(self.a.device)), -2), torch.cat((v, tv.to(self.a.device)), -2))
                     for (k, v), (tk, tv) in zip(base, state['tail']))

    @torch.no_grad()
    def collect(self):
        manifest_path = self.out / 'config.json'
        config = dict(vars(self.a), vocab=self.vocab, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                      model_layers=[len(self.teacher.model.layers), len(self.student.model.layers)],
                      model_heads=[self.teacher.config.num_attention_heads, self.student.config.num_attention_heads],
                      protocol='Frozen BF16 Qwen2; eager attention; native historical KV; original token/RoPE positions; disjoint document titles',
                      data_manifest=json.loads((Path(self.a.data) / 'manifest.json').read_text()),
                      source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in Path(__file__).parent.glob('*.py')})
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            for k in ('data_manifest', 'budgets', 'query_steps', 'recent', 'chunk', 'max_new_tokens', 'seed', 'teacher', 'student'):
                if old.get(k) != config.get(k):
                    raise ValueError(f'Resume configuration changed: {k}')
        else:
            write(manifest_path, config)
        for split, items in self.data.items():
            for index, item in enumerate(items):
                if self.doc_path(item).exists():
                    continue
                self.tc.enabled = self.sc.enabled = False
                self.tc.capture = self.sc.capture = False
                ids = item['ids']
                tout = core_forward(self.teacher, tensor_ids(ids[:-1], self.a.device))
                tc = legacy_cache(tout.past_key_values)
                sout = core_forward(self.student, tensor_ids(ids[:-1], self.a.device))
                sc = legacy_cache(sout.past_key_values)
                initial_student = cpu_cache(sc)
                prefix = len(ids)-1
                states, reference = [], []
                query = ids[-1]
                for step in range(self.a.max_new_tokens):
                    capture = step in self.a.query_steps
                    self.tc.capture = self.sc.capture = capture
                    tout = self.forward(self.teacher, query, tc)
                    target = self.logit(self.teacher, tout)
                    next_token = int(target.argmax())
                    if capture:
                        sout = self.forward(self.student, query, sc)
                        native = self.logit(self.student, sout)
                        trow, _, _ = self.tc.snapshot()
                        srow, _, qhidden = self.sc.snapshot()
                        tail = tuple((k[:, :, prefix:].cpu(), v[:, :, prefix:].cpu()) for k, v in sc)
                        states.append(dict(step=step, input_id=query, length=prefix+step+1,
                                           teacher=target.cpu(), trow=trow.cpu(), srow=srow.cpu(), query=qhidden.cpu(),
                                           tail=tail, native=metrics(native, target)))
                        sc = legacy_cache(sout.past_key_values)
                    elif step < max(self.a.query_steps):
                        sout = self.forward(self.student, query, sc)
                        sc = legacy_cache(sout.past_key_values)
                    tc = legacy_cache(tout.past_key_values)
                    reference.append(next_token)
                    query = next_token
                    if query == self.st.eos_token_id:
                        break
                bundle = dict(item=item, initial_student=initial_student, states=states, reference=reference)
                temporary = self.doc_path(item).with_suffix('.tmp')
                torch.save(bundle, temporary)
                temporary.replace(self.doc_path(item))
                print(f"collect {split} {index+1}/{len(items)} states={len(states)} reference={len(reference)}", flush=True)
        self.tc.capture = self.sc.capture = False
        self.match_layers()

    def match_layers(self):
        matrices = {'all': [], 'remote': [], 'recent': []}
        for item in self.data['train']:
            doc_m = {k: [] for k in matrices}
            bundle = torch.load(self.doc_path(item), weights_only=False)
            for state in bundle['states']:
                t, s = state['trow'], state['srow']
                n = t.shape[-1]
                doc_m['all'].append(js_matrix(s, t))
                doc_m['remote'].append(js_matrix(s[:, 1:n-self.a.recent], t[:, 1:n-self.a.recent]))
                doc_m['recent'].append(js_matrix(s[:, n-self.a.recent:], t[:, n-self.a.recent:]))
            for k in matrices:
                matrices[k].append(torch.stack(doc_m[k]).mean(0))
        mean = {k: torch.stack(v).mean(0) for k, v in matrices.items()}
        depth = [round(i * (mean['remote'].shape[1]-1)/(mean['remote'].shape[0]-1)) for i in range(mean['remote'].shape[0])]
        pairing = mean['remote'].argmin(-1).tolist()
        random_pair = torch.randperm(mean['remote'].shape[1], generator=torch.Generator().manual_seed(self.a.seed)).tolist()
        write(self.out / 'layer_pairs.json', dict(pairing=pairing, depth=depth, random=random_pair,
                                                train_docs=len(matrices['all']), matrices={k: v.tolist() for k, v in mean.items()}))
        torch.save(mean, self.out / 'layer_heatmaps.pt')
        print('E4 train layer pairing:', pairing, flush=True)

    @torch.no_grad()
    def prediction(self, cache, state, selection=None, bias=None):
        if bias is None:
            self.sc.configure(selection)
        else:
            layer, teacher_layer, strength, shuffled = bias
            prior = state['trow'][teacher_layer].to(self.a.device)
            if shuffled:
                n = prior.numel()
                positions = torch.arange(1, n-self.a.recent, device=prior.device)
                generator = torch.Generator(device=prior.device).manual_seed(self.a.seed + state['step'])
                permutation = torch.randperm(positions.numel(), device=prior.device, generator=generator)
                prior = prior.clone()
                prior[positions] = prior[positions[permutation]]
            self.sc.configure(selection, bias_layer=layer, prior=prior, strength=strength)
        out = self.forward(self.student, state['input_id'], cache)
        return metrics(self.logit(self.student, out), state['teacher'].to(self.a.device))

    @torch.no_grad()
    def labels(self, split):
        path = self.out / f'utility_{split}.jsonl'
        done = {(r['source_index'], r['step'], r['budget']) for r in rows(path)}
        for doc_index, item in enumerate(self.data[split]):
            bundle = torch.load(self.doc_path(item), weights_only=False)
            base = device_cache(bundle['initial_student'], self.a.device)
            for state in bundle['states'][:2]:
                cache = self.state_cache(base, state)
                t, s, q = (state[k].to(self.a.device) for k in ('trow', 'srow', 'query'))
                for b in self.a.budgets:
                    if (item['source_index'], state['step'], b) in done:
                        continue
                    selected = retrieve('teacher_token', t, s, b, self.a.recent)
                    fixed = mandatory(state['length'], self.a.recent, t.device)
                    free_selected = selected[~torch.isin(selected, fixed)]
                    anchor = int(free_selected[torch.argmin(t[-1, free_selected])])
                    base_metrics = self.prediction(cache, state, selected)
                    available = torch.arange(state['length'], device=t.device)
                    available = available[~torch.isin(available, selected)]
                    generator = torch.Generator(device=t.device).manual_seed(self.a.seed+item['source_index']+state['step'])
                    tr = available[torch.argsort(t[-1, available], descending=True, stable=True)]
                    sr = available[torch.argsort(s[-1, available], descending=True, stable=True)]
                    rr = available[torch.randperm(len(available), generator=generator, device=t.device)]
                    candidates = []
                    # 8 teacher, 8 student, 8 random alternatives; deduplicate,
                    # then fill from teacher order, within the same candidate pool.
                    for token in torch.cat((tr[:8], sr[:8], rr[:8], tr)).tolist():
                        if token not in candidates:
                            candidates.append(token)
                        if len(candidates) == self.a.utility_candidates:
                            break
                    features = utility_features(t, s, q, anchor)
                    gains = []
                    for candidate in candidates:
                        change = selected.clone()
                        change[change == anchor] = candidate
                        change = change.sort().values
                        prediction = self.prediction(cache, state, change)
                        gains.append(base_metrics['kl'] - prediction['kl'])
                    record = dict(source_index=item['source_index'], split=split, step=state['step'], budget=b,
                                  selected=selected.tolist(), anchor=anchor, candidates=candidates,
                                  features=features[candidates].cpu().tolist(), gains=gains, baseline=base_metrics)
                    append(path, record)
            print(f'utility labels {split} {doc_index+1}/{len(self.data[split])}', flush=True)

    def fit_selector(self):
        train, val = rows(self.out / 'utility_train.jsonl'), rows(self.out / 'utility_validation.jsonl')
        x = torch.tensor([f for r in train for f in r['features']], device=self.a.device)
        y = torch.tensor([g for r in train for g in r['gains']], device=self.a.device)
        vx = torch.tensor([f for r in val for f in r['features']], device=self.a.device)
        vy = torch.tensor([g for r in val for g in r['gains']], device=self.a.device)
        torch.manual_seed(self.a.seed)
        network = UtilitySelector(x.shape[-1]).to(self.a.device)
        network.mean.copy_(x.mean(0))
        network.scale.copy_(x.std(0).clamp_min(1e-4))
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.002, weight_decay=0.01)
        # Optimize conditional equal-budget swap utility. True output losses
        # provide labels only; no language-model weights receive gradients.
        best_loss, best_state, best_epoch = float('inf'), None, None
        history = []
        for epoch in range(1, 201):
            network.train()
            optimizer.zero_grad()
            prediction = network(x)
            loss = F.mse_loss(prediction, y)
            loss.backward()
            optimizer.step()
            network.eval()
            with torch.no_grad():
                vprediction = network(vx)
                vloss = float(F.mse_loss(vprediction, vy))
            if epoch % 10 == 0:
                history.append(dict(epoch=epoch, train_mse=float(loss.detach()), validation_mse=vloss))
            if vloss < best_loss:
                best_loss, best_epoch = vloss, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
        network.load_state_dict(best_state)
        torch.save(dict(state=best_state, dimensions=x.shape[-1], epoch=best_epoch, validation_mse=best_loss), self.out / 'selector.pt')
        with torch.no_grad():
            vp = network(vx).cpu().numpy()
            truth = vy.cpu().numpy()
        write(self.out / 'selector_training.json', dict(history=history, selected_epoch=best_epoch,
              validation_mse=best_loss, zero_predictor_mse=float(vy.square().mean()),
              validation_correlation=float(np.corrcoef(vp, truth)[0, 1]),
              parameters=sum(p.numel() for p in network.parameters()), labels=len(y), features='Teacher attention across layers plus frozen current-query summaries; candidate and anchor feature difference'))
        self.network = network.eval()
        print(f'E9 selector epoch={best_epoch} validation MSE={best_loss:.6g}', flush=True)

    def load_selector(self):
        checkpoint = torch.load(self.out / 'selector.pt', weights_only=False)
        self.network = UtilitySelector(checkpoint['dimensions']).to(self.a.device)
        self.network.load_state_dict(checkpoint['state'])
        self.network.eval()

    @torch.no_grad()
    def learned_retrieval(self, t, s, q, b):
        base = retrieve('teacher_token', t, s, b, self.a.recent)
        fixed = mandatory(t.shape[-1], self.a.recent, t.device)
        free = base[~torch.isin(base, fixed)]
        anchor = int(free[torch.argmin(t[-1, free])])
        score = self.network(utility_features(t, s, q, anchor))
        pool = torch.arange(t.shape[-1], device=t.device)
        pool = pool[~torch.isin(pool, fixed)]
        selected = pool[torch.argsort(score[pool], descending=True, stable=True)[:min(b,t.shape[-1])-fixed.numel()]]
        return torch.cat((fixed, selected)).sort().values

    @torch.no_grad()
    def calibrate(self):
        if self.selected_path.exists():
            return json.loads(self.selected_path.read_text())
        pairs = json.loads((self.out / 'layer_pairs.json').read_text())
        records = []
        for item in self.data['validation']:
            bundle = torch.load(self.doc_path(item), weights_only=False)
            base = device_cache(bundle['initial_student'], self.a.device)
            for state in bundle['states']:
                cache = self.state_cache(base, state)
                t, s = (state[k].to(self.a.device) for k in ('trow', 'srow'))
                for b in self.a.budgets:
                    for method in ('union', 'difference_core'):
                        for ratio in (0.25, 0.5, 0.75):
                            selected = retrieve(method,t,s,b,self.a.recent,ratio=ratio)
                            result = self.prediction(cache,state,selected)
                            records.append(dict(source_index=item['source_index'], step=state['step'], kind='ratio',
                                                method=method,budget=b,ratio=ratio,**result))
                for layer in (0, len(pairs['pairing'])//2, len(pairs['pairing'])-1):
                    for route in ('depth', 'pairing', 'random', 'shuffled'):
                        tl = pairs['pairing'][layer] if route == 'shuffled' else pairs[route][layer]
                        for strength in (0.25, 0.5, 1.0):
                            result = self.prediction(cache,state,None,(layer,tl,strength,route=='shuffled'))
                            records.append(dict(source_index=item['source_index'],step=state['step'],kind='bias',
                                                layer=layer, teacher_layer=tl, route=route, strength=strength, **result))
            print(f'calibration {item["source_index"]}', flush=True)
        write(self.out / 'calibration.json', records)
        ratios = {}
        for b in self.a.budgets:
            for method in ('union','difference_core'):
                scores = {r: np.mean([x['kl'] for x in records if x['kind']=='ratio' and x['budget']==b and x['method']==method and x['ratio']==r]) for r in (0.25,0.5,0.75)}
                ratios[f'{method}_{b}'] = min(scores, key=scores.get)
        configurations = {}
        for route in ('depth','pairing','random','shuffled'):
            options = []
            for layer in (0,len(pairs['pairing'])//2,len(pairs['pairing'])-1):
                for strength in (0.25,0.5,1.0):
                    subset = [r for r in records if r['kind']=='bias' and r['route']==route and r['layer']==layer and r['strength']==strength]
                    options.append((np.mean([r['kl'] for r in subset]),layer,strength,subset[0]['teacher_layer']))
            _, layer, strength, tl = min(options)
            configurations[route] = [layer, tl, strength, route=='shuffled']
        # Same-layer/same-strength controls distinguish correspondence from tuning.
        selected = dict(ratios=ratios, bias=configurations, paired_common_layer=configurations['pairing'][0],
                        paired_common_strength=configurations['pairing'][2], selection_metric='mean validation output KL; no test tuning')
        write(self.selected_path, selected)
        return selected

    @torch.no_grad()
    def evaluate(self):
        self.load_selector()
        chosen = self.calibrate()
        pairs = json.loads((self.out / 'layer_pairs.json').read_text())
        path = self.out / 'offline_test.jsonl'
        done = {(r['source_index'],r['step'],r['budget'],r['method']) for r in rows(path)}
        layer_metrics = []
        for index,item in enumerate(self.data['test']):
            bundle = torch.load(self.doc_path(item), weights_only=False)
            base = device_cache(bundle['initial_student'], self.a.device)
            for state in bundle['states']:
                cache = self.state_cache(base,state)
                t,s,q = (state[k].to(self.a.device) for k in ('trow','srow','query'))
                remote = js_matrix(s[:,1:-self.a.recent],t[:,1:-self.a.recent])
                si = torch.arange(s.shape[0],device=t.device)
                layer_metrics.append(dict(source_index=item['source_index'],step=state['step'],
                     stability=float((remote.argmin(-1)==torch.tensor(pairs['pairing'],device=t.device)).float().mean()),
                     paired_js=float(remote[si,pairs['pairing']].mean()),depth_js=float(remote[si,pairs['depth']].mean()),
                     random_js=float(remote[si,pairs['random']].mean())))
                for b in self.a.budgets:
                    variants = {}
                    for method in ('recent','random','student_chunk','teacher_chunk','student_token','teacher_token','union','difference','difference_core'):
                        ratio = chosen['ratios'].get(f'{method}_{b}',0.5)
                        variants[method] = (retrieve(method,t,s,b,self.a.recent,self.a.chunk,ratio,self.a.seed+item['source_index']),None)
                    for route in ('depth','pairing','random'):
                        bylayer = [retrieve('teacher_token',t[pairs[route][l]:pairs[route][l]+1],s[l:l+1],b,self.a.recent)
                                   for l in range(s.shape[0])]
                        variants[f'layer_{route}'] = (bylayer,None)
                    variants['learned_global'] = (self.learned_retrieval(t,s,q,b),None)
                    for method,(selection,bias) in variants.items():
                        key=(item['source_index'],state['step'],b,method)
                        if key in done:
                            continue
                        result = self.prediction(cache,state,selection,bias)
                        selected_positions = selection[-1] if isinstance(selection,list) else selection
                        evidence = set(item['evidence_positions'])
                        covered = len(evidence & set(selected_positions.tolist()))/max(1,len(evidence))
                        append(path,dict(source_index=item['source_index'],step=state['step'],budget=b,method=method,
                                         retained=int(selected_positions.numel()),evidence_doc_coverage=covered,
                                         native_match=state['native']['match'],**result))
                biases = {f'bias_{route}_tuned': tuple(v) for route,v in chosen['bias'].items()}
                layer,strength=chosen['paired_common_layer'],chosen['paired_common_strength']
                for route in ('depth','pairing','random','shuffled'):
                    tl=pairs['pairing'][layer] if route=='shuffled' else pairs[route][layer]
                    biases[f'bias_{route}_common']=(layer,tl,strength,route=='shuffled')
                biases['native_full']=None
                for method,bias in biases.items():
                    key=(item['source_index'],state['step'],0,method)
                    if key not in done:
                        result=self.prediction(cache,state,None,bias)
                        append(path,dict(source_index=item['source_index'],step=state['step'],budget=0,method=method,
                                         retained=state['length'],native_match=state['native']['match'],**result))
            print(f'offline test {index+1}/{len(self.data["test"])}',flush=True)
        write(self.out/'layer_test.json',layer_metrics)
        self.labels('test')
        self.evaluate_utility()

    @torch.no_grad()
    def evaluate_utility(self):
        label_rows=rows(self.out/'utility_test.jsonl')
        bydoc={}
        for r in label_rows:
            bydoc.setdefault(r['source_index'],[]).append(r)
        path=self.out/'utility_joint_test.jsonl'
        done={(r['source_index'],r['step'],r['budget'],r['method']) for r in rows(path)}
        for item in self.data['test']:
            bundle=torch.load(self.doc_path(item),weights_only=False)
            base=device_cache(bundle['initial_student'],self.a.device)
            states={s['step']:s for s in bundle['states']}
            for r in bydoc[item['source_index']]:
                state=states[r['step']]
                cache=self.state_cache(base,state)
                t,s=(state[k].to(self.a.device) for k in ('trow','srow'))
                candidates=torch.tensor(r['candidates'],device=t.device)
                selected=torch.tensor(r['selected'],device=t.device)
                fixed=mandatory(state['length'],self.a.recent,t.device)
                free=selected[~torch.isin(selected,fixed)]
                removals=free[torch.argsort(t[-1,free])[:self.a.utility_joint]]
                retained=selected[~torch.isin(selected,removals)]
                scores=self.network(torch.tensor(r['features'],device=t.device))
                generator=torch.Generator(device=t.device).manual_seed(self.a.seed+item['source_index']+r['step'])
                rankings={'random':torch.randperm(len(candidates),device=t.device,generator=generator),
                    'teacher':torch.argsort(t[-1,candidates],descending=True),
                    'difference':torch.argsort((t[-1]*(t[-1]-s[-1]).clamp_min(0))[candidates],descending=True),
                    'oracle':torch.argsort(torch.tensor(r['gains'],device=t.device),descending=True),
                    'learned':torch.argsort(scores,descending=True)}
                prediction=scores.cpu().numpy()
                truth=np.asarray(r['gains'])
                corr=float(np.corrcoef(prediction,truth)[0,1]) if np.std(truth)>0 and np.std(prediction)>0 else 0.0
                for method,ranking in rankings.items():
                    if (item['source_index'],r['step'],r['budget'],method) in done:
                        continue
                    change=torch.cat((retained,candidates[ranking[:len(removals)]] )).sort().values
                    result=self.prediction(cache,state,change)
                    append(path,dict(source_index=item['source_index'],step=r['step'],budget=r['budget'],method=method,
                        baseline_kl=r['baseline']['kl'],joint_kl_gain=r['baseline']['kl']-result['kl'],
                        predicted_utility_correlation=corr,selected_candidates=candidates[ranking[:len(removals)]].tolist(),**result))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['collect','labels','fit','evaluate','runtime','all','smoke'],default='all')
    p.add_argument('--data',default='exp/attention_selection/data_fresh')
    p.add_argument('--output',default='exp/attention_selection/out_stage2')
    p.add_argument('--cache',default='/mnt/8T/cl/rescue_experiment_cache/attention_selection_stage2')
    p.add_argument('--teacher',default='/mnt/8T/cl/models/Qwen2-7B-Instruct')
    p.add_argument('--student',default='/mnt/8T/cl/models/Qwen2-1.5B-Instruct')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--budgets',type=int,nargs='+',default=[256,512])
    p.add_argument('--query-steps',type=int,nargs='+',default=[0,2,4,6])
    p.add_argument('--recent',type=int,default=64)
    p.add_argument('--chunk',type=int,default=32)
    p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--draft-k',type=int,default=4)
    p.add_argument('--utility-candidates',type=int,default=24)
    p.add_argument('--utility-joint',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    e=Experiment(a,load=a.phase!='fit')
    if a.phase=='smoke':
        from verify import verify
        verify(e)
        return
    if a.phase in ('collect','all'):
        e.collect()
    if a.phase in ('labels','all'):
        e.labels('train')
        e.labels('validation')
    if a.phase in ('fit','all'):
        e.fit_selector()
    if a.phase in ('evaluate','all'):
        e.evaluate()
    if a.phase in ('runtime','all'):
        from runtime import run_runtime
        run_runtime(e)


if __name__=='__main__':
    main()
