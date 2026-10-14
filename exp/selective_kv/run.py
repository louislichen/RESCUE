#!/usr/bin/env python3
"""Train performance-supervised selective teacher-V injection and evaluate SD."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'exp/attention_selection'))
from exp.attention_selection.run import (Experiment, append, core_forward, cpu_cache,
                                         device_cache, legacy_cache, metrics, rows,
                                         tensor_ids, trim, write)
from selection import mandatory, retrieve
from runtime import teacher_reference, sync
from bridge import Injection, ValueMapper


class Pilot(Experiment):
    def __init__(self, args):
        super().__init__(args)
        self.teacher.to(dtype=torch.float32)
        pairs = json.loads((ROOT / 'exp/attention_selection/out_stage2/layer_pairs.json').read_text())
        self.layers = [24, 25, 26, 27]
        self.teacher_layers = [pairs['pairing'][l] for l in self.layers]
        self.inject = Injection(self.sc)
        self.networks = {}
        self.current_bundle = None
        self.device_base = None
        protocol_path = self.out / 'protocol.json'
        if protocol_path.exists():
            old = json.loads(protocol_path.read_text())
            for key, value in vars(args).items():
                if key != 'phase' and old.get(key) != value:
                    raise ValueError(f'Resume configuration changed: {key}; use a new output/cache')
            manifest = json.loads((Path(args.data) / 'manifest.json').read_text())
            if old['manifest'] != manifest:
                raise ValueError('Resume data manifest changed; use a new output/cache')

    @torch.no_grad()
    def collect(self):
        protocol = dict(vars(self.a), layers=self.layers, teacher_layers=self.teacher_layers,
                        teacher_dtype='float32', student_dtype='bfloat16', source_storage='bfloat16',
                        source='Only completed prefix KV; teacher-attention prior from previous completed query',
                        intervention='Read-only bounded V residual on remote selected tokens; K unchanged',
                        labels='FP32 teacher output KL and teacher argmax CE; no direct greedy acceptance gradient',
                        manifest=json.loads((Path(self.a.data) / 'manifest.json').read_text()),
                        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in [*Path(__file__).parent.glob('*.py'),
                                                 ROOT/'exp/attention_selection/attention.py']})
        if not (self.out / 'protocol.json').exists():
            write(self.out / 'protocol.json', protocol)
        for split, items in self.data.items():
            for index, item in enumerate(items):
                if self.doc_path(item).exists():
                    continue
                self.inject.clear()
                self.sc.enabled = False
                self.sc.capture = False
                self.tc.capture = True
                ids = item['ids']
                tout = core_forward(self.teacher, tensor_ids(ids[:-1], self.a.device))
                tc = legacy_cache(tout.past_key_values)
                prior, _, _ = self.tc.snapshot()
                sout = core_forward(self.student, tensor_ids(ids[:-1], self.a.device))
                sc = legacy_cache(sout.past_key_values)
                prefix = len(ids)-1
                initial_student = cpu_cache(sc)
                initial_source = torch.stack([tc[l][1][0].transpose(0, 1).reshape(prefix, -1)
                                              for l in self.teacher_layers]).to(torch.bfloat16).cpu()
                states, reference = [], []
                current = ids[-1]
                for step in range(self.a.max_new_tokens):
                    if step in self.a.query_steps:
                        tail = tuple((k[:, :, prefix:].cpu(), v[:, :, prefix:].cpu()) for k, v in sc)
                        source_tail = torch.stack([tc[l][1][0, :, prefix:]
                                                    .transpose(0, 1).reshape(step, 512)
                                                    for l in self.teacher_layers]).to(torch.bfloat16).cpu()
                        state = dict(step=step, input_id=current, length=prefix+step,
                                     tail=tail, source_tail=source_tail, trow=prior.cpu())
                    tout = self.forward(self.teacher, current, tc)
                    target = self.logit(self.teacher, tout)
                    if step in self.a.query_steps:
                        state['teacher'] = target.cpu()
                        states.append(state)
                    tc = legacy_cache(tout.past_key_values)
                    prior, _, _ = self.tc.snapshot()
                    if step < max(self.a.query_steps):
                        sout = self.forward(self.student, current, sc)
                        sc = legacy_cache(sout.past_key_values)
                    current = int(target.argmax())
                    reference.append(current)
                    if current == self.st.eos_token_id:
                        break
                bundle = dict(item=item, initial_student=initial_student, initial_source=initial_source,
                              states=states, reference=reference)
                torch.save(bundle, self.doc_path(item))
                print(f'collect {split} {index+1}/{len(items)} states={len(states)}', flush=True)
        self.tc.capture = False

    def load_bundle(self, item):
        bundle = torch.load(self.doc_path(item), weights_only=False)
        base = device_cache(bundle['initial_student'], self.a.device)
        source = bundle['initial_source'].to(self.a.device)
        return bundle, base, source

    def state_inputs(self, base, source, state):
        cache = self.state_cache(base, state)
        tv = torch.cat((source, state['source_tail'].to(self.a.device)), 1)
        prior = state['trow'].to(self.a.device)
        selected = retrieve('teacher_token', prior, prior, self.a.budget, self.a.recent)
        remote = selected[~torch.isin(selected, mandatory(state['length'], self.a.recent, prior.device))]
        order = torch.argsort(prior[-1, remote], descending=True, stable=True)
        positions = remote[order[:self.a.inject_tokens]].sort().values
        return cache, tv[:, positions].float(), prior[-1], selected, positions

    def predict(self, state, inputs, network=None, mode='gated', alpha=1.0, shuffle=False,
                full=False):
        cache, source, prior, selected, positions = inputs
        self.sc.configure(None if full else selected)
        if network is None:
            self.inject.clear()
        else:
            self.inject.bind(network, positions, source, prior, mode, alpha, shuffle)
        out = self.forward(self.student, state['input_id'], cache)
        return self.logit(self.student, out)

    def loss(self, logits, target):
        lp, lq = F.log_softmax(target.float(), -1), F.log_softmax(logits.float(), -1)
        kl = (lp.exp() * (lp-lq)).sum()
        ce = -lq[target.argmax()]
        return kl + 0.25 * ce, kl, ce

    @torch.no_grad()
    def validation(self, network, mode, alpha):
        network.eval()
        docs = []
        for item in self.data['validation']:
            bundle, base, source = self.load_bundle(item)
            values = []
            for state in bundle['states']:
                inputs = self.state_inputs(base, source, state)
                logits = self.predict(state, inputs, network, mode, alpha)
                loss, kl, ce = self.loss(logits, state['teacher'].to(self.a.device))
                values.append([float(loss), float(kl), float(ce),
                               int(logits.argmax()) == int(state['teacher'].argmax()),
                               self.inject.stats()['mean']])
            docs.append(np.mean(values, axis=0))
        score = np.mean(docs, axis=0)
        return dict(objective=float(score[0]), kl=float(score[1]), ce=float(score[2]),
                    match=float(score[3]), gate_mean=float(score[4]))

    def new_network(self):
        torch.manual_seed(self.a.seed)
        return ValueMapper(self.layers, self.teacher_layers, rank=self.a.rank,
                           seed=self.a.seed).to(self.a.device)

    def save_network(self, name, network, epoch, validation):
        torch.save(dict(config=network.config, state_dict=network.state_dict(), epoch=epoch,
                        validation=validation), self.out / f'{name}.pt')

    def load_network(self, name):
        payload = torch.load(self.out / f'{name}.pt', map_location='cpu', weights_only=False)
        network = ValueMapper(**payload['config']).to(self.a.device)
        network.load_state_dict(payload['state_dict'])
        self.networks[name] = network.eval()
        return network

    def reconstruct(self):
        if (self.out / 'reconstruction.pt').exists():
            return
        network = self.new_network()
        source_rows, target_rows = [[] for _ in self.layers], [[] for _ in self.layers]
        for item in self.data['train']:
            bundle, base, source = self.load_bundle(item)
            state = bundle['states'][0]
            cache, tv, _, _, positions = self.state_inputs(base, source, state)
            for r, layer in enumerate(self.layers):
                native = cache[layer][1][0, :, positions].transpose(0, 1).reshape(len(positions), -1)
                source_rows[r].append(tv[r].detach())
                target_rows[r].append(native.float().detach())
        sources, targets = [torch.cat(v) for v in source_rows], [torch.cat(v) for v in target_rows]
        params = list(network.encoders.parameters()) + list(network.decoders.parameters())
        optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
        history = []
        best = float('inf')
        for epoch in range(1, self.a.reconstruction_epochs+1):
            order = torch.randperm(len(sources[0]), device=self.a.device)
            total = []
            for indices in order.split(256):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for r in range(len(self.layers)):
                    sv = targets[r][indices]
                    predicted, _, scale = network.proposal(r, sources[r][indices], sv)
                    losses.append(((predicted-sv) / scale).square().mean())
                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()
                total.append(float(loss.detach()))
            score = self.validation(network, 'fixed', 0.25)
            history.append(dict(epoch=epoch, training_nmse=float(np.mean(total)), **score))
            if score['objective'] < best:
                best = score['objective']
                self.save_network('reconstruction', network, epoch, score)
            print(f'reconstruction epoch={epoch} nmse={np.mean(total):.4f} val={score}', flush=True)
        write(self.out / 'train_reconstruction.json', history)

    def train_performance(self, name, mode):
        if (self.out / f'{name}.pt').exists():
            return
        network = self.load_network('reconstruction')
        network.train()
        if mode == 'fixed':
            params = list(network.encoders.parameters()) + list(network.decoders.parameters())
        else:
            params = list(network.parameters())
        optimizer = torch.optim.AdamW(params, lr=0.0003, weight_decay=0.01)
        alpha = 0.25 if mode == 'fixed' else 1.0
        initial = self.validation(network, mode, alpha)
        best = initial['objective']
        self.save_network(name, network, 0, initial)
        history = [dict(epoch=0, **initial)]
        for epoch in range(1, self.a.epochs+1):
            network.train()
            order = list(self.data['train'])
            random.Random(self.a.seed+epoch).shuffle(order)
            losses, norms = [], []
            for item in order:
                bundle, base, source = self.load_bundle(item)
                for state in bundle['states']:
                    optimizer.zero_grad(set_to_none=True)
                    inputs = self.state_inputs(base, source, state)
                    logits = self.predict(state, inputs, network, mode, alpha)
                    target = state['teacher'].to(self.a.device)
                    objective, kl, ce = self.loss(logits, target)
                    loss = objective + 0.01 * self.inject.penalty()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                    assert torch.isfinite(norm), 'Non-finite mapper gradients'
                    assert all(p.grad is None for p in self.student.parameters())
                    optimizer.step()
                    losses.append([float(objective.detach()), float(kl.detach()), float(ce.detach())])
                    norms.append(float(norm))
                    self.inject.clear()
            score = self.validation(network, mode, alpha)
            history.append(dict(epoch=epoch, train=np.mean(losses, axis=0).tolist(),
                                gradient_norm=float(np.mean(norms)), **score))
            if score['objective'] < best:
                best = score['objective']
                self.save_network(name, network, epoch, score)
            write(self.out / f'train_{name}.json', history)
            print(f'{name} epoch={epoch} train={np.mean(losses, axis=0).tolist()} val={score}', flush=True)
        self.inject.clear()

    @torch.no_grad()
    def calibrate(self):
        selected = {}
        calibration = []
        for name, mode in [('reconstruction', 'fixed'), ('performance_fixed', 'fixed'),
                           ('performance_gated', 'gated')]:
            network = self.load_network(name)
            alphas = [0, 0.05, 0.1, 0.25, 0.5] if mode == 'fixed' else [0, 0.25, 0.5, 1.0]
            trials = []
            for alpha in alphas:
                score = self.validation(network, mode, alpha)
                row = dict(name=name, mode=mode, alpha=alpha, **score)
                trials.append(row)
                calibration.append(row)
            positive = min((r for r in trials if r['alpha'] > 0), key=lambda r: r['objective'])
            guarded = min(trials, key=lambda r: r['objective'])
            selected[name] = dict(forced=positive, guarded=guarded)
        write(self.out / 'calibration.json', calibration)
        write(self.out / 'selected.json', selected)
        self.inject.clear()

    @torch.no_grad()
    def verify(self):
        network = self.new_network()
        item = self.data['train'][0]
        bundle, base, source = self.load_bundle(item)
        state = bundle['states'][0]
        inputs = self.state_inputs(base, source, state)
        cache_before = [(k.clone(), v.clone()) for k, v in inputs[0]]
        baseline = self.predict(state, inputs)
        zero = self.predict(state, inputs, network, 'gated', 0)
        assert torch.equal(baseline, zero), 'Zero gate is not exactly native selection'
        active = self.predict(state, inputs, network, 'gated', 1.0)
        for old, now in zip(cache_before, inputs[0]):
            assert all(torch.equal(a, b) for a, b in zip(old, now)), 'Backing KV mutated'
        assert inputs[-1].numel() == self.a.inject_tokens
        assert not bool(torch.isin(inputs[-1], mandatory(state['length'], self.a.recent, self.a.device)).any())
        self.inject.clear()
        # Compare end-to-end gradients against central finite differences in FP32.
        small = self.new_network().float()
        sv = torch.randn(5, 256, device=self.a.device)
        tv = torch.randn(5, 512, device=self.a.device)
        q = torch.randn(1536, device=self.a.device)
        features = torch.randn(5, 2, device=self.a.device)
        with torch.enable_grad():
            corrected, _, _ = small.correction(0, tv, sv, q, features)
            loss = corrected.square().mean()
            parameter = small.gates[0][-1].bias
            derivative = float(torch.autograd.grad(loss, parameter)[0])
        original = parameter.clone()
        parameter.add_(0.001)
        plus = float(small.correction(0, tv, sv, q, features)[0].square().mean())
        parameter.copy_(original-0.001)
        minus = float(small.correction(0, tv, sv, q, features)[0].square().mean())
        finite = (plus-minus)/0.002
        assert abs(derivative-finite) < 0.0001
        write(self.out / 'verification.json', dict(zero_gate_exact=True, native_backing_unchanged=True,
              k_unchanged_by_identity=True, remote_positions_only=True,
              active_logit_max_change=float((baseline-active).abs().max()),
              gate_derivative=derivative, finite_difference=finite))
        print('injection verification passed', flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['collect', 'verify', 'train', 'evaluate', 'runtime', 'all'], default='all')
    p.add_argument('--data', default='exp/selective_kv/data')
    p.add_argument('--output', default='exp/selective_kv/out_pilot')
    p.add_argument('--cache', default='/mnt/8T/cl/rescue_experiment_cache/selective_kv_pilot')
    p.add_argument('--teacher', default='/mnt/8T/cl/models/Qwen2-7B-Instruct')
    p.add_argument('--student', default='/mnt/8T/cl/models/Qwen2-1.5B-Instruct')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--budget', type=int, default=512)
    p.add_argument('--inject-tokens', type=int, default=128)
    p.add_argument('--recent', type=int, default=64)
    p.add_argument('--query-steps', nargs='+', type=int, default=[0, 2, 4, 6])
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--draft-k', type=int, default=4)
    p.add_argument('--rank', type=int, default=64)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--reconstruction-epochs', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    return p


def main():
    args = parser().parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    experiment = Pilot(args)
    if args.phase in ('collect', 'all'):
        experiment.collect()
    if args.phase in ('verify', 'all'):
        experiment.verify()
    if args.phase in ('train', 'all'):
        experiment.reconstruct()
        experiment.train_performance('performance_fixed', 'fixed')
        experiment.train_performance('performance_gated', 'gated')
        experiment.calibrate()
    if args.phase in ('evaluate', 'runtime', 'all'):
        from evaluate import evaluate, run_runtime
        if args.phase in ('evaluate', 'all'):
            evaluate(experiment)
        if args.phase in ('runtime', 'all'):
            run_runtime(experiment)


if __name__ == '__main__':
    main()
