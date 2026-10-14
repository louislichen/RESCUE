"""Censored competing-risk likelihood; causal boundary/within-block forecasts."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from selection import FEATURE_GROUPS


class RiskNet(torch.nn.Module):
    def __init__(self, group, hidden=32, horizon=12):
        super().__init__()
        self.group, self.horizon = group, horizon
        indices = FEATURE_GROUPS[group]
        self.register_buffer('indices', torch.tensor(indices))
        self.register_buffer('mean', torch.zeros(len(indices)))
        self.register_buffer('scale', torch.ones(len(indices)))
        self.net = torch.nn.Sequential(torch.nn.Linear(len(indices), hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, horizon*3))
        self.temperature = 1.0

    def forward(self, x):
        z = (x[..., self.indices]-self.mean)/self.scale.clamp_min(1e-4)
        return self.net(z).reshape(*x.shape[:-1], self.horizon, 3)/self.temperature

    def continuation(self, x):
        p = self(x).softmax(-1)
        return p[..., 0].cumprod(-1)


def outcome_labels(draft, accepted, eos, horizon=12):
    labels = np.zeros(horizon, dtype=np.int64)
    mask = np.zeros(horizon, dtype=bool)
    for j, token in enumerate(draft):
        mask[j] = True
        if j < accepted:
            labels[j] = 1 if token == eos else 0
            if token == eos:
                break
        else:
            labels[j] = 2
            break
    return labels.tolist(), mask.tolist()


def load(path, device='cuda:0'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = RiskNet(payload['group'], horizon=payload['horizon']).to(device)
    model.load_state_dict(payload['state_dict'])
    model.temperature = payload['temperature']
    return model.eval(), payload


def fit(out, seeds=(42,43,44), epochs=120):
    from experiment import read_rows, write
    dataset = {}
    for split in ('train','validation'):
        blocks = read_rows(out/f'branches_{split}.jsonl')
        features, labels, masks, weights, keys = [], [], [], [], []
        docs = sorted({r['source_index'] for r in blocks})
        ndoc = {d:sum(r['source_index']==d for r in blocks) for d in docs}
        for block in blocks:
            for frame in block['frames']:
                features.append(frame['features'])
                labels.append(block['labels'])
                masks.append(block['mask'])
                weights.append(1/(ndoc[block['source_index']]*len(block['frames'])))
                keys.append((block['source_index'],block['rule'],block['block_index'],frame['offset']))
        dataset[split] = dict(x=torch.tensor(features, device='cuda:0'), y=torch.tensor(labels,device='cuda:0'),
            mask=torch.tensor(masks,device='cuda:0'), weight=torch.tensor(weights,device='cuda:0'), keys=keys)
    train, val = dataset['train'], dataset['validation']
    selected, history = {}, []
    for group in FEATURE_GROUPS:
        trials = []
        for seed in seeds:
            path = out/f'risk_{group}_{seed}.pt'
            if path.exists():
                model, payload = load(path)
                trials.append(dict(seed=seed,path=path.name,validation_nll=payload['validation_nll']))
                continue
            torch.manual_seed(seed)
            model = RiskNet(group).cuda()
            model.mean.copy_(train['x'][:,model.indices].mean(0))
            model.scale.copy_(train['x'][:,model.indices].std(0).clamp_min(1e-4))
            opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
            best, best_state, best_epoch = float('inf'), None, 0
            for epoch in range(1,epochs+1):
                model.train()
                order = torch.randperm(len(train['x']),device='cuda:0')
                for indices in order.split(512):
                    opt.zero_grad(set_to_none=True)
                    logits = model(train['x'][indices])
                    loss = F.cross_entropy(logits.flatten(0,1),train['y'][indices].flatten(),reduction='none').reshape(-1,12)
                    w = train['weight'][indices]
                    valid_weight=train['mask'][indices]*w[:,None]
                    loss=(loss*valid_weight).sum()/valid_weight.sum().clamp_min(1e-9)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    opt.step()
                model.eval()
                with torch.no_grad():
                    ce = F.cross_entropy(model(val['x']).flatten(0,1),val['y'].flatten(),reduction='none').reshape(-1,12)
                    valid_weight=val['mask']*val['weight'][:,None]
                    loss=(ce*valid_weight).sum()/valid_weight.sum()
                    value = float(loss)
                if value < best:
                    best, best_epoch = value, epoch
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best_state)
            temps = []
            for temperature in (.5,.75,1.,1.5,2.):
                model.temperature = temperature
                with torch.no_grad():
                    ce = F.cross_entropy(model(val['x']).flatten(0,1),val['y'].flatten(),reduction='none').reshape(-1,12)
                    valid_weight=val['mask']*val['weight'][:,None]
                    score=float((ce*valid_weight).sum()/valid_weight.sum())
                temps.append((score,temperature))
            score, temperature = min(temps)
            payload = dict(group=group,horizon=12,seed=seed,state_dict=best_state,epoch=best_epoch,
                           temperature=temperature,validation_nll=score)
            torch.save(payload,path)
            trials.append(dict(seed=seed,path=path.name,validation_nll=score,epoch=best_epoch,temperature=temperature))
            print(f'risk {group} seed={seed} epoch={best_epoch} val_nll={score:.4f}',flush=True)
        selected[group] = min(trials,key=lambda r:r['validation_nll'])
        history.append(dict(group=group,trials=trials))
    write(out/'risk_selection.json', selected)
    write(out/'risk_training.json', history)
    return selected
