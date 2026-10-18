"""Collect verified joint rollouts; fit mapper and injection-aware length risk."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from .baselines import BaselineEngine
from .compare_baselines import append, write
from .config import SystemConfig
from .engine import RescueEngine
from .loading import ROOT, DEFAULT_MAPPER, DEFAULT_RISK, load_models, load_modules
from .long_data import context
from .policy import RiskModel
from .rollout_training import RolloutCollector, replay, output_loss


def collect(args):
    if args.output.exists():
        raise ValueError('Use a fresh collection output')
    args.output.mkdir(parents=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    data_path = ROOT/'exp/long_system/data/train.jsonl'
    data = [json.loads(l) for l in data_path.read_text().splitlines()][:args.documents]
    cfg = SystemConfig(selector='hierarchical', length_policy='fixed', fixed_k=12,
        max_new_tokens=args.max_new_tokens, injection_layers=(27,), inject_tokens=64,
        prefill_precision=args.prefill_precision)
    sources = [*sorted((ROOT/'rescue').glob('*.py')), *sorted((ROOT/'tests').glob('*.py'))]
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        dest = args.output/'source'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    write(args.output/'collection_protocol.json', dict(config=asdict(cfg),
        split='train only', source_indices=[r['source_index'] for r in data], lengths=args.lengths,
        target_dtype='float32', student_dtype='bfloat16', tf32=False,
        source_sha256=hashes, data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        checkpoint_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (args.mapper, DEFAULT_RISK)},
        objective='Conditional output KL+0.25 teacher CE+0.01 normalized residual; real SD first-token states for length',
        replay='Last Draft layer only, immutable snapshots, discard query targets after first mismatch',
        cache=str(args.cache), max_records_per_request=args.records))
    teacher, student, tokenizer = load_models(efficient=True)
    mapper, risk, cost = load_modules(teacher.device, mapper=args.mapper)
    for doc_index, item in enumerate(data):
        for length in args.lengths:
            ids = context(item, length)
            with BaselineEngine(teacher, student, cfg, vocab_size=len(tokenizer)) as engine:
                reference = engine.reference(ids, tokenizer.eos_token_id)['generated']
            for tokens in (0, 64):
                from dataclasses import replace
                current = replace(cfg, inject_tokens=tokens)
                collector = RolloutCollector(student, max_records=args.records)
                collector.eos_token_id = tokenizer.eos_token_id
                with RescueEngine(teacher, student, current, mapper, risk, cost,
                    vocab_size=len(tokenizer), draft_observer=collector, round_observer=collector.round) as engine:
                    collector.engine = engine
                    result = engine.generate(ids, tokenizer.eos_token_id)
                result.update(source_index=item['source_index'], length=length, inject_tokens=tokens,
                              output_agrees=result['generated'] == reference,
                              replay_records=len(collector.records), risk_frames=len(collector.frames))
                append(args.output/'collection_runtime.jsonl', result)
                if not result['output_agrees']:
                    write(args.output/'collection_mismatch.json', dict(reference=reference, result=result))
                    raise AssertionError('Collection differs from matched Target')
                path = args.cache/f"doc{item['source_index']}_n{length}_i{tokens}.pt"
                if path.exists():
                    raise ValueError('Refuse to overwrite recorded tensors')
                torch.save(dict(source_index=item['source_index'], length=length, inject_tokens=tokens,
                    document_sha256=item['document_sha256'], records=collector.records, frames=collector.frames), path)
                append(args.output/'bundles.jsonl', dict(path=str(path), source_index=item['source_index'],
                    length=length, inject_tokens=tokens, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                print('collected', doc_index, item['source_index'], length, tokens,
                      len(collector.records), len(collector.frames), flush=True)


def train(args):
    bundles = [json.loads(l) for l in (args.output/'bundles.jsonl').read_text().splitlines()]
    payloads = []
    for row in bundles:
        path = Path(row['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Recorded replay tensors changed')
        payloads.append(torch.load(path, weights_only=False))
    documents = sorted({r['source_index'] for r in payloads})
    if len(documents) < 3:
        raise ValueError('At least three training documents for internal fit/monitor split')
    held = documents[-1]
    fit = [r for p in payloads if p['source_index'] != held for r in p['records']]
    monitor = [r for p in payloads if p['source_index'] == held for r in p['records']]
    if not fit or not monitor:
        raise ValueError('Missing nonzero injection replay records')
    teacher, student, _ = load_models(efficient=True)
    # Teacher is not used by replay; release it before gradient work.
    del teacher
    torch.cuda.empty_cache()
    student.eval().requires_grad_(False)
    mapper, _, _ = load_modules(student.device, mapper=args.mapper)
    mapper = copy.deepcopy(mapper).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=args.learning_rate, weight_decay=.01)
    def score():
        mapper.eval()
        with torch.no_grad():
            losses = []
            for r in monitor:
                logits, penalty = replay(student, mapper, r)
                losses.append(float(output_loss(logits, r['target'], penalty)))
        mapper.train()
        return sum(losses)/len(losses)
    best, history = score(), []
    best_state = copy.deepcopy(mapper.state_dict())
    initial = best
    rng = random.Random(42)
    for epoch in range(args.epochs):
        order = list(range(len(fit)))
        rng.shuffle(order)
        total = 0.
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            r = fit[index]
            logits, penalty = replay(student, mapper, r)
            loss = output_loss(logits, r['target'], penalty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())
        value = score()
        history.append(dict(epoch=epoch+1, training_loss=total/len(fit), monitor_loss=value))
        if value < best:
            best, best_state = value, copy.deepcopy(mapper.state_dict())
        print('mapper epoch', epoch+1, history[-1], flush=True)
    torch.save(dict(config=mapper.config, state_dict=best_state,
        training_scope='Joint-rollout last-layer replay, training-only document monitor',
        original_monitor_loss=initial, selected_monitor_loss=best), args.output/'mapper_joint.pt')
    write(args.output/'mapper_training.json', dict(fit_documents=[d for d in documents if d != held],
        monitor_document=held, fit_queries=len(fit), monitor_queries=len(monitor), original_loss=initial,
        selected_loss=best, epochs=history, learning_rate=args.learning_rate))
    # Hazard targets are right-censored after the first rejection or EOS.
    frames_fit = [r for p in payloads if p['source_index'] != held for r in p['frames']]
    frames_monitor = [r for p in payloads if p['source_index'] == held for r in p['frames']]
    risk = RiskModel(group='attention_injected').cpu()
    x = torch.stack([r['features'] for r in frames_fit])
    with torch.no_grad():
        risk.mean.copy_(x.mean(0)); risk.scale.copy_(x.std(0, unbiased=False).clamp_min(.01))
    def hazards(frames):
        inputs = torch.stack([r['features'] for r in frames])
        labels = torch.full((len(frames), 12), -100, dtype=torch.long)
        for i, frame in enumerate(frames):
            labels[i, :len(frame['labels'])] = torch.tensor(frame['labels'])
        return inputs, labels
    xf, yf = hazards(frames_fit)
    xm, ym = hazards(frames_monitor)
    opt = torch.optim.AdamW(risk.parameters(), lr=.001, weight_decay=.01)
    best, best_state, risk_history = float('inf'), None, []
    for epoch in range(250):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(risk(xf).flatten(0, 1), yf.flatten())
        loss.backward(); opt.step()
        with torch.no_grad():
            monitor_loss = float(F.cross_entropy(risk(xm).flatten(0, 1), ym.flatten()))
        if monitor_loss < best:
            best, best_state = monitor_loss, copy.deepcopy(risk.state_dict())
        if epoch % 10 == 0:
            risk_history.append(dict(epoch=epoch+1, training_nll=float(loss.detach()), monitor_nll=monitor_loss))
    torch.save(dict(group='attention_injected', horizon=12, state_dict=best_state, temperature=1.,
        features='31 attention/read features plus gate, applied residual, working attention mass, layer fraction, token fraction'),
        args.output/'risk_injected.pt')
    write(args.output/'risk_training.json', dict(fit_frames=len(frames_fit), monitor_frames=len(frames_monitor),
        monitor_document=held, selected_monitor_nll=best, history=risk_history,
        limitation='Single seed, internal training monitor; controller is not trained after newly fitted mapper rollouts yet'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['collect', 'train'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path('/mnt/8T/cl/rescue_experiment_cache/long_system_v1/rollout_v1'))
    p.add_argument('--documents', type=int, default=6)
    p.add_argument('--lengths', type=int, nargs='+', default=[4096, 16384])
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--records', type=int, default=24)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--learning-rate', type=float, default=.0001)
    p.add_argument('--mapper', type=Path, default=DEFAULT_MAPPER)
    p.add_argument('--prefill-precision', choices=['native', 'bfloat16'], default='native')
    args = p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (collect if args.phase == 'collect' else train)(args)


if __name__ == '__main__':
    main()
