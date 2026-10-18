"""Diagnostic CPU wall / CUDA timeline spans; never used for speed claims."""
from collections import defaultdict
from contextlib import contextmanager
import time
from unittest.mock import patch

import torch


class Timeline:
    def __init__(self, device):
        self.device = torch.device(device)
        self.entries = defaultdict(list)

    @contextmanager
    def span(self, name):
        cuda = self.device.type == 'cuda'
        begin = torch.cuda.Event(enable_timing=True) if cuda else None
        end = torch.cuda.Event(enable_timing=True) if cuda else None
        if cuda:
            begin.record()
        tick = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter()-tick
            if cuda:
                end.record()
            self.entries[name].append((elapsed, begin, end))

    def summary(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return {name: dict(calls=len(rows), cpu_wall_ms=sum(x[0] for x in rows)*1000,
                           cuda_timeline_ms=sum(x[1].elapsed_time(x[2]) for x in rows) if rows[0][1] else None)
                for name, rows in self.entries.items()}


@contextmanager
def instrument(engine, micro=False):
    """Nested spans include their children; CPU/GPU columns must not be added.

    CUDA spans include launch starvation inside the interval, not only busy
    kernel time. Wrapper instrumentation itself perturbs performance.
    """
    from contextlib import ExitStack
    from .attention import DraftAttention
    from .cache import WorkingCache
    from .injection import ContentInjector
    from .selection import ContextSelector, ReadState, LazyReadState
    from .scheduler import Scheduler
    from . import features, scheduler
    timeline = Timeline(engine.device)

    def wrap(target, method, label):
        original = getattr(target, method)
        def call(*args, **kwargs):
            name = label(*args, **kwargs) if callable(label) else label
            with timeline.span(name):
                return original(*args, **kwargs)
        stack.enter_context(patch.object(target, method, call))

    with ExitStack() as stack:
        wrap(ReadState, '__init__', 'read_state')
        wrap(LazyReadState, '__init__', 'read_state')
        wrap(ContextSelector, 'select', 'selection')
        wrap(WorkingCache, 'bind', 'working_bind')
        wrap(ContentInjector, 'bind', 'injection_bind')
        if micro:
            wrap(ContentInjector, 'apply', lambda self, layer, *a: 'gate_active' if layer in self.active else 'gate_noop')
        wrap(DraftAttention, 'step', 'draft_forward')
        wrap(features, 'prepare', 'length_prepare')
        wrap(features, 'finish', 'length_finish')
        wrap(scheduler, 'choose', 'length_score')
        wrap(Scheduler, 'publish', 'length_publish')
        wrap(Scheduler, 'resolve', 'length_wait')
        original = engine._forward
        def forward(model, tokens, cache):
            name = ('target' if model is engine.teacher else 'student')+('_prefill' if len(tokens) > 13 else '_verify')
            with timeline.span(name):
                return original(model, tokens, cache)
        stack.enter_context(patch.object(engine, '_forward', forward))
        yield timeline


def main():
    import argparse
    from dataclasses import replace
    import hashlib
    import json
    from pathlib import Path
    from .baselines import BaselineEngine
    from .config import SystemConfig
    from .engine import RescueEngine
    from .loading import ROOT, load_models, load_modules
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--documents', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    sources = sorted((ROOT/'rescue').glob('*.py'))
    hashes = {}
    for path in sources:
        dest = args.output/'source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher, student, tokenizer = load_models()
    mapper, risk, _ = load_modules(teacher.device)
    cost = json.loads((ROOT/'exp/system_validation/out_validation_v1_20261017/calibrated_cost.json').read_text())
    records = [json.loads(line) for line in (ROOT/'exp/attention_horizon/data/train.jsonl').read_text().splitlines()][:args.documents]
    config = SystemConfig(max_new_tokens=args.max_new_tokens)
    modes = ('native', 'retrieval', 'coverage_fixed', 'injection_fixed', 'complete')
    output = []
    for mode in modes:
        cfg = config if mode == 'complete' else replace(config, length_policy='fixed', fixed_k=4)
        if mode == 'coverage_fixed':
            cfg = replace(cfg, inject_tokens=0)
        factory = lambda: RescueEngine(teacher, student, cfg, mapper, risk, cost, vocab_size=len(tokenizer))
        if mode in ('native', 'retrieval'):
            factory = lambda: BaselineEngine(teacher, student, cfg, vocab_size=len(tokenizer),
                                              top_chunks=16 if mode == 'retrieval' else None)
        with factory() as engine:
            engine.generate(records[0]['ids'], tokenizer.eos_token_id)
        for item in records:
            with factory() as engine:
                reference = engine.reference(item['ids'], tokenizer.eos_token_id)
                with instrument(engine) as timeline:
                    result = engine.generate(item['ids'], tokenizer.eos_token_id)
                assert result['generated'] == reference['generated']
                output.append(dict(mode=mode, source_index=item['source_index'], result=result, spans=timeline.summary()))
                (args.output/'profiles.json').write_text(json.dumps(output, indent=2)+'\n')
                print(mode, item['source_index'], result['blocks'], round(result['decode_tps'], 2), flush=True)
    (args.output/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,
        scope='Instrumented diagnostic only, nested CPU/CUDA timelines; training documents',
        source_indices=[r['source_index'] for r in records], max_new_tokens=args.max_new_tokens), indent=2)+'\n')


if __name__ == '__main__':
    main()
