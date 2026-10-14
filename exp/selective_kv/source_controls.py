"""Post-pilot frozen source ablations; no checkpoint or strength retuning."""
import hashlib
import json
import random
import time

import torch

from run import Pilot, parser
from evaluate import speculative
from run import append, metrics, rows, write, teacher_reference


@torch.no_grad()
def main():
    args = parser().parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    e = Pilot(args)
    chosen = json.loads((e.out/'selected.json').read_text())['performance_gated']['forced']
    network = e.load_network('performance_gated')
    configs = {name: dict(name='performance_gated', mode='gated', alpha=chosen['alpha'],
                          source_policy=policy)
               for name, policy in [('aligned', 'aligned'), ('mean', 'mean'), ('zero', 'zero')]}
    write(e.out/'source_controls_protocol.json', dict(
        purpose='Frozen exploratory follow-up after primary shuffled content retained gains',
        changes='Mean or zero teacher V for BOTH mapper and gate; teacher-attention selection unchanged',
        checkpoint='performance_gated.pt', alpha=chosen['alpha'], tuning='None',
        timing='All three methods rerun in shuffled per-document order in this supplementary session',
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in e.out.parent.glob('*.py')}))
    path = e.out/'source_controls_offline.jsonl'
    done = {(r['source_index'], r['step'], r['method']) for r in rows(path)}
    for item in e.data['test']:
        bundle, base, source = e.load_bundle(item)
        for state in bundle['states']:
            original = e.state_inputs(base, source, state)
            for name, config in configs.items():
                if (item['source_index'], state['step'], name) in done:
                    continue
                inputs = list(original)
                if name == 'mean':
                    inputs[1] = inputs[1].mean(1, keepdim=True).expand_as(inputs[1])
                elif name == 'zero':
                    inputs[1] = torch.zeros_like(inputs[1])
                logits = e.predict(state, inputs, network, 'gated', chosen['alpha'])
                append(path, dict(source_index=item['source_index'], step=state['step'], method=name,
                                  gate=e.inject.stats(), **metrics(logits, state['teacher'].to(e.a.device))))
    for method, config in configs.items():
        speculative(e, e.data['train'][0], method, config)
    runtime_path = e.out/'source_controls_runtime.jsonl'
    done = {(r['source_index'], r['method']) for r in rows(runtime_path)}
    refs = {r['source_index']: r for r in rows(e.out/'runtime_test.jsonl') if r['method'] == 'teacher_greedy'}
    for i, item in enumerate(e.data['test']):
        order = list(configs.items())
        random.Random(args.seed+item['source_index']).shuffle(order)
        for method, config in order:
            if (item['source_index'], method) in done:
                continue
            result = speculative(e, item, method, config)
            result['output_agrees'] = result['generated'] == refs[item['source_index']]['generated']
            append(runtime_path, result)
        print(f'source controls {i+1}/30', flush=True)


if __name__ == '__main__':
    main()
