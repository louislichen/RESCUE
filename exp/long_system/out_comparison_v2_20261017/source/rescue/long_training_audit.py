"""Audit long-context training provenance, selected weights and hazard NLL."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .compare_baselines import write
from .policy import RiskModel


def audit(directory):
    collection = json.loads((directory/'collection_protocol.json').read_text())
    training = json.loads((directory/'training_protocol.json').read_text())
    for protocol, subdirectory in ((collection, 'source'), (training, 'training_source')):
        for name, digest in protocol['source_sha256'].items():
            assert hashlib.sha256((directory/subdirectory/name).read_bytes()).hexdigest() == digest, name
    rows = [json.loads(l) for l in (directory/'collection_runtime.jsonl').read_text().splitlines()]
    expected = {(d, n, i) for d in collection['source_indices'] for n in collection['lengths'] for i in (0, 64)}
    assert len(rows) == len(expected)
    assert {(r['source_index'], r['length'], r['inject_tokens']) for r in rows} == expected
    assert all(r['output_agrees'] and len(r['generated']) == r['tokens'] for r in rows)
    assert len({r['request_id'] for r in rows}) == len(rows)
    mapper_info = json.loads((directory/'mapper_training.json').read_text())
    chosen_loss = min(mapper_info['original_loss'], *(r['monitor_loss'] for r in mapper_info['epochs']))
    assert mapper_info['selected_loss'] == chosen_loss
    assert mapper_info['monitor_document'] not in mapper_info['fit_documents']
    selected = torch.load(directory/'mapper_joint.pt', map_location='cpu', weights_only=False)
    original = torch.load(training['mapper'], map_location='cpu', weights_only=False)
    changed = [name for name, tensor in selected['state_dict'].items()
               if not torch.equal(tensor, original['state_dict'][name])]
    if mapper_info['selected_loss'] == mapper_info['original_loss']:
        assert not changed
    else:
        assert changed
    frames = json.loads((directory/'risk_frames.json').read_text())
    assert len(frames) == sum(r['risk_frames'] for r in rows)
    fit = [r for r in frames if not r['monitor']]
    monitor = [r for r in frames if r['monitor']]
    model = RiskModel.load(directory/'risk_injected.pt', 'cpu')
    x = torch.tensor([r['features'] for r in fit])
    torch.testing.assert_close(model.mean, x.mean(0))
    torch.testing.assert_close(model.scale, x.std(0, unbiased=False).clamp_min(.01))
    assert all(len(r['features']) == 36 and 1 <= len(r['labels']) <= 12 for r in frames)
    for r in frames:
        assert all(v == 0 for v in r['labels'][:-1])
        assert 0 <= r['labels'][-1] <= 2
        if not r['inject_tokens']:
            assert all(v == 0 for v in r['features'][31:])
    y = torch.full((len(monitor), 12), -100, dtype=torch.long)
    for i, frame in enumerate(monitor):
        y[i, :len(frame['labels'])] = torch.tensor(frame['labels'])
    with torch.no_grad():
        nll = float(F.cross_entropy(model(torch.tensor([r['features'] for r in monitor])).flatten(0, 1), y.flatten()))
    info = json.loads((directory/'risk_training.json').read_text())
    assert abs(nll-info['selected_monitor_nll']) < 1e-6
    assert info['fit_frames'] == len(fit) and info['monitor_frames'] == len(monitor)
    return dict(passed=True, collection_requests=len(rows), exact_collection_outputs=len(rows),
        replay_records=sum(r['replay_records'] for r in rows), risk_frames=len(frames),
        mapper_changed_tensors=changed, selected_monitor_nll=nll,
        scope='Frozen-source/selection/checkpoint and CPU hazard audit; no reexecution of mapper GPU gradients')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.directory)
    write(args.directory/'training_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
