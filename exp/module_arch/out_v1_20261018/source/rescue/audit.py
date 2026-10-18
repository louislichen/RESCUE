"""Independently validate saved system records, cache counters and source copies."""
import argparse
import hashlib
import json
from pathlib import Path

from .loading import ROOT


def audit(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        if hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f'Frozen source mismatch: {name}')
    references = json.loads((directory/'references.json').read_text())
    rows = [json.loads(line) for line in (directory/'runtime.jsonl').read_text().splitlines()]
    wanted = {(d, r, mode) for d in protocol['source_indices'] for r in range(protocol['args']['repeats'])
              for mode in protocol['args']['modes']}
    seen, request_ids = set(), set()
    paired = {}
    for row in rows:
        identity = row['source_index'], row['repeat'], row['mode']
        if identity not in wanted or identity in seen:
            raise AssertionError('Unexpected or duplicate request identity')
        seen.add(identity)
        if row['generated'] != references[str(row['source_index'])]['generated'] or row['output_agrees'] is not True:
            raise AssertionError('Saved output differs from Target reference')
        if len(row['generated']) > protocol['config']['max_new_tokens']:
            raise AssertionError('Output budget exceeded')
        if sum(b['proposed'] for b in row['trace']) != row['proposed'] or sum(b['accepted'] for b in row['trace']) != row['accepted']:
            raise AssertionError('Acceptance accounting differs')
        if row['blocks'] != len(row['trace']) or not row['trace'][-1]['terminal']:
            raise AssertionError('Missing terminal verification')
        paired.setdefault(identity[:2], {})[row['mode']] = row
        if row['mode'] == 'prototype':
            continue
        if row['request_id'] in request_ids:
            raise AssertionError('Request UUID reused')
        request_ids.add(row['request_id'])
        for i, block in enumerate(row['trace']):
            if not 0 <= block['accepted'] <= block['proposed'] <= block['cap'] <= protocol['config']['max_draft']:
                raise AssertionError('Invalid length/acceptance boundary')
            if block['key'] != dict(request=row['request_id'], round_index=i,
                                    prefix=block['prefix'], workset_version=i+1):
                raise AssertionError('Stale key or context version')
            if i and block['prefix'] != row['trace'][i-1]['prefix']+row['trace'][i-1]['accepted']+1:
                raise AssertionError('Committed prefix did not advance correctly')
            if block['terminal'] is not (i == len(row['trace'])-1):
                raise AssertionError('Extra work after termination')
            if block['workset_tokens'] > protocol['config']['budget']:
                raise AssertionError('Working set token budget exceeded')
        catchup = sum(not b['terminal'] and b['accepted'] == b['proposed'] for b in row['trace'])
        if row['draft_forwards'] != row['proposed']+catchup:
            raise AssertionError('Extra/missing Draft forward')
        if row['working_gather_layers'] != row['blocks']*protocol['student_layers']:
            raise AssertionError('Working KV was gathered more/less than once per round')
    if seen != wanted:
        raise AssertionError('Incomplete registered request matrix')
    same = {}
    def trajectory(row):
        return [(b['cap'], b['proposed'], b['accepted']) for b in row['trace']]
    for a, b in [('compat', 'prototype'), ('fast', 'compat'), ('rpc', 'coverage')]:
        if a in protocol['args']['modes'] and b in protocol['args']['modes']:
            same[f'{a}_vs_{b}'] = dict(matched=sum(trajectory(group[a]) == trajectory(group[b]) for group in paired.values()),
                                       pairs=len(paired))
    return dict(passed=True, requests=len(rows), exact_outputs=len(rows), unique_system_request_ids=len(request_ids),
                frozen_sources=len(protocol['source_sha256']), trajectory_comparisons=same,
                scope='Saved-record and frozen-source verification, not an independent GPU rerun')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    (args.directory/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
