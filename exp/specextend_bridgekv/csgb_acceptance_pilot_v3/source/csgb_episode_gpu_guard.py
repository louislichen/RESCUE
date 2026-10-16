"""Read-only idle gate for one explicitly registered physical GPU. No killing."""
import argparse
import json
import re
import subprocess
import time


def observe(index, wanted, run=subprocess.run):
    result = run(['nvidia-smi','-i',str(index),
                  '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu',
                  '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    if result.stderr != '' or len(result.stdout.strip().splitlines()) != 1:
        raise RuntimeError('GPU identity observation unavailable; never infer idle')
    parts = [p.strip() for p in result.stdout.strip().split(',')]
    if len(parts) != 5 or parts[0] != wanted or parts[1] != 'NVIDIA GeForce RTX 4090':
        raise RuntimeError('GPU identity/model differs from frozen pilot')
    try: total,used,util = map(int,parts[2:])
    except ValueError as exc: raise RuntimeError('Invalid numeric GPU observation') from exc
    if total < 45000 or not 0 <= used <= total or not 0 <= util <= 100:
        raise RuntimeError('Unexpected capacity or malformed observation')
    apps = run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    if apps.stderr != '': raise RuntimeError('GPU process observation emitted stderr')
    pids = []
    for line in apps.stdout.splitlines():
        match = re.fullmatch(r'\s*(GPU-[0-9a-fA-F-]{36})\s*,\s*([0-9]+)\s*',line)
        if match is None: raise RuntimeError('Malformed process observation; never infer empty')
        if match[1] == wanted: pids.append(int(match[2]))
    return {'gpu_uuid':wanted,'physical_gpu_index':index,'memory_total_mib':total,
            'memory_used_mib':used,'utilization_percent':util,'pids':pids,
            'idle':not pids and used <= 128 and util == 0}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index',required=True,type=int,choices=range(8))
    ap.add_argument('--uuid',required=True)
    ap.add_argument('--timeout-seconds',type=int,default=21600)
    args=ap.parse_args()
    if re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',args.uuid) is None:
        raise ValueError('Explicit UUID required')
    if not 1 <= args.timeout_seconds <= 43200: raise ValueError('Bounded wait required')
    start,consecutive,previous=time.monotonic(),0,None
    while True:
        state=observe(args.index,args.uuid)
        key=(state['idle'],tuple(state['pids']))
        if key != previous: print(json.dumps({'kind':'idle_gate_observation',**state}),flush=True)
        previous=key
        consecutive=consecutive+1 if state['idle'] else 0
        if consecutive == 3:
            print(json.dumps({'kind':'idle_gate_passed','consecutive_observations':3,**state}),flush=True)
            return
        if time.monotonic()-start >= args.timeout_seconds:
            raise TimeoutError('GPU remained occupied; no process was touched')
        time.sleep(2 if state['idle'] else 30)


if __name__=='__main__': main()
