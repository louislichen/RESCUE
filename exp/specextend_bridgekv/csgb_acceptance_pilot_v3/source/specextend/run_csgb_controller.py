"""Exploratory document-held-out CSGB experiment, Native-V only.

All policies use the same round timer. Feature extraction is charged to its
policy. No future outcome enters a decision; training updates only chosen arms.
"""
import argparse
import hashlib
import json
import os
import random
import time
import types
from pathlib import Path
import numpy as np
import torch
from classic.model_classic import SPModel
from baseline_profiler import exact_input, load_texts
from csgb import features, LinUCB, BUDGETS
from csgb_cache_geometry import root_readable_positions


def atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


class Harness:
    def __init__(self, model):
        self.model = model
        self.original_draft = model.draft
        self.original_refresh = model.update_working_cache_retrieval
        model.draft = types.MethodType(self.draft, model)
        model.update_working_cache_retrieval = types.MethodType(self.refresh, model)

    def reset(self, policy, learner=None, explore=False, rho=0., clean=False):
        self.policy, self.learner, self.explore, self.rho = policy, learner, explore, rho
        self.anchor = None
        self.pending = None
        self.rows = []
        self.ema = 0.
        self.last_accept = 0.
        self.round_index = 0
        self.last_x = None
        self.latest_attention = None
        self.clean = clean
        self.last_feature_signature = None
        self.feature_cache_hits = 0
        self.feature_cache_misses = 0

    def refresh(self, model, *a, **kw):
        initial = model.selected_chunks is None
        attn = model.attn_scores_final
        anchor = (attn.detach().float().cpu().numpy().copy()
                  if self.policy in {'coverage', 'shift', 'csgb'} and
                  attn is not None and kw.get('do_retrieval', False) else None)
        result = self.original_refresh(*a, **kw)
        if not initial and anchor is not None:
            self.anchor = anchor
        return result

    def complete_pending(self, now, next_root_input_length, terminal=False):
        """Close a verified round without selecting or growing another tree."""
        if self.pending is not None:
            row = self.pending
            row['committed'] = next_root_input_length-row['prefix']
            if row['committed'] < 1:
                raise RuntimeError('Verified round must commit at least one token')
            submitted_seconds = now-row.pop('start')
            row['seconds'] = None if self.clean else submitted_seconds
            row['reward'] = None if self.clean else row['committed']-self.rho*row['seconds']
            self.last_accept = max(0, row['committed']-1)
            self.ema = .75*self.ema+.25*self.last_accept
            if self.explore:
                self.learner.update(None if row['x'] is None else np.array(row['x']),
                                    row['budget'], row['committed'], row['seconds'], self.rho)
            self.rows.append(row)
            if terminal:
                row['terminal_commit'] = True
            self.pending = None

    def finalize_terminal(self, committed_ids):
        """Terminal callback receives committed IDs, without an uncommitted Root."""
        if self.pending is None:
            raise RuntimeError('Terminal pending round is missing or already finalized')
        if not self.clean:
            torch.cuda.synchronize()
        # Ordinary draft entries contain committed prefix + the next Root.
        # Reproduce that length convention logically, without creating a token.
        self.complete_pending(time.perf_counter(),int(committed_ids.shape[-1])+1,True)

    def draft(self, model, input_ids, nodes, threshold, max_depth):
        if not self.clean:
            torch.cuda.synchronize()
        now = time.perf_counter()
        # A call from verify contains the just-committed prefix plus the next
        # root. Prefix-length difference therefore equals the prior commitment.
        self.complete_pending(now,int(input_ids.shape[-1]))
        decision_start = now
        x = None
        c = d = None
        if self.policy in {'coverage', 'shift', 'csgb'}:
            a = getattr(model, 'attn_scores_final', None)
            selected = getattr(model, 'selected_chunks', None)
            if a is not None and selected:
                self.latest_attention = a.detach().float().cpu().numpy().copy()
            if self.latest_attention is not None and selected:
                n=len(self.latest_attention)
                ranges=tuple((lo,min(hi,n)) for _,lo,hi in selected if lo<n)
                signature=(id(self.latest_attention),id(self.anchor),ranges)
                corrected_root=getattr(model,'csgb_root_coverage',False)
                if corrected_root:
                    signature+=(int(model.total_seq_len),int(input_ids.shape[-1]))
                if signature!=self.last_feature_signature:
                    readable = (root_readable_positions(selected,int(model.total_seq_len),n,
                                                       int(input_ids.shape[-1])) if corrected_root
                                else [i for lo,hi in ranges for i in range(lo,hi)])
                    self.last_x = features(self.latest_attention, readable, self.anchor)
                    self.last_feature_signature=signature
                    self.feature_cache_misses+=1
                else:
                    self.feature_cache_hits+=1
            # Attention is observed every 4 rounds; explicitly carry the latest
            # observed signature between observations, never fabricate new data.
            if self.last_x is not None:
                c, d = float(self.last_x[1]), float(self.last_x[2])
                x = {'coverage': np.array([1., c]), 'shift': np.array([1., d]),
                     'csgb': self.last_x}[self.policy]
        elif self.policy == 'history':
            x = np.array([1., self.last_accept/10., self.ema/10.])
        elif self.policy == 'constant':
            x = np.ones(1)
        if self.round_index == 0:
            x = None
        if self.policy.startswith('fixed'):
            budget = int(self.policy[5:])
        else:
            budget = self.learner.choose(x, explore=self.explore)
        self.pending = {'proposal': self.round_index, 'prefix': int(input_ids.shape[-1]),
                        'budget': budget, 'x': None if x is None else x.tolist(),
                        'coverage': c, 'shift': d, 'start': decision_start,
                        'feature_cache_hits':self.feature_cache_hits,
                        'feature_cache_misses':self.feature_cache_misses,
                        'decision_cpu_seconds': time.perf_counter()-decision_start}
        self.round_index += 1
        result = self.original_draft(input_ids, budget, threshold, max_depth)
        self.pending['actual_draft_nodes'] = int(result[0].shape[-1])
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--draft', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--target-fp32', action='store_true')
    p.add_argument('--fresh-data', action='store_true')
    p.add_argument('--budget-screen', action='store_true',
                   help='Training documents 0..3 only; fixed budget diagnostic, no controller selection')
    p.add_argument('--min-exploration', type=int, default=1)
    p.add_argument('--epsilon', type=float, default=0.)
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--budgets',type=int,nargs='+',default=list(BUDGETS))
    p.add_argument('--fallback-budget',type=int,default=64)
    args = p.parse_args()
    budgets=tuple(args.budgets)
    # Validate before creating output or loading expensive model weights.
    LinUCB(budgets=budgets,fallback=args.fallback_budget)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible GPU required')
    torch.manual_seed(42)
    np.random.seed(42)
    model = SPModel.from_pretrained(base_model_path=args.base, draft_model_path=args.draft,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={'': 0}).eval()
    if args.target_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        model.base_model.float()
    h = Harness(model)
    train_docs = list(range(24 if args.fresh_data else 8))
    dev_docs = list(range(24,32)) if args.fresh_data else list(range(8,12))
    test_docs = list(range(32,48)) if args.fresh_data else list(range(12,20))
    if args.budget_screen:
        train_docs, dev_docs, test_docs = list(range(4)), [], []
    texts = load_texts(args.source, 4 if args.budget_screen else (48 if args.fresh_data else 20))
    # All 20 documents are historical benchmark examples; held-out here means
    # controller-held-out only, not untouched by prior research.
    manifest = {'phase': 'training_budget_screen' if args.budget_screen else ('new_protocol' if args.fresh_data else 'exploratory'), 'train': train_docs, 'dev': dev_docs,
        'test': test_docs, 'contexts': [4096,8192], 'budgets': list(budgets),
        'fallback_budget':args.fallback_budget,
        'train_epochs': 2, 'alpha': args.alpha, 'ridge': 1., 'seed': 42,
        'min_exploration':args.min_exploration, 'epsilon':args.epsilon,
        'target_dtype':'float32' if args.target_fp32 else 'float16',
        'feature_cache':'exact reuse keyed by observed attention, anchor and readable intervals',
        'max_new_tokens': args.tokens, 'timing_repeats': args.repeats,
        'features_between_refreshes': 'latest observed attention recomputed against actual root cache and its anchor',
        'runtime': 'round-synchronized instrumented; eager Target attention',
        'document_hashes': [hashlib.sha256(t.encode()).hexdigest() for t in texts],
        'source_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic(out/'protocol.json', manifest)
    def run(doc, context, policy, learner=None, explore=False, rho=0., phase='test', repeat=0,clean=False):
        if clean and explore:
            raise ValueError('Clean timing cannot produce online reward labels')
        h.reset(policy, learner, explore, rho,clean=clean)
        ids = exact_input(model.tokenizer, texts[doc], context, torch, torch.device('cuda:0'))
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            result = model.spgenerate(ids, nodes=64, max_depth=10, threshold=.7,
                max_new_tokens=args.tokens, temperature=0, use_specextend=True,
                retrieval_chunk_size=32, retrieve_top_k=32, retrieve_every_n_steps=4,
                retrieval_num_queries=1, exact_target_attention=True,
                return_output_ids=True, verbose=False, output_result_line=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        row = {'phase': phase, 'document': doc, 'context': context, 'policy': policy,
            'repeat': repeat, 'seconds': elapsed, 'tps': result['total_generated']/elapsed,
            'timing_mode':'request_boundary_sync_only' if clean else 'round_synchronized',
            'result': result, 'rounds': h.rows,
            'input_hash': hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()}
        atomic(out/f'{phase}_{context}_{doc}_{policy}_{repeat}.json', row)
        print(json.dumps({'phase': phase,'context':context,'doc':doc,'policy':policy,
                          'repeat':repeat,'tps':row['tps']}), flush=True)
        return row
    # Warm all shapes before calibration/training; saved separately.
    screen_budgets = (8,16,24,32,64,96)
    # Fixed-64 reward calibration is retained across action-range revisions.
    warm_budgets=tuple(dict.fromkeys(budgets+(64,)))
    for budget in (screen_budgets if args.budget_screen else warm_budgets):
        run(0,4096,f'fixed{budget}',phase='warmup')
    if args.budget_screen:
        atomic(out/'screen_protocol.json', {'documents':[0,1,2,3],
            'contexts':[4096,8192], 'budgets':list(screen_budgets),
            'repeats':args.repeats, 'purpose':'training-only budget feasibility diagnostic',
            'timing':'request boundary synchronization only; eager Target attention',
            'confirmation_opened':False})
        failures=[]
        for doc in range(4):
            for context in [4096,8192]:
                ids=exact_input(model.tokenizer,texts[doc],context,torch,torch.device('cuda:0'))
                with torch.inference_mode():
                    ref=model.ar_generate(ids,max_new_tokens=args.tokens,verbose=False,return_metrics=True)
                atomic(out/f'reference_{context}_{doc}.json',ref)
                for repeat in range(args.repeats):
                    order=list(screen_budgets)
                    random.Random(9101+doc*100+context+repeat).shuffle(order)
                    for budget in order:
                        row=run(doc,context,f'fixed{budget}',phase='budget_screen',repeat=repeat,clean=True)
                        if row['result']['output_token_ids']!=ref['output_token_ids']:
                            failures.append([doc,context,budget,repeat])
        atomic(out/'SCREEN_DONE.json',{'failures':failures,'passed_exactness':not failures,
            'cells':4*2*len(screen_budgets)*args.repeats,'confirmation_opened':False})
        return
    if args.audit_only:
        failures=[]
        for doc in [14,18]:
            ids=exact_input(model.tokenizer,texts[doc],8192,torch,torch.device('cuda:0'))
            with torch.inference_mode():
                ref=model.ar_generate(ids,max_new_tokens=args.tokens,verbose=False,return_metrics=True)
            atomic(out/f'reference_8192_{doc}.json',ref)
            for budget in budgets:
                row=run(doc,8192,f'fixed{budget}',phase='numerical_audit')
                if row['result']['output_token_ids']!=ref['output_token_ids']:
                    failures.append([doc,budget])
        atomic(out/'AUDIT_DONE.json',{'failures':failures,'passed':not failures,
            'target_dtype':manifest['target_dtype'],'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
        return
    if args.smoke:
        for policy,width in [('csgb',4),('history',3),('constant',1)]:
            learner=LinUCB(width=width,budgets=budgets,fallback=args.fallback_budget)
            run(0,4096,policy,learner,True,20.,'smoke')
            print(json.dumps({'policy':policy,'counts':learner.counts.tolist()}),flush=True)
        atomic(out/'DONE.json',{'smoke':True})
        return
    rho_rows = [run(doc,c,'fixed64',phase='rho') for doc in range(2) for c in [4096,8192]]
    rho = sum(r['result']['total_generated'] for r in rho_rows)/sum(r['seconds'] for r in rho_rows)
    atomic(out/'rho.json', {'rho_tokens_per_second':rho,'source':'train documents 0,1 only; includes prefill'})
    widths = {'constant':1,'history':3,'coverage':2,'shift':2,'csgb':4}
    learners = {k:LinUCB(width=v,alpha=args.alpha,min_exploration=args.min_exploration,
                       epsilon=args.epsilon,budgets=budgets,fallback=args.fallback_budget) for k,v in widths.items()}
    for epoch in range(2):
        cases = [(d,c,k) for d in train_docs for c in [4096,8192] for k in learners]
        random.Random(42+epoch).shuffle(cases)
        for doc,c,k in cases:
            run(doc,c,k,learners[k],True,rho,'train',epoch)
    frozen = {k:{'a':v.a.tolist(),'y':v.y.tolist(),'counts':v.counts.tolist(),
                 'budgets':list(v.budgets),'fallback':v.fallback} for k,v in learners.items()}
    atomic(out/'frozen_models.json', frozen)
    fixed = [f'fixed{b}' for b in budgets]
    dev = [run(d,c,k,phase='dev') for d in dev_docs for c in [4096,8192] for k in fixed]
    best = {str(c):max(fixed,key=lambda k:sum(r['tps'] for r in dev if r['context']==c and r['policy']==k)) for c in [4096,8192]}
    atomic(out/'frozen_best_fixed.json', best)
    if args.fresh_data:
        # New confirmation documents remain sealed unless the frozen CSGB
        # clears both context-level development point-estimate gates.
        development = [run(d,c,'csgb',learners['csgb'],False,rho,'dev_controller')
                       for d in dev_docs for c in [4096,8192]]
        gains = {}
        for c in [4096,8192]:
            changes=[]
            for d in dev_docs:
                candidate=next(r for r in development if r['context']==c and r['document']==d)
                baseline=next(r for r in dev if r['context']==c and r['document']==d and r['policy']==best[str(c)])
                changes.append(candidate['tps']/baseline['tps']-1)
            gains[str(c)]=float(np.mean(changes))
        passed=all(g>0 for g in gains.values())
        atomic(out/'DEVELOPMENT_GATE.json',{'passed':passed,'paired_tps_point_gain':gains,
            'confirmation_opened':passed})
        if not passed:
            atomic(out/'DEV_STOP.json',{'reason':'CSGB did not beat best fixed at both contexts',
                                      'confirmation_opened':False})
            return
    for doc in test_docs:
        for c in [4096,8192]:
            ids = exact_input(model.tokenizer,texts[doc],c,torch,torch.device('cuda:0'))
            with torch.inference_mode():
                reference = model.ar_generate(ids,max_new_tokens=args.tokens,verbose=False,return_metrics=True)
            atomic(out/f'reference_{c}_{doc}.json',reference)
            for repeat in range(args.repeats):
                policies = fixed+list(learners)
                random.Random(42+doc*100+c+repeat).shuffle(policies)
                for k in policies:
                    r = run(doc,c,k,learners.get(k),False,rho,'test',repeat)
                    if r['result']['output_token_ids'] != reference['output_token_ids']:
                        atomic(out/'EXACTNESS_FAILURE.json',{'context':c,'doc':doc,'policy':k,'repeat':repeat})
                if args.fresh_data:
                    # Predeclared final system timing: no per-round profiler
                    # synchronization, same frozen policy, no model selection.
                    clean_policies=[best[str(c)],'csgb']
                    random.Random(711+doc*100+c+repeat).shuffle(clean_policies)
                    for k in clean_policies:
                        r=run(doc,c,k,learners.get(k),False,rho,'clean_test',repeat,clean=True)
                        if r['result']['output_token_ids']!=reference['output_token_ids']:
                            atomic(out/'EXACTNESS_FAILURE.json',{'context':c,'doc':doc,'policy':k,
                                'repeat':repeat,'timing_mode':'clean'})
    if frozen != {k:{'a':v.a.tolist(),'y':v.y.tolist(),'counts':v.counts.tolist(),
                    'budgets':list(v.budgets),'fallback':v.fallback} for k,v in learners.items()}:
        raise RuntimeError('Test modified frozen learner')
    atomic(out/'DONE.json',{'complete':True,'exactness_passed':not (out/'EXACTNESS_FAILURE.json').exists()})


if __name__ == '__main__':
    main()
