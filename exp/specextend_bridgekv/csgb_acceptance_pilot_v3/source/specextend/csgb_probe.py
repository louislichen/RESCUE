"""Read-only Native-V feature feasibility probe; not a speed benchmark."""
import argparse
import json
import os
import types
import hashlib
import numpy as np
import torch
from classic.model_classic import SPModel
from baseline_profiler import exact_input, load_texts
from csgb import features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--draft', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--budgets', nargs='+', type=int, default=[64])
    args = p.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible GPU required')
    model = SPModel.from_pretrained(base_model_path=args.base, draft_model_path=args.draft,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={'': 0}).eval()
    original_draft = model.draft
    original_refresh = model.update_working_cache_retrieval
    state = {'anchor': None, 'rows': [], 'proposal': 0}

    def refresh(self, *a, **kw):
        initial = self.selected_chunks is None
        # Selection clears attn_scores_final, so capture its source beforehand.
        source = self.attn_scores_final
        anchor = (source.detach().float().cpu().numpy().copy()
            if source is not None and kw.get('do_retrieval', False) else None)
        result = original_refresh(*a, **kw)
        if not initial and anchor is not None:
            state['anchor'] = anchor
        return result

    def draft(self, input_ids, nodes, threshold, max_depth):
        # This executes before root forward and before the subsequent refresh.
        a = getattr(self, 'attn_scores_final', None)
        selected = getattr(self, 'selected_chunks', None)
        if a is not None and selected:
            readable = [i for _, start, end in selected for i in range(start, end)]
            x = features(a.detach().float().cpu().numpy(), readable, state['anchor'])
            if x is not None:
                state['rows'].append({'coverage': float(x[1]), 'shift': float(x[2]),
                    'budget': nodes, 'root_positions': len(set(readable)),
                    'proposal_index': state['proposal'],
                    'prefix_tokens': int(input_ids.shape[-1])})
        state['proposal'] += 1
        return original_draft(input_ids, nodes, threshold, max_depth)

    model.draft = types.MethodType(draft, model)
    model.update_working_cache_retrieval = types.MethodType(refresh, model)
    texts = load_texts(args.source, args.samples)
    os.makedirs(args.output, exist_ok=True)
    for length in [4096, 8192]:
        for budget in args.budgets:
            for doc, text in enumerate(texts):
                state.update(anchor=None, rows=[], proposal=0)
                ids = exact_input(model.tokenizer, text, length, torch, torch.device('cuda:0'))
                with torch.inference_mode():
                    result = model.spgenerate(ids, nodes=budget, max_depth=10, threshold=.7,
                        max_new_tokens=128, temperature=0, use_specextend=True,
                        retrieval_chunk_size=32, retrieve_top_k=32,
                        retrieve_every_n_steps=4, retrieval_num_queries=1,
                        exact_target_attention=True, return_output_ids=True,
                        verbose=False, output_result_line=False)
                accepts = result['raw_accept_length_list']
                observations = []
                for row in state['rows']:
                    proposal = row['proposal_index']
                    if proposal < len(accepts):
                        row['next_accept_length'] = accepts[proposal]
                        observations.append(row)
                payload = {'phase': 'feature_probe_not_timing', 'context': length,
                    'budget': budget, 'document': doc,
                    'input_hash': hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                    'features': observations, 'result': result,
                    'attention_sampling': 'fresh attention at existing retrieval rounds only',
                    'timing_eligible': False}
                path = os.path.join(args.output, f'{length}_{budget}_{doc}.json')
                with open(path, 'x') as f:
                    json.dump(payload, f)
                print(json.dumps({'saved': path, 'feature_rounds': len(state['rows'])}), flush=True)


if __name__ == '__main__':
    main()
