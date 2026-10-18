"""Profile reliable long-context SpecExtend baselines.

The runner keeps prompts, seeds, tree construction, chunk size, and generation
length fixed across Full/Recent/Random/Attention.  It records both end-to-end
metrics and per-round diagnostics needed to distinguish cache-size effects from
retrieval-quality effects.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict


LENGTHS = [1024, 2048, 4096, 8192, 16384]
LABELS = {1024: "1K", 2048: "2K", 4096: "4K", 8192: "8K", 16384: "16K"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["classic", "eagle"], required=True)
    p.add_argument("--source_file", default="data/govreport/govreport_16K.jsonl")
    p.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--max_gen_len", type=int, default=256)
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--retrieve_every", type=int, default=4)
    p.add_argument("--tree_nodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--draft_model_path", required=True)
    p.add_argument("--device_map", default="auto", choices=["auto", "balanced"])
    p.add_argument("--attention_query_counts", nargs="+",
                   help="Attention ablation query counts; use 'all' for the full accepted path")
    p.add_argument("--attention_aggregations", nargs="+", choices=["mean", "decay", "max"],
                   default=["mean"])
    p.add_argument("--attention_decay_lambda", type=float, default=0.7)
    p.add_argument("--cache_top_ks", nargs="+", type=int,
                   help="Fixed attention-cache budgets to compare, e.g. 16 24 32 48 64")
    p.add_argument("--dynamic_topk_pilot", action="store_true",
                   help="Add accept-, KL-, and joint dynamic K controllers")
    p.add_argument("--dynamic_alpha", type=float, default=0.25)
    p.add_argument("--accept_low", type=float, default=1.0)
    p.add_argument("--accept_high", type=float, default=3.0)
    p.add_argument("--kl_low", type=float, default=0.2)
    p.add_argument("--kl_high", type=float, default=2.5)
    p.add_argument("--dynamic_patience", type=int, default=2)
    p.add_argument("--shuffle_policy_order", action="store_true",
                   help="Deterministically shuffle policy order within each paired sample")
    return p.parse_args()


def load_texts(path, limit):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            value = json.loads(line)
            if isinstance(value.get("text"), str):
                rows.append(value["text"])
            if len(rows) == limit:
                break
    if len(rows) < limit:
        raise RuntimeError(f"Need {limit} texts, found {len(rows)}")
    return rows


def split_prompt(text):
    text_marker, summary_marker = "### Text:", "### Summary:"
    text_start = text.index(text_marker) + len(text_marker)
    summary_start = text.rindex(summary_marker)
    return text[:text_start] + "\n", text[text_start:summary_start], "\n\n" + text[summary_start:]


def exact_input(tokenizer, text, target_tokens, torch, device):
    prefix, document, suffix = split_prompt(text)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    document_ids = tokenizer.encode(document, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    complete_ids = prefix_ids + document_ids + suffix_ids
    # The released "16K" GovReport split contains 15.1K--15.4K Vicuna
    # tokens after templating.  Preserve the complete example for that nominal
    # benchmark bin and record its exact token count in every result row.
    if target_tokens == 16384 and len(complete_ids) <= target_tokens:
        return torch.tensor([complete_ids], dtype=torch.long, device=device)
    budget = target_tokens - len(prefix_ids) - len(suffix_ids)
    if budget <= 0 or len(document_ids) < budget:
        raise ValueError({"target": target_tokens, "available_document_tokens": len(document_ids), "budget": budget})
    ids = prefix_ids + document_ids[:budget] + suffix_ids
    assert len(ids) == target_tokens
    return torch.tensor([ids], dtype=torch.long, device=device)


class RuntimeProfiler:
    """Low-overhead CUDA-event profiler installed around an engine instance."""

    def __init__(self, torch, module, model, engine):
        self.torch = torch
        self.module = module
        self.model = model
        self.engine = engine
        self.owner = model if engine == "classic" else model.ea_layer
        self.events = defaultdict(list)
        self.kl_tensors = []
        self.rounds = []
        self.retrievals = []
        self._previous_selection = None
        self._install()

    def reset(self):
        # Rebind rather than clear: completed result rows retain the previous
        # lists until they are serialized.
        self.events = defaultdict(list)
        self.kl_tensors = []
        self.rounds = []
        self.retrievals = []
        self._previous_selection = None

    def timed(self, label, fn, *args, **kwargs):
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        self.events[label].append((start, end))
        return result

    def _install(self):
        draft_name = "draft" if self.engine == "classic" else "topK_genrate"
        original_draft = getattr(self.owner, draft_name)

        def profiled_draft(*args, **kwargs):
            return self.timed("draft", original_draft, *args, **kwargs)

        setattr(self.owner, draft_name, profiled_draft)

        original_retrieval = self.owner.update_working_cache_retrieval

        def profiled_retrieval(*args, **kwargs):
            do_retrieval = kwargs.get("do_retrieval", args[1] if len(args) > 1 else False)
            initial = getattr(self.owner, "selected_chunks", None) is None
            result = self.timed("retrieval", original_retrieval, *args, **kwargs)
            if do_retrieval or initial:
                chunks = list(getattr(self.owner, "selected_chunks", []) or [])
                ids = [int(c[0]) for c in chunks]
                current = set(ids)
                previous = self._previous_selection
                union = current | previous if previous is not None else set()
                jaccard = (len(current & previous) / len(union)) if previous is not None and union else None
                total = max(int(getattr(self.owner, "total_seq_len", 0)), 1)
                self.retrievals.append({
                    "round": int(getattr(self.owner, "timestep", 0)),
                    "kind": "retrieval" if do_retrieval else "initial",
                    "selected_chunk_ids": ids,
                    "selected_chunk_ranges": [[int(c[1]), int(c[2])] for c in chunks],
                    "selected_chunk_centers_normalized": [((c[1] + c[2]) / 2) / total for c in chunks],
                    "jaccard_with_previous": jaccard,
                    "working_cache_tokens": int(result[0][0].shape[2]),
                    "full_draft_cache_tokens": int(getattr(self.owner, "total_seq_len", 0)),
                })
                self._previous_selection = current
            return result

        self.owner.update_working_cache_retrieval = profiled_retrieval

        original_tree = self.module.tree_decoding

        def profiled_tree(*args, **kwargs):
            result = self.timed("verification", original_tree, *args, **kwargs)
            kl = getattr(self.owner, "last_root_target_draft_kl", None)
            if kl is not None:
                self.kl_tensors.append(float(kl))
            return result

        self.module.tree_decoding = profiled_tree

        original_verify = self.module.verify

        def profiled_verify(*args, **kwargs):
            result = original_verify(*args, **kwargs)
            accept = int(result[2].item() if hasattr(result[2], "item") else result[2])
            stable = getattr(self.owner, "draft_stable_kv", None)
            working = int(stable[0][0].shape[2]) if stable else 0
            self.rounds.append({
                "round": len(self.rounds),
                "accept_length": accept,
                "rejected_position": accept + 1,
                "working_cache_tokens": working,
                "requested_cache_top_k": int(getattr(self.owner, "retrieve_top_k", 0)),
                "dynamic_cache_state": int(getattr(self.owner, "dynamic_cache_state", 0)),
                "ema_accept_length": getattr(self.owner, "dynamic_cache_ema_accept", None),
                "ema_root_target_draft_kl": getattr(self.owner, "dynamic_cache_ema_kl", None),
            })
            return result

        self.module.verify = profiled_verify

    def finish(self):
        self.torch.cuda.synchronize()
        stage_ms = {
            label: sum(start.elapsed_time(end) for start, end in pairs)
            for label, pairs in self.events.items()
        }
        kls = [float(x.item()) if hasattr(x, "item") else float(x) for x in self.kl_tensors]
        for i, value in enumerate(kls):
            if i < len(self.rounds):
                self.rounds[i]["root_target_draft_kl"] = value
        return {
            "stage_time_ms": stage_ms,
            "rounds": self.rounds,
            "retrieval_events": self.retrievals,
            "mean_root_target_draft_kl": sum(kls) / len(kls) if kls else None,
        }


def save_atomic(path, args, torch, records):
    payload = {
        "config": vars(args),
        "metric_definitions": {
            "draft_time_ms": "CUDA time in draft tree construction, including retrieval-cache use",
            "verification_time_ms": "CUDA time in target-model tree forward",
            "retrieval_time_ms": "CUDA time selecting/materializing draft working KV; subset of draft time",
            "root_target_draft_kl": "KL(target || draft) at the root distribution of each draft tree",
            "rejected_position": "1-indexed position of the first rejected draft token (accept_length + 1)",
            "16K": "Complete examples from the released GovReport 16K split; exact Vicuna-tokenized length is recorded per run (15.1K--15.4K here)",
        },
        "environment": {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "raw_runs": records,
    }
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(temporary, path)


def main():
    args = parse_args()
    query_specs = []
    for value in args.attention_query_counts or []:
        query_count = 0 if value.lower() == "all" else int(value)
        if query_count < 0:
            raise ValueError("attention query counts must be positive, or 'all'")
        query_specs.append((value.lower(), query_count))
    if query_specs and args.cache_top_ks:
        raise ValueError("attention_query_counts and cache_top_ks are separate ablations")
    if args.cache_top_ks and any(value <= 0 for value in args.cache_top_ks):
        raise ValueError("cache_top_ks must contain positive integers")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    import torch

    if args.engine == "classic":
        import classic.model_classic as engine_module
        model = engine_module.SPModel.from_pretrained(
            base_model_path=args.base_model_path, draft_model_path=args.draft_model_path,
            torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=args.device_map,
        ).eval()
        generator_name = "spgenerate"
        cache_top_k = 32
    else:
        import eagle.model_eagle as engine_module
        model = engine_module.EaModel.from_pretrained(
            base_model_path=args.base_model_path, ea_model_path=args.draft_model_path,
            torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=args.device_map,
        ).eval()
        generator_name = "eagenerate"
        cache_top_k = 64

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    texts = load_texts(args.source_file, args.samples)
    profiler = RuntimeProfiler(torch, engine_module, model, args.engine)
    generate = getattr(model, generator_name)

    if os.path.exists(args.output):
        records = json.load(open(args.output, encoding="utf-8")).get("raw_runs", [])
    else:
        records = []
    completed = {(r["length_bin_tokens"], r["policy"], r["sample_index"]) for r in records}

    warmup = exact_input(tokenizer, texts[0], min(args.lengths), torch, device)
    policy_config = {}
    controller_config = {}
    if args.cache_top_ks:
        policies = [f"attention_k{value}" for value in args.cache_top_ks]
        policy_config = {policy: (1, "mean", value)
                         for policy, value in zip(policies, args.cache_top_ks)}
        controller_config = {policy: "none" for policy in policies}
        if args.dynamic_topk_pilot:
            dynamic = ["dynamic_accept", "dynamic_kl", "dynamic_accept_kl"]
            policies += dynamic
            policy_config.update({policy: (1, "mean", 32) for policy in dynamic})
            controller_config.update({
                "dynamic_accept": "accept",
                "dynamic_kl": "kl",
                "dynamic_accept_kl": "accept_kl",
            })
    elif query_specs:
        policies = []
        for query_label, query_count in query_specs:
            aggregations = ["mean"] if query_count == 1 else args.attention_aggregations
            for aggregation in aggregations:
                policy = f"attention_{aggregation}_q{query_label}"
                policies.append(policy)
                policy_config[policy] = (query_count, aggregation, cache_top_k)
                controller_config[policy] = "none"
    else:
        policies = ["full", "full_hybrid", "recent", "random", "attention"]
        policy_config = {policy: (1, "mean", cache_top_k) for policy in policies}
        controller_config = {policy: "none" for policy in policies}
    for warm_policy in policies:
        query_count, aggregation, policy_top_k = policy_config[warm_policy]
        profiler.reset()
        generate(
            warmup, temperature=0, max_new_tokens=8, nodes=args.tree_nodes,
            verbose=False, output_result_line=False,
            use_specextend=warm_policy not in {"full", "full_hybrid"},
            retrieval_chunk_size=args.chunk_size,
            retrieve_top_k=policy_top_k,
            retrieve_every_n_steps=args.retrieve_every,
            retrieval_policy="attention" if warm_policy.startswith(("attention", "dynamic")) or warm_policy in {"full", "full_hybrid"} else warm_policy,
            retrieval_seed=args.seed,
            force_flash_prefill=True,
            force_hybrid_tree_attn=warm_policy == "full_hybrid",
            retrieval_num_queries=query_count,
            retrieval_query_aggregation=aggregation,
            retrieval_decay_lambda=args.attention_decay_lambda,
            dynamic_cache_controller=controller_config[warm_policy],
            dynamic_cache_alpha=args.dynamic_alpha,
            dynamic_cache_accept_low=args.accept_low,
            dynamic_cache_accept_high=args.accept_high,
            dynamic_cache_kl_low=args.kl_low,
            dynamic_cache_kl_high=args.kl_high,
            dynamic_cache_patience=args.dynamic_patience,
        )
        profiler.finish()

    total = len(args.lengths) * len(policies) * args.samples
    job = 0
    for length in args.lengths:
        for sample_index, text in enumerate(texts):
            input_ids = exact_input(tokenizer, text, length, torch, device)
            sample_policies = list(policies)
            if args.shuffle_policy_order:
                random.Random(args.seed + length * 1000 + sample_index).shuffle(sample_policies)
            for execution_order, policy in enumerate(sample_policies):
                query_count, aggregation, policy_top_k = policy_config[policy]
                job += 1
                key = (length, policy, sample_index)
                if key in completed:
                    continue
                profiler.reset()
                torch.cuda.synchronize()
                wall_start = time.perf_counter()
                result = generate(
                    input_ids, temperature=0, max_new_tokens=args.max_gen_len,
                    nodes=args.tree_nodes, output_result_line=False, verbose=False,
                    use_specextend=policy not in {"full", "full_hybrid"},
                    retrieval_chunk_size=args.chunk_size,
                    retrieve_top_k=policy_top_k,
                    retrieve_every_n_steps=args.retrieve_every,
                    retrieval_policy="attention" if policy.startswith(("attention", "dynamic")) or policy in {"full", "full_hybrid"} else policy,
                    retrieval_seed=args.seed,
                    force_flash_prefill=True,
                    force_hybrid_tree_attn=policy == "full_hybrid",
                    retrieval_verbose=False,
                    retrieval_num_queries=query_count,
                    retrieval_query_aggregation=aggregation,
                    retrieval_decay_lambda=args.attention_decay_lambda,
                    dynamic_cache_controller=controller_config[policy],
                    dynamic_cache_alpha=args.dynamic_alpha,
                    dynamic_cache_accept_low=args.accept_low,
                    dynamic_cache_accept_high=args.accept_high,
                    dynamic_cache_kl_low=args.kl_low,
                    dynamic_cache_kl_high=args.kl_high,
                    dynamic_cache_patience=args.dynamic_patience,
                )
                torch.cuda.synchronize()
                wall_time = time.perf_counter() - wall_start
                diagnostics = profiler.finish()
                row = {
                    "engine": args.engine, "input_tokens": int(input_ids.shape[1]),
                    "length_bin_tokens": length,
                    "length_label": LABELS.get(length, str(length)),
                    "policy": policy, "sample_index": sample_index,
                    "cache_top_k": policy_top_k, "chunk_size": args.chunk_size,
                    "seed": args.seed, "nodes": args.tree_nodes,
                    "retrieval_num_queries": query_count,
                    "retrieval_query_aggregation": aggregation,
                    "retrieval_decay_lambda": args.attention_decay_lambda,
                    "dynamic_cache_controller": controller_config[policy],
                    "execution_order": execution_order,
                    **result, "wall_time": wall_time,
                    "wall_tokens_per_sec": result["total_generated"] / wall_time,
                    **diagnostics,
                }
                records.append(row)
                completed.add(key)
                save_atomic(args.output, args, torch, records)
                print(f"[{job}/{total}] {args.engine} {LABELS.get(length,length)} {policy} "
                      f"sample={sample_index} accept={result['avg_accept_length']:.3f} "
                      f"tok/s={row['wall_tokens_per_sec']:.2f}", flush=True)


if __name__ == "__main__":
    main()
