"""Run the three causal experiments motivated by the short-context sweep.

Suites
------
dense
    Nested 512--1280-token length sweep on the same GovReport documents.
content
    Hold total length at 1024 tokens and vary added content: related
    continuation, duplicated prefix, or an unrelated report.
policy
    Compare full cache, random, recent, and target-attention selection at
    several cache budgets and input lengths.
"""

import argparse
import json
import math
import os
import random
import time


DENSE_LENGTHS = [512, 640, 768, 896, 1024, 1280]
POLICY_LENGTHS = [768, 1024, 1280]
POLICY_BUDGETS = [8, 16, 24, 32]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", choices=["dense", "content", "policy"], default=["dense", "content", "policy"])
    parser.add_argument("--source_file", default="data/govreport/govreport_2K.jsonl")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--dense_lengths", nargs="+", type=int, default=DENSE_LENGTHS)
    parser.add_argument("--policy_lengths", nargs="+", type=int, default=POLICY_LENGTHS)
    parser.add_argument("--policy_budgets", nargs="+", type=int, default=POLICY_BUDGETS)
    parser.add_argument("--output_dir", default="mechanism_results")
    parser.add_argument("--base_model_path", default="lmsys/vicuna-7b-v1.5-16k")
    parser.add_argument("--draft_model_path", default="double7/vicuna-68m")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_texts(path, limit):
    texts = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if isinstance(value.get("text"), str):
                texts.append(value["text"])
            if len(texts) == limit:
                break
    return texts


def split_prompt(text):
    text_marker = "### Text:"
    summary_marker = "### Summary:"
    text_start = text.index(text_marker) + len(text_marker)
    summary_start = text.rindex(summary_marker)
    return text[:text_start] + "\n", text[text_start:summary_start], "\n\n" + text[summary_start:]


def tokenized_parts(tokenizer, text):
    prefix, document, suffix = split_prompt(text)
    return (
        tokenizer.encode(prefix, add_special_tokens=True),
        tokenizer.encode(document, add_special_tokens=False),
        tokenizer.encode(suffix, add_special_tokens=False),
    )


def tensor_from_ids(ids, torch, device):
    return torch.tensor([ids], dtype=torch.long, device=device)


def nested_input(tokenizer, text, target_tokens, torch, device):
    prefix, document, suffix = tokenized_parts(tokenizer, text)
    budget = target_tokens - len(prefix) - len(suffix)
    if budget < 0 or len(document) < budget:
        raise ValueError((target_tokens, len(prefix), len(document), len(suffix)))
    ids = prefix + document[:budget] + suffix
    assert len(ids) == target_tokens
    return tensor_from_ids(ids, torch, device)


def content_input(tokenizer, primary, distractor, condition, torch, device):
    prefix, primary_doc, suffix = tokenized_parts(tokenizer, primary)
    _, distractor_doc, _ = tokenized_parts(tokenizer, distractor)
    fixed = len(prefix) + len(suffix)
    base_budget = 512 - fixed
    total_budget = 1024 - fixed
    base = primary_doc[:base_budget]

    if condition == "related":
        document = primary_doc[:total_budget]
    elif condition == "duplicate":
        repeats = math.ceil(total_budget / len(base))
        document = (base * repeats)[:total_budget]
    elif condition == "unrelated":
        remaining = total_budget - len(base)
        if len(distractor_doc) < remaining:
            repeats = math.ceil(remaining / len(distractor_doc))
            distractor_doc = distractor_doc * repeats
        document = base + distractor_doc[:remaining]
    else:
        raise ValueError(condition)

    ids = prefix + document + suffix
    assert len(ids) == 1024
    return tensor_from_ids(ids, torch, device)


def synchronized_call(torch, function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return dict(result), time.perf_counter() - started


def generate(model, input_ids, config, max_gen_len, torch):
    use_specextend = config["policy"] != "full"
    result, wall_time = synchronized_call(
        torch,
        lambda: model.spgenerate(
            input_ids,
            temperature=0,
            max_new_tokens=max_gen_len,
            output_result_line=False,
            verbose=False,
            use_specextend=use_specextend,
            retrieval_chunk_size=32,
            retrieve_top_k=config.get("k", 32),
            retrieve_every_n_steps=config.get("retrieve_every", 4),
            retrieval_policy=config.get("retrieval_policy", "attention"),
            retrieval_seed=config.get("retrieval_seed", 0),
            retrieval_verbose=False,
        ),
    )
    result["wall_time"] = wall_time
    result["wall_tokens_per_sec"] = result["total_generated"] / wall_time
    return result


def result_key(row):
    return (
        row["suite"], row.get("input_tokens"), row.get("condition"),
        row["policy"], row.get("k"), row["sample_index"],
    )


def save(path, args, torch, records):
    payload = {
        "config": vars(args),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "raw_runs": records,
    }
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    import torch
    from classic.model_classic import SPModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = SPModel.from_pretrained(
        base_model_path=args.base_model_path,
        draft_model_path=args.draft_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    texts = load_texts(args.source_file, args.samples)
    if len(texts) < args.samples:
        raise RuntimeError(f"Need {args.samples} source documents, found {len(texts)}")

    # Compile kernels and initialize model-specific caches outside measurements.
    warmup = nested_input(tokenizer, texts[0], 512, torch, device)
    generate(model, warmup, {"policy": "full"}, 5, torch)
    generate(model, warmup, {"policy": "attention", "retrieval_policy": "attention", "k": 16, "retrieve_every": 1}, 5, torch)

    for suite in args.suites:
        path = os.path.join(args.output_dir, f"{suite}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                records = json.load(handle).get("raw_runs", [])
        else:
            records = []
        completed = {result_key(row) for row in records}

        jobs = []
        if suite == "dense":
            configs = [
                {"policy": "full"},
                {"policy": "attention", "retrieval_policy": "attention", "k": 32, "retrieve_every": 4},
            ]
            for length in args.dense_lengths:
                for sample_index in range(args.samples):
                    for config in configs:
                        jobs.append((length, None, sample_index, config))
        elif suite == "content":
            configs = [
                {"policy": "full"},
                {"policy": "attention", "retrieval_policy": "attention", "k": 32, "retrieve_every": 4},
            ]
            for condition in ("related", "duplicate", "unrelated"):
                for sample_index in range(args.samples):
                    for config in configs:
                        jobs.append((1024, condition, sample_index, config))
        else:
            configs = [{"policy": "full"}]
            for policy in ("random", "recent", "attention"):
                for k in args.policy_budgets:
                    configs.append({
                        "policy": policy,
                        "retrieval_policy": policy,
                        "k": k,
                        "retrieve_every": 1,
                        "retrieval_seed": args.seed,
                    })
            for length in args.policy_lengths:
                for sample_index in range(args.samples):
                    for config in configs:
                        jobs.append((length, None, sample_index, config))

        total = len(jobs)
        for job_index, (length, condition, sample_index, config) in enumerate(jobs, start=1):
            probe = {
                "suite": suite,
                "input_tokens": length,
                "condition": condition,
                "policy": config["policy"],
                "k": config.get("k"),
                "sample_index": sample_index,
            }
            if result_key(probe) in completed:
                continue
            if suite == "content":
                input_ids = content_input(
                    tokenizer, texts[sample_index], texts[(sample_index + 1) % args.samples],
                    condition, torch, device,
                )
            else:
                input_ids = nested_input(
                    tokenizer, texts[sample_index], length, torch, device,
                )
            result = generate(model, input_ids, config, args.max_gen_len, torch)
            row = {**probe, **result}
            records.append(row)
            completed.add(result_key(row))
            save(path, args, torch, records)
            print(
                f"[{suite} {job_index}/{total}] len={length} condition={condition} "
                f"policy={config['policy']} k={config.get('k')} sample={sample_index} "
                f"accept={result['avg_accept_length']:.3f} tok/s={result['tokens_per_sec']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
