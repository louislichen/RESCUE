"""Reproduce the Vicuna-7B/68M SpecExtend results on paired inputs.

This runner measures naive autoregressive decoding, standard speculative
decoding, and SpecExtend with the same model instance and the same examples.
It writes both raw per-run measurements and aggregated metrics.
"""

import argparse
import glob
import json
import os
import random
import time

torch = None
SPModel = None


MODEL_PATHS = {
    "vicuna_7b": ("lmsys/vicuna-7b-v1.5-16k", "double7/vicuna-68m"),
    "longchat_7b": ("lmsys/longchat-7b-16k", "JackFram/llama-68m"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/govreport")
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--samples_per_length", type=int, default=20)
    parser.add_argument(
        "--sample_start",
        type=int,
        default=0,
        help="Zero-based first sample to run; useful for resuming or isolating failures.",
    )
    parser.add_argument("--runs_per_sample", type=int, default=2)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--model_name", choices=sorted(MODEL_PATHS), default="vicuna_7b")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["naive", "standard", "specextend"],
        default=["naive", "standard", "specextend"],
        help="Generation modes to run, in order. Useful when a baseline does not fit in GPU memory.",
    )
    parser.add_argument("--base_model_path")
    parser.add_argument("--draft_model_path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_file", default="reproduction_classic.json")
    parser.add_argument("--check_only", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_texts(path, limit, start=0):
    texts = []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            if len(texts) >= limit:
                break
            value = json.loads(line)
            if isinstance(value.get("text"), str):
                texts.append(value["text"])
    return texts


def resolve_files(data_dir, lengths):
    resolved = {}
    for length in lengths:
        matches = glob.glob(os.path.join(data_dir, f"*_{length}K.jsonl"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one '*_{length}K.jsonl' in {data_dir}, found {matches}"
            )
        resolved[length] = matches[0]
    return resolved


def synchronized_call(function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def run_mode(model, input_ids, mode, max_gen_len):
    if mode == "naive":
        result, wall_time = synchronized_call(
            lambda: model.ar_generate(
                input_ids,
                max_new_tokens=max_gen_len,
                verbose=False,
                return_metrics=True,
            )
        )
    else:
        result, wall_time = synchronized_call(
            lambda: model.spgenerate(
                input_ids,
                temperature=0,
                max_new_tokens=max_gen_len,
                output_result_line=False,
                verbose=False,
                use_specextend=(mode == "specextend"),
                retrieval_chunk_size=32,
                retrieve_top_k=32,
                retrieve_every_n_steps=4,
                retrieval_verbose=False,
            )
        )
    result = dict(result)
    result["wall_time"] = wall_time
    result["wall_tokens_per_sec"] = result["total_generated"] / wall_time
    return result


def aggregate(records):
    grouped = {}
    for row in records:
        key = (row["length_k"], row["mode"])
        grouped.setdefault(key, []).append(row)

    output = {}
    for (length, mode), rows in grouped.items():
        total_tokens = sum(row["total_generated"] for row in rows)
        total_time = sum(row["wall_time"] for row in rows)
        item = {
            "runs": len(rows),
            "total_generated": total_tokens,
            "total_wall_time": total_time,
            "tokens_per_sec": total_tokens / total_time,
        }
        acceptance = [row["avg_accept_length"] for row in rows if "avg_accept_length" in row]
        if acceptance:
            item["avg_accept_length"] = sum(acceptance) / len(acceptance)
        output.setdefault(f"{length}K", {})[mode] = item

    for length_data in output.values():
        naive = length_data.get("naive", {}).get("tokens_per_sec")
        standard = length_data.get("standard", {}).get("tokens_per_sec")
        for mode, values in length_data.items():
            if naive:
                values["speedup_vs_naive"] = values["tokens_per_sec"] / naive
            if standard:
                values["speedup_vs_standard_sd"] = values["tokens_per_sec"] / standard
    return output


def main():
    global torch, SPModel
    args = parse_args()
    files = resolve_files(args.data_dir, args.lengths)
    print(json.dumps({"resolved_data": files}, indent=2))
    if args.check_only:
        return
    import torch as torch_module
    from classic.model_classic import SPModel as model_class

    torch = torch_module
    SPModel = model_class
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; the paper uses one A100 80GB GPU.")

    seed_everything(args.seed)
    default_base, default_draft = MODEL_PATHS[args.model_name]
    model = SPModel.from_pretrained(
        base_model_path=args.base_model_path or default_base,
        draft_model_path=args.draft_model_path or default_draft,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    modes = tuple(args.modes)

    first_text = load_texts(files[args.lengths[0]], 1, args.sample_start)[0]
    warmup_ids = tokenizer.encode(
        first_text, return_tensors="pt", add_special_tokens=True
    ).to(device)
    for mode in modes:
        run_mode(model, warmup_ids, mode, max_gen_len=5)

    records = []
    for length in args.lengths:
        texts = load_texts(files[length], args.samples_per_length, args.sample_start)
        if len(texts) != args.samples_per_length:
            raise RuntimeError(
                f"{files[length]} contains {len(texts)} usable samples; "
                f"expected {args.samples_per_length}."
            )
        for sample_index, text in enumerate(texts, start=args.sample_start):
            input_ids = tokenizer.encode(
                text, return_tensors="pt", add_special_tokens=True
            ).to(device)
            for run_index in range(args.runs_per_sample):
                for mode in modes:
                    result = run_mode(model, input_ids, mode, args.max_gen_len)
                    records.append(
                        {
                            "length_k": length,
                            "sample_index": sample_index,
                            "run_index": run_index,
                            "mode": mode,
                            **result,
                        }
                    )
                    payload = {
                        "config": vars(args),
                        "environment": {
                            "torch": torch.__version__,
                            "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(0),
                        },
                        "aggregate": aggregate(records),
                        "raw_runs": records,
                    }
                    with open(args.output_file, "w", encoding="utf-8") as handle:
                        json.dump(payload, handle, indent=2)
                    print(
                        f"{length}K sample={sample_index} run={run_index} mode={mode}: "
                        f"{result['wall_tokens_per_sec']:.2f} tok/s"
                    )


if __name__ == "__main__":
    main()
