"""Measure speculative acceptance below 1K tokens on GovReport prompts.

The official benchmark begins at 1K tokens.  This runner keeps the same
prompt template and samples, truncates only the document body, and preserves
the ``### Summary:`` suffix so every point remains a summarization prompt.
"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", default="data/govreport/govreport_1K.jsonl")
    parser.add_argument(
        "--token_lengths",
        nargs="+",
        type=int,
        default=[38, 64, 128, 256, 512, 768],
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--output_file", default="results_short_context.json")
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
    marker = "### Text:"
    suffix_marker = "### Summary:"
    marker_end = text.index(marker) + len(marker)
    suffix_start = text.rindex(suffix_marker)
    return text[:marker_end] + "\n", text[marker_end:suffix_start], "\n\n" + text[suffix_start:]


def build_exact_length_input(tokenizer, text, target_tokens, torch, device):
    prefix, document, suffix = split_prompt(text)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    document_ids = tokenizer.encode(document, add_special_tokens=False)
    fixed = len(prefix_ids) + len(suffix_ids)
    if target_tokens < fixed:
        raise ValueError(
            f"Requested {target_tokens} tokens, but the valid prompt template needs {fixed}."
        )
    ids = prefix_ids + document_ids[: target_tokens - fixed] + suffix_ids
    if len(ids) != target_tokens:
        raise AssertionError((len(ids), target_tokens))
    return torch.tensor([ids], dtype=torch.long, device=device)


def aggregate(records):
    grouped = {}
    for row in records:
        grouped.setdefault((row["input_tokens"], row["mode"]), []).append(row)
    output = {}
    for (tokens, mode), rows in grouped.items():
        total_generated = sum(row["total_generated"] for row in rows)
        total_wall_time = sum(row["wall_time"] for row in rows)
        accept_values = [row["avg_accept_length"] for row in rows]
        output.setdefault(str(tokens), {})[mode] = {
            "runs": len(rows),
            "tokens_per_sec": total_generated / total_wall_time,
            "avg_accept_length": sum(accept_values) / len(accept_values),
        }
    return output


def main():
    args = parse_args()
    import torch
    import reproduce_classic as shared
    from classic.model_classic import SPModel

    shared.torch = torch
    shared.seed_everything(args.seed)
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
    if len(texts) != args.samples:
        raise RuntimeError(f"Expected {args.samples} samples, found {len(texts)}")

    warmup_ids = build_exact_length_input(
        tokenizer, texts[0], min(args.token_lengths), torch, device
    )
    for mode in ("standard", "specextend"):
        shared.run_mode(model, warmup_ids, mode, max_gen_len=5)

    records = []
    for target_tokens in args.token_lengths:
        for sample_index, text in enumerate(texts):
            input_ids = build_exact_length_input(
                tokenizer, text, target_tokens, torch, device
            )
            for run_index in range(args.runs):
                for mode in ("standard", "specextend"):
                    result = shared.run_mode(model, input_ids, mode, args.max_gen_len)
                    records.append(
                        {
                            "input_tokens": target_tokens,
                            "length_k": target_tokens / 1024,
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
                        f"tokens={target_tokens} sample={sample_index} run={run_index} "
                        f"mode={mode} accept={result['avg_accept_length']:.3f} "
                        f"tok/s={result['tokens_per_sec']:.2f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
