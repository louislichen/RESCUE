import os
import re
import glob
import json
import argparse
from tqdm import tqdm
import torch
from accelerate import Accelerator
from eagle.model_eagle import EaModel
from termcolor import colored

def load_texts_from_jsonl(path: str, max_samples: int = None):
    """
    Load up to `max_samples` texts from a JSONL file.
    Each line must be a JSON object with a "text" key.
    """
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if text is None:
                continue
            texts.append(text)
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SpecExtend performance across multiple sequence lengths"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/govreport",
        help="Directory containing .jsonl files (e.g., govreport_1K.jsonl, govreport_2K.jsonl, ...)"
    )
    parser.add_argument(
        "--samples_per_length", "-n", type=int, default=20,
        help="Number of samples to evaluate per sequence length"
    )
    parser.add_argument(
        "--runs_per_sample", "-r", type=int, default=2,
        help="Number of runs per sample"
    )
    parser.add_argument(
        "--model_name", "-m", choices=["vicuna_7b", "longchat_7b"],
        default="vicuna_7b",
        help="Which base model to use"
    )
    parser.add_argument(
        "--use_specextend", action="store_true",
        help="Enable SpecExtend during inference"
    )
    parser.add_argument(
        "--max_gen_len", type=int, default=256,
        help="Max tokens to generate per run"
    )
    parser.add_argument(
        "--output_file", type=str, default="eval_results_eagle.json",
        help="File to write aggregated results"
    )
    args = parser.parse_args()

    # Map model names to HF repo IDs
    base_map = {
        "vicuna_7b":  "lmsys/vicuna-7b-v1.5-16k",
        "longchat_7b": "lmsys/longchat-7b-16k"
    }
    draft_map = {
        "vicuna_7b":  "jycha-98/EAGLE-vicuna-7b-v1.5-16k",
        "longchat_7b": "jycha-98/EAGLE-longchat-7b-16k"
    }
    base_path = base_map[args.model_name]
    draft_path = draft_map[args.model_name]

    # Load and prepare model
    model = EaModel.from_pretrained(
        base_model_path=base_path,
        ea_model_path=draft_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    ).eval()
    tokenizer = model.tokenizer
    accelerator = Accelerator()
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Gather all JSONL files in the data directory
    pattern = os.path.join(args.data_dir, "*.jsonl")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {args.data_dir}")

    # Sort by sequence length extracted from filename (_<len>K.jsonl)
    def extract_length(path):
        basename = os.path.basename(path)
        m = re.search(r'_(\d+)K\.jsonl$', basename)
        if not m:
            raise ValueError(f"Filename {basename} does not match '_<len>K.jsonl'")
        return int(m.group(1))
    files = sorted(files, key=extract_length)

    # Warmup GPUs on the first sample of the first file
    print(colored('Warming up GPUs...', 'yellow'))
    first_texts = load_texts_from_jsonl(files[0], max_samples=1)
    if first_texts:
        input_ids = tokenizer.encode(
            first_texts[0], return_tensors="pt", add_special_tokens=True
        ).to(accelerator.device)
        for _ in range(3):
            _ = model.eagenerate(
                input_ids,
                temperature=0,
                max_new_tokens=5,
                output_result_line=False,
                verbose=False,
                use_specextend=args.use_specextend,
                retrieval_chunk_size=32,
                retrieve_top_k=64,
                retrieve_every_n_steps=4,
                retrieval_verbose=False
            )
    print(colored('Warmup complete!', 'yellow'))

    results = {}
    for filepath in tqdm(files, desc="Sequence lengths"):
        length = extract_length(filepath)
        texts = load_texts_from_jsonl(filepath, max_samples=args.samples_per_length)
        if not texts:
            continue

        runs_data = []
        for text in tqdm(texts, desc=f"{length}K samples", leave=False):
            input_ids = tokenizer.encode(
                text, return_tensors="pt", add_special_tokens=True
            ).to(accelerator.device)
            for _ in tqdm(range(args.runs_per_sample), desc="runs", leave=False):
                res = model.eagenerate(
                    input_ids,
                    temperature=0,
                    max_new_tokens=args.max_gen_len,
                    output_result_line=False,
                    verbose=False,
                    use_specextend=args.use_specextend,
                    retrieval_chunk_size=32,
                    retrieve_top_k=64,
                    retrieve_every_n_steps=4,
                    retrieval_verbose=False
                )
                runs_data.append(res)

        # Aggregate metrics
        total_runs = len(runs_data)
        # total generated tokens across all runs
        total_generated = sum(r['total_generated'] for r in runs_data)
        # total inference time across all runs (seconds)
        total_time = round(sum(r['inference_time'] for r in runs_data),3)
        # average tokens/sec for this length
        avg_tokens_per_sec = round(total_generated / total_time if total_time > 0 else 0.0, 3)
        # average accepted length across runs
        avg_accept_length = sum(r['avg_accept_length'] for r in runs_data) / total_runs

        results[f"{length}K"] = {
            "avg_accept_length": avg_accept_length,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "total_generated_tokens": total_generated,
            "total_inference_time (s)": total_time,
            "runs": total_runs,
        }

        # Checkpoint after each length
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"Done! Full results written to {args.output_file}")

if __name__ == "__main__":
    main()