from eagle.model_eagle import EaModel
import torch
import json
from accelerate import Accelerator 
from termcolor import colored
import argparse

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
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON decode error at line {i}: {e}")
                continue
            text = obj.get("text")
            if text is None:
                print(f"[Warning] No 'text' field at line {i}, skipping.")
                continue
            texts.append(text)
    return texts

def main():
    parser = argparse.ArgumentParser(
        description="Run SpecExtend inference on a JSONL file of texts."
    )
    parser.add_argument(
        "--input_file", "-i",
        required=True,
        help="Path to input JSONL file (one JSON obj per line, with a 'text' field)."
    )
    parser.add_argument(
        "--max_samples", "-n",
        type=int, default=1,
        help="Maximum number of samples to read (default: 1)."
    )
    parser.add_argument(
        "--model_name", "-m",
        choices=["vicuna_7b", "longchat_7b"],
        default="vicuna_7b",
        help="Which base model to use (default: vicuna_7b)."
    )
    parser.add_argument(
        "--max_gen_len", "-max",
        type=int, default=256,
        help="Maximum number of tokens to generate(default: 256)."
    )
    parser.add_argument(
        "--use_specextend",
        action="store_true",
        help="Enable SpecExtend speculative decoding (default: False)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging from the model."
    )
    parser.add_argument(
        "--output_result_line",
        action="store_true",
        help="If set, print result line-by-line instead of as a block."
    )
    args = parser.parse_args()

    base_model_map = {
        "vicuna_7b":  "lmsys/vicuna-7b-v1.5-16k",
        "longchat_7b": "lmsys/longchat-7b-16k",
    }
    draft_model_map = {
        "vicuna_7b":  "jycha-98/EAGLE-vicuna-7b-v1.5-16k",
        "longchat_7b": "jycha-98/EAGLE-longchat-7b-16k",
    }

    base_model_path  = base_model_map[args.model_name]
    draft_model_path = draft_model_map[args.model_name]

    texts = load_texts_from_jsonl(args.input_file, args.max_samples)
    if not texts:
        print("No valid texts loaded; exiting.")
        return

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=draft_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    ).eval()
    tokenizer = model.tokenizer
    accelerator = Accelerator()
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Warmup GPUs
    print(colored(f'Warming up GPUs...', 'yellow'))
    for idx, text in enumerate(texts[:1]):
        input_ids = tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=True
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
    print(colored(f'Warmup complete!', 'yellow'))

    for idx, text in enumerate(texts):
        print(colored(f"\n=== Sample {idx+1}/{len(texts)} ===", 'yellow'))
        input_ids = tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=True
        ).to(accelerator.device)

        results = model.eagenerate(
            input_ids,
            temperature=0,
            max_new_tokens=args.max_gen_len,
            output_result_line=args.output_result_line,
            verbose=args.verbose,
            use_specextend=args.use_specextend,
            retrieval_chunk_size=32,
            retrieve_top_k=64,
            retrieve_every_n_steps=4,
            retrieval_verbose=False
        )

if __name__ == "__main__":
    main()