#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Freeze a fixed HotpotQA evaluation set from streaming data."""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import Any, Dict

from tqdm import tqdm
from transformers import AutoTokenizer

from run_hotpot_attn_transfer import (
    build_docs_and_query,
    load_hotpot_dataset,
    make_prompt,
    str2bool,
)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def parse_args():
    p = argparse.ArgumentParser("Freeze streaming HotpotQA samples")
    p.add_argument("--teacher_model_path", default="/mnt/8T/cl/models/Qwen2-7B-Instruct")
    p.add_argument("--student_model_path", default="/mnt/8T/cl/models/Qwen2-1.5B-Instruct")
    p.add_argument("--dataset_name", default="hotpotqa/hotpot_qa")
    p.add_argument("--dataset_config", default="distractor")
    p.add_argument("--dataset_split", default="validation")
    p.add_argument("--hf_parquet_url", default="")
    p.add_argument("--doc_mode", default="all", choices=["all", "gold_only"])
    p.add_argument("--max_samples", type=int, default=300)
    p.add_argument("--max_L", type=int, default=2048)
    p.add_argument("--output_path", default="exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl")
    p.add_argument("--summary_path", default="exp/attn_transfer/data/hotpot_stream_300_maxL2048.summary.json")
    p.add_argument("--overwrite", type=str2bool, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.overwrite and os.path.exists(args.output_path):
        os.remove(args.output_path)
    if args.overwrite and os.path.exists(args.summary_path):
        os.remove(args.summary_path)

    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model_path, use_fast=True, local_files_only=True)
    student_tok = AutoTokenizer.from_pretrained(args.student_model_path, use_fast=True, local_files_only=True)

    ds_args = SimpleNamespace(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_path="",
        streaming=True,
        hf_parquet_url=args.hf_parquet_url,
    )
    dataset = load_hotpot_dataset(ds_args)

    seen = 0
    kept = 0
    skipped_bad = 0
    skipped_too_long = 0
    tokenizer_mismatch = 0

    pbar = tqdm(total=args.max_samples, desc="freeze", dynamic_ncols=True)
    for sample in dataset:
        seen += 1
        doc_texts, query = build_docs_and_query(sample, doc_mode=args.doc_mode)
        prompt = make_prompt(doc_texts, query)
        gold = str(sample.get("answer", "")).strip()
        question = str(sample.get("question", "")).strip()
        if not prompt or not gold or not question:
            skipped_bad += 1
            continue

        teacher_ids = teacher_tok(prompt, add_special_tokens=False)["input_ids"]
        student_ids = student_tok(prompt, add_special_tokens=False)["input_ids"]
        teacher_len = len(teacher_ids)
        student_len = len(student_ids)
        if max(teacher_len, student_len) > args.max_L:
            skipped_too_long += 1
            continue
        same_token_ids = teacher_ids == student_ids
        if not same_token_ids:
            tokenizer_mismatch += 1

        record = {
            "source_index": seen - 1,
            "question": question,
            "gold": gold,
            "prompt": prompt,
            "num_docs": len(doc_texts),
            "prompt_chars": len(prompt),
            "teacher_prompt_tokens": teacher_len,
            "student_prompt_tokens": student_len,
            "same_token_ids": same_token_ids,
            "doc_mode": args.doc_mode,
        }
        append_jsonl(args.output_path, record)
        kept += 1
        pbar.update(1)
        if kept >= args.max_samples:
            break

    pbar.close()
    summary = {
        "seen": seen,
        "kept": kept,
        "skipped_bad": skipped_bad,
        "skipped_too_long": skipped_too_long,
        "tokenizer_mismatch": tokenizer_mismatch,
        "max_L": args.max_L,
        "doc_mode": args.doc_mode,
        "output_path": args.output_path,
    }
    os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
