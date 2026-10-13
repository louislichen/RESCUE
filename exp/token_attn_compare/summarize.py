#!/usr/bin/env python3
"""Question-paired comparisons with the teacher-attention-free controls."""
import argparse
import json
from pathlib import Path

from run_compare import paired_interval, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    summary = json.loads((output/"summary.json").read_text())
    seed = summary["config"]["seed"]
    methods = summary["methods"]
    analysis = {"paired_controls": {}, "teacher_output_differences": {}, "token_counts": {}}
    for split in ("train", "calibration"):
        rows = [row for row in read_jsonl(str(output/"train_states.jsonl")) if row["split"] == split]
        analysis["token_counts"][split] = dict(questions=len(rows), tokens=sum(len(row["labels"]) for row in rows))
    for route, control in (("adapter", "adapter_uniform"), ("q_attention", "q_output_only"),
                           ("q_native_attention", "q_native_output_only")):
        if route not in methods:
            continue
        records = list(read_jsonl(str(output/f"test_{route}.jsonl")))
        baseline = list(read_jsonl(str(output/f"test_{control}.jsonl")))
        analysis["paired_controls"][route+"_vs_"+control] = {
            "delta_accept_rate": methods[route]["accept_rate"]-methods[control]["accept_rate"],
            "paired_95pct_delta_interval": paired_interval(records, baseline, seed)}
    baseline = list(read_jsonl(str(output/"test_baseline.jsonl")))
    for kind in methods:
        if kind == "baseline":
            continue
        records = list(read_jsonl(str(output/f"test_{kind}.jsonl")))
        differences = []
        for row, base in zip(records, baseline):
            if row["generated_ids"] != base["generated_ids"]:
                a, b = row["generated_ids"], base["generated_ids"]
                position = next((i for i, (x,y) in enumerate(zip(a,b)) if x != y), min(len(a),len(b)))
                differences.append(dict(cursor=row["cursor"], first_difference_position=position,
                                        method_text=row["text"], baseline_text=base["text"]))
        analysis["teacher_output_differences"][kind] = differences
    write_json(output/"analysis.json", analysis)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
