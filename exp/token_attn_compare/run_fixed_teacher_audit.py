#!/usr/bin/env python3
"""Greedy SD acceptance replay against one invariant teacher continuation.

Teacher reference tokens are produced with single-query cached decoding. This
avoids differing BF16 verifier batch widths changing the evaluation targets.
The draft still uses the actual interventions, KV rollback, corrections and
full-accept bonus. This audit measures acceptance, not end-to-end latency.
"""
import argparse
import json
from pathlib import Path

import torch

from run_compare import (aggregate, core_forward, encoded_sample, legacy_cache,
                         load_model_and_tokenizer, logits_of, new_network,
                         paired_interval, prefix_cache, read_jsonl, tensor_ids,
                         trim_cache, write_json, append)
from methods import Intervention


@torch.no_grad()
def teacher_reference(teacher, student, item, tokenizer, controller, args):
    ids = item["ids"]
    tc = prefix_cache(teacher, ids, args.device)
    controller.enabled = False
    sc = prefix_cache(student, ids, args.device)
    generated, margins = [], []
    next_input, bank = ids[-1], None
    for position in range(args.max_new_tokens):
        out = core_forward(teacher, tensor_ids([next_input], args.device), tc, attentions=position == 0)
        logits = logits_of(teacher, out)
        next_input = int(logits.argmax(-1))
        top = logits[0].topk(2).values
        margins.append(float(top[0]-top[1]))
        tc = legacy_cache(out.past_key_values)
        generated.append(next_input)
        if bank is None:
            bank = torch.cat([out.attentions[i][0, :, -1, :].float() for i in controller.layers])
            bank /= bank.sum(-1, keepdim=True)
        if next_input == tokenizer.eos_token_id:
            break
    return dict(item, reference=generated, margins=margins, cache=sc, bank=bank)


@torch.no_grad()
def replay(student, frame, tokenizer, controller, args):
    ids, reference = frame["ids"], frame["reference"]
    controller.set_context(frame["bank"], frame["docs"])
    controller.enabled = controller.kind != "baseline"
    cache = frame["cache"]
    generated, blocks, last_id = [], [], ids[-1]
    while len(generated) < len(reference):
        position = len(generated)
        k = min(args.draft_k, args.max_new_tokens-position)
        draft, next_input = [], last_id
        for _ in range(k):
            out = core_forward(student, tensor_ids([next_input], args.device), cache)
            cache = legacy_cache(out.past_key_values)
            next_input = int(logits_of(student, out).argmax(-1))
            draft.append(next_input)
            if next_input == tokenizer.eos_token_id:
                break
        accepted = 0
        for index, proposed in enumerate(draft):
            if position+index >= len(reference) or proposed != reference[position+index]:
                break
            accepted += 1
            if proposed == tokenizer.eos_token_id:
                break
        blocks.append(dict(proposed=len(draft), accepted=accepted, position=position))
        generated.extend(draft[:accepted])
        if len(generated) == len(reference):
            break
        correction = reference[len(generated)]
        if accepted < len(draft):
            cache = trim_cache(cache, len(ids)+position+accepted)
        else:
            out = core_forward(student, tensor_ids([draft[-1]], args.device), cache)
            cache = legacy_cache(out.past_key_values)
        generated.append(correction)
        last_id = correction
        assert cache[0][0].shape[-2] == len(ids)+len(generated)-1
    assert generated == reference
    return dict(cursor=frame["cursor"], proposed=sum(b["proposed"] for b in blocks),
                accepted=sum(b["accepted"] for b in blocks), blocks=blocks,
                generated_ids=generated, seconds=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-output", required=True)
    parser.add_argument("--native-output", required=True)
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    compare, native, output = Path(cli.compare_output), Path(cli.native_output), Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists():
        raise FileExistsError("Audit already completed")
    config = json.loads((native/"config.json").read_text())
    original_config = json.loads((compare/"config.json").read_text())
    for key in ("data_sha256", "samples", "test_start", "test_count", "draft_k", "max_new_tokens", "seed"):
        assert config[key] == original_config[key], key
    args = argparse.Namespace(**config)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    teacher_tokenizer, teacher = load_model_and_tokenizer(args.teacher, torch.bfloat16, args.device)
    tokenizer, student = load_model_and_tokenizer(args.student, torch.bfloat16, args.device)
    teacher.requires_grad_(False)
    student.requires_grad_(False)
    layers = list(range(student.config.num_hidden_layers-4, student.config.num_hidden_layers))
    controller = Intervention(student, layers, teacher.config.num_attention_heads)
    controller.install()
    samples = list(read_jsonl(args.samples))
    frames = []
    for index in range(args.test_count):
        cursor = args.test_start+index
        item = encoded_sample(samples[cursor], cursor, teacher_tokenizer, tokenizer, args)
        frame = teacher_reference(teacher, student, item, tokenizer, controller, args)
        frames.append(frame)
        append(output/"teacher_reference.jsonl", dict(cursor=cursor, generated_ids=frame["reference"],
               logit_margins=frame["margins"], text=tokenizer.decode(frame["reference"], skip_special_tokens=True)))
        if (index+1)%10 == 0:
            print(f"reference {index+1}/{args.test_count}", flush=True)
    methods = ["baseline", "adapter", "bias", "q_native_attention", "adapter_uniform", "q_native_output_only"]
    records_by_method, summaries = {}, {}
    selections = json.loads((compare/"selection.json").read_text())["methods"]
    native_selections = json.loads((native/"selection.json").read_text())["methods"]
    for kind in methods:
        network, zero = None, False
        if kind not in ("baseline", "bias"):
            source = native if kind.startswith("q_native") else compare
            checkpoint = torch.load(source/f"{kind}.pt", map_location=args.device, weights_only=False)
            network = new_network(kind, student, controller, args)
            if checkpoint["state"] is not None:
                network.load_state_dict(checkpoint["state"])
            zero = checkpoint["selected_epoch"] == 0
        controller.configure(kind, network, selections["bias"]["strength"])
        controller.force_zero = zero
        records = []
        for index, frame in enumerate(frames):
            row = replay(student, frame, tokenizer, controller, args)
            records.append(row)
            append(output/f"test_{kind}.jsonl", row)
            if (index+1)%10 == 0:
                print(f"replay {kind} {index+1}/{args.test_count}", flush=True)
        score = aggregate(records)
        score.pop("total_seconds")
        if kind != "baseline":
            score["delta_accept_rate"] = score["accept_rate"]-summaries["baseline"]["accept_rate"]
            score["paired_95pct_delta_interval"] = paired_interval(records, records_by_method["baseline"], args.seed)
        records_by_method[kind], summaries[kind] = records, score
        print("result " + kind + " " + json.dumps(score), flush=True)
    controls = {}
    for kind, control in (("adapter", "adapter_uniform"), ("q_native_attention", "q_native_output_only")):
        controls[kind+"_vs_"+control] = dict(
            delta_accept_rate=summaries[kind]["accept_rate"]-summaries[control]["accept_rate"],
            paired_95pct_delta_interval=paired_interval(records_by_method[kind], records_by_method[control], args.seed))
    summary = dict(config=config, evaluation="fixed single-query greedy teacher continuation; SD draft replay",
                   compare_output=str(compare), native_output=str(native), methods=summaries,
                   paired_controls=controls, identical_teacher_output_questions=args.test_count,
                   selection=dict(original=selections, native=native_selections))
    controller.restore()
    write_json(output/"summary.json", summary)


if __name__ == "__main__":
    main()
