#!/usr/bin/env python3
"""Train/calibrate three token-attention routes, then evaluate greedy SD.

No parameters or strengths are fitted to the held-out test questions. The
teacher prompt attention is available to the two inference-time routes once
per question; Q updates use teacher attention only as a training objective.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp" / "attn_adapter_transfer"))
from common import load_model_and_tokenizer, read_jsonl, sample_to_prompt, tensor_ids
from run_attention_oracle import core_forward, legacy_cache, logits_of, prefix_cache, write_json
from methods import FullQUpdate, Intervention, TokenAdapter


def append(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def fast_window(student, frame, controller, *, attentions=False):
    cache = DynamicCache.from_legacy_cache(frame["cache"])
    hidden = frame["hidden"]
    prefix, length = frame["prefix_len"], frame["inputs"].shape[-1]
    keys = prefix + length
    rows = prefix + torch.arange(length, device=hidden.device)
    forbidden = torch.arange(keys, device=hidden.device)[None, :] > rows[:, None]
    mask = torch.zeros(length, keys, device=hidden.device, dtype=hidden.dtype)
    mask.masked_fill_(forbidden, torch.finfo(hidden.dtype).min)
    mask = mask[None, None]
    saved = []
    for layer in student.model.layers[controller.layers[0]:]:
        out = layer(hidden, attention_mask=mask, position_ids=rows[None],
                    past_key_value=cache, output_attentions=attentions, use_cache=True)
        hidden = out[0]
        if attentions:
            saved.append(out[1][0].float().mean(0))
    logits = student.lm_head(student.model.norm(hidden))[0].float()
    return logits, torch.stack(saved) if attentions else None


def encoded_sample(sample, cursor, teacher_tokenizer, tokenizer, args):
    prompt, gold, question = sample_to_prompt(sample, "all")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if len(ids) > args.max_prompt_tokens:
        raise ValueError(f"Sample {cursor} exceeds the fixed prompt limit")
    if ids != teacher_tokenizer(prompt, add_special_tokens=False)["input_ids"]:
        raise ValueError(f"Tokenization mismatch at {cursor}")
    boundary = prompt.rfind("\n\nQuestion:")
    if boundary < 0:
        raise ValueError("Missing document/question boundary")
    docs = [i for i, (start, end) in enumerate(encoded["offset_mapping"])
            if i > 0 and end > start and end <= boundary]
    return {"cursor": cursor, "ids": ids, "docs": docs, "gold": gold, "question": question}


@torch.no_grad()
def collect_frame(teacher, student, item, controller, args, policy):
    controller.enabled = False
    ids = item["ids"]
    teacher_prefix = prefix_cache(teacher, ids, args.device)
    student_prefix = prefix_cache(student, ids, args.device)
    tc, sc = teacher_prefix, student_prefix
    next_input = ids[-1]
    inputs, labels, target_attention = [], [], []
    bank = None
    for step in range(args.train_tokens):
        inputs.append(next_input)
        tout = core_forward(teacher, tensor_ids([next_input], args.device), tc, attentions=True)
        target = int(logits_of(teacher, tout).argmax(-1).item())
        labels.append(target)
        rows = torch.stack([tout.attentions[i][0, :, -1, :].float().mean(0)
                            for i in controller.layers])
        target_attention.append(rows)
        if bank is None:
            bank = torch.cat([tout.attentions[i][0, :, -1, :].float()
                              for i in controller.layers])
            bank = bank / bank.sum(-1, keepdim=True)
        sout = core_forward(student, tensor_ids([next_input], args.device), sc)
        proposed = int(logits_of(student, sout).argmax(-1).item())
        tc, sc = legacy_cache(tout.past_key_values), legacy_cache(sout.past_key_values)
        next_input = target if policy == "teacher" else proposed
        if next_input == args.eos_id:
            break
    query_inputs = tensor_ids(inputs, args.device)
    out = core_forward(student, query_inputs, student_prefix, hidden_states=True)
    key_len = len(ids) - 1 + len(inputs)
    aux = torch.stack([F.pad(row, (0, key_len - row.shape[-1])) for row in target_attention], dim=1)
    frame = dict(item, cache=student_prefix, prefix_len=len(ids)-1,
                 inputs=query_inputs, labels=torch.tensor(labels, device=args.device),
                 hidden=out.hidden_states[controller.layers[0]].detach(),
                 bank=bank.detach(), target_attention=aux.detach(), policy=policy)
    logits, _ = fast_window(student, frame, controller)
    frame["baseline_ids"] = logits.argmax(-1)
    return frame


@torch.no_grad()
def evaluate_frames(student, frames, controller):
    hits = tokens = repaired = broken = mismatches = matches = 0
    ce = 0.0
    was_enabled = controller.enabled
    controller.enabled = controller.kind != "baseline"
    for frame in frames:
        controller.set_context(frame["bank"], frame["docs"])
        logits, _ = fast_window(student, frame, controller)
        good = logits.argmax(-1) == frame["labels"]
        base_good = frame["baseline_ids"] == frame["labels"]
        hits += int(good.sum())
        tokens += good.numel()
        repaired += int((good & ~base_good).sum())
        broken += int((~good & base_good).sum())
        mismatches += int((~base_good).sum())
        matches += int(base_good.sum())
        ce += float(F.cross_entropy(logits, frame["labels"], reduction="sum"))
    controller.enabled = was_enabled
    return dict(match=hits/tokens, ce=ce/tokens, hits=hits, tokens=tokens,
                repaired=repaired, broken=broken, baseline_mismatches=mismatches, baseline_matches=matches)


def new_network(kind, student, controller, args):
    if kind.startswith("adapter"):
        return TokenAdapter(len(controller.layers), student.config.num_attention_heads,
                            len(controller.layers)*controller.teacher_heads,
                            student.config.hidden_size, args.max_alpha).to(args.device)
    return FullQUpdate(len(controller.layers), student.config.hidden_size).to(args.device)


def train_route(kind, teacher, student, train, calibration, controller, args, output):
    # Reset seeds so the adapter/uniform and Q/CE-only comparisons have identical
    # initial parameters and ordering. All checkpoints include the zero baseline.
    torch.manual_seed(args.seed)
    network = new_network(kind, student, controller, args)
    controller.configure(kind, network)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.adapter_lr if kind.startswith("adapter") else args.q_lr)
    controller.enabled = True
    controller.force_zero = True
    base_score = evaluate_frames(student, calibration, controller)
    controller.force_zero = False
    best_score, best_state, best_epoch = base_score, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        ordering = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(ordering)
        total_ce = total_aux = total_tokens = 0
        started = time.perf_counter()
        network.train()
        for order, index in enumerate(ordering):
            frame = train[index]
            optimizer.zero_grad(set_to_none=True)
            controller.set_context(frame["bank"], frame["docs"])
            use_aux = kind in ("q_attention", "q_native_attention")
            logits, rows = fast_window(student, frame, controller, attentions=use_aux)
            ce = F.cross_entropy(logits, frame["labels"])
            aux = torch.zeros((), device=args.device)
            if use_aux:
                target = frame["target_attention"]
                target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
                rows = rows / rows.sum(-1, keepdim=True).clamp_min(1e-12)
                aux = (target * (target.clamp_min(1e-12).log() - rows.clamp_min(1e-12).log())).sum(-1).mean()
            loss = ce + args.attention_weight * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            count = frame["labels"].numel()
            total_ce += float(ce.detach())*count
            total_aux += float(aux.detach())*count
            total_tokens += count
            if (order + 1) % 32 == 0:
                print(f"train {kind} epoch={epoch} {order+1}/{len(train)} ce={total_ce/total_tokens:.4f}", flush=True)
        network.eval()
        score = evaluate_frames(student, calibration, controller)
        row = dict(method=kind, epoch=epoch, train_ce=total_ce/total_tokens,
                   train_attention_kl=total_aux/total_tokens, calibration=score,
                   seconds=time.perf_counter()-started)
        history.append(row)
        append(output / "training.jsonl", row)
        print(json.dumps(row), flush=True)
        if (score["match"], -score["ce"]) > (best_score["match"], -best_score["ce"]):
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
    # Keep the trained endpoint for diagnosis even if calibration selects the
    # zero intervention. A selected zero route is a valid negative result.
    torch.save({"method": kind, "state": network.state_dict(), "epoch": args.epochs}, output / f"{kind}_final.pt")
    if best_state is not None:
        network.load_state_dict(best_state)
    checkpoint = {"method": kind, "state": best_state, "selected_epoch": best_epoch,
                  "calibration": best_score, "parameter_count": sum(p.numel() for p in network.parameters())}
    torch.save(checkpoint, output / f"{kind}.pt")
    return network, checkpoint, history


def trim_cache(cache, length):
    return tuple((k[:, :, :length, :], v[:, :, :length, :]) for k, v in legacy_cache(cache))


@torch.no_grad()
def speculative_decode(teacher, student, item, tokenizer, controller, args):
    started = time.perf_counter()
    controller.enabled = False
    ids = item["ids"]
    tc = prefix_cache(teacher, ids, args.device)
    sc = prefix_cache(student, ids, args.device)
    if controller.kind.startswith("adapter") or controller.kind == "bias":
        out = core_forward(teacher, tensor_ids(ids[-1:], args.device), tc, attentions=True)
        bank = torch.cat([out.attentions[i][0, :, -1, :].float() for i in controller.layers])
        bank /= bank.sum(-1, keepdim=True)
        controller.set_context(bank, item["docs"])
    controller.enabled = controller.kind != "baseline"
    generated, blocks = [], []
    last_id = ids[-1]
    while len(generated) < args.max_new_tokens:
        context_len = len(ids) + len(generated)
        k = min(args.draft_k, args.max_new_tokens - len(generated))
        draft, next_input = [], last_id
        for _ in range(k):
            out = core_forward(student, tensor_ids([next_input], args.device), sc)
            sc = legacy_cache(out.past_key_values)
            next_input = int(logits_of(student, out).argmax(-1).item())
            draft.append(next_input)
            if next_input == tokenizer.eos_token_id:
                break
        out = core_forward(teacher, tensor_ids([last_id] + draft, args.device), tc)
        checks = teacher.lm_head(out.last_hidden_state).float().argmax(-1)[0].tolist()
        accepted = 0
        for proposed, target in zip(draft, checks):
            if proposed != target:
                break
            accepted += 1
        blocks.append({"proposed": len(draft), "accepted": accepted, "position": len(generated)})
        generated.extend(draft[:accepted])
        if generated and generated[-1] == tokenizer.eos_token_id:
            break
        if len(generated) >= args.max_new_tokens:
            break
        if accepted < len(draft):
            correction = checks[accepted]
            tc = trim_cache(out.past_key_values, context_len + accepted)
            sc = trim_cache(sc, context_len + accepted)
            generated.append(correction)
            last_id = correction
        else:
            tc = legacy_cache(out.past_key_values)
            sout = core_forward(student, tensor_ids([draft[-1]], args.device), sc)
            sc = legacy_cache(sout.past_key_values)
            generated.append(checks[-1])
            last_id = checks[-1]
        assert sc[0][0].shape[-2] == len(ids)+len(generated)-1
        assert tc[0][0].shape[-2] == len(ids)+len(generated)-1
        if generated[-1] == tokenizer.eos_token_id:
            break
    torch.cuda.synchronize()
    return dict(cursor=item["cursor"], proposed=sum(b["proposed"] for b in blocks),
                accepted=sum(b["accepted"] for b in blocks), blocks=blocks,
                generated_ids=generated, text=tokenizer.decode(generated, skip_special_tokens=True),
                seconds=time.perf_counter()-started)


def aggregate(records):
    accepted = sum(r["accepted"] for r in records)
    proposed = sum(r["proposed"] for r in records)
    blocks = [b for r in records for b in r["blocks"]]
    return dict(accept_rate=accepted/proposed, accepted=accepted, proposed=proposed,
                questions=len(records), blocks=len(blocks), mean_accept_length=accepted/len(blocks),
                full_accept_rate=sum(b["accepted"] == b["proposed"] for b in blocks)/len(blocks),
                first_token_match=sum(b["accepted"] > 0 for b in blocks)/len(blocks),
                verifier_call_reduction_proxy=sum(len(r["generated_ids"]) for r in records)/len(blocks),
                total_seconds=sum(r["seconds"] for r in records))


def paired_interval(records, baseline, seed):
    rng = np.random.default_rng(seed)
    numerator = np.array([r["accepted"] for r in records])
    denominator = np.array([r["proposed"] for r in records])
    bn = np.array([r["accepted"] for r in baseline])
    bd = np.array([r["proposed"] for r in baseline])
    indices = rng.integers(0, len(records), (2000, len(records)))
    deltas = numerator[indices].sum(1)/denominator[indices].sum(1) - bn[indices].sum(1)/bd[indices].sum(1)
    return [float(x) for x in np.quantile(deltas, [0.025, 0.975])]


def sanity(student, frame, controller, args):
    controller.configure("baseline")
    controller.enabled = False
    with torch.no_grad():
        base, _ = fast_window(student, frame, controller)
        out = core_forward(student, frame["inputs"], frame["cache"])
        full = student.lm_head(out.last_hidden_state)[0].float()
    torch.testing.assert_close(base, full, rtol=0, atol=0)
    results = {"fast_window_full_max_logit_error": float((base-full).abs().max())}
    check_routes = ["adapter", "bias", "q_attention"]
    if any(kind.startswith("q_native") for kind in args.methods):
        check_routes.append("q_native_attention")
    for kind in check_routes:
        network = None if kind == "bias" else new_network(kind, student, controller, args)
        controller.configure(kind, network, 0.5)
        controller.enabled = True
        controller.force_zero = True
        controller.set_context(frame["bank"], frame["docs"])
        with torch.no_grad():
            zero, _ = fast_window(student, frame, controller)
        torch.testing.assert_close(zero, base, rtol=0, atol=0)
        results[f"{kind}_zero_max_logit_error"] = float((zero-base).abs().max())
        controller.force_zero = False
        if kind == "bias":
            continue
        fast, _ = fast_window(student, frame, controller)
        loss = F.cross_entropy(fast, frame["labels"])
        fast_grads = torch.autograd.grad(loss, tuple(network.parameters()))
        controller.set_context(frame["bank"], frame["docs"])
        out = core_forward(student, frame["inputs"], frame["cache"])
        full = student.lm_head(out.last_hidden_state)[0].float()
        full_grads = torch.autograd.grad(F.cross_entropy(full, frame["labels"]), tuple(network.parameters()))
        torch.testing.assert_close(fast, full, rtol=0, atol=0)
        for left, right in zip(fast_grads, full_grads):
            torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-6)
        results[f"{kind}_gradient_norm"] = float(torch.stack([g.norm() for g in fast_grads]).norm())
        results[f"{kind}_fast_full_gradient_max_error"] = max(float((l-r).abs().max()) for l, r in zip(fast_grads, full_grads))
        if not results[f"{kind}_gradient_norm"] > 0:
            raise RuntimeError("No training gradient")
    controller.force_zero = False
    controller.enabled = False
    controller.configure("baseline")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", default=str(ROOT/"exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl"))
    parser.add_argument("--teacher", default="/mnt/8T/cl/models/Qwen2-7B-Instruct")
    parser.add_argument("--student", default="/mnt/8T/cl/models/Qwen2-1.5B-Instruct")
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--cal-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--cal-start", type=int, default=128)
    parser.add_argument("--test-start", type=int, default=200)
    parser.add_argument("--train-tokens", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--adapter-lr", type=float, default=0.005)
    parser.add_argument("--q-lr", type=float, default=0.00003)
    parser.add_argument("--attention-weight", type=float, default=0.05)
    parser.add_argument("--max-alpha", type=float, default=0.5)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--methods", nargs="+",
                        choices=("adapter", "bias", "q_attention", "adapter_uniform", "q_output_only",
                                 "q_native_attention", "q_native_output_only"),
                        default=["adapter", "bias", "q_attention", "adapter_uniform", "q_output_only"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists():
        raise FileExistsError("Use a new output directory for a completed run")
    config = vars(args).copy()
    config["data_sha256"] = hashlib.sha256(Path(args.samples).read_bytes()).hexdigest()
    config["torch"] = torch.__version__
    config["gpu"] = torch.cuda.get_device_name()
    write_json(output/"config.json", config)
    teacher_tokenizer, teacher = load_model_and_tokenizer(args.teacher, torch.bfloat16, args.device)
    tokenizer, student = load_model_and_tokenizer(args.student, torch.bfloat16, args.device)
    teacher.requires_grad_(False)
    student.requires_grad_(False)
    layers = list(range(student.config.num_hidden_layers-4, student.config.num_hidden_layers))
    if teacher.config.num_hidden_layers != student.config.num_hidden_layers:
        raise ValueError("This initial comparison expects equal layer counts")
    args.eos_id = tokenizer.eos_token_id
    controller = Intervention(student, layers, teacher.config.num_attention_heads)
    controller.install()
    samples = list(read_jsonl(args.samples))
    splits = {"train": list(range(args.train_count)),
              "calibration": list(range(args.cal_start, args.cal_start+args.cal_count)),
              "test": list(range(args.test_start, args.test_start+args.test_count))}
    if any(set(splits[a]) & set(splits[b]) for a,b in (("train","calibration"),("train","test"),("calibration","test"))):
        raise ValueError("Overlapping question splits")
    write_json(output/"splits.json", splits)
    items = {name: [encoded_sample(samples[i], i, teacher_tokenizer, tokenizer, args) for i in indices]
             for name, indices in splits.items()}
    networks, selections = {}, {}
    training_order = [kind for kind in ("adapter", "adapter_uniform", "q_attention", "q_output_only",
                                        "q_native_attention", "q_native_output_only") if kind in args.methods]
    evaluation_order = ["baseline"] + args.methods
    if not args.eval_only:
        frames = {}
        for name in ("train", "calibration"):
            frames[name] = []
            started = time.perf_counter()
            for index, item in enumerate(items[name]):
                policy = "teacher" if name == "train" and index % 2 == 0 else "student"
                frame = collect_frame(teacher, student, item, controller, args, policy)
                frames[name].append(frame)
                append(output/"train_states.jsonl", {"split": name, "cursor": item["cursor"],
                       "policy": policy, "input_ids": frame["inputs"][0].tolist(),
                       "labels": frame["labels"].tolist(), "baseline_ids": frame["baseline_ids"].tolist()})
                if (index+1)%8 == 0:
                    print(f"collect {name} {index+1}/{len(items[name])} seconds={time.perf_counter()-started:.1f}", flush=True)
            if name == "train":
                checks = sanity(student, frames[name][0], controller, args)
                write_json(output/"validation.json", checks)
                print("sanity " + json.dumps(checks), flush=True)
        controller.configure("baseline")
        baseline_cal = evaluate_frames(student, frames["calibration"], controller)
        bias_scores = []
        for strength in (0.0, 0.125, 0.5, 1.0):
            controller.configure("bias", strength=strength)
            score = evaluate_frames(student, frames["calibration"], controller)
            bias_scores.append(dict(strength=strength, **score))
        bias = max(bias_scores, key=lambda row:(row["match"], -row["ce"]))
        selections["bias"] = bias
        write_json(output/"bias_calibration.json", bias_scores)
        for kind in training_order:
            network, checkpoint, history = train_route(kind, teacher, student, frames["train"],
                                                      frames["calibration"], controller, args, output)
            networks[kind] = network
            selections[kind] = {k: v for k,v in checkpoint.items() if k != "state"}
        write_json(output/"selection.json", {"baseline_calibration": baseline_cal, "methods": selections})
        del frames
        torch.cuda.empty_cache()
    else:
        selections = json.loads((output/"selection.json").read_text())["methods"]
        for kind in training_order:
            network = new_network(kind, student, controller, args)
            checkpoint = torch.load(output/f"{kind}.pt", map_location=args.device, weights_only=False)
            if checkpoint["state"] is not None:
                network.load_state_dict(checkpoint["state"])
            networks[kind] = network
    # A shared, frozen set of baseline-draft prefixes measures both repairs and
    # collateral damage. It is evaluated only after checkpoint selection.
    controller.configure("baseline")
    controller.force_zero = False
    test_frames = []
    for index, item in enumerate(items["test"]):
        frame = collect_frame(teacher, student, item, controller, args, "student")
        test_frames.append(frame)
        append(output/"test_states.jsonl", {"cursor": item["cursor"],
               "input_ids": frame["inputs"][0].tolist(), "labels": frame["labels"].tolist(),
               "baseline_ids": frame["baseline_ids"].tolist()})
        if (index+1)%10 == 0:
            print(f"collect heldout common prefixes {index+1}/{len(items['test'])}", flush=True)
    fixed_scores = {}
    for kind in evaluation_order:
        controller.configure(kind, networks.get(kind), selections.get("bias", {}).get("strength", 0.0))
        controller.force_zero = kind in networks and selections[kind]["selected_epoch"] == 0
        fixed_scores[kind] = evaluate_frames(student, test_frames, controller)
    write_json(output/"test_fixed_context.json", fixed_scores)
    del test_frames
    torch.cuda.empty_cache()
    results, summary = {}, {"config": config, "selection": selections,
                            "fixed_context": fixed_scores, "methods": {}}
    for kind in evaluation_order:
        controller.configure(kind, networks.get(kind), selections.get("bias", {}).get("strength", 0.0))
        controller.force_zero = kind in networks and selections[kind]["selected_epoch"] == 0
        records = []
        for index, item in enumerate(items["test"]):
            record = speculative_decode(teacher, student, item, tokenizer, controller, args)
            records.append(record)
            append(output/f"test_{kind}.jsonl", record)
            if (index+1)%5 == 0:
                print(f"test {kind} {index+1}/{len(items['test'])} accept={aggregate(records)['accept_rate']:.4f}", flush=True)
        results[kind] = records
        row = aggregate(records)
        if kind != "baseline":
            row["delta_accept_rate"] = row["accept_rate"] - summary["methods"]["baseline"]["accept_rate"]
            row["paired_95pct_delta_interval"] = paired_interval(records, results["baseline"], args.seed)
            row["identical_teacher_output_questions"] = sum(a["generated_ids"] == b["generated_ids"]
                                                           for a,b in zip(records, results["baseline"]))
        summary["methods"][kind] = row
        write_json(output/"summary.partial.json", summary)
        print("result " + kind + " " + json.dumps(row), flush=True)
    controller.restore()
    write_json(output/"summary.json", summary)


if __name__ == "__main__":
    main()
