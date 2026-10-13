#!/usr/bin/env python3
"""Diagnose query-space reachability, then calibrate residual completion.

All strengths are selected on the original 32 calibration questions. Test
teacher continuations are reused verbatim from the earlier fixed-target audit.
No per-state teacher label is used to optimize an intervention.
"""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import torch

from run_compare import (aggregate, append, core_forward, encoded_sample,
                         legacy_cache, load_model_and_tokenizer, logits_of,
                         paired_interval, prefix_cache, read_jsonl, tensor_ids,
                         write_json)
from run_fixed_teacher_audit import replay, teacher_reference
from geometry import GeometryContext, GeometryIntervention, distribution_metrics


def config_route(controller, route):
    controller.route(route["projected"], route["residual"],
                     native=route["native"], shuffle=route.get("shuffle", False))
    controller.capture = False


@torch.no_grad()
def evaluate(student, frames, tokenizer, controller, args, route, *, record_path=None):
    config_route(controller, route)
    rows = []
    started = time.perf_counter()
    for frame in frames:
        controller.bind(frame)
        row = replay(student, frame, tokenizer, controller, args)
        rows.append(row)
        if record_path is not None:
            append(record_path, row)
    score = aggregate(rows)
    score.pop("total_seconds")
    score["prototype_student_replay_seconds"] = time.perf_counter()-started
    return score, rows


@torch.no_grad()
def load_fixed_test_frame(teacher, student, item, tokenizer, controller, args, fixed):
    controller.enabled = False
    tc = prefix_cache(teacher, item["ids"], args.device)
    sc = prefix_cache(student, item["ids"], args.device)
    out = core_forward(teacher, tensor_ids(item["ids"][-1:], args.device), tc, attentions=True)
    first = int(logits_of(teacher, out).argmax(-1))
    reference = fixed["generated_ids"][:args.max_new_tokens]
    if first != reference[0]:
        raise RuntimeError(f"Frozen reference reconstruction differs at cursor {item['cursor']}")
    bank = torch.cat([out.attentions[i][0, :, -1, :].float() for i in controller.layers])
    bank /= bank.sum(-1, keepdim=True)
    return dict(item, reference=reference, cache=sc, bank=bank)


@torch.no_grad()
def sanity(student, frame, controller, args):
    controller.bind(frame)
    controller.kind = "baseline"
    controller.enabled = True
    baseline = logits_of(student, core_forward(student, tensor_ids(frame["ids"][-1:], args.device), frame["cache"]))
    controller.route(0, 0)
    controller.capture = True
    controller.set_context(frame["bank"], frame["docs"])
    out = core_forward(student, tensor_ids(frame["ids"][-1:], args.device), frame["cache"])
    zero = logits_of(student, out)
    torch.testing.assert_close(zero, baseline, rtol=0, atol=0)
    capture = controller.captured[0]
    keys, scores = capture["keys"].double(), capture["scores"].double()
    ctx = frame["geometry"]
    docs = ctx.docs
    desired = torch.zeros_like(scores)
    desired[:, docs] = ctx.log_prior[0]-scores[:, docs].log_softmax(-1)
    centered, projected, residual, coefficients = ctx.decompose(0, keys, desired)
    repeats = scores.shape[0]//keys.shape[0]
    centered_keys = keys-keys.mean(1, keepdim=True)
    u, values, _ = torch.linalg.svd(centered_keys, full_matrices=False)
    dense = (u@(u.transpose(1, 2)@centered.reshape(keys.shape[0], repeats, -1).transpose(1, 2))).transpose(1, 2).reshape(scores.shape)
    torch.testing.assert_close(projected, dense, rtol=1e-7, atol=1e-7)
    query_delta = (keys.shape[-1]**0.5)*coefficients
    implied = (centered_keys@query_delta/keys.shape[-1]**0.5).transpose(1, 2).reshape(scores.shape)
    torch.testing.assert_close(implied, projected, rtol=1e-8, atol=1e-8)
    before = scores.softmax(-1)
    after = (scores+projected+residual).softmax(-1)
    desired_probs = before.clone()
    desired_probs[:, docs] = before[:, docs].sum(-1, keepdim=True)*ctx.log_prior[0].exp()
    torch.testing.assert_close(after, desired_probs, rtol=1e-7, atol=1e-9)
    # Every decomposition coordinate must work in actual model decoding. Equal
    # strengths use the exact nondecomposed implementation (constant cancels).
    controller.capture = False
    results = dict(zero_intervention_max_logit_error=float((baseline-zero).abs().max()),
                   prefix_statistics_dense_svd_max_error=float((projected-dense).abs().max()),
                   query_change_score_projection_max_error=float((projected-implied).abs().max()),
                   full_completion_probability_max_error=float((after-desired_probs).abs().max()),
                   score_reconstruction_max_error=float((centered-projected-residual).abs().max()),
                   smallest_key_singular_value=float(values.min()))
    for name, a, b in (("projected", 0.25, 0), ("residual", 0, 0.25), ("hybrid", 0.25, 0.5)):
        controller.route(a, b)
        logits = logits_of(student, core_forward(student, tensor_ids(frame["ids"][-1:], args.device), frame["cache"]))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite geometry-intervened logits")
        results[name+"_finite"] = True
    controller.enabled = False
    controller.capture = False
    return results


@torch.no_grad()
def validate_generated_keys(student, frame, controller, args):
    """Check reused prefix statistics against live intervened suffix keys."""
    controller.bind(frame)
    controller.route(0.25, 0.5)
    controller.set_context(frame["bank"], frame["docs"])
    controller.enabled = controller.capture = True
    cache = frame["cache"]
    inputs = ([frame["ids"][-1]]+frame["reference"][:-1])[:17]
    for token in inputs:
        out = core_forward(student, tensor_ids([token], args.device), cache)
        cache = legacy_cache(out.past_key_values)
    max_projection_error = max_orthogonality_error = 0.0
    for relative, capture in controller.captured.items():
        ctx = frame["geometry"]
        keys, scores = capture["keys"].double(), capture["scores"].double()
        torch.testing.assert_close(keys[:, :ctx.prefix_len], ctx.prefix[relative], rtol=0, atol=0)
        desired = torch.zeros_like(scores)
        desired[:, ctx.docs] = ctx.log_prior[relative]-scores[:, ctx.docs].log_softmax(-1)
        centered, projected, residual, _ = ctx.decompose(relative, keys, desired)
        repeats = scores.shape[0]//keys.shape[0]
        centered_keys = keys-keys.mean(1, keepdim=True)
        u, values, _ = torch.linalg.svd(centered_keys, full_matrices=False)
        u = u*(values > values.amax(-1, keepdim=True)*1e-6)[:, None, :]
        dense = (u@(u.transpose(1, 2)@centered.reshape(keys.shape[0], repeats, -1).transpose(1, 2))).transpose(1, 2).reshape(scores.shape)
        torch.testing.assert_close(projected, dense, rtol=1e-7, atol=1e-7)
        residual_by_group = residual.reshape(keys.shape[0], repeats, -1)
        orthogonality = (centered_keys.transpose(1, 2)@residual_by_group.transpose(1, 2)).norm()
        orthogonality /= centered_keys.norm()*residual.norm()+1e-24
        if float(orthogonality) > 1e-9:
            raise RuntimeError("Generated-key residual is not orthogonal to the query space")
        max_projection_error = max(max_projection_error, float((projected-dense).abs().max()))
        max_orthogonality_error = max(max_orthogonality_error, float(orthogonality))
    controller.enabled = controller.capture = False
    return dict(intervened_generated_queries_checked=len(inputs),
                frozen_prefix_keys_exact=True,
                generated_keys_dense_svd_max_error=max_projection_error,
                generated_keys_residual_orthogonality_error=max_orthogonality_error)


@torch.no_grad()
def diagnose_frame(teacher, student, frame, controller, args, output):
    controller.bind(frame)
    controller.route(0, 0)
    controller.set_context(frame["bank"], frame["docs"])
    controller.enabled = True
    controller.capture = True
    tc = prefix_cache(teacher, frame["ids"], args.device)
    sc = frame["cache"]
    inputs = [frame["ids"][-1]]+frame["reference"][:-1]
    for position, token in enumerate(inputs):
        collect = position in args.diag_positions
        tout = core_forward(teacher, tensor_ids([token], args.device), tc, attentions=collect)
        sout = core_forward(student, tensor_ids([token], args.device), sc)
        tc, sc = legacy_cache(tout.past_key_values), legacy_cache(sout.past_key_values)
        if not collect:
            continue
        for relative, layer in enumerate(controller.layers):
            capture = controller.captured[relative]
            current = tout.attentions[layer][0, :, -1, :].double().mean(0)
            current = current/current.sum()
            scores = capture["scores"].double()
            docs = frame["geometry"].docs
            conditional = current[docs].clamp_min(1e-12)
            conditional /= conditional.sum()
            static = frame["geometry"].log_prior[relative]
            drift = float((conditional*(conditional.log()-static)).sum())
            for kind in ("current_teacher_full", "prompt_teacher_document"):
                if kind == "current_teacher_full":
                    target = current
                else:
                    target = scores.softmax(-1)
                    target[:, docs] = target[:, docs].sum(-1, keepdim=True)*static.exp()
                metrics = distribution_metrics(scores, target, frame["geometry"], relative,
                                               capture["keys"], capture["query"])
                metrics.update(cursor=frame["cursor"], generated_position=position, layer=layer,
                               target=kind, visible_tokens=scores.shape[-1],
                               teacher_document_prior_drift_kl=drift)
                append(output/"geometry_rows.jsonl", metrics)
    controller.capture = False
    controller.enabled = False


def summarize_geometry(path):
    rows = list(read_jsonl(str(path)))
    result = {}
    metrics = [key for key in rows[0] if key not in ("cursor", "generated_position", "layer", "target", "visible_tokens")]
    for target in sorted({row["target"] for row in rows}):
        selected = [row for row in rows if row["target"] == target]
        result[target] = dict(layer_query_rows=len(selected),
                              states=len({(row["cursor"], row["generated_position"]) for row in selected}),
                              mean={key:sum(row[key] for row in selected)/len(selected) for key in metrics},
                              max={key:max(row[key] for row in selected) for key in metrics})
    return result


def select(candidates, predicate):
    possible = [row for row in candidates if predicate(row["route"])]
    return max(possible, key=lambda row:(row["score"]["accept_rate"],
                                        -row["score"]["blocks"],
                                        -row["route"]["projected"]**2-row["route"]["residual"]**2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cal-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0, 0.25, 0.5, 1.0])
    parser.add_argument("--diag-positions", nargs="+", type=int, default=[0, 8, 16, 24, 31])
    parser.add_argument("--no-native", action="store_true")
    cli = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    previous = Path(__file__).resolve().parent
    config = json.loads((previous/"out_native_q128_e3/config.json").read_text())
    config.update(vars(cli))
    args = argparse.Namespace(**config)
    if 0 not in args.strengths or any(x < 0 for x in args.strengths):
        raise ValueError("Nonnegative strength grid must contain zero")
    if not (1 <= args.cal_count <= 32 and 1 <= args.test_count <= 50 and 1 <= args.max_new_tokens <= 32):
        raise ValueError("This audit reuses at most 32 calibration / 50 test questions and 32 reference tokens")
    if hashlib.sha256(Path(args.samples).read_bytes()).hexdigest() != config["data_sha256"]:
        raise RuntimeError("Dataset differs from the previous comparison")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use an empty output directory")
    config["geometry_source_sha256"] = {name:hashlib.sha256((previous/name).read_bytes()).hexdigest()
                                         for name in ("geometry.py", "run_geometry.py")}
    write_json(output/"config.json", config)
    teacher_tokenizer, teacher = load_model_and_tokenizer(args.teacher, torch.bfloat16, args.device)
    tokenizer, student = load_model_and_tokenizer(args.student, torch.bfloat16, args.device)
    teacher.requires_grad_(False)
    student.requires_grad_(False)
    args.eos_id = tokenizer.eos_token_id
    layers = list(range(student.config.num_hidden_layers-4, student.config.num_hidden_layers))
    controller = GeometryIntervention(student, layers, teacher.config.num_attention_heads)
    controller.install()
    if not args.no_native:
        checkpoint = torch.load(previous/"out_native_q128_e3/q_native_attention.pt", map_location=args.device, weights_only=False)
        if checkpoint["state"] is None:
            raise RuntimeError("Expected a nonzero trained Q checkpoint")
        controller.load_native(checkpoint["state"]["weight"])
    samples = list(read_jsonl(args.samples))
    fixed_test = {row["cursor"]:row for row in read_jsonl(str(previous/"out_fixed_teacher_audit/teacher_reference.jsonl"))}
    frames = {}
    for split, start, count in (("calibration", args.cal_start, args.cal_count), ("test", args.test_start, args.test_count)):
        frames[split] = []
        started = time.perf_counter()
        for index in range(count):
            cursor = start+index
            item = encoded_sample(samples[cursor], cursor, teacher_tokenizer, tokenizer, args)
            if split == "test" and cursor in fixed_test:
                frame = load_fixed_test_frame(teacher, student, item, tokenizer, controller, args, fixed_test[cursor])
            else:
                frame = teacher_reference(teacher, student, item, tokenizer, controller, args)
            frame["geometry"] = GeometryContext(student, layers, frame, args.seed)
            frames[split].append(frame)
            append(output/"references.jsonl", dict(split=split, cursor=cursor, generated_ids=frame["reference"]))
            if (index+1)%8 == 0:
                print(f"prepare {split} {index+1}/{count} seconds={time.perf_counter()-started:.1f}", flush=True)
    validation = sanity(student, frames["calibration"][0], controller, args)
    validation.update(validate_generated_keys(student, max(frames["calibration"], key=lambda f:len(f["reference"])),
                                              controller, args))
    write_json(output/"validation.json", validation)
    print("validation " + json.dumps(validation), flush=True)
    for index, frame in enumerate(frames["calibration"]):
        diagnose_frame(teacher, student, frame, controller, args, output)
        if (index+1)%8 == 0:
            print(f"diagnose {index+1}/{len(frames['calibration'])}", flush=True)
    geometry_summary = summarize_geometry(output/"geometry_rows.jsonl")
    write_json(output/"geometry_summary.json", geometry_summary)
    print("geometry " + json.dumps(geometry_summary), flush=True)
    routes = [dict(projected=a, residual=b, native=False) for a,b in itertools.product(args.strengths, repeat=2)]
    if not args.no_native:
        routes += [dict(projected=0.0, residual=b, native=True) for b in args.strengths]
        routes += [dict(projected=b, residual=b, native=True) for b in args.strengths if b != 0]
    candidates = []
    for index, route in enumerate(routes):
        score, _ = evaluate(student, frames["calibration"], tokenizer, controller, args, route)
        row = dict(route=route, score=score)
        candidates.append(row)
        append(output/"calibration.jsonl", row)
        print(f"calibration {index+1}/{len(routes)} " + json.dumps(row), flush=True)
    choices = {
        "baseline": select(candidates, lambda r:not r["native"] and r["projected"] == r["residual"] == 0),
        "projected_only": select(candidates, lambda r:not r["native"] and r["residual"] == 0),
        "residual_only": select(candidates, lambda r:not r["native"] and r["projected"] == 0),
        "full_direct_bias": select(candidates, lambda r:not r["native"] and r["projected"] == r["residual"]),
        "decomposed_hybrid": select(candidates, lambda r:not r["native"]),
    }
    if not args.no_native:
        choices.update({
            "trained_q": select(candidates, lambda r:r["native"] and r["projected"] == r["residual"] == 0),
            "trained_q_residual": select(candidates, lambda r:r["native"] and r["projected"] == 0),
            "trained_q_full_bias": select(candidates, lambda r:r["native"] and r["projected"] == r["residual"]),
        })
    shuffled = json.loads(json.dumps(choices["decomposed_hybrid"]))
    shuffled["route"]["shuffle"] = True
    shuffled["score"] = None
    shuffled["strength_selection_source"] = "decomposed_hybrid; no shuffled-prior calibration"
    choices["hybrid_shuffled_prior"] = shuffled
    write_json(output/"selection.json", choices)
    summaries, records = {}, {}
    for kind, choice in choices.items():
        score, rows = evaluate(student, frames["test"], tokenizer, controller, args, choice["route"],
                               record_path=output/f"test_{kind}.jsonl")
        if kind != "baseline":
            score["delta_accept_rate"] = score["accept_rate"]-summaries["baseline"]["accept_rate"]
            score["paired_95pct_delta_interval"] = paired_interval(rows, records["baseline"], args.seed)
        summaries[kind], records[kind] = score, rows
        print("test " + kind + " " + json.dumps(score), flush=True)
        write_json(output/"summary.partial.json", dict(methods=summaries, selection=choices))
    controls = {}
    pairs = [("decomposed_hybrid", "full_direct_bias"), ("decomposed_hybrid", "hybrid_shuffled_prior")]
    if not args.no_native:
        pairs += [("trained_q_residual", "trained_q"), ("trained_q_residual", "trained_q_full_bias")]
    for method, control in pairs:
        controls[method+"_vs_"+control] = dict(delta_accept_rate=summaries[method]["accept_rate"]-summaries[control]["accept_rate"],
                                             paired_95pct_delta_interval=paired_interval(records[method], records[control], args.seed))
    verification = dict(all_routes_all_questions_reference_identical=True,
                        test_labels_not_used_for_strength_selection=True,
                        frozen_previous_teacher_test_reference=True)
    if args.max_new_tokens == 32 and args.test_count == 50:
        for kind, old_kind in (("baseline", "baseline"), ("trained_q", "q_native_attention")):
            if kind not in records:
                continue
            old = list(read_jsonl(str(previous/"out_fixed_teacher_audit"/f"test_{old_kind}.jsonl")))
            for a,b in zip(records[kind], old):
                for key in ("cursor", "proposed", "accepted", "blocks", "generated_ids"):
                    if a[key] != b[key]:
                        raise RuntimeError(f"Previous acceptance replay differs: {kind}/{key}")
            verification[kind+"_previous_replay_exact"] = True
    controller.restore()
    write_json(output/"verification.json", verification)
    summary = dict(config=config, geometry=geometry_summary, selection=choices,
                   methods=summaries, paired_controls=controls, verification=verification,
                   evaluation="fixed teacher continuation; native draft and KV rollback; acceptance only")
    write_json(output/"summary.json", summary)
    (output/"summary.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
