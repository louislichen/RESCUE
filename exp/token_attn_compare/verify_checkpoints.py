#!/usr/bin/env python3
"""Replay saved selected checkpoints and check trained multi-query gradients."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from run_compare import (collect_frame, core_forward, encoded_sample, fast_window,
                         load_model_and_tokenizer, new_network, read_jsonl,
                         speculative_decode, write_json)
from methods import Intervention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    output = Path(cli.output)
    config = json.loads((output/"config.json").read_text())
    args = argparse.Namespace(**config)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    teacher_tokenizer, teacher = load_model_and_tokenizer(args.teacher, torch.bfloat16, args.device)
    tokenizer, student = load_model_and_tokenizer(args.student, torch.bfloat16, args.device)
    teacher.requires_grad_(False)
    student.requires_grad_(False)
    args.eos_id = tokenizer.eos_token_id
    layers = list(range(student.config.num_hidden_layers-4, student.config.num_hidden_layers))
    controller = Intervention(student, layers, teacher.config.num_attention_heads)
    controller.install()
    samples = list(read_jsonl(args.samples))
    selection = json.loads((output/"selection.json").read_text())["methods"]
    checks = {}
    item = encoded_sample(samples[args.test_start], args.test_start, teacher_tokenizer, tokenizer, args)
    frame = collect_frame(teacher, student, item, controller, args, "student")
    for kind in ["baseline"] + config.get("methods", ["adapter", "bias", "q_attention", "adapter_uniform", "q_output_only"]):
        network = None
        zero = False
        if kind not in ("baseline", "bias"):
            network = new_network(kind, student, controller, args)
            checkpoint = torch.load(output/f"{kind}.pt", map_location=args.device, weights_only=False)
            if checkpoint["state"] is not None:
                network.load_state_dict(checkpoint["state"])
            zero = checkpoint["selected_epoch"] == 0
        controller.configure(kind, network, selection["bias"]["strength"])
        controller.force_zero = zero
        controller.enabled = kind != "baseline"
        controller.set_context(frame["bank"], frame["docs"])
        if kind in ("adapter", "q_attention", "q_native_attention") and not zero:
            fast, _ = fast_window(student, frame, controller)
            fast_grad = torch.autograd.grad(F.cross_entropy(fast, frame["labels"]), tuple(network.parameters()))
            controller.set_context(frame["bank"], frame["docs"])
            out = core_forward(student, frame["inputs"], frame["cache"])
            full = student.lm_head(out.last_hidden_state)[0].float()
            full_grad = torch.autograd.grad(F.cross_entropy(full, frame["labels"]), tuple(network.parameters()))
            torch.testing.assert_close(fast, full, rtol=0, atol=0)
            for left, right in zip(fast_grad, full_grad):
                torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-6)
            checks[kind+"_trained_logit_max_error"] = float((fast-full).abs().max())
            checks[kind+"_trained_gradient_max_error"] = max(float((a-b).abs().max()) for a,b in zip(fast_grad, full_grad))
        replay = speculative_decode(teacher, student, item, tokenizer, controller, args)
        recorded = next(read_jsonl(str(output/f"test_{kind}.jsonl")))
        assert replay["cursor"] == recorded["cursor"]
        for key in ("generated_ids", "blocks", "proposed", "accepted"):
            assert replay[key] == recorded[key], f"Checkpoint replay mismatch: {kind}/{key}"
        checks[kind+"_checkpoint_replay"] = True
    controller.restore()
    checks["replayed_test_cursor"] = item["cursor"]
    checks["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (Path(__file__), Path(__file__).with_name("run_compare.py"),
                                           Path(__file__).with_name("methods.py"))}
    write_json(output/"checkpoint_verification.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
