"""READ-ONLY device/autograd inspection for
RESCUE-TEACHER-DECISION-ATTRIBUTION-IMPLEMENTATION-REVIEW-001.

This computes NO attribution and runs NO backward.  It only:
  * loads the frozen 7B exactly as real_k4_runner.build_target does,
  * records the dispatch map, dtypes, parameter grad flags and hook classes,
  * runs ONE 8-token forward to record live tensor devices,
  * reproduces the exact failing expression as a controlled minimal reproduction,
  * forward-only checks whether a captured last-query attention row would retain a graph.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k4-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--validation-prior-cache", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target-max-memory", default="8GiB")
    ap.add_argument("--target-dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(args.k4_dir).resolve()))
    sys.path.insert(0, str(args.harness_root))
    sys.path.insert(0, str(args.source_lock))
    k4 = __import__("real_k4_runner")
    report: dict = {"mode": "READ_ONLY_INSPECTION", "no_backward_run": True}

    model, device, dmap = k4.build_target(args)
    report["build_target_call"] = {
        "source": "real_k4_runner.build_target (frozen, unmodified)",
        "dtype": args.target_dtype, "attn_implementation": "eager", "device_map": "balanced",
        "max_memory_per_gpu": args.target_max_memory, "low_cpu_mem_usage": True,
        "returns_device": str(device),
        "freezes_parameters": "for p in model.parameters(): p.requires_grad_(False)",
    }
    report["device_map"] = dmap
    report["hf_device_map_first_module"] = next(iter(dmap.items())) if dmap else None
    report["param_requires_grad_true_count"] = sum(1 for p in model.parameters()
                                                   if p.requires_grad)
    report["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    report["model_training_flag"] = bool(model.training)
    report["torch_is_grad_enabled"] = bool(torch.is_grad_enabled())

    hooks = {}
    for idx in (0, 24, 27):
        att = model.model.layers[idx].self_attn
        h = getattr(att, "_hf_hook", None)
        hooks[str(idx)] = {
            "module_device": str(next(att.parameters()).device),
            "hook_class": None if h is None else type(h).__name__,
            "hook_has_weights_map": bool(h is not None and hasattr(h, "weights_map")),
            "hook_offload_flag": bool(getattr(h, "offload", False)) if h is not None else None,
            "hook_execution_device": str(getattr(h, "execution_device", None)) if h else None,
        }
    report["hooks"] = hooks
    report["any_offload_hook"] = any(v["hook_offload_flag"] for v in hooks.values())

    live: dict = {}
    att24 = model.model.layers[24].self_attn

    def probe_hook(module, inputs, kwargs, output):
        hs = kwargs.get("hidden_states")
        live["hidden_states_device"] = str(hs.device if hs is not None else inputs[0].device)
        live["attention_mask_device"] = str(kwargs["attention_mask"].device) \
            if kwargs.get("attention_mask") is not None else None
        live["attention_mask_shape"] = list(kwargs["attention_mask"].shape) \
            if kwargs.get("attention_mask") is not None else None
        live["position_ids_device"] = str(kwargs["position_ids"].device) \
            if kwargs.get("position_ids") is not None else None
        pe = kwargs.get("position_embeddings")
        live["position_embeddings_devices"] = ([str(t.device) for t in pe]
                                               if isinstance(pe, (tuple, list)) else None)
        live["past_key_values"] = kwargs.get("past_key_values") is not None
        live["attn_output_device"] = str(output[0].device)
        live["attn_weights_is_none"] = output[1] is None
        live["module_device"] = str(next(module.parameters()).device)
        return None

    handle = att24.register_forward_hook(probe_hook, with_kwargs=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device)
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        z = o.logits[0, -1, :].float()
        live["logits_device"] = str(z.device)
        live["logits_dtype_after_float"] = str(z.dtype)
        live["lm_head_device"] = str(next(model.lm_head.parameters()).device)
        live["embed_tokens_device"] = str(next(model.model.embed_tokens.parameters()).device)
    handle.remove()
    report["live_forward_8_tokens"] = live
    del o, z
    torch.cuda.empty_cache()

    repro: dict = {"expression": "z.clone().index_fill(0, torch.tensor([y]), float('-inf')) "
                                 "where z is the CUDA logits slice and y is a Python int"}
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        z = o.logits[0, -1, :].float()
        try:
            z.clone().index_fill(0, torch.tensor([5]), float("-inf")).max()
            repro["raised"] = False
        except RuntimeError as exc:
            repro["raised"] = True
            repro["exception_type"] = type(exc).__name__
            repro["exception_message"] = str(exc).split("\n")[0]
        try:
            z.clone().index_fill(0, torch.tensor([5], device=z.device), float("-inf")).max()
            repro["fixed_variant_ok"] = True
        except RuntimeError as exc:
            repro["fixed_variant_ok"] = False
            repro["fixed_variant_error"] = str(exc).split("\n")[0]
    del o, z
    torch.cuda.empty_cache()
    report["minimal_reproduction"] = repro

    src = None
    sys.path.insert(0, str(args.harness_root))
    from rescue_harness import harness as H
    src = H.STEERING_SOURCE.replace(
        "state.capture_rows[layer_index] = last_probs[0, :, 0, :].detach().float()",
        "state.capture_rows[layer_index] = last_probs")
    cm = H.build_steering_context(src, {"contextlib": __import__("contextlib"),
                                        "math": __import__("math"), "types": types,
                                        "torch": torch, "F": F,
                                        "_STEERING_STATE_CLS": H.SteeringState})
    graph: dict = {}
    emb = model.model.embed_tokens(ids).detach().requires_grad_(True)
    with torch.enable_grad():
        with cm(model, {}, None, 1e-8, selected_layers={24}, phase="REVIEW_PROBE",
                capture_rows={}) as st:
            o = model(inputs_embeds=emb, attention_mask=torch.ones_like(ids), use_cache=False)
            row = st.capture_rows[24]
            graph["captured_row_shape"] = list(row.shape)
            graph["captured_row_requires_grad"] = bool(row.requires_grad)
            graph["captured_row_grad_fn"] = (type(row.grad_fn).__name__
                                             if row.grad_fn is not None else None)
            graph["logits_requires_grad"] = bool(o.logits.requires_grad)
            graph["logits_grad_fn"] = (type(o.logits.grad_fn).__name__
                                       if o.logits.grad_fn is not None else None)
            graph["logits_device"] = str(o.logits.device)
            graph["input_embeds_device"] = str(emb.device)
            graph["selected_layers_captured"] = sorted(st.capture_rows.keys())
    del o, emb, row
    torch.cuda.empty_cache()
    report["forward_only_graph_check"] = graph
    report["nvidia_smi"] = None
    try:
        import subprocess
        report["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
    except Exception as exc:
        report["nvidia_smi"] = f"unavailable: {exc}"

    out.joinpath("review_device_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("param_requires_grad_true_count", "model_training_flag",
                       "torch_is_grad_enabled", "any_offload_hook",
                       "live_forward_8_tokens", "minimal_reproduction",
                       "forward_only_graph_check", "nvidia_smi")},
                     ensure_ascii=False, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
