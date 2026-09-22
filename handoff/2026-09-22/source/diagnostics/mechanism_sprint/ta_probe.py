"""Connectivity probe for the Track-A gradient: where does the chain to M break?"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import build_backend, load_config, load_harness, load_modules, maps_for  # noqa: E402

DETACH_LINE = "state.capture_rows[layer_index] = last_probs[0, :, 0, :].detach().float()"
GRAPH_LINE = "state.capture_rows[layer_index] = last_probs[0, :, 0, :]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--validation-prior-cache", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dr-rows", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    H = load_harness(args.harness_root)
    backend_mod, runner_mod, steering = load_modules(H, args.source_dir, args.source_lock)
    src = H.STEERING_SOURCE.replace(DETACH_LINE, GRAPH_LINE)
    diag = H.build_steering_context(src, {"contextlib": __import__("contextlib"),
                                          "math": __import__("math"),
                                          "types": __import__("types"), "torch": torch,
                                          "F": F, "_STEERING_STATE_CLS": H.SteeringState})
    backend = build_backend(backend_mod, args, steering)
    layers = sorted(backend.student_layers)
    backbone = backend.student.model
    deep = backbone.layers[len(backbone.layers) - len(layers):]
    vocab = int(backend.student.config.vocab_size)

    rows = [json.loads(l) for l in Path(args.dr_rows).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = [r for r in rows if "__result_header__" not in r and r["group"] == "S!=T"]
    rows.sort(key=lambda r: (int(r["source_index"]), r["position"]))
    dr = rows[0]
    src_rows = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    by_id = {r["canonical_source_id"]: r for r in src_rows}
    row = by_id[dr["canonical_source_id"]]
    target = backend.prompt_target(row["prompt"], row["answer"])
    base_tokens = list(map(int, dr["base_tokens"]))
    prefix = list(map(int, target.input_ids)) + base_tokens[:int(dr["position"])]
    y_T, y_S = int(dr["y_target"]), int(dr["y_base"])
    ids = torch.tensor([prefix], device=backend.device)
    amask = torch.ones_like(ids)

    cap = {}

    def pre_hook(module, hargs, hkwargs):
        cap["h"] = hargs[0].detach().clone()
        for k in ("attention_mask", "position_ids", "cache_position", "position_embeddings"):
            cap[k] = hkwargs.get(k)
        return None

    handle = deep[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    with torch.no_grad():
        oA = backend.student(input_ids=ids, attention_mask=amask, use_cache=False,
                             output_attentions=False)
        z_full = oA.logits[0, -1, :vocab].float().cpu()
    handle.remove()
    del oA

    h_leaf = cap["h"].requires_grad_(True)
    outs = []
    with torch.enable_grad():
        with diag(backend.student, {}, None, backend.eps, selected_layers=set(layers),
                  phase="PROBE", capture_rows={}) as st:
            h = h_leaf
            for layer in deep:
                h = layer(h, attention_mask=cap["attention_mask"],
                          position_ids=cap["position_ids"], past_key_values=None,
                          use_cache=False, cache_position=cap["cache_position"],
                          position_embeddings=cap["position_embeddings"])
                outs.append(h)
            hh = backbone.norm(h)
            logits = backend.student.lm_head(hh)[0, -1, :vocab].float()
        M = logits[y_T] - logits[y_S]
        p_rows = [st.capture_rows[L] for L in layers]
        local = {}
        for i, L in enumerate(layers):
            try:
                gl = torch.autograd.grad(outs[i].abs().sum(), [p_rows[i]], allow_unused=True,
                                         retain_graph=True)[0]
                local[f"L{L}_local_grad_abs_sum"] = None if gl is None else float(gl.abs().sum())
            except Exception as exc:
                local[f"L{L}_local_grad_abs_sum"] = f"{type(exc).__name__}: {exc}"
        targets = {"h_leaf": h_leaf, "after_layer24": outs[0], "after_layer25": outs[1],
                   "after_layer26": outs[2], "after_layer27": outs[3], "after_norm": hh,
                   "logits": logits}
        for i, L in enumerate(layers):
            targets[f"p_row_L{L}"] = p_rows[i]
        out = {}
        for name, t in targets.items():
            try:
                g = torch.autograd.grad(M, [t], allow_unused=True, retain_graph=True)[0]
                out[name] = {"requires_grad": bool(t.requires_grad),
                             "grad_fn": None if t.grad_fn is None else type(t.grad_fn).__name__,
                             "grad_is_none": g is None,
                             "grad_abs_sum": None if g is None else float(g.abs().sum()),
                             "shape": list(t.shape)}
            except Exception as exc:
                out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps({"src": dr["source_index"], "pos": dr["position"],
                      "prefix_len": len(prefix), "M": float(M.detach()),
                      "recon_delta": float((logits.detach().cpu() - z_full).abs().max()),
                      "patched_forward_call_log": st.call_log,
                      "capture_keys": sorted(st.capture_rows.keys()),
                      "mix_log_len": len(st.mix_log),
                      "local_connectivity": local,
                      "connectivity": out}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
