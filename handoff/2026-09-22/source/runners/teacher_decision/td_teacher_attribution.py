"""TD â€” Teacher decision attribution (RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001, Phase C).

Canonical object, frozen by TEACHER_DECISION_ATTRIBUTION_IMPLEMENTATION_REVIEW:

    M_T          = z_T[y_T] - z_T[y_S]
    score[l,h,j] = d M_T / d a[l,h,j]

`a` is the LAST-QUERY-ROW attention probability at the four frozen teacher layers (24-27),
post-mask, post-softmax (fp32), pre-dropout, post-repeat_kv, pre-AV -- exactly the tensor the
frozen `steering.qwen2_teacher_row_observer` produces, kept attached to the autograd graph.

Extraction mechanism (same one the accepted Track A used on the Student): the harness V3
STEERING_SOURCE is installed on the four frozen teacher layers with an EMPTY prior map (capture
only; `_apply_mix` is never reached), the capture line is replaced in memory by
`last_probs.retain_grad()`, and the gradient is read from `last_probs.grad[0, :, 0, :]` after
`backward()`.  `autograd.grad` on the sliced view returns None in this graph (established earlier
in this project by an isolated reproduction).  Nothing on disk in the frozen tree is touched.

Autograd scope: layers 0-23 run once under no_grad through the frozen model forward (pass A) and
their captured layer-24 inputs seed pass B, which re-runs exactly the frozen modules of layers
24-27.  No parameter requires grad, so no weight gradients are ever allocated.

Teacher-space question 1 of the sprint (does the Teacher possess the decision-aware signal): the
signed score is turned into a direction by the SAME deterministic greedy mass transfer the accepted
Track A used for the Student oracle direction, at the same per-head L1 budget (both rows are
probability distributions over the same keys, so the budget is directly comparable), then applied
through the real frozen attention path and the ACTUAL dM_T of the Teacher's own decision is
measured.  Oracle-style, non-deployable.

Usage:
  td_teacher_attribution.py --harness-root H --config CFG --population POP --positions-set SET \\
      --k4-dir KD --source-dir S --source-lock SL --teacher-model TM --student-model SM \\
      --dataset-path DS --validation-ledger V --validation-prior-cache P --checkpoint CK \\
      --set-name short|long|pilot|all_population --shard-lo 0 --shard-hi 0 --shard-name t0 \\
      --out OUTDIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import (build_backend, load_config, load_harness, load_modules,  # noqa: E402
                       sha256_file)

TASK_ID = "RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001"
TEACHER_LAYER_SCOPE = "last_4_layers"

DETACH_LINE = "state.capture_rows[layer_index] = last_probs[0, :, 0, :].detach().float()"
GRAPH_LINES = ("last_probs.retain_grad()\n"
               "                state.capture_rows[layer_index] = last_probs")
MIX_CALL = "last_probs = _apply_mix(state, layer_index, last_probs, key_len)"
DELTA_CALL = "last_probs = _apply_delta(state, layer_index, last_probs, key_len)"

DELTA_HELPER = '''

def _apply_delta(state, layer_index, attn_probs, key_len):
    """Diagnostic-only: apply a FIXED per-layer delta under the same L1 budget."""
    delta = state.layer_to_prior[layer_index].to(attn_probs.device, dtype=torch.float32)
    row = attn_probs[:, :, -1, :].float()
    new = (row + delta).clamp_min(0.0)
    new = new / new.sum(dim=-1, keepdim=True).clamp_min(state.eps)
    state.mix_log.append({"layer": layer_index, "key_len": int(key_len), "diagnostic": True})
    return torch.cat((attn_probs[:, :, :-1, :], new.to(attn_probs.dtype).unsqueeze(2)), dim=2)

'''


def prefix_sha(prompt_ids, toks) -> str:
    """The frozen parent's prefix identity hash (dr_stream.prefix_sha)."""
    return hashlib.sha256(json.dumps([list(map(int, prompt_ids)), list(map(int, toks))],
                                     separators=(",", ":")).encode()).hexdigest()


def greedy_transfer(score, p, budget):
    """Deterministic greedy mass transfer, ||d||_1 = budget, ordered by `score` ascending.

    Byte-for-byte the construction of the accepted Track A oracle direction
    (`ta_controllability.oracle_delta_per_head`), with the Student controllability gradient
    replaced by the signed Teacher attribution.  Taking from the most negative and giving to the
    most positive attributed keys is the signed-input analogue of the same operator; the simplex
    and non-negativity hold by construction.
    """
    import numpy as np
    snp = np.asarray(score, dtype=np.float64)
    pnp = np.asarray(p, dtype=np.float64)
    d = np.zeros_like(pnp)
    m = float(budget) / 2.0
    if m <= 0.0:
        return torch.from_numpy(d).float()
    order = np.argsort(snp)
    cum = np.cumsum(pnp[order])
    prev = np.concatenate(([0.0], cum[:-1]))
    take = np.minimum(pnp[order], np.maximum(0.0, m - prev))
    d[order] -= take
    orderd = order[::-1]
    room = np.maximum(0.0, 1.0 - pnp[orderd] - d[orderd])
    cum2 = np.cumsum(room)
    prev2 = np.concatenate(([0.0], cum2[:-1]))
    give = np.minimum(room, np.maximum(0.0, m - prev2))
    d[orderd] += give
    return torch.from_numpy(d).float()


def select_positions(pop_rows, set_name, set_file):
    """Frozen selection: never re-selected after seeing attribution values."""
    if set_name == "all_population":
        return list(pop_rows), None
    spec = json.loads(Path(set_file).read_text(encoding="utf-8"))
    if set_name == "short":
        wanted = [spec["one_short_position"]]
    elif set_name == "long":
        wanted = [spec["one_long_position"]]
    elif set_name == "smoke":
        wanted = [spec["one_short_position"], spec["one_long_position"]]
    elif set_name == "pilot":
        wanted = list(spec["eight_position_pilot"])
    else:
        raise SystemExit(f"unknown --set-name {set_name}")
    order = {(w["sample_id"], int(w["position"])): i for i, w in enumerate(wanted)}
    keys = set(order)
    sel = [r for r in pop_rows if (r["sample_id"], int(r["position"])) in keys]
    sel.sort(key=lambda r: order[(r["sample_id"], int(r["position"]))])
    if len(sel) != len(wanted):
        raise SystemExit(f"selection incomplete: {len(sel)} of {len(wanted)}")
    for r in sel:
        w = wanted[order[(r["sample_id"], int(r["position"]))]]
        if int(w["prefix_len"]) != int(r["prefix_len"]):
            raise SystemExit(f"prefix_len mismatch for {r['sample_id']} pos {r['position']}")
        if w["prefix_sha256"] != r["prefix_sha256"]:
            raise SystemExit(f"prefix_sha256 mismatch for {r['sample_id']} pos {r['position']}")
    return sel, spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--positions-set", required=True)
    ap.add_argument("--set-name", default="all_population")
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
    ap.add_argument("--shard-lo", type=int, default=0)
    ap.add_argument("--shard-hi", type=int, default=0)
    ap.add_argument("--shard-name", default="t0")
    ap.add_argument("--gpu-total-mib", type=float, default=11264.0)
    ap.add_argument("--abort-vram-fraction", type=float, default=0.90)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    H = load_harness(args.harness_root)
    backend_mod, runner_mod, steering = load_modules(H, args.source_dir, args.source_lock)
    sys.path.insert(0, str(Path(args.k4_dir).resolve()))
    sys.path.insert(0, str(args.source_lock))
    import run_hotpot_attn_transfer as RHT  # noqa: E402
    import real_k4_runner
    from transformers import AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    from steering import qwen2_teacher_row_observer

    # instrumented steering source -- capture only; prior map stays empty
    src = H.STEERING_SOURCE
    if src.count(DETACH_LINE) != 1:
        raise RuntimeError("steering capture line missing/duplicated")
    diag_src = src.replace(DETACH_LINE, GRAQH_LINES)
    if diag_src.count(MIX_CALL) != 1:
        raise RuntimeError("steering mix call missing/duplicated")
    delta_src = diag_src.replace(MIX_CALL, DELTA_CALL) + DELTA_HELPER
    diff = {"capture": {"from": DETACH_LINE, "to": GRAPH_LINES.replace("\n", "\\n")},
            "diagnostic_delta": {"from": MIX_CALL, "to": DELTA_CALL},
            "frozen_file_modified": False, "frozen_steering_sha256": sha256_file(Path(args.source_lock) / "steering.py")}

    def build(text):
        return H.build_steering_context(text, {"contextlib": __import__("contextlib"),
                                               "math": __import__("math"),
                                               "types": __import__("types"),
                                               "torch": torch, "F": F,
                                               "_STEERING_STATE_CLS": H.SteeringState})
    steering_diag = build(diag_src)
    steering_delta = build(delta_src)

    backend = build_backend(backend_mod, args, steering)
    layers = sorted(backend.student_layers)
    eps = backend.eps
    vocab_s = int(backend.student.config.vocab_size)
    backbone = backend.student.model
    deep = backbone.layers[len(backbone.layers) - len(layers):]

    # target/Teacher on all 4 cards, the accepted bf16 policy
    t_args = argparse.Namespace(teacher_model=args.teacher_model,
                                target_dtype="bfloat16",
                                target_max_memory=args.target_max_memory)
    model, t_device, dmap = real_k4_runner.build_target(t_args)
    vocab = min(vocab_s, int(model.config.vocab_size))
    t_layers: list[int] = list(RHT.selected_layer_indices(int(model.config.num_hidden_layers),
                                                  TEATHER_LAYER_SCOPE))
    t_tok = AutoTokenizer.from_pretrained(args.teacher_model, local_files_only=True, use_fast=True)
    s_tok = backend.student_tokenizer

    # rows / priors / population
    rows_src = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    by_sid = {r["canonical_source_id"]: r for r in rows_src}
    prior_cache = torch.load(Path(args.validation_prior_cache), map_location="cpu", weights_only=True)

    pop_all = []
    for line in Path(args.population).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "__result_header__" not in r and r["group"] == "S!=T":
                pop_all.append(r)
    sel, set_spec = select_positions(pop_all, args.set_name, args.positions_set)
    lo = args.shard_lo
    hi = args.shard_hi if args.shard_hi > 0 else len(sel)
    sel = sel[lo:hi]

    scores_path = out / f"td_scores_{args.shard_name}.pt"
    body = out / f"td_teacher_rows_{args.shard_name}.jsonl"
    records = []
    raw = {}
    t_shard = time.time()

    def peak_mib():
        return {str(i): round(torch.cuda.max_memory_allocated(i) / (1024 * 1024), 1)
                for i in range(torch.cuda.device_count())}

    for n, r in enumerate(sel):
        sid, t = r["sample_id"], int(r["position"])
        row = by_sid[sid]
        p_ids = s_tok(row["prompt"], add_special_tokens=False)["input_ids"]
        prefix = p_ids + [int(x) for x in r["prefix_generated_tokens"]]
        L = len(prefix)
        sha = prefix_sha(p_ids, r["prefix_generated_tokens"])
        if L != int(r["prefix_len"]) or sha != r["prefix_sha256"]:
            raise RuntimeError(f"prefix identity drift {sid} {t}")
        y_T, y_S = int(r["y_target"]), int(r["y_base"])
        if y_T == y_S:
            raisYIS = RuntimeError("selection included an S=T position")
        t_text = s_tok.decode(prefix, skip_special_tokens=False)

        # The frozen Teacher top-1 must be recomputed and exactly match y_T before any attribution.
        with torch.inference_mode():
            enc = t_tok(t_text, return_tensors="pt", add_special_tokens=False)
            enc = {k: v.to(t_device) for k, v in enc.items()}
            z_full = model(**enc, use_cache=False, output_attentions=False).logits[0, -1, :vocab].float().detach()
            t_top1 = int(torch.argmax(z_full¤¹¥Ñ•´ ¤¤(€€€€€€€€€€€5}Q}™Õ±°€ô™±½…Ğ¡é}™Õ±±må}Qt€´é}™Õ±±må}Mt¤(€€€€€€€¥˜Ñ}Ñ½ÀÄ€„ôå}Pè(€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È¡˜‰ÍÑ½¬Q•…¡•ÈÑ½À´ÄíÑ}Ñ½ÀÅô€„ô™É½é•¸å}Píå}Qô…ĞíÍ¥‘ôíÑôˆ¤(€€€€€€€±½Á}‰…Í”€ô™±½…Ğ¡¹±½}Í½™Ñµ…à¡é}™Õ±°¹ÁÔ ¤°‘¥´ô´Ä¥må}Qt¤(€€€€€€€‘•°é}™Õ°(€€€€€€€Ñ½É ¹Õ‘„¹•µÁÑå}…¡” ¤((€€€€€€€ÁÉ”€ôÁÉ¥½É}…¡•mÍ¥‘t¹Ñ¼¡Ñ½É ¹™±½…ĞÌÈ¤€€ŒmP°!Ğ°-tÁÉ½©•Ñ•Q•…¡•ÈÁÉ½ˆÉ½İÌ(€€€€€€€€ŒUMP€ôÁÉ”¥Ì…±É•…‘äÁÉ½©•Ñ•Ñ¼MÑÕ‘•¹Ğ­•åÌ°‰ÕĞ…ÑÑÉ¥‰ÕÑ¥½¸µÕÍĞ‰”½µÁÕÑ•(€€€€€€€€Œ½¸Q•…¡•ÈÁÉ½‰Ì‰•™½É”Ñ¡”¡…¹”½˜Ñ½­•¹¥é•ÈÍÁ…”¸€A•Èµ!•…0Ä‰Õ‘•ÑÌ½µ”™É½´Ñ¡”(€€€€€€€€ŒÁÉ½©•Ñ•É½İÌ°İ¡¥ İ…Ì…±Í¼¡½ÜÑ¡”MÑÕ‘•¹Ğ=É…±”‰Õ‘•Ğİ…Ì‘•™¥¹•¸(€€€€€€€Ñ}ÁÉ½¨€ôÁÉ”(€€€€€€€P€ôÑ}ÁÉ½¨¹Í¡…Á•lÁt(€€€€€€€!Ğ€ôÑ}ÁÉ½¨¹Í¡…Á•lÅt(€€€€€€€,€ôÑ}ÁÉ½¨¹Í¡…Á•lÉt(€€€€€€€Í½É•}É½İÌ€ômt(€€€€€€€Ñ±…å•ÉÌ€ômt(€€€€€€€™É••}‰•™½É”€ôíÍÑÈ¡¤¤èÉ½Õ¹¡Ñ½É ¹Õ‘„¹µ•µ}•Ñ}¥¹™¼¡¤¥lÁt€¼€ ÄÀÈĞ€¨€ÄÀÈĞ¤°€Ä¤(€€€€€€€€€€€€€€€€€€€€€€€€™½È¤¥¸É…¹”¡Ñ½É ¹Õ‘„¹‘•Ù¥•}½Õ¹Ğ ¤¥ô(€€€€€€€™½È¤¥¸É…¹”¡Ñ½É ¹Õ‘„¹‘•Ù¥•}½Õ¹Ğ ¤¤è(€€€€€€€€€€€Ñ½É ¹Õ‘„¹É•Í•Ñ}Á•…­}µ•µ½Éå}ÍÑ…ÑÌ9¤¤((€€€€€€€€ŒA…ÍÌ€´´™É½é•¸™Õ±°Q•…¡•È™½Éİ…ÉÕ¹‘•È¹½}É…€¬•á…Ğ…ÁÑÕÉ”¥¹ÁÕÑÌÑ¼±…å•È´ÈĞ¸(€€€€€€€Ñ}„€ôÑ¥µ”¹Ñ¥µ” ¤(€€€€€€€…ÁÑÕÉ•€ôíô(€€€€€€€‘•˜…ÁÑÕÉ•}ÁÉ”¡µ½‘Õ±”°¥¹ÁÕÑÌ¤è(€€€€€€€€€€€…ÁÑÕÉ•‘l‰…ÉÌ‰t€ôÑÕÁ±”¡à¹‘•Ñ…  ¤¥˜Ñ½É ¹¥Í}Ñ•¹Í½È¡à¤•±Í”à™½Èà¥¸¥¹ÁÕÑÌ¤(€€€€€€€¡½½­}¡…¹‘±”€ôµ½‘•°¹µ½‘•°¹±…å•ÉÍmÑ±…å•ÉÍlÁut¹É•¥ÍÑ•É}™½Éİ…É‘}ÁÉ•}¡½½¬¡…ÁÑÕÉ•}ÁÉ”¤(€€€€€€€İ¥Ñ Ñ½É ¹¹½}É… ¤è(€€€€€€€€€€€•¹Œ€ôÑ}Ñ½¬¡Ñ}Ñ•áĞ°É•ÑÕÉ¹}Ñ•¹Í½ÉÌô‰ÁĞˆ°…‘‘}ÍÁ•¥…±}Ñ½­•¹Ìõ…±Í”¤(€€€€€€€€€€€•¹Œ€ôìè¤¹Ñ¼¡Ñ}‘•Ù¥”¤™½È¬°¤¥¸•¹Œ¹¥Ñ•µÌ ¥ô(€€€€€€€€€€€|€ôµ½‘•°  ¨©•¹Œ°ÕÍ•}…¡”õ…±Í”°½ÕÑÁÕÑ}…ÑÑ•¹Ñ¥½¹Ìõ…±Í”¤(€€€€€€€¡½½­}¡…¹‘±”¹É•µ½Ù” ¤(€€€€€€€Ñ}Á…ÍÍ}„€ôÑ¥µ”¹Ñ¥µ” ¤€´Ñ}„(€€€€€€€¥˜€‰…ÉÌˆ¹½Ğ¥¸…ÁÑÕÉ•è(€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰™…¥±•Ñ¼…ÁÑÕÉ”Ñ•…¡•È±…å•È´ÈĞ¥¹ÁÕÑÌˆ¤((€€€€€€€€ŒA…ÍÌ€´´•á…Ğ™É½é•¸±…å•ÉÌ€ÈĞ´ÈÜ°É…Á µ•¹…‰±•°5AQdÁÉ¥½Èµ…À¸(€€€€€€€Ñ}ˆ€ôÑ¥µ”¹Ñ¥µ” ¤(€€€€€€€•µÁÑå}ÍÑ…Ñ”€ôÍÑ••É¥¹}‘¥…œ¹}}±½‰…±Í}l‰}MQI%9}MQQ}1L‰t (€€€€€€€€€€€±…å•É}Ñ½}ÁÉ¥½Èõíô°±…å•É}Ñ½}…±Á¡„õíô°•ÁÌõ‰…­•¹¹•ÁÌ°(€€€€€€€€€€€Á¡…Í”ô‰Q}AQUI}=91dˆ°Í•±•Ñ•‘}±…å•ÉÌõÍ•Ğ¡Ñ±…å•ÉÌ¤¤(€€€€€€€•µÁÑå}ÍÑ…Ñ”¹…ÁÑÕÉ•}É½İÌ€ôíô(€€€€€€€•Ñ…ÑÑÈ¡ÍÑ••É¥¹}‘¥…œ°ÍÑ••É¥¹}‘¥…œ¹}}¹…µ•}|¤¹}}Í•Ñ}|¡µ½‘•°°•µÁÑå}ÍÑ…Ñ”¤(€€€€€€€…ÉœÁ}…ÁÑÕÉ•€ô…ÁÑÕÉ•‘l‰…ÉÌ‰t(€€€€€€€à€ô…ÉœÁ}…ÁÑÕÉ•‘lÁt¹Ñ¼¡µ¥‘‘±•}‘•Ù¥”¤€¥˜Ñ½É ¹¥Í}Ñ•¹Í½È¡…ÉœÁ}…ÁÑÕÉ•‘lÁt¤•±Í”…ÉœÁ}…ÁÑÕÉ•‘lÁt(€€€€€€€•áÑÉ…}…ÉÌ€ôl¡„¹Ñ¼¡µ¥‘‘±•}‘•Ù¥”¤€¥˜Ñ½É ¹¥Í}Ñ•¹Í½È¡„¤•±Í”„¤™½È„¥¸…ÉœÁ}…ÁÑÕÉ•‘lÄéut(€€€€€€€¡¥‘‘•¸€ôµ½‘•°¹µ½‘•°¹±…å•ÉÍmÑ±…å•ÉÍlÁut¹¥¹ÁÕÑ}±…å•É¹½É´¡à¤(€€€€€€€™½È¥‘à¥¸É…¹”¡±•¸¡±…å•ÉÌ¤¤è(€€€€€€€€€€€±…å•É}¥‘à€ôÑ±…å•ÉÍm¥‘át(€€€€€€€€€€€±…å•È€ôµ½‘•°¹µ½‘•°¹±…å•ÉÍm±…å•É}¥‘át(€€€€€€€€€€€¥˜¡¥‘‘•¸¹‘•Ù¥”€„ô¹•áĞ¡±…å•È¹Á…É…µ•Ñ•ÉÌ ¤¤¹‘•Ù¥”è(€€€€€€€€€€€€€€€¡¥‘‘•¸€ô¡¥‘‘•¸¹Ñ¼¡¹•áĞ¡±…å•È¹Á…É…µ•Ñ•ÉÌ ¤¤¹‘•Ù¥”¤(€€€€€€€€€€€±…å•É}…ÉÌ€ôm„¹Ñ¼¡¡¥‘‘•¸¹‘•Ù¥”¤¥˜Ñ½É ¹¥Í}Ñ•¹Í½È¡„¤•±Í”„™½È„¥¸•áÑÉ…}…ÉÍt(€€€€€€€€€€€¡¥‘‘•¸€ô±…å•È¡¡¥‘‘•¸°€©±…å•É}…ÉÌ¤(€€€€€€€€€€€¥˜±…å•É}¥‘à¹½Ğ¥¸•µÁÑå}ÍÑ…Ñ”¹…ÁÑÕÉ•}É½İÌè(€€€€€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È¡˜‰¹¼Q•…¡•È…ÁÑÕÉ”™½È±…å•Èí±…å•É}¥‘áôˆ¤(€€€€€€€€€€€ÁÉ½‰Ì€ô•µÁÑå}ÍÑ…Ñ”¹…ÁÑÕÉ•}É½İÍm±…å•É}¥‘át(€€€€€€€€€€€¥˜ÁÉ½‰Ì¹É•ÅÕ¥É•Í}É…¥Ì…±Í”è(€€€€€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È¡˜‰…ÁÑÕÉ•Q•…¡•ÈÉ½Ü±½ÍĞÉ……Ğ±…å•Èí±…å•É}¥‘áôˆ¤(€€€€€€€€€€€ÁÉ½‰Ì¹É•Ñ…¥¹}É… ¤(€€€€€€€¡´Ğ€ôµ½‘•°¹µ½‘•°¹¹½É´¡¡¥‘‘•¸¤(€€€€€€€±…ÍÑ}‘•Ù¥”€ô¹•áĞ¡µ½‘•°¹±µ}¡•…¹Á…É…µ•Ñ•ÉÌ ¤¤¹‘•Ù¥”(€€€€€€€¥˜¡´Ğ¹‘•Ù¥”€„ô±…ÍÑ}‘•Ù¥”è(€€€€€€€€€€€¡´Ğ€ô¡´Ğ¹Ñ¼¡±…ÍÑ}‘•Ù¥”¤(€€€€€€€é}É•½¸€ôµ½‘•°¹±µ}¡•…‘¡´Ğ¥lÀ°€´Ä°€éÙ½…‰t¹™±½…Ğ ¤(€€€€€€€Ñ}Ñ½ÀÅ}É•½¸€ô¥¹Ğ¡Ñ½É ¹…Éµ…à¡é}É•½¸¤¹¥Ñ•´ ¤¤(€€€€€€€5}P€ôé}É•½¹må}Qt€´é}É•½¹må}Mt(€€€€€€€5}Q}­••À€ô™±½…Ğ¡5}P¹‘•Ñ…  ¤¤(€€€€€€€5}P¹‰…­İ…É ¤(€€€€€€€Ğ_pass_b = time.time() - t_b
        score_rows = [empty_state.capture_rows[L].grad[0, :, 0, :].detach().torch.float32).cpu()
                     for L in tlayers]
        if not all(torch.isfinite(x).all() for x in score_rows):
            raise RuntimeError ("non-finite Teacher attribution")

        # reconstruction check vs the frozen full Teacher forward
        recon = float((z_recon.detach().cpu() - z_full).abs().max())
        recon_argmax_equal = bool(t_top1_recon == t_top1)
        del x_recon, hm4, hidden, z_recon, M_T
        torch.cuda.empty_cache()

        # observer cross-check for the same prefix (MUST be exact -- cap signature)
        t_c = time.time()
        with qwen2_teacher_row_observer(model, tlayers) as cap:
            with torch.inference_mode():
                enc = t_tok(t_text, return_tensors="pt", add_special_tokens=False)
                enc = {k: v.to(t_device) for k, v in enc.items()}
                _ = model(**enc, use_cache=False, output_attentions=False)
        t_pass_c = time.time() - t_c
        observer = {}
        for L inlayers :
            obs = cap["attn"][L].to(torch.float32)
            sc = score_rows[tlayers.index(L)]
            observer[str(L)] = {
                "max_abs_delta": float((obs - sc).abs().max()),
                "exact": bool(torch.equal(obs, sc)),
                "len": int(obs.shape[-1]),
            }

        # Teacher-space question 1 diagnostic: deterministic greedy mass transfer of signed scores.
        p_teacher = prior_cache[sid].to(torch.float32)
        per_layer = {}
        d_teacher = {}
        v_td_teacher = 0.0
        t_direction = {}
        for li, L in enumerate(tlayers):
            p = p_teacher[li]
          sc = score_rows[li]
          budget_head_mean = float(budgets[L]|raise if false else 0])
            d_list = []
            v_list = []
            for h in range(p.shape[0]):
                budget = budget_head:
                d_h = greedy_transfer(sc[h].cpu().numpy(), p[h].cpu().numpy(), budget)
                d_list.append(d_h)
                v_list.append(float((sc[h].cpu() * d_h).sum())
            d = torch.stack(d_list, 0)
            d_teacher[L] = d
            v_td_teacher += sum(v_list)
            p_new = p + d
            p_new = p_new.clamp_min(0.0)
            p_new = p_new / p_new.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            t_direction[L] = p_new
            per_layer[str(L)] = {
                "score_abs_max": float(sc.abs().max()),
                "score_nonzero_fraction": float((sc != 0).float().mean()),
                "budget_l1_per_head": [float(b) for b in budgets[L]},
                "V_TD_teacher_per_head": v_list,
                "V_TD_teacher": sum(v_list),
                "direction_min": float(p_new.min()),
                "direction_sum_dev_max": float(obs(p_new.sum(dim=-1) - 1).max()),
            }

        # apply same-budget Telecher-space direction and measure actual Teacher decision
        with torch.inference_mode():
            with qwen2_teacher_row_observer(model, tlayers) as cap_act:
                enc = t_tok(t_text, return_tensors="pt", add_special_tokens=False)
                enc = {k: v.to(t_device) for k, v in enc.items()}
                st = steering_diag.__globals_["_STEERING_STATE_CLS"](
                    layer_to_prior={L: t_direction[L].to(next(model.model.layers[L].parameters()).device) for L in tlayers},
                    layer_to_alpha={}, eps=1e-12, phase="TD_TEACHER_SPACE", selected_layers=set(layers))
                with steering_diag.__globals_["qwen2_adapter_steering"](model, st.layer_to_prior, {}, 1e-12, selected_layers=set(layers), phase="TD_TEACHER_SPACE"):
                    o_act = model(**enc, use_cache=False, output_attentions=False)
                    z_act = o_act.logits[0, -1, :vocab].float().detach()
        t_top1_act = int(torch.argmax(z_act).item())
        M_T_act = float(z_act[y_T] - z_act[y_S])
        logp_act = float(F.log_softmax(z_act.cpu(), dim=-1)[y_T])
        n_inter = len(st.mix_log)

        # construct record
        rec = {"canonical_source_id": sid, "source_index": r["source_index"],
               "position": t, "prefix_len": L, "prefix_sha256": sha,
               "y_target": y_T, "y_base": y_S, "M_T_base": M_T_full,
               "V_TD_teacher_first_order": v_td_teacher,
               "dST_T_actual": M_T_act - M_T_full,
               "dlogp_yT_actual": logp_act - logp_base,
               "top1_under_TD_teacher": t_top1_act,
               "reconstruction_max_abs_logit_delta": recon,
               "teacher_interventions": n_inter, "per_layer": per_layer,
                "observer_cross_check": observer,
               "harness_manifest_sha256": cfg["harness_manifest_sha256"],
               "run_config_hash": cfg["config_hash"], "checkpoint_sha256": cfg["checkpoint_sha256"]}
        records.append(rec)
        raw[f"{sid}|{t}"] = {"score": torch.stack(score_rows, 0), "prefix_sha256": sha,
                          "y_T": y_T, "y_S": y_S, "layers": layers}
        if (n + 1) % 8 == 0 or n == len(sel) - 1:
            torch.save(raw, scores_path)
        H.write_result_envelope(body, task_id=TASK_ID, stage="teacher_decision_attribution",
                records=records, arms=["decision_position"],
                sample_ids=sorted({r["canonical_source_id"] for r in records}),
                config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])

    # finalize
    torch.save(raw, scores_path)
    H.write_result_envelope(body, task_id=TASK_ID, stage="teacher_decision_attribution",
                        records=records, arms=["decision_position"],
                        sample_ids=sorted({r["canonical_source_id"] for r in records}),
                        config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    meta = {"shard": args.shard_name, "set": args.set_name, "positions": len(records),
            "scores_sha256": sha256_file(scores_path), "body_sha256": sha256_file(body),
            "verify": H.verify_result(body, expected_config_hash=cfg["config_hash"],
                                      expected_checkpoint_sha256=cfg["checkpoint_sha256"])}
    H.atomic_write_text(out / f"td_teacher_rows_{args.shard_name}_meta.json",
                        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"shard": args.shard_name, "positions": len(records),
                      "verify": meta["verify"]["status"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
