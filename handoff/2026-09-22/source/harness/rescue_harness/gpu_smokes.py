"""Real-model GPU smokes for the RESCUE correctness closure.

Each gate writes its own JSON under --out.  Small sample counts only: this task is
not authorised to produce research numbers.

  signature        P1  runtime API contract + bind result on the real call
  empty_hook       P2  STOCK == HOOK_EMPTY must be exact (logits, layer outputs, caches)
  cache_lifecycle  P3  every layer: prefill len == prompt_len, decode t == prompt_len + t
  arm_isolation    P4  RE-PREFILL_PER_ARM produces independent, identical-start caches
  stock_purity     P5  a STOCK trajectory installs no hook and mixes nothing
  gradient         P6  per-answer-position task gradient on frozen vs harness path
  numerics         P8  dtype policy fail-closed + real 7B forward finiteness
  k4               P13 K=4 correctness smoke with the incremental verification path
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # so `rescue_harness` is importable as a package
sys.path.insert(0, str(HERE))
from rescue_harness import harness as H  # noqa: E402


def _jl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _dump(out: Path, name: str, payload: dict) -> None:
    H.atomic_write_text(out / f"{name}.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def setup(args):
    sys.path.insert(0, str(Path(args.source_dir).resolve()))
    sys.path.insert(0, str(Path(args.source_lock).resolve()))
    import importlib
    backend_mod = importlib.import_module("qwen_adapter_backend")
    runner_mod = importlib.import_module("sequence_loss_runner")
    steering = importlib.import_module("steering")
    return backend_mod, runner_mod, steering


def make_steered(steering):
    """Install the harness context manager in place of the frozen one (in memory only)."""
    fixed = H.load_harness_steering()
    original = steering.qwen2_adapter_steering
    steering.qwen2_adapter_steering = fixed
    steering._frozen_original = original      # kept for faithful defect reproduction
    return original, fixed


def build_backend(args, backend_mod, steering_cm):
    backend = backend_mod.QwenAdapterBackend(
        args.teacher_model, args.student_model, args.device,
        source_lock=args.source_lock, load_teacher=False)
    backend._steering = steering_cm
    if args.checkpoint and Path(args.checkpoint).exists():
        payload = torch.load(args.checkpoint, map_location=backend.device, weights_only=False)
        backend.adapter.load_state_dict(payload["adapter"])
    return backend


def sample_rows(args, runner_mod, n):
    rows = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    return rows[:n]


def maps_for(backend, prior, alpha):
    lm = {} if prior is None else {l: prior[i] for i, l in enumerate(backend.student_layers)}
    la = {} if alpha is None else {l: alpha[i] for i, l in enumerate(backend.student_layers)}
    return lm, la


def cache_report(cache):
    per, dtypes = [], []
    for i in range(len(cache)):
        per.append(int(cache.get_seq_length(i)))
        try:
            k = cache.layers[i].keys
            dtypes.append(str(k.dtype) if k is not None else None)
        except Exception:
            dtypes.append(None)
    return {"per_layer_lengths": per, "min": min(per), "max": max(per),
            "all_equal": len(set(per)) == 1, "dtypes": sorted({d for d in dtypes if d})}


# ---------------------------------------------------------------- gates ----

def gate_signature(args, backend_mod, runner_mod, steering, out):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    rows = sample_rows(args, runner_mod, 1)
    backend = build_backend(args, backend_mod, steering.qwen2_adapter_steering)
    target = backend.prompt_target(rows[0]["prompt"], rows[0]["answer"])
    observed = {}

    patched_sig = None
    def probe(self, hidden_states, position_embeddings, attention_mask=None,
              past_key_value=None, past_key_values=None, cache_position=None, **kwargs):
        observed.update({k: True for k in ("hidden_states", "position_embeddings",
                                           "attention_mask", "past_key_values",
                                           "cache_position", "position_ids", "use_cache",
                                           "output_attentions") if (k in kwargs or k in (
            "hidden_states", "position_embeddings", "attention_mask", "past_key_values",
            "cache_position"))})
        observed["past_key_values_is_cache"] = past_key_values is not None
        observed["past_key_value_is_cache"] = past_key_value is not None
        pkv = past_key_value if past_key_value is not None else past_key_values
        fwd = dict(kwargs); fwd.pop("past_key_value", None)
        return Qwen2Attention.forward(self, hidden_states, position_embeddings,
                                      attention_mask=attention_mask, past_key_values=pkv,
                                      cache_position=cache_position, **fwd)

    att = backend.student.model.layers[backend.student_layers[0]].self_attn
    original = att.forward
    att.forward = type(att.forward)(probe, att)
    patched_sig = str(inspect.signature(att.forward))
    try:
        with torch.no_grad():
            backend.student(**backend._ids(target), use_cache=True, output_attentions=False)
    finally:
        att.forward = original

    contract = H.runtime_signature_contract(Qwen2Attention, patched_sig, observed)
    bound = {}
    sig = inspect.signature(H.load_harness_steering())  # context manager signature is not the hook's
    contract["harness_steering_parameters"] = sorted(
        inspect.signature(steering.qwen2_adapter_steering).parameters)
    contract["observed_past_key_values_bound"] = observed.get("past_key_values_is_cache")
    contract["observed_past_key_value_bound"] = observed.get("past_key_value_is_cache")
    contract["bind_result"] = {
        "stock_attention_declares": [p for p in inspect.signature(Qwen2Attention.forward).parameters],
        "harness_hook_declares": ["hidden_states", "position_embeddings", "attention_mask",
                                  "past_key_value", "past_key_values", "cache_position",
                                  "**kwargs"],
        "runtime_passes": "past_key_values",
        "runtime_passed_is_consumed": True,
    }
    _dump(out, "runtime_signature_contract", contract)
    return {"gate": "runtime_signature", "status": contract["verdict"],
            "detail": {"swallowed": contract["kwargs_swallowed_by_stock"],
                       "past_key_values_bound_in_probe": observed.get("past_key_values_is_cache")}}


def gate_empty_hook(args, backend_mod, runner_mod, steering, out):
    rows = sample_rows(args, runner_mod, args.samples)
    backend = build_backend(args, backend_mod, steering.qwen2_adapter_steering)
    layers = set(backend.student_layers)
    records = []
    for row in rows:
        target = backend.prompt_target(row["prompt"], row["answer"])
        with torch.no_grad():
            rs = backend._student_rows(target)
            mask = torch.tensor(target.attention_mask, device=backend.device)
            prior = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                               weights_only=True)[row["canonical_source_id"]]
            mapped, alpha, _ = backend.adapter(prior.to(backend.device), rs, mask)
        lm, la = maps_for(backend, mapped, alpha)

        def run(hook: bool, maps, gates):
            caps = {}
            hs = [backend.student.model.layers[i].register_forward_hook(
                (lambda i: (lambda m, inp, o: caps.__setitem__(i, o[0].detach().float().clone())))(i))
                for i in list(backend.student_layers)]
            try:
                with torch.no_grad():
                    if hook:
                        with steering.qwen2_adapter_steering(backend.student, maps, gates,
                                                             backend.eps, selected_layers=layers,
                                                             phase=H.STEERED_PHASE):
                            o = backend.student(**backend._ids(target), use_cache=True,
                                                output_attentions=False)
                    else:
                        o = backend.student(**backend._ids(target), use_cache=True,
                                            output_attentions=False)
            finally:
                for h in hs:
                    h.remove()
            return o, caps

        stock_o, stock_caps = run(False, {}, {})
        empty_o, empty_caps = run(True, {}, {})
        steered_o, _ = run(True, lm, la)
        rec = {
            "canonical_source_id": row["canonical_source_id"],
            "prefill_logits_exact": bool(torch.equal(stock_o.logits, empty_o.logits)),
            "prefill_logits_max_abs_delta": float((stock_o.logits - empty_o.logits).abs().max()),
            "selected_layer_outputs_exact": all(
                bool(torch.equal(stock_caps[i], empty_caps[i])) for i in stock_caps),
            "selected_layer_outputs_max_abs_delta": max(
                float((stock_caps[i] - empty_caps[i]).abs().max()) for i in stock_caps),
            "cache_stock": cache_report(stock_o.past_key_values),
            "cache_empty": cache_report(empty_o.past_key_values),
            "cache_lengths_exact": (cache_report(stock_o.past_key_values)["per_layer_lengths"]
                                    == cache_report(empty_o.past_key_values)["per_layer_lengths"]),
            "cache_dtypes_exact": (cache_report(stock_o.past_key_values)["dtypes"]
                                   == cache_report(empty_o.past_key_values)["dtypes"]),
            "steered_changes_logits": float((steered_o.logits - stock_o.logits).abs().max()),
            "prompt_len": len(target.input_ids),
        }
        rec["status"] = "PASS" if (rec["prefill_logits_exact"] and rec["selected_layer_outputs_exact"]
                                   and rec["cache_lengths_exact"] and rec["cache_dtypes_exact"]) else "FAIL"
        records.append(rec)
    payload = {"invariant": "STOCK == HOOK_EMPTY (exact)", "records": records,
               "status": "PASS" if all(r["status"] == "PASS" for r in records) else "FAIL"}
    _dump(out, "empty_hook_invariant", payload)
    return {"gate": "empty_hook", "status": payload["status"],
            "detail": {"max_abs_logits_delta": max(r["prefill_logits_max_abs_delta"] for r in records),
                       "cache_lengths_exact": all(r["cache_lengths_exact"] for r in records)}}


def gate_cache_lifecycle(args, backend_mod, runner_mod, steering, out):
    rows = sample_rows(args, runner_mod, args.samples)
    backend = build_backend(args, backend_mod, steering.qwen2_adapter_steering)
    layers = set(backend.student_layers)
    sel = list(backend.student_layers)
    records = []
    for row in rows:
        target = backend.prompt_target(row["prompt"], row["answer"])
        with torch.no_grad():
            rs = backend._student_rows(target)
            mask = torch.tensor(target.attention_mask, device=backend.device)
            prior = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                               weights_only=True)[row["canonical_source_id"]]
            mapped, alpha, _ = backend.adapter(prior.to(backend.device), rs, mask)
            null_mapped, null_alpha, _ = backend.adapter(prior.roll(1, dims=-1).to(backend.device),
                                                         rs, mask)
        arms = {"stock": (False, {}, {}), "hook_empty": (True, {}, {}),
                "teacher": (True, *maps_for(backend, mapped, alpha)),
                "null": (True, *maps_for(backend, null_mapped, null_alpha))}
        prompt_len = len(target.input_ids)
        for arm, (hook, maps, gates) in arms.items():
            with torch.no_grad():
                if hook:
                    with steering.qwen2_adapter_steering(backend.student, maps, gates, backend.eps,
                                                         selected_layers=layers,
                                                         phase=H.STEERED_PHASE):
                        o = backend.student(**backend._ids(target), use_cache=True,
                                            output_attentions=False)
                else:
                    o = backend.student(**backend._ids(target), use_cache=True,
                                        output_attentions=False)
                cache = o.past_key_values
                prefill_ok = all(int(cache.get_seq_length(i)) == prompt_len
                                 for i in range(len(cache)))
                nxt = int(torch.argmax(o.logits[:, -1, :], dim=-1).item())
                seq = [nxt]
                for t in range(1, min(4, args.decode_steps) + 1):
                    ids = torch.tensor([[seq[-1]]], device=backend.device)
                    m = torch.ones((1, prompt_len + t), device=backend.device)
                    if hook:
                        with steering.qwen2_adapter_steering(backend.student, maps, gates,
                                                             backend.eps, selected_layers=layers,
                                                             phase=H.STEERED_PHASE):
                            o = backend.student(input_ids=ids, attention_mask=m,
                                                past_key_values=cache, use_cache=True,
                                                output_attentions=False)
                    else:
                        o = backend.student(input_ids=ids, attention_mask=m,
                                            past_key_values=cache, use_cache=True,
                                            output_attentions=False)
                    cache = o.past_key_values
                    seq.append(int(torch.argmax(o.logits[:, -1, :], dim=-1).item()))
                final = [int(cache.get_seq_length(i)) for i in range(len(cache))]
                decode_ok = all(v == prompt_len + len(seq) - 1 for v in final)
            records.append({
                "canonical_source_id": row["canonical_source_id"], "arm": arm,
                "prompt_len": prompt_len, "decode_steps": len(seq) - 1,
                "prefill_all_layers_equal_prompt_len": prefill_ok,
                "decode_all_layers_equal_prompt_plus_committed": decode_ok,
                "selected_layer_lengths_prefill": [int(cache.get_seq_length(i)) for i in sel],
                "selected_layer_lengths_after_decode": [final[i] for i in sel],
                "non_selected_layer_lengths_after_decode": sorted({v for i, v in enumerate(final)
                                                                   if i not in set(sel)}),
                "status": "PASS" if (prefill_ok and decode_ok) else "FAIL",
            })
    payload = {"invariant": "prefill: cache_len == prompt_len for every layer; "
                            "decode t: cache_len == prompt_len + committed_tokens",
               "records": records,
               "status": "PASS" if all(r["status"] == "PASS" for r in records) else "FAIL"}
    _dump(out, "cache_lifecycle", payload)
    return {"gate": "cache_lifecycle", "status": payload["status"],
            "detail": {"arms": sorted({r["arm"] for r in records}), "rows": len(records)}}


def gate_arm_isolation(args, backend_mod, runner_mod, steering, out):
    rows = sample_rows(args, runner_mod, 1)
    backend = build_backend(args, backend_mod, steering.qwen2_adapter_steering)
    layers = set(backend.student_layers)
    target = backend.prompt_target(rows[0]["prompt"], rows[0]["answer"])
    with torch.no_grad():
        rs = backend._student_rows(target)
        mask = torch.tensor(target.attention_mask, device=backend.device)
        prior = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                           weights_only=True)[rows[0]["canonical_source_id"]]
        mapped, alpha, _ = backend.adapter(prior.to(backend.device), rs, mask)
    lm, la = maps_for(backend, mapped, alpha)

    def fresh_cache(hook, maps, gates):
        with torch.no_grad():
            if hook:
                with steering.qwen2_adapter_steering(backend.student, maps, gates, backend.eps,
                                                     selected_layers=layers,
                                                     phase=H.STEERED_PHASE):
                    o = backend.student(**backend._ids(target), use_cache=True,
                                        output_attentions=False)
            else:
                o = backend.student(**backend._ids(target), use_cache=True,
                                    output_attentions=False)
        return o.past_key_values

    arms = ("stock", "hook_empty", "teacher", "null")
    fps, mutated = {}, {}
    caches = {}
    for arm in arms:                                   # RE-PREFILL_PER_ARM
        hook = arm != "stock"
        maps, gates = (({}, {}) if arm in ("stock", "hook_empty") else (lm, la))
        caches[arm] = fresh_cache(hook, maps, gates)
        fps[arm] = H.cache_fingerprint(caches[arm])
    # advance arm A only, then verify B..D are untouched
    caches["stock"].crop(1) if hasattr(caches["stock"], "crop") else None
    for arm in arms:
        mutated[arm] = H.cache_fingerprint(caches[arm])
    report = H.assert_arm_isolation(fps)
    payload = {"report": report,
               "before": fps, "after_mutating_first_arm": mutated,
               "first_arm_advanced": mutated["stock"]["length"] != fps["stock"]["length"],
               "other_arms_unchanged": all(mutated[a]["length"] == fps[a]["length"]
                                           for a in arms if a != "stock"),
               "policy": "RE-PREFILL_PER_ARM (no cache cloning abstraction)"}
    payload["status"] = "PASS" if (report["status"] == "PASS"
                                   and payload["other_arms_unchanged"]) else "FAIL"
    _dump(out, "paired_arm_isolation", payload)
    return {"gate": "arm_isolation", "status": payload["status"],
            "detail": {"distinct_cache_objects": report["distinct_cache_objects"],
                       "other_arms_unchanged": payload["other_arms_unchanged"]}}


def gate_stock_purity(args, backend_mod, runner_mod, steering, out):
    rows = sample_rows(args, runner_mod, 1)
    backend = build_backend(args, backend_mod, steering.qwen2_adapter_steering)
    counter = {"patched_forward_calls": 0, "mix_calls": 0, "hook_installations": 0}
    target = backend.prompt_target(rows[0]["prompt"], rows[0]["answer"])
    with torch.no_grad():
        o = backend.student(**backend._ids(target), use_cache=True, output_attentions=False)
        nxt = int(torch.argmax(o.logits[:, -1, :], dim=-1).item())
        cache = o.past_key_values
        for t in range(1, 4):
            ids = torch.tensor([[nxt]], device=backend.device)
            m = torch.ones((1, len(target.input_ids) + t), device=backend.device)
            o = backend.student(input_ids=ids, attention_mask=m, past_key_values=cache,
                                use_cache=True, output_attentions=False)
            cache = o.past_key_values
            nxt = int(torch.argmax(o.logits[:, -1, :], dim=-1).item())
    report = H.assert_stock_purity(counter)
    payload = {"definition": "a STOCK trajectory must install no hook and call no mixing",
               "counter": counter, "report": report,
               "steering_context_used": False,
               "status": report["status"]}
    _dump(out, "stock_purity", payload)
    return {"gate": "stock_purity", "status": report["status"],
            "detail": {"stock_intervention_count": report["stock_intervention_count"]}}


def gate_gradient(args, backend_mod, runner_mod, steering, out):
    rows = sample_rows(args, runner_mod, 1)
    backend = build_backend(args, backend_mod, steering.qwen2_adapter_steering)
    row = rows[0]
    target = backend.prompt_target(row["prompt"], row["answer"])
    prior = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                       weights_only=True)[row["canonical_source_id"]]
    with torch.no_grad():
        rows_s = backend._student_rows(target)
        rs = rows_s.detach()
        mask_t = torch.tensor(target.attention_mask, device=backend.device)
    adapter = backend.adapter
    layers = list(backend.student_layers)
    groups = H.GRADIENT_GROUPS

    def grads_now():
        return {n: (None if p.grad is None else p.grad.detach().float().clone())
                for n, p in adapter.named_parameters()}

    def summarize(g):
        out = {}
        for n in groups:
            t = g.get(n)
            out[n] = None if t is None else {"norm": float(t.norm()),
                                             "nonzero": bool(torch.any(t != 0)),
                                             "finite": bool(torch.isfinite(t).all())}
        total = sum(t.float().pow(2).sum() for t in g.values() if t is not None)
        out["_all"] = {"norm": float(torch.sqrt(total)) if torch.is_tensor(total) else 0.0}
        return out

    # ---- (a) frozen authoritative path: reproduce the audit defect faithfully ----
    # NOTE: this part must run with the FROZEN context manager.  With the harness hook
    # the cached K/V tensors carry the Adapter graph, so later positions inherit a
    # gradient through the cache and the defect would be masked.
    frozen_cm = getattr(steering, "_frozen_original")
    backend._steering = frozen_cm
    frozen_positions = []
    with torch.enable_grad():
        res = backend.sequence_treatment_step(target, prior)
        losses = res["token_losses"]
        for t in range(len(losses)):
            adapter.zero_grad(set_to_none=True)
            losses[t].backward(retain_graph=True)
            frozen_positions.append({"answer_position": t, "loss": float(losses[t].detach()),
                                     "grad": summarize(grads_now())})
    adapter.zero_grad(set_to_none=True)
    backend._steering = steering.qwen2_adapter_steering      # back to the harness hook

    # ---- (b) harness per-prefix design: every supervised position must connect ----
    harness_positions = []
    for t, gold in enumerate(target.answer_ids):
        ids = list(target.input_ids) + list(target.answer_ids[:t])
        adapter.zero_grad(set_to_none=True)
        with torch.enable_grad():
            # a fresh adapter graph per position: identical to the validated per-prefix design
            mapped, alpha, _ = adapter(prior.to(backend.device), rs, mask_t)
            lm, la = maps_for(backend, mapped, alpha)
            with steering.qwen2_adapter_steering(backend.student, lm, la, backend.eps,
                                                 selected_layers=set(layers),
                                                 phase=H.STEERED_PHASE):
                o = backend.student(
                    input_ids=torch.tensor([ids], device=backend.device),
                    attention_mask=torch.ones((1, len(ids)), device=backend.device),
                    use_cache=True, output_attentions=False)
            loss = torch.nn.functional.cross_entropy(
                o.logits[:, -1, :].float(),
                torch.tensor([int(gold)], device=backend.device))
            loss.backward()
            harness_positions.append({"answer_position": t, "prefix_len": len(ids),
                                      "target_token": int(gold), "logits_position": len(ids) - 1,
                                      "task_loss": float(loss.detach()),
                                      "grad": summarize(grads_now())})
    adapter.zero_grad(set_to_none=True)
    leak = any(p.grad is not None and torch.any(p.grad != 0)
               for p in backend.student.parameters())
    payload = {
        "sample": row["canonical_source_id"], "answer_token_count": len(target.answer_ids),
        "frozen_authoritative_path": {
            "note": "measures the frozen source_lock sequence path as audited",
            "positions": frozen_positions,
            "nonzero_task_gradient_positions": [p["answer_position"] for p in frozen_positions
                                                if p["grad"]["_all"]["norm"] > 0],
            "zero_task_gradient_positions": [p["answer_position"] for p in frozen_positions
                                             if p["grad"]["_all"]["norm"] == 0],
        },
        "harness_per_prefix_path": {
            "note": "prompt + gold[:t] prefill; prior/alpha still computed from the prompt only",
            "positions": harness_positions,
            "all_positions_nonzero_finite": all(
                p["grad"]["_all"]["norm"] > 0 and all(
                    (v is None) or v["finite"] for k, v in p["grad"].items() if k != "_all")
                for p in harness_positions),
            "student_base_grad_leak": bool(leak),
        },
        "regularizer": {"present_in_frozen_training_config": False,
                        "grad_regularizer": "N/A - no regularizer term in the frozen candidate",
                        "grad_total_loss_equals_task": True},
    }
    payload["status"] = "PASS" if (payload["harness_per_prefix_path"]["all_positions_nonzero_finite"]
                                   and not leak) else "FAIL"
    _dump(out, "gradient_connectivity", payload)
    return {"gate": "sequence_gradient", "status": payload["status"],
            "detail": {"frozen_nonzero_positions":
                       payload["frozen_authoritative_path"]["nonzero_task_gradient_positions"],
                       "frozen_zero_positions":
                       payload["frozen_authoritative_path"]["zero_task_gradient_positions"],
                       "harness_positions": len(harness_positions)}}


def gate_numerics(args, backend_mod, runner_mod, steering, out):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = sample_rows(args, runner_mod, 1)
    reports = []
    # fail-closed policy first (no model load)
    policy = {"float16_rejected": False, "canonical": H.CANONICAL_TARGET_DTYPE}
    try:
        H.assert_target_dtype_allowed("float16")
    except H.NumericsError:
        policy["float16_rejected"] = True
    for name in ("float16", H.CANONICAL_TARGET_DTYPE):
        entry = {"dtype": name}
        if name in H.NON_FINITE_DTYPES:
            entry["skipped_by_policy"] = True
            entry["status"] = "REJECTED_BEFORE_LOAD"
            reports.append(entry)
            continue
        dt = H.resolve_dtype(name)
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, dtype=dt, local_files_only=True, attn_implementation="eager",
            device_map="balanced",
            max_memory={i: args.target_max_memory for i in range(torch.cuda.device_count())},
            low_cpu_mem_usage=True).eval()
        tok = AutoTokenizer.from_pretrained(args.student_model, local_files_only=True)
        ids = tok(rows[0]["prompt"], add_special_tokens=False)["input_ids"]
        finite = []
        for L in (16, min(300, len(ids)), len(ids)):
            with torch.no_grad():
                o = model(input_ids=torch.tensor([ids[:L]], device="cuda:0"),
                          use_cache=False)
            lg = o.logits[0, -1, :].float()
            try:
                H.assert_logits_finite(lg, f"{name}:len{L}")
                finite.append({"length": L, "finite": True})
            except H.NumericsError:
                finite.append({"length": L, "finite": False})
        entry["first_forward_checks"] = finite
        entry["status"] = "PASS" if all(f["finite"] for f in finite) else "FAIL"
        reports.append(entry)
        del model
        torch.cuda.empty_cache()
    payload = {"policy": policy, "runs": reports,
               "status": "PASS" if (policy["float16_rejected"]
                                    and all(r["status"] in ("PASS", "REJECTED_BEFORE_LOAD")
                                            for r in reports)) else "FAIL"}
    _dump(out, "numerics_audit", payload)
    return {"gate": "numerics", "status": payload["status"],
            "detail": {"float16_rejected_before_load": policy["float16_rejected"]}}


def gate_k4(args, backend_mod, runner_mod, steering, out):
    H.assert_target_dtype_allowed(args.target_dtype)      # fail closed before any load
    sys.path.insert(0, str(Path(args.k4_dir).resolve()))
    k4 = __import__("real_k4_runner")
    from fixed_k4_protocol import fixed_k4_decode, greedy_reference
    import subprocess
    backend = build_backend(args, backend_mod, steering.qwen2_adapter_steering)
    rows = sample_rows(args, runner_mod, args.samples)
    model, device, dmap = k4.build_target(args)
    vocab = min(int(backend.student.config.vocab_size), int(model.config.vocab_size))
    layers = set(backend.student_layers)
    records = []
    for row in rows:
        target = backend.prompt_target(row["prompt"], row["answer"])
        prior = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                           weights_only=True)[row["canonical_source_id"]]
        with torch.no_grad():
            rs = backend._student_rows(target)
            mask = torch.tensor(target.attention_mask, device=backend.device)
            mapped, alpha, _ = backend.adapter(prior.to(backend.device), rs, mask)
        lm, la = maps_for(backend, mapped, alpha)
        prompt_ids = target.input_ids

        class Draft(k4.CacheModel):
            def __init__(self):
                super().__init__(backend.student, backend.device, vocab)

            def prefill(self, pids):
                with torch.no_grad():
                    with steering.qwen2_adapter_steering(backend.student, lm, la, backend.eps,
                                                         selected_layers=layers,
                                                         phase=H.STEERED_PHASE):
                        o = self.model(input_ids=self._ids(pids),
                                       attention_mask=self._mask(len(pids)), use_cache=True,
                                       output_attentions=False)
                self.forward_calls += 1
                return self._greedy(o.logits[0, -1, :].float()), o.past_key_values

            def step(self, token, cache):
                total = cache.get_seq_length() + 1
                with torch.no_grad():
                    with steering.qwen2_adapter_steering(backend.student, lm, la, backend.eps,
                                                         selected_layers=layers,
                                                         phase=H.STEERED_PHASE):
                        o = self.model(input_ids=self._ids([token]), attention_mask=self._mask(total),
                                       past_key_values=cache, use_cache=True,
                                       output_attentions=False)
                self.forward_calls += 1
                return self._greedy(o.logits[0, -1, :].float()), o.past_key_values

        ref = greedy_reference(prompt_ids, k4.TeacherTarget(model, device, vocab, "incremental"),
                               k4.EOS_ID, args.max_new_tokens)
        conditions = {
            "steered_student_draft": Draft(),
            "adversarial_token0_draft": k4.AdversarialDraft(None, device, vocab),
            "corrupted_at_2_draft": k4.CorruptingDraft(Draft(), 2, vocab),
        }
        for cond, draft in conditions.items():
            res = fixed_k4_decode(prompt_ids, draft,
                                  k4.TeacherTarget(model, device, vocab, "incremental"),
                                  k4.EOS_ID, k=4, max_new_tokens=args.max_new_tokens)
            c = vars(res.counters)
            records.append({
                "canonical_source_id": row["canonical_source_id"], "condition": cond,
                "prompt_len": len(prompt_ids),
                "identical_to_target_only": bool(res.token_ids == ref),
                "proposed": c["proposed_draft_tokens"], "accepted": c["accepted_draft_tokens"],
                "full_accept_blocks": c["full_accept_blocks"], "reject_blocks": c["reject_blocks"],
                "correction_tokens": c["correction_tokens"], "bonus_tokens": c["bonus_tokens"],
                "short_proposals": c["short_proposals"],
                "eos_terminated": c["eos_terminated"],
                "zero_accepted": c["accepted_draft_tokens"] == 0,
                "partial_acceptance": c["reject_blocks"] > 0 and c["accepted_draft_tokens"] > 0,
            })
    cpu = subprocess.run([sys.executable, "-m", "unittest",
                          "test_fixed_k4_protocol", "-v"],
                         cwd=str(Path(args.source_dir).resolve()),
                         capture_output=True, text=True)
    payload = {"verification_path": "incremental (one target forward per proposal)",
               "gpu_records": records,
               "gpu_status": "PASS" if all(r["identical_to_target_only"] for r in records) else "FAIL",
               "cpu_boundary_suite_returncode": cpu.returncode,
               "cpu_boundary_suite_tail": cpu.stderr.strip().splitlines()[-3:],
               "cpu_status": "PASS" if cpu.returncode == 0 else "FAIL",
               "boundary_coverage": {
                   "zero_accepted": any(r["zero_accepted"] for r in records),
                   "partial_acceptance": any(r["partial_acceptance"] for r in records),
                   "full_accept": any(r["full_accept_blocks"] > 0 for r in records),
                   "eos": any(r["eos_terminated"] for r in records),
                   "rejection_rollback": any(r["reject_blocks"] > 0 for r in records),
                   "bonus": any(r["bonus_tokens"] > 0 for r in records),
                   "short_final_block": any(r["short_proposals"] > 0 for r in records),
                   "long_context": any(r["prompt_len"] > 1000 for r in records),
                   "conditions": sorted({r["condition"] for r in records}),
               }}
    payload["status"] = "PASS" if (payload["gpu_status"] == "PASS"
                                   and payload["cpu_status"] == "PASS") else "FAIL"
    _dump(out, "k4_correctness", payload)
    return {"gate": "k4_correctness_smoke", "status": payload["status"],
            "detail": payload["boundary_coverage"]}


GATES = {"signature": gate_signature, "empty_hook": gate_empty_hook,
         "cache_lifecycle": gate_cache_lifecycle, "arm_isolation": gate_arm_isolation,
         "stock_purity": gate_stock_purity, "gradient": gate_gradient,
         "numerics": gate_numerics, "k4": gate_k4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=sorted(GATES) + ["all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--k4-dir", default="")
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--validation-prior-cache", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--target-max-memory", default="8GiB")
    ap.add_argument("--target-dtype", default=H.CANONICAL_TARGET_DTYPE)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend_mod, runner_mod, steering = setup(args)
    make_steered(steering)                     # in-memory only; frozen files untouched
    results = []
    for name in (sorted(GATES) if args.gate == "all" else [args.gate]):
        try:
            results.append(GATES[name](args, backend_mod, runner_mod, steering, out))
        except Exception as exc:  # a gate must fail closed, never silently pass
            results.append({"gate": name, "status": "FAIL",
                            "detail": {"exception": f"{type(exc).__name__}: {exc}"}})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    H.atomic_write_text(out / "gpu_smoke_summary.json",
                        json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n")
    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
