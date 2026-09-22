#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write `e_artifacts.local.json` — handoff-local wiring for the FIX-E CPU regression.

The frozen `e_artifacts.json` records execution-host absolute paths.  This script
rebuilds the same structure from THIS package plus your environment, so the FIX-E
tests (`test_scenario_e_all_pass`, `test_bool_and_placeholder_claims_cannot_pass`)
can actually run.

Everything that is bundled is resolved relative to this file.  Only the two locked
model snapshots must come from your environment.

    usage:
        cd <package>/harness
        TEACHER_MODEL=<path to locked Qwen2-7B> \\
        STUDENT_MODEL=<path to locked Qwen2-1.5B> \\
        python results/make_e_artifacts_local.py

        RESCUE_E_ARTIFACTS=results/e_artifacts.local.json \\
        PYTHONPATH=. python -m unittest discover -s rescue_harness -t . -p test_harness.py

Optional overrides: CHECKPOINT (default <package>/external/checkpoint_080.pt),
OUT (default <package>/harness/results/e_artifacts.local.json).

Expected result with the bundled checkpoint and both models present:
    Ran 44 tests ... OK        (skipped=0)
which reproduces harness/results/cpu_regression_v3.log.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # <package>/harness/results
PKG = HERE.parent.parent                        # <package>
HARNESS = PKG / "harness"


def need(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(
            "error: %s is not set.\n"
            "       set TEACHER_MODEL and STUDENT_MODEL to the locked Qwen2 "
            "snapshot directories\n"
            "       (see ENVIRONMENT.md section 3 and EXTERNAL_ARTIFACTS.md section 3)." % name)
    p = Path(v).expanduser()
    if not p.is_dir():
        sys.exit("error: %s=%s is not a directory" % (name, p))
    if not (p / "config.json").is_file():
        sys.exit("error: %s=%s has no config.json -- is that a HF model snapshot?" % (name, p))
    return str(p.resolve())


TRUTHFUL_CONFIG = {
    "adapter_config": {
        "eps": 1e-08,
        "max_alpha": 0.35,
        "num_student_heads": 12,
        "num_student_layers": 4,
        "num_teacher_heads": 28,
        "num_teacher_layers": 4,
    },
    "dtype": "bfloat16",
    "initialization": "from_scratch_no_warm_start",
    "layer_scope": "last_4_layers",
    "learning_rate": 0.003,
    "method_variant": "attention_prior_adapter+entropy_conditioned_alpha"
                      "+per_prefix_sequence_supervision",
    "optimizer": "AdamW",
    "regularization": {"beta_kl": 0.0, "delta_sharpness": 0.0, "gamma_alpha": 0.0},
    "seed": 20260830,
    "supervised_target_mode": "per_prefix_full_answer_CE",
    "train_split_identity": {
        "name": "train.csv",
        "row_count": 128,
        "sha256": "785adf5a76a6acdf2174bf9f4e4bf4abd1c7303610cb81e7191f1218534eb508",
    },
    "training_steps": 128,
    "validation_split_identity": {
        "name": "validation.csv",
        "row_count": 128,
        "sha256": "d4686c26f50cbaaf3788eef79a3a12af49a5224671e1d3b5a02195f299627ae7",
    },
}


def main() -> int:
    ckpt = os.environ.get("CHECKPOINT") or str(
        (PKG / "external" / "checkpoint_080.pt").resolve())
    if not Path(ckpt).is_file():
        sys.exit("error: checkpoint not found at %s\n"
                 "       (it ships with the package at external/checkpoint_080.pt)"
                 % ckpt)

    art = {
        "_generated_by": "harness/results/make_e_artifacts_local.py",
        "_note": ("handoff-local FIX-E wiring; every path is absolute and resolved from this "
                  "package, except the two locked model snapshots which come from the "
                  "environment."),
        "source_lock": str((PKG / "runtime" / "source_lock").resolve()),
        "source_dirs": [
            str((PKG / "runtime" / "source").resolve()),
            str((PKG / "runtime" / "prefix_fix").resolve()),
        ],
        "checkpoint": str(Path(ckpt).resolve()),
        "train_ledger": str((PKG / "data" / "train.csv").resolve()),
        "validation_ledger": str((PKG / "data" / "validation.csv").resolve()),
        "teacher_model": need("TEACHER_MODEL"),
        "student_model": need("STUDENT_MODEL"),
        "envelope": str((HERE / "run_provenance_080.json").resolve()),
        "envelope_tampered": str((HERE / "_e2e_tampered_envelope.json").resolve()),
        "truthful_config": TRUTHFUL_CONFIG,
    }

    out = Path(os.environ.get("OUT") or (HERE / "e_artifacts.local.json"))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(art, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("wrote", out)
    for k in ("source_lock", "checkpoint", "teacher_model", "student_model", "envelope"):
        print("  %-16s %s" % (k, art[k]))
    print()
    print("now run:")
    print("  cd %s" % HARNESS)
    print("  RESCUE_E_ARTIFACTS=%s PYTHONPATH=. \\" % out.relative_to(HARNESS))
    print("    python -m unittest discover -s rescue_harness -t . -p test_harness.py")
    print("expect: Ran 44 tests ... OK   (skipped=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
