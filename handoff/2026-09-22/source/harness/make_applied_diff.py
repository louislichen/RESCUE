"""Generate minimal_applied.diff from the frozen sources vs the harness working copies.

Run with the frozen file paths as arguments; writes a unified diff plus a manifest of
sha256 before/after for every touched file.  Nothing is written to the frozen tree.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import sys
from pathlib import Path

PAIRS = [
    ("source_lock/steering.py", "harness_freeze_001/rescue_harness/harness.py",
     "STEERING_SOURCE"),
    ("source_lock/q3_teacher_prior_worker.py",
     "harness_freeze_001/patched_sources/q3_teacher_prior_worker.py", None),
    ("prefix_fix_001/prefix_seqloss.py", "harness_freeze_001/patched_sources/prefix_seqloss_notes.md", None),
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def extract_steering_source(harness_py: Path) -> str:
    text = harness_py.read_text(encoding="utf-8")
    start = text.index("STEERING_SOURCE = '''") + len("STEERING_SOURCE = '''")
    end = text.index("'''", start)
    return text[start:end]


def main() -> int:
    root = Path(sys.argv[1])                 # harness_freeze_001
    frozen_steering = Path(sys.argv[2])      # .../source_lock/steering.py
    frozen_worker = Path(sys.argv[3])        # .../prefix_fix_001/source/q3_teacher_prior_worker.py
    out = Path(sys.argv[4])
    chunks = []
    manifest = {"applied": True, "frozen_tree_modified": False, "files": []}
    new_steering = extract_steering_source(root / "rescue_harness/harness.py")
    old = frozen_steering.read_text(encoding="utf-8")
    chunks.append("".join(difflib.unified_diff(
        old.splitlines(keepends=True), new_steering.splitlines(keepends=True),
        fromfile="a/source_lock/steering.py",
        tofile="b/harness/rescue_harness/harness.py:STEERING_SOURCE")))
    manifest["files"].append({
        "frozen": str(frozen_steering), "frozen_sha256": sha(frozen_steering),
        "frozen_modified": False,
        "replacement": "harness/rescue_harness/harness.py:STEERING_SOURCE (in-process, frozen file untouched)",
        "changes": ["past_key_values is bound and used to update the cache",
                    "idle layers delegate to the stock forward (exact transparency)",
                    "row capture preserved when capture_rows is requested",
                    "steered path refuses to run without a cache handle or with output_attentions"],
    })
    worker_new = root / "patched_sources/q3_teacher_prior_worker.py"
    if frozen_worker.exists() and worker_new.exists():
        chunks.append("".join(difflib.unified_diff(
            frozen_worker.read_text(encoding="utf-8").splitlines(keepends=True),
            worker_new.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile="a/" + frozen_worker.name,
            tofile="b/harness/patched_sources/q3_teacher_prior_worker.py")))
        manifest["files"].append({
            "frozen": str(frozen_worker),
            "frozen_sha256": sha(frozen_worker), "frozen_modified": False,
            "replacement": "harness/patched_sources/q3_teacher_prior_worker.py",
            "changes": ["teacher dtype from the canonical policy (float16 refused)",
                        "first-forward finiteness check fails closed",
                        "identity record binds student tokenizer, both weight manifests, "
                        "projection code and source-lock manifest",
                        "DONE reuse goes through the identity-aware resume check"],
        })
    manifest["files"].append({
        "frozen": "prefix_fix_001/prefix_seqloss.py", "frozen_modified": False,
        "replacement": "harness/rescue_harness/harness.py:check_gradients_before_step",
        "changes": ["grads_ok is no longer advisory: missing / non-finite / all-zero / "
                    "frozen-leak gradients raise BEFORE optimizer.step()"],
    })
    manifest["frozen_source_lock_sha256"] = {
        n: sha(frozen_steering.parent / n)
        for n in ("adapter.py", "common.py", "steering.py")
        if (frozen_steering.parent / n).exists()}
    out.write_text(
        "# RESCUE correctness closure — minimal applied changes\n"
        "# The frozen tree is NOT modified. Each entry is an in-process replacement or a\n"
        "# separate working copy; the diff below is frozen-source vs replacement.\n\n"
        + json.dumps(manifest, ensure_ascii=False, indent=2) + "\n\n"
        + "".join(chunks), encoding="utf-8")
    print(json.dumps({"out": str(out), "files": len(manifest["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
