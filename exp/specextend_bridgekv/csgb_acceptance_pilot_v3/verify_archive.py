#!/usr/bin/env python3
"""Verify this saved CPU evidence archive, NOT rerun GPU/model/source audits.

No extraction, model loading, old absolute-path reads or analysis-module import.
NumPy is optional. Bootstrap follows the ORIGINAL Python Random(42).choices
index sequence, not NumPy's different RNG. Without NumPy, CIs are not rerun.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import statistics
import sys
import tarfile


REMOTE = "/storage/dzj/spec_repro/CSGB_episode_pilot_20261016_v3/"
POLICIES = tuple("fixed%d_native" % b for b in (8, 16, 24, 32)) + ("csgb", "constant", "history_native")
MODES = POLICIES[4:]
PINS = {"source/stage.sha256": "f2bf537ca7ee448ece75a9c09ceb4772730d49e5ffeb55f594867d561cb10ee4",
        "source/pilot_protocol.json": "06186865f89065dc54f200e9326030a09cc215baa8540814cc4c50c812de4928",
        "analysis/manifest.json": "0f94517b1dab293dab67d3a642faabf7918d3d460291a381b64e8c2dff976ca4"}
DONE_SHA = "64b403a1ba8d24bcbb41cb220b45d9e903812a725c45d01586c187a1f6de10f0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def bad(value):
        raise ValueError("Nonfinite JSON literal: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def relative(name):
    require(type(name) is str and name and "\\" not in name and "\x00" not in name,
            "Unsafe archive path")
    require(not PurePosixPath(name).is_absolute() and all(p not in ("", ".", "..") for p in name.split("/")),
            "Unsafe/noncanonical archive path")
    return name


def finite(value, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (not positive or value > 0),
            "Invalid finite measurement")
    return value


def close(value, expected):
    require(math.isclose(finite(value), finite(expected), rel_tol=1e-11, abs_tol=1e-11),
            "Saved arithmetic/statistic differs")


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid integer identity/count")
    return value


def read_tar(path):
    """Stream regular flat pilot JSON members only; retain no tensors/raw rows."""
    hashes, controls, refs, requests = {}, {}, {}, []
    seen, total = set(), 0
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = member.name.rstrip("/") if member.isdir() else member.name
            relative(name)
            require(name not in seen, "Duplicate tar member")
            seen.add(name)
            if member.isdir():
                require(name == "pilot", "Unexpected tar directory")
                continue
            require(member.isreg() and name.startswith("pilot/") and len(name.split("/")) == 2
                    and re.fullmatch(r"[A-Za-z0-9_.-]+\.json", name.split("/")[1]) is not None,
                    "Nonregular/unsafe/unregistered raw tar member")
            require(0 < member.size <= 64 * 1024 * 1024, "Excessive raw JSON member size")
            total += member.size
            require(total <= 256 * 1024 * 1024, "Excessive uncompressed raw archive")
            stream = archive.extractfile(member)
            require(stream is not None, "Unreadable tar member")
            data = stream.read(member.size + 1)
            require(len(data) == member.size, "Truncated tar member")
            base = name.split("/")[1]
            hashes[base] = sha_bytes(data)
            record = load_json(data)
            if base in ("DONE.json", "protocol.json", "calibration.json", "frozen_policies.json", "training_order.json"):
                controls[base] = record
            elif base.startswith("reference_4096_"):
                doc = int(base.removeprefix("reference_4096_").removesuffix(".json"))
                require(doc not in refs and 0 <= doc < 24, "Duplicate/out-of-scope AR reference")
                refs[doc] = record["output_token_ids"]
            else:
                requests.append(check_request(record))
    return hashes, controls, refs, requests


def check_request(record):
    d = integer(record["document"])
    require(d < 24 and type(record["context"]) is int and record["context"] == 4096,
            "Request outside original TRAIN24/4K")
    require(record["policy"] in POLICIES, "Unexpected policy")
    e = record["episode"]; result = e["result"]
    output = result["output_token_ids"]
    require(type(output) is list and 0 < len(output) <= 128
            and all(type(t) is int and t >= 0 for t in output), "Invalid returned tokens")
    raw, scored, rows = result["raw_accept_length_list"], result["accept_length_list"], e["rows"]
    require(type(raw) is list and type(scored) is list and type(rows) is list
            and len(raw) == len(scored) == len(rows) > 0, "Round/acceptance count differs")
    cursor = 0
    for i, (a, b, row) in enumerate(zip(raw, scored, rows)):
        integer(a); integer(b)
        useful = min(cursor + a + 1, len(output)) - min(cursor, len(output))
        require(useful > 0 and b == useful - 1 and row["raw_committed"] == a + 1
                and row["useful_committed"] == useful, "Raw/scored terminal clipping differs")
        cursor += a + 1
        require(i == len(rows) - 1 or cursor < len(output), "Unused post-terminal proposal")
    require(result["total_generated"] == integer(len(output), 1)
            and type(result["total_generated"]) is int
            and type(result["raw_total_generated"]) is int and result["raw_total_generated"] == cursor
            and type(result["generation_overshoot"]) is int and result["generation_overshoot"] == cursor - len(output),
            "Generation totals differ")
    eos = record["eos_token_id"]
    require((eos in output and output.index(eos) == len(output) - 1 and cursor == len(output))
            or (eos not in output and len(output) == 128), "EOS/maximum-output accounting differs")
    seconds = finite(e["seconds"], positive=True)
    tps, acceptance = len(output) / seconds, statistics.fmean(scored)
    for key, value in (("useful_n", len(output)), ("request_tps", tps), ("scored_acceptance", acceptance)):
        close(e[key], value)
    # These two display-only engine fields are rounded in the frozen engine.
    # Primary metrics above use the unrounded lists/full-request wall time.
    close(result["avg_accept_length"], round(acceptance, 3))
    close(result["tokens_per_sec"], round(len(output) / finite(result["inference_time"], positive=True), 2))
    require(e["exact"] is True and e["reference_output_token_ids"] == output,
            "Saved canonical equality differs")
    return {"key": (record["phase"], d, record["policy"], record["pass_index"], record["stream"], record["timing_repeat"]),
            "document": d, "policy": record["policy"], "phase": record["phase"],
            "stream": record["stream"], "repeat": record["timing_repeat"],
            "output": output, "tps": tps, "acceptance": acceptance,
            "input_sha256": record["input_sha256"], "seed": record["request_seed"],
            "before": record["parameters_before"], "after": record["parameters_after"]}


def verify(root):
    require(root.is_dir() and not root.is_symlink(), "Archive root missing/aliased")
    manifest = load_json((root / "ARCHIVE_MANIFEST.json").read_bytes())
    files = manifest["files"]
    require(type(files) is dict and files, "Missing archive file manifest")
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Archive contains a symlink")
        if path.is_file() and path != root / "ARCHIVE_MANIFEST.json":
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(files), "Archive manifest does not cover exact regular-file inventory")
    for name, declaration in files.items():
        relative(name)
        digest = declaration if type(declaration) is str else declaration["sha256"]
        require(sha_file(root / name) == digest, "Archive file SHA differs: " + name)
        if type(declaration) is dict and "bytes" in declaration:
            require((root / name).stat().st_size == declaration["bytes"], "Archive file size differs")
    for name, wanted in PINS.items():
        require(sha_file(root / name) == wanted, "Original experiment pin differs: " + name)
    stage = {}
    for line in (root / "source/stage.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = relative(name.strip())
        require(name not in stage, "Duplicate original source stage member")
        stage[name] = digest
    require(len(stage) == 60, "Original60-member stage is incomplete")
    excluded = manifest.get("excluded_originals")
    require(type(excluded) is dict and set(excluded) == {"source/predecessor_terminal.log"},
            "Only the explicitly registered process-log exclusion is permitted")
    for name, item in excluded.items():
        require(item["sha256"] == stage[name.removeprefix("source/")]
                and type(item["bytes"]) is int and item["bytes"] > 0
                and item["reason"] == "repository excludes process logs"
                and not (root / name).exists(), "Excluded-log metadata differs from original stage")
    included_stage = {name: digest for name, digest in stage.items() if "source/" + name not in excluded}
    for name, wanted in included_stage.items():
        require(sha_file(root / "source" / name) == wanted, "Original included source stage differs")
    require({p.relative_to(root / "source").as_posix() for p in (root / "source").rglob("*") if p.is_file()}
            == set(included_stage) | {"stage.sha256"}, "Frozen source has missing or extra files")
    source = load_json((root / "source/pilot_protocol.json").read_bytes())
    require(len(source["source_manifest"]) == 52, "Original52-source closure differs")
    for name, wanted in source["source_manifest"].items():
        require(name.startswith(REMOTE), "Unexpected source provenance prefix")
        require(stage.get(relative(name[len(REMOTE):])) == wanted, "Runtime source/stage binding differs")
    original = load_json((root / "analysis/manifest.json").read_bytes())
    for name, wanted in original["artifacts"].items():
        relative(name)
        if name == "plot_data.json":
            digest = hashlib.sha256(); size = 0
            with gzip.open(root / "analysis/plot_data.json.gz", "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(block); require(size <= 128 * 1024 * 1024, "Excessive compressed plot data")
                    digest.update(block)
            require(digest.hexdigest() == wanted, "Plot gzip changes original uncompressed bytes")
        else:
            require(sha_file(root / "analysis" / name) == wanted, "Original analysis artifact SHA differs")
    hashes, controls, refs, requests = read_tar(root / "raw_results.tar.gz")
    require(len(hashes) == 585 and hashes.get("DONE.json") == DONE_SHA and len(refs) == 24,
            "Original585JSON/24AR/DONE identity differs")
    done = controls["DONE.json"]
    bound = {}
    for name, wanted in done["raw_sha256"].items():
        require(name.startswith(REMOTE + "pilot/") and "/" not in name[len(REMOTE + "pilot/"):], "Unexpected raw provenance key")
        base = name[len(REMOTE + "pilot/"):]
        require(base not in bound and hashes.get(base) == wanted, "Original raw SHA differs")
        bound[base] = wanted
    require(len(bound) == 584 and set(bound) == set(hashes) - {"DONE.json"}, "Incomplete raw binding")
    for name in ("frozen_policies.json", "calibration.json", "protocol.json", "DONE.json"):
        require(sha_file(root / "frozen" / name) == hashes[name], "Readable frozen copy changes raw bytes")
    protocol = controls["protocol.json"]
    require(protocol["pilot_protocol"] == source and protocol["source_manifest"] == source["source_manifest"],
            "Runtime protocol differs from original source protocol")
    for name, wanted in protocol["input_manifest"].items():
        require(name.startswith(REMOTE) and sha_file(root / "source" / relative(name[len(REMOTE):])) == wanted,
                "Runtime input/source SHA differs")
    expected = {( "calibration", d, p, None, None, None) for d in range(16) for p in POLICIES[:4]}
    expected |= {("fit_warmup", 0, p, None, None, -1) for p in POLICIES[:4]}
    expected |= {("train", d, p, epoch, None, None) for d in range(16) for p in MODES for epoch in (0, 1)}
    expected |= {("monitor_warmup", d, p, None, 0, -1) for d in range(16, 24) for p in POLICIES}
    expected |= {("monitor", d, p, None, s, r) for d in range(16, 24) for p in POLICIES for s in range(3) for r in (0, 1)}
    found = {}
    for r in requests:
        require(r["key"] in expected and r["key"] not in found and r["output"] == refs[r["document"]],
                "Request matrix/canonical AR equality differs")
        found[r["key"]] = r
    require(len(requests) == 556 and set(found) == expected and done["requests_including_warmups"] == 556,
            "Original556-request matrix is incomplete")
    require(Counter(r["phase"] for r in requests) == {"calibration": 64, "fit_warmup": 4,
            "train": 96, "monitor_warmup": 56, "monitor": 336}, "Request phase counts differ")
    frozen = controls["frozen_policies.json"]
    require(hashes["frozen_policies.json"] == done["frozen_policies_sha256"], "Frozen model SHA differs")
    for r in requests:
        if r["phase"] in ("monitor", "monitor_warmup") and r["policy"] in MODES:
            require(r["before"] == r["after"] == frozen["models"][r["policy"]], "Monitor changed frozen parameters")
    cells = {}
    for d in range(16, 24):
        require(len({r["input_sha256"] for r in requests if r["document"] == d}) == 1, "Unpaired monitor prompt")
        for p in POLICIES:
            cells[d, p] = {metric: statistics.fmean(statistics.median(
                found["monitor", d, p, None, s, r][metric] for r in (0, 1)) for s in range(3))
                for metric in ("acceptance", "tps")}
    calibration = controls["calibration.json"]
    fixed_means = {p: statistics.fmean(found["calibration", d, p, None, None, None]["tps"] for d in range(16)) for p in POLICIES[:4]}
    best = max(POLICIES[:4], key=lambda p: (fixed_means[p], -int(p[5:].split("_")[0])))
    require(best == calibration["best_fixed"], "Fit-only fixed baseline selection differs")
    summary = load_json((root / "analysis/summary.json").read_bytes())
    require(summary["primary_baseline"] == best and summary["monitor_matrix_complete"] is True,
            "Saved primary baseline/completeness differs")
    primary = summary["continuation_gate"]["csgb_vs_fit_best_fixed"]
    differences = {m: [100 * (cells[d, "csgb"][m] / cells[d, best][m] - 1) if m == "tps"
                        else cells[d, "csgb"][m] - cells[d, best][m] for d in range(16, 24)] for m in ("acceptance", "tps")}
    estimates = {}
    try:
        import numpy as np
    except ImportError:
        np = None
    for m, values in differences.items():
        saved = primary[m]; close(saved["mean"], statistics.fmean(values))
        require(saved["documents"] == 8, "Paired unit must remain n8 documents")
        for d, value in zip(range(16, 24), values): close(saved["per_document"][str(d)], value)
        ci = None
        if np is not None:
            rng = random.Random(42)
            draws = [statistics.fmean(rng.choices(values, k=8)) for _ in range(10000)]
            ci = [float(v) for v in np.quantile(draws, [.025, .975], method="linear")]
            for x, y in zip(saved["ci95"], ci): close(x, y)
        estimates[m] = {"mean": statistics.fmean(values), "ci95_recomputed": ci, "documents": 8}
    return {"passed": True, "archive_file_hashes": len(files), "source_stage_members": 60,
        "source_stage_files_verified": len(included_stage), "excluded_originals": excluded,
        "excluded_log_bytes_reverified": False,
        "runtime_source_files": 52, "raw_json_files": 585, "done_raw_hashes": 584,
        "canonical_exact_requests": 556, "AR_references": 24, "monitor_requests": 336,
        "primary_baseline": best, "primary": estimates, "CI_recomputed": np is not None,
        "CI_note": "10000 Python Random42 paired-document draws, linear95% interval" if np is not None else "NumPy unavailable: CI NOT recomputed; points/hashes/exactness verified",
        "scope": "Saved archive byte/canonical/arithmetic verification only; no GPU rerun, no model-file/external-provenance or full RNG/gradient audit",
        "production_speed_claim": False, "full_study_success_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        result = verify(args.root)
    except Exception as error:
        print(json.dumps({"passed": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
