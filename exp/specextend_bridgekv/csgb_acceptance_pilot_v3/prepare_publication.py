"""Mechanically copy/hash a saved experiment for publication; never run models."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(experiment, out):
    experiment, out = Path(experiment).resolve(), Path(out).absolute()
    if out.exists():
        raise ValueError("Publication output must be fresh; never overwrite")
    local = Path(__file__).resolve().parent
    for required in ("README.md", "verify_archive.py"):
        if not (local / required).is_file():
            raise ValueError("Missing publication entry: " + required)
    out.mkdir(parents=True, exist_ok=False)
    originals, excluded = {}, {}
    def copy(source, relative):
        data = source.read_bytes()
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(data)
        originals[relative] = {"original_relative_path": str(source.relative_to(experiment)),
                               "sha256": sha(data), "bytes": len(data)}
    for source in sorted((experiment / "frozen_source").rglob("*")):
        if source.is_symlink():
            raise ValueError("No symlink in frozen code")
        if source.is_file():
            relative = "source/" + str(source.relative_to(experiment / "frozen_source"))
            if source.suffix in (".log", ".pid"):
                data = source.read_bytes()
                excluded[relative] = {"sha256": sha(data), "bytes": len(data),
                                      "reason": "repository excludes process logs"}
            else:
                copy(source, relative)
    for source in sorted((experiment / "analysis").iterdir()):
        if source.is_symlink() or not source.is_file():
            raise ValueError("Expected original regular analysis files only")
        if source.name == "plot_data.json":
            data = source.read_bytes()
            (out / "analysis/plot_data.json.gz").write_bytes(gzip.compress(data, mtime=0))
            originals["analysis/plot_data.json.gz"] = {
                "original_relative_path": "analysis/plot_data.json", "original_sha256": sha(data),
                "original_bytes": len(data), "transformation": "lossless gzip; original JSON bytes unchanged"}
        else:
            copy(source, "analysis/" + source.name)
    for name in ("frozen_policies.json", "calibration.json", "protocol.json", "DONE.json"):
        copy(experiment / "pilot" / name, "frozen/" + name)
    copy(experiment / "pilot_completion_verification_v1.json",
         "provenance/pilot_completion_verification_v1.json")
    for name in ("README.md", "verify_archive.py", "prepare_publication.py"):
        copy(local / name, name)
    # Deterministic, flat regular-file tar. No extraction is needed for audits.
    files = sorted((experiment / "pilot").glob("*.json"))
    if len(files) != 585:
        raise ValueError("Expected the complete585 JSON pilot files")
    with (out / "raw_results.tar.gz").open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as bundle:
                for source in files:
                    if source.is_symlink():
                        raise ValueError("No symlink in raw results")
                    data = source.read_bytes()
                    info = tarfile.TarInfo("pilot/" + source.name)
                    info.size, info.mtime, info.mode = len(data), 0, 0o644
                    bundle.addfile(info, io.BytesIO(data))
    manifest = {"schema": 1, "kind": "csgb_acceptance_pilot_publication_archive",
        "experiment": "CSGB_episode_pilot_20261016_v3", "not_a_new_experiment": True,
        "scope": "NativeV; actual4096; node budgets8/16/24/32; fixed max_depth10; historical TRAIN-only",
        "raw_json_files": 585, "requests_including_warmups": 556,
        "runtime_models_included": False, "new_depth_controller_included": False,
        "originals": originals, "excluded_originals": excluded, "files": {}}
    for source in sorted(out.rglob("*")):
        if source.is_file():
            data = source.read_bytes()
            git_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            manifest["files"][str(source.relative_to(out))] = {
                "sha256": sha(data), "bytes": len(data), "git_blob_sha1": git_blob}
    with (out / "ARCHIVE_MANIFEST.json").open("x") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {"files": len(manifest["files"]) + 1, "bytes": sum(v["bytes"] for v in manifest["files"].values()),
            "raw_archive_bytes": (out / "raw_results.tar.gz").stat().st_size,
            "manifest_sha256": sha((out / "ARCHIVE_MANIFEST.json").read_bytes())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.experiment, args.out), sort_keys=True))
