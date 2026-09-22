#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-freeze the package after you edit a document in place.

Run this only if you changed files inside `RESCUE_CODE_HANDOFF_20260922/` — for example after
filling the artifact-URL block in `ARTIFACT_PUBLICATION.md`.  It rehashes every file, updates
`CODE_MANIFEST.json`, and regenerates `SHA256SUMS.txt`, so `sha256sum -c SHA256SUMS.txt` passes again.

    cd <package>
    python tools/refresh_hashes.py            # dry run: report what would change
    python tools/refresh_hashes.py --write    # apply

If you would rather leave the package byte-frozen, record the published URLs in a companion file
*outside* the package instead (see ARTIFACT_PUBLICATION.md §"After the upload").  That keeps every
hash in this package exactly as published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF_EXCLUDE = {"SHA256SUMS.txt"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def walk(root: str):
    for r, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache", ".git")]
        for fn in sorted(files):
            p = os.path.join(r, fn)
            yield os.path.relpath(p, root).replace("\\", "/"), p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply the changes")
    args = ap.parse_args()

    man_path = os.path.join(PKG, "CODE_MANIFEST.json")
    if not os.path.isfile(man_path):
        sys.exit("error: CODE_MANIFEST.json not found under %s" % PKG)
    man = json.load(open(man_path, encoding="utf-8"))
    by_path = {f["packaged_path"]: f for f in man["files"]}

    disk = dict(walk(PKG))
    changed, added, gone = [], [], []

    for rel, p in disk.items():
        if rel in SELF_EXCLUDE:
            continue
        h = sha256(p)
        size = os.path.getsize(p)
        if rel == "CODE_MANIFEST.json":
            continue
        e = by_path.get(rel)
        if e is None:
            added.append(rel)
            continue
        if e.get("sha256") != h or e.get("size_bytes") != size:
            changed.append((rel, e.get("sha256", "")[:16], h[:16]))

    for rel in by_path:
        if rel not in disk:
            gone.append(rel)

    print("package        :", PKG)
    print("files on disk  :", len(disk))
    print("manifest       :", len(by_path), "entries")
    print("changed        :", len(changed))
    for rel, old, new in changed:
        print("   %-58s %s -> %s" % (rel, old, new))
    print("new (unlisted) :", len(added))
    for rel in added:
        print("   ", rel)
    print("missing        :", len(gone))
    for rel in gone:
        print("   ", rel)

    if not args.write:
        print()
        print("dry run — re-run with --write to apply")
        return 0

    for rel, p in disk.items():
        if rel in SELF_EXCLUDE or rel == "CODE_MANIFEST.json":
            continue
        e = by_path.get(rel)
        if e is None:
            continue
        e["sha256"] = sha256(p)
        e["size_bytes"] = os.path.getsize(p)

    man["files"] = sorted(by_path.values(), key=lambda f: f["packaged_path"])
    man["file_count"] = len(man["files"])
    with open(man_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
        f.write("\n")

    sums = []
    for rel, p in disk.items():
        if rel in SELF_EXCLUDE:
            continue
        sums.append("%s  %s" % (sha256(p), rel))
    sums.sort(key=lambda s: s.split("  ", 1)[1])
    with open(os.path.join(PKG, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sums) + "\n")

    print()
    print("rewrote CODE_MANIFEST.json and SHA256SUMS.txt")
    print("verify with:  cd %s && sha256sum -c SHA256SUMS.txt" % PKG)
    print("note: the ZIP must be rebuilt separately — this script only re-freezes the tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
