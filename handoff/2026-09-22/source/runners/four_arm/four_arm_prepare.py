"""§4/§5 preparation: dataset identity + the frozen experiment configuration.

Writes, before any research row is produced:
  out/dataset_identity.json          dataset + split identity and the ORDERED sample ids
  out/four_arm_experiment_config.json the frozen experiment config; its sha256 is the
                                      config_hash carried by every result envelope
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from four_arm_common import dataset_identity, experiment_config, load_harness  # noqa: E402

SOURCE_LOCK_CONTRACT = "frozen_experiment_and_statistics_contract.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--validation-prior-cache", required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--planned-n", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    H = load_harness(args.harness_root)

    contract_path = Path(args.source_lock) / SOURCE_LOCK_CONTRACT
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    dataset = dataset_identity(Path(args.validation_ledger), Path(args.dataset_path), contract)
    dataset["statistics_contract"] = {"path": str(contract_path),
                                      "sha256": H.sha256_file(contract_path)}
    dataset["planned_N"] = args.planned_n
    dataset["N_matches_contract"] = dataset["row_count"] == args.planned_n

    cfg = experiment_config(args, args.manifest_sha256)
    cfg["dataset"]["row_count"] = dataset["row_count"]
    cfg["dataset"]["ordered_sample_ids_sha256"] = __import__("hashlib").sha256(
        json.dumps(dataset["ordered_sample_ids"], sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()
    cfg["config_hash"] = __import__("hashlib").sha256(
        json.dumps({k: v for k, v in cfg.items() if k != "config_hash"},
                   sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    H.atomic_write_text(out / "dataset_identity.json",
                        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n")
    H.atomic_write_text(out / "four_arm_experiment_config.json",
                        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"planned_N": args.planned_n,
                      "ledger_rows": dataset["row_count"],
                      "ledger_sha256": dataset["ledger_sha256"],
                      "ordered_sample_ids_sha256": cfg["dataset"]["ordered_sample_ids_sha256"],
                      "first_ids": [x["canonical_source_id"] for x in
                                    dataset["ordered_sample_ids"][:3]],
                      "last_id": dataset["ordered_sample_ids"][-1]["canonical_source_id"],
                      "config_hash": cfg["config_hash"],
                      "checkpoint_sha256": cfg["checkpoint_sha256"],
                      "source_lock_sha256": cfg["source_lock_sha256"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
