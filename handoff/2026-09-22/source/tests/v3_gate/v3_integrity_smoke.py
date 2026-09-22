"""§10 RESULT INTEGRITY SMOKE — RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001.

CPU only. Uses the frozen harness envelope writer/verifier on a tiny fake shard and then
simulates stale/truncated/missing/mismatched result artifacts. Every damaged variant
must be judged incomplete.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
CONFIG_HASH="c"*64
CKPT_SHA="6532daad4e3ae54a7e76063e6aef3fef9b59ba6ef709394bf361c30228f19ace"

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--harness-root",required=True); ap.add_argument("--out",required=True)
    args=ap.parse_args(); H=Path(args.harness_root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(H)); from rescue_harness import harness as Hm
    work=out/"_integrity_work"
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    body=work/"fake_shard.jsonl"; records=[{"sample_id":f"s{i}","arm":"teacher","value":i} for i in range(4)]
    marker=Hm.write_result_envelope(body,task_id="RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001",
      stage="result_integrity_smoke",records=records,arms=["teacher","null"],
      sample_ids=[r["sample_id"] for r in records],config_hash=CONFIG_HASH,checkpoint_sha256=CKPT_SHA)
    cases=[]
    def record(name,expect_incomplete,result,detail=""):
        got=result["status"]; ok=(got!="PASS") if expect_incomplete else (got=="PASS")
        cases.append({"case":name,"expected":"INCOMPLETE" if expect_incomplete else "PASS",
          "observed":got,"reason":result.get("reason"),"detail":detail,"ok":bool(ok),"returned":result})
    clean=Hm.verify_result(body); record("valid_body_bound_marker",False,clean)
    header=Hm.read_result_header(body); parsed=Hm.read_result_records(body)
    stale=body.read_text(encoding="utf-8")+json.dumps({"sample_id":"s9","arm":"teacher","value":9})+"\n"
    body.write_text(stale,encoding="utf-8"); record("stale_marker_body_appended",True,Hm.verify_result(body))
    body.write_text(stale[:len(stale)//2],encoding="utf-8"); record("truncated_body",True,Hm.verify_result(body))
    marker=Hm.write_result_envelope(body,task_id="RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001",
      stage="result_integrity_smoke",records=records,arms=["teacher","null"],
      sample_ids=[r["sample_id"] for r in records],config_hash=CONFIG_HASH,checkpoint_sha256=CKPT_SHA)
    flag=Hm.completion_path(body); saved=flag.read_text(encoding="utf-8"); flag.unlink()
    record("marker_missing",True,Hm.verify_result(body)); flag.write_text(saved,encoding="utf-8")
    body.unlink(); record("body_missing",True,Hm.verify_result(body))
    Hm.write_result_envelope(body,task_id="RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001",
      stage="result_integrity_smoke",records=records,arms=["teacher","null"],
      sample_ids=[r["sample_id"] for r in records],config_hash=CONFIG_HASH,checkpoint_sha256=CKPT_SHA)
    lines=body.read_text(encoding="utf-8").splitlines(keepends=True); body.write_text("".join(lines[:-1]),encoding="utf-8")
    lying=json.loads(saved); lying["body_sha256"]=Hm.sha256_file(body); flag.write_text(json.dumps(lying,indent=2)+"\n",encoding="utf-8")
    record("marker_lies_about_record_count",True,Hm.verify_result(body))
    Hm.write_result_envelope(body,task_id="RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001",
      stage="result_integrity_smoke",records=records,arms=["teacher","null"],
      sample_ids=[r["sample_id"] for r in records],config_hash=CONFIG_HASH,checkpoint_sha256=CKPT_SHA)
    record("config_hash_mismatch",True,Hm.verify_result(body,expected_config_hash="d"*64))
    record("checkpoint_sha_mismatch",True,Hm.verify_result(body,expected_checkpoint_sha256="e"*64))
    payload={"stage":"result_integrity_smoke",
      "generated_at":datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z"),
      "invariant":"a reader must trust a completion marker only after full re-verification",
      "marker_written":{k:marker[k] for k in ("status","generation_id","body_sha256","record_count","config_hash","checkpoint_sha256")},
      "body_header":header,"body_records_parsed":len(parsed),
      "record_count_bound":marker["record_count"]==len(parsed),
      "generation_id_bound":header.get("generation_id")==marker["generation_id"],
      "config_hash_bound":header.get("config_hash")==marker["config_hash"],
      "checkpoint_sha_bound":header.get("checkpoint_sha256")==marker["checkpoint_sha256"],"cases":cases}
    payload["status"]="PASS" if all(c["ok"] for c in cases) else "FAIL"
    (out/"v3_result_integrity.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if payload["status"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
