# RESCUE Evidence Index — 2026-09-21 Archive

This file indexes important locally returned evidence. Paths are preserved for provenance; large evidence bundles are not copied into GitHub by this web archive.

## Harness / correctness

### Harness freeze V3
- Manifest:
  - `HARNESS_FREEZE_CANDIDATE_MANIFEST_V3.json`
  - SHA256 `0e569ea84d36ae91d7ad73dd7299b60c9dcab1ff0abb14e270f447449ff43788`
- Independent review:
  - `HARNESS_FREEZE_FINAL_IDENTITY_DELTA_REVIEW.md`
  - SHA256 `330615041a4a56b86710f59d52d1cbcc0fb23ae2ba29e45d5a9ff5686fc3724a`
- PRE gate report:
  - `GATE_V3_REPORT.md`
  - SHA256 `8e8bd2e6ff425686ef830d535657abb64b433f5bd922650f23461624199324d5`

Representative local root:
`E:\桌面\rescue\_rescue_20260920_work\harness_freeze_001\`

Representative remote root:
`/home/ztl/rescue_runs/parallel_closure_003/k4_seqloss_dev_closure_002/harness_freeze_001/`

## Four-arm valid rerun

Task:
`RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001`

Remote:
`/home/ztl/rescue_runs/parallel_closure_003/k4_seqloss_dev_closure_002/four_arm_run_001/out/`

Local:
`E:\桌面\rescue\_rescue_20260920_work\four_arm_run_001\out\`

Report:
`FOUR_ARM_RESEARCH_SUMMARY.md`

Report SHA256:
`9b277340…`

Archive tar SHA256:
`4618c7a24a74570d17476f4a81c7b45fe842272b64d7c4e6656291b2fd9abed7`

## Guidance representation mechanism sprint

Task:
`RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001`

Remote:
`/home/ztl/rescue_runs/parallel_closure_003/k4_seqloss_dev_closure_002/mech_run_001/out/`

Local:
`E:\桌面\rescue\_rescue_20260920_work\mech_run_001\out\`

Report:
`GUIDANCE_REPRESENTATION_MECHANISM_REPORT.md`

Report SHA256:
`ae9f8333…`

Candidate card:
`NEXT_METHOD_CANDIDATE_CARD.md`

Candidate card SHA256:
`b86678d6…`

Archive tar SHA256:
`879487c44d08051fe2d1ce5065997712576895c1467b2af12b2bb284e959845d`

## R1 pilot

Task:
`RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001`

Result:
`R1-FAIL`

Cached R1 mapped priors:
`r1_mapped_priors.pt`

Reported archive tar SHA256:
`2d0d69b7…`

## Teacher decision attribution Phase C

Task family:
`RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001`

Remote evidence root:
`…/td_run_001/out/`

Key files:
- `TEACHER_DECISION_ATTRIBUTION_REPORT.md`
  - SHA256 `518cb65b…`
- `teacher_attribution_rows.jsonl`
  - SHA256 `2104ac02…`
- `teacher_attribution_summary.json`
  - SHA256 `7adf095f…`

Archive tar:
`td_run_001_out.tar.gz`

SHA256:
`5b55ff6c7a3ba8f474e368ab6e93be554b9aca86954a448da115d875618ae14d`

Reported evidence count:
71 files, SHA256 verification passed locally and remotely.

## Control-plane snapshots

Web-available historical bundle:
- V8: `RESCUE_WEB_SYNC_BUNDLE_20260921_V8.zip`
- SHA256 `6d1fdfe7cdd2c4542beddd3eb0b6faa25e8e27ff3de51bb1d213090d1fc617fd`

Latest local state reported after V8:
- PROJECT_STATE_VERSION `2026-09-21.11`
- corresponding V10 bundle SHA was reported locally as beginning `d50778e5…`
- V10 bundle itself was not supplied to the web session when this GitHub archive was created.

## Important validity rule

Evidence listed as INVALIDATED in the canonical control plane must stay invalidated even if retained for history. Archival presence is not evidence validity.
