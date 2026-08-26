# R7 selective span repair — Harness 实现版

独立 runner：`code/r7_span_repair_runner.py`（import R6 runner + `r7_pure_logic.py`）。
纯逻辑校验：`python tests_r7_span_repair.py`（22 assertions，无 torch 依赖）。
GPU smoke / 完整 Pilot 曲线由 Harness 在远程环境运行。

## 三种条件

| 条件 | 语义 | budget |
|---|---|---|
| `oracle_span` | 全部 gold supporting sentence 对应连续 token spans 的 KV 写回 | 固定 100% |
| `oracle_budget` | gold spans 按 teacher attention 排序，预算截断写回 | 1/2/5/10/20%（需要 `--teacher_attention_jsonl`，缺失则显式失败） |
| `random_matched_budget` | non-gold 随机 span，written token 数与 span 数与同预算 Oracle 匹配（seed 20260827+idx，确定性） | 同 oracle_budget 各档 |

## 远程示例

```bash
cd /home/ztl/RESCUE-baseline-fc46ae2a/RESCUE_R6_R11_SPRINT/code
PY=/home/ztl/anaconda3/envs/rescue-r0/bin/python
MODEL=/home/ztl/jwt/models/Qwen2-1_5B_Instruct
DATA=/home/ztl/jwt/datasets/hotpotqa/hotpot_qa/distractor
OUT=/home/ztl/rescue-evidence/r7-001/r6

# oracle_span（=100%）
CUDA_VISIBLE_DEVICES=0 $PY r7_span_repair_runner.py --r7_condition oracle_span \
  --model_path $MODEL --dataset_path $DATA --no_fallback_online \
  --max_samples 50 --max_L 2050 --max_new_tokens 32 \
  --dtype float16 --device_map cuda:0 --output_dir $OUT/oracle_span --jsonl_name decode_eval.jsonl

# oracle_budget 各档（需要 teacher attention 资产）
CUDA_VISIBLE_DEVICES=1 $PY r7_span_repair_runner.py --r7_condition oracle_budget --budget_pct 10 \
  --teacher_attention_jsonl /home/ztl/rescue-evidence/r7-001/teacher_attn_order.jsonl \
  --model_path $MODEL --dataset_path $DATA --no_fallback_online \
  --max_samples 50 --max_L 2050 --max_new_tokens 32 \
  --dtype float16 --device_map cuda:0 --output_dir $OUT/oracle_b10 --jsonl_name decode_eval.jsonl

# random_matched_budget 各档
CUDA_VISIBLE_DEVICES=2 $PY r7_span_repair_runner.py --r7_condition random_matched_budget --budget_pct 10 \
  --model_path $MODEL --dataset_path $DATA --no_fallback_online \
  --max_samples 50 --max_L 2050 --max_new_tokens 32 \
  --dtype float16 --device_map cuda:0 --output_dir $OUT/rand_b10 --jsonl_name decode_eval.jsonl
```

## 输出字段（每个 evaluated 记录，除 R6 公共列外）

- `gold_sentence_spans` / `gold_token_spans`：char/token 级定位证据
- `selected_spans`：实际写回区间（doc_index, ts, te）
- `written_kv_tokens` / `processed_tokens` / `attended_prefix_tokens`
- `written_token_ratio` / `processed_token_ratio`（分母 = Route-A full prompt tokens, L_A）
- `em_repair_clean` / `f1_repair_clean`：repair 后 decode 的 Clean 指标；同列含 Route A/B 基线
- `teacher_attention_used`：oracle_budget/random 为 true
- drop 记录：`max_prompt_len>2050`（6 行：idx 7/13/43/44/46/47）、
  `RANDOM_LENGTH_MATCHED_UNCONSTRUCTIBLE`（机械排除，不崩溃不回退）

## 契约红线

- teacher attention 缺失 → `TEACHER_ATTENTION_ASSET_UNAVAILABLE` 显式失败，不伪造排序
- 未选中下游文档绝不被完整 recompute；processed 成本显式披露
- 最小单位为 contiguous span，不做 token 内拆分
- 不改 R6 prompt/顺序/evaluator；共享逻辑 runner 与测试同一代码路径