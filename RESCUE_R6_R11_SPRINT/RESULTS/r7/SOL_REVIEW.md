# R7 SOL_REVIEW（Harness 实现版，待 Sol/high 复核）

实现：`code/r7_span_repair_runner.py` + 共享纯逻辑 `r7_pure_logic.py`（runner/tests 同路径）。
复核对照以下冻结语义审查 7 项，逐条指向真实代码行：

## 1. selected token/span 绝对位置
- `supporting_sentence_spans_in_doc_text`（r7_pure_logic.py L26-59）：从原 context 句子列表按 R6
  相同 join 规则（`" ".join` 单空格）重建句子 char 区间，**绝不按空格 split 文本**；doc 坐标 =
  `len(title)+1` 起，逐句累计 `len(s)+1`。
- `char_spans_to_token_spans`（L62-92）：经 `tokenizer(doc_input, return_offsets_mapping=True,
  add_special_tokens=False)` 把 char 区间映射为半开 token 区间；doc 内 token 起点再叠加
  `[Doc N]\n` 前缀长度与本 doc 在 fused KV 中的起点 `sum(doc_lengths[:di])`
  （runner `repair_span_kv` 中 `abs_start = prefix_starts[di] + ts`）。
- 测试：tests_r7_span_repair.py「gold supporting sentences located in both docs /
  doc0/doc1 span matches」精确断言 (0,6,16)/(1,15,26)；(1,15,26) 按人工逐字符累加校验。

## 2. RoPE / position IDs
- `repair_span_kv`（runner）：每个 span forward 传 `position_ids = arange(abs_start,
  abs_start+slen)`，与 fused KV 中该区间绝对位置一致；query append 再从 `fused_len` 继续
  （`prefill_route_b_with_span_repair`），与 R6 `prefill_route_b_from_ids` 相同。
- 断言：`abs_end <= total_len`，越界即 RuntimeError，位置不允许重排。

## 3. causal mask
- span forward 用 `attention_mask = ones((1, abs_start+slen))`（past prefix + span 全可见），
  因果性由 Qwen 标准 causal mask 提供；未选中后缀完全不出现在该次 forward 的 attention 里。

## 4. 每层 hidden-state 依赖
- prefix 经 `past_key_values` 传给模型（`r6._from_legacy_cache(prefix)`），每层 KV 由
  prefix（已有 KV）+ selected span 的真实 forward 产生；绝不可能只重算 selected token 而
  跳过依赖前缀（past 前缀 KV 是读取旧值，span 的 K/V 是本次 forward 产出）。
- 代价披露：prefix 规模记入 `attended_prefix_tokens`，span 规模记入 `processed_tokens`。

## 5. selected KV write-back
- 只替换每层 `[abs_start, abs_end)` 区间（`merged_k = cat(old[:abs_start], rep, old[abs_end:])`），
  并断言 `merged_k.shape[2] == total_len`（长度不变不变量，代码内 RuntimeError 兜底）。
- torch 相关不变量由远程 GPU smoke 的 `repair_span_kv` 长度断言覆盖
  （smoke 输出含 written/processed/attended 且总长不变）。

## 6. query append 一致性
- 修复完成后 `fused_len = past_seq_len(fused)` 重新读取，query 从该长度继续；与 Route-B
  naive（无 repair）query append 完全同构，仅 past 内容不同。

## 7. 未选择的下游文档不得暗中完整 recompute
- 单次 forward 仅输入 span ids（`span_ids = doc_inputs_ids[di][:, ts:te]`），后缀 token 不
  进入任何 forward；`processed_tokens` 只累计 span 长度（不含前缀重算、不含下游文档），
  `attended_prefix_tokens` 单独披露前缀读取，绝不隐藏 processed 成本。

## Teacher-attention 排序
- `oracle_budget` 需要 `--teacher_attention_jsonl`（每行 sample_index + span_order）；
  缺失时 `TEACHER_ATTENTION_ASSET_UNAVAILABLE` 显式失败（runner main 顶部），**不伪造
  排序、不降级**。`truncate_spans_by_budget` 按 attention 顺序贪心截断，预算
  `int(budget_pct/100*total_prompt_tokens)`，最小单位为整个 span（不做 token 内拆分），
  预算小于单个 span 时保留最高注意力整 span（注释明示）。

## 固定基线
- Route A（FULL_CONTINUOUS_PREFILL）与 Route B（NAIVE_DOCUMENT_KV_REUSE）逐行对齐 R6
  runner 的 `prefill_route_a` / `prefill_route_b_from_ids(repair_mode="none")`；
  prompt 文本、doc 顺序、evaluator（exact_match/f1 + extract_final_answer）全部复用 R6
  模块导入，零改动，保证 paired population 位一致。

## 待 Sol/high 复核点
- (a) `abs_start` 逻辑核对：`prefix_starts` 用 doc_ids_list 各段实际 token 长度累加；
      confirm doc 内 offset 与 fused KV 布局完全对应（should hold: fused = concat(doc segments)，见 R6 fuse_past_key_values）。
- (b) span forward 时 prefix 以 **legacy cache** 传入（`_from_legacy_cache(prefix)`），
      与 R6 `repair_document_spans` 的 doc 级 rewrite 同构；spans 会重复读取各自前缀 —
      每个 span 独立 forward（非批量），processed 按 span 线性累计，属已披露的保守成本模型。
- (c) `budget_tokens = int(pct/100 * L_A)`；L_A 为 Route-A full prompt tokens（含 query），
      ratio 与契约「真实 processed_token_ratio」分母一致（L_A）。