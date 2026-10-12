# SpecExtend reproduction plan

This checkout is pinned to upstream commit `a307ace6960a8bcf39f919b2d8fdcac587fa444a`.

## Target result

The first milestone reproduces the Vicuna-7B/Vicuna-68M rows of Table 2 on
GovReport. The paper uses one A100 80GB GPU, FP16, greedy decoding, 256 generated
tokens, 20 inputs per length, and two runs per input.

| Input | Standard tau | Standard tok/s | SpecExtend tau | SpecExtend tok/s |
|---|---:|---:|---:|---:|
| 1K | 2.73 | 100.31 | 3.80 | 128.59 |
| 2K | 1.64 | 55.64 | 3.52 | 109.58 |
| 4K | 1.60 | 41.91 | 3.04 | 76.48 |
| 8K | 1.62 | 25.71 | 3.06 | 55.16 |
| 16K | 1.59 | 16.56 | 3.07 | 33.84 |

The reproduction runner additionally measures naive autoregressive decoding so
that speedup relative to the non-speculative baseline is available from the same
machine and examples.

## Environment

Use Linux, Python 3.10, CUDA 12.1, and an A100 80GB GPU. From the repository:

```bash
cd specextend
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

The two model repositories are downloaded from Hugging Face:

- `lmsys/vicuna-7b-v1.5-16k`
- `double7/vicuna-68m`

Run a data-only preflight before downloading model weights:

```bash
python reproduce_classic.py \
  --data_dir data/govreport \
  --lengths 1 2 4 8 16 \
  --check_only
```

Run a one-sample smoke test:

```bash
python reproduce_classic.py \
  --data_dir data/govreport \
  --lengths 1 \
  --samples_per_length 1 \
  --runs_per_sample 1 \
  --max_gen_len 32 \
  --output_file smoke_classic.json
```

Run the Table 2 reproduction:

```bash
python reproduce_classic.py \
  --data_dir data/govreport \
  --lengths 1 2 4 8 16 \
  --samples_per_length 20 \
  --runs_per_sample 2 \
  --max_gen_len 256 \
  --output_file govreport_classic_reproduction.json
```

The JSON output contains environment metadata, aggregate results, and every raw
run. Keep the raw results for paired significance tests and later method audits.

Use `--modes naive standard specextend` to select only the paths that fit the
available GPU. `--sample_start N` starts at a particular zero-based sample,
which supports failure isolation and resumable per-sample execution.

The completed RTX 4090 experiment and raw measurements are in
`results/rtx4090_reproduction/REPORT.md`.

## Known upstream reproduction issues

1. `eval_classic.py` scans every JSONL file but its filename parser rejects the
   included `*_512.jsonl` file.
2. It evaluates only one SD mode per invocation and does not measure the naive
   autoregressive baseline used to calculate the paper's reported speedups.
3. Retrieval parameters are hard-coded. The classic configuration matches the
   appendix: chunk size 32, top-k 32, retrieval every four steps.
4. The paper reports A100 80GB results. Numbers from other GPUs should be treated
   as functional validation, not an exact throughput reproduction.
5. Standard 16K speculative decoding OOMs on a 48GB RTX 4090 during dense draft
   prefill. SpecExtend itself fits.
6. Reusing one model instance across multiple 16K samples can trigger a CUDA
   index assertion. Running each 16K sample in a fresh process avoids the stale
   cross-sample state and completed all 40 SpecExtend measurements.

## Mechanism experiments

`mechanism_experiments.py` runs three paired studies on the first 20 documents
from the bundled 2K GovReport split:

1. `dense`: exact nested inputs at 512/640/768/896/1024/1280 tokens.
2. `content`: 1024-token prompts whose added half is a related continuation,
   a duplicate of the first half, or text from an unrelated report.
3. `policy`: full cache versus random/recent/target-attention chunk selection
   at 768/1024/1280 tokens and K=8/16/24/32.

Greedy decoding is deterministic, so each document is generated once per
condition; the 20 documents, not repeated runs, are the independent statistical
units. Results are atomically checkpointed after every generation and existing
records are skipped when the command is resumed.

```bash
cd specextend
CUDA_VISIBLE_DEVICES=0 python mechanism_experiments.py \
  --suites dense content policy \
  --source_file data/govreport/govreport_2K.jsonl \
  --samples 20 \
  --max_gen_len 256 \
  --base_model_path /path/to/vicuna-7b-v1.5-16k \
  --draft_model_path /path/to/vicuna-68m \
  --output_dir /path/to/mechanism_results
```

The mechanism patch also resets `selected_chunks` for every request. Upstream
left this field populated across samples, which contaminated the first draft
step of subsequent requests. The runner additionally exposes deterministic
`random`, `recent`, and `attention` retrieval policies for controlled ablations.

## Reliable long-context baselines

`baseline_profiler.py` evaluates paired 1K/2K/4K/8K/16K inputs with chunk size
32 and 256-token generation. The completed reliable runs use a 64-node
candidate tree and seed 42. Classic
uses a 32-chunk working-cache budget and EAGLE uses 64 chunks. The nominal 16K
GovReport examples contain 15.1K--15.4K Vicuna tokens after templating; the
runner preserves each complete example and records the exact token count.

Five conditions are retained. The fifth is a diagnostic control beyond the
four requested baselines:

1. `full`: complete draft KV with Flash Prefill;
2. `full_hybrid`: complete draft KV with the same Hybrid Tree Attention used by
   the retrieval variants;
3. `recent`: recent-only working KV;
4. `random`: deterministic random chunks;
5. `attention`: original target-attention Top-K.

This separates efficient target verification (`full_hybrid - full`), cache
shortening (`recent/random - full_hybrid`), and retrieval quality
(`attention - recent/random`). Per run, the profiler records end-to-end
throughput, acceptance lengths, CUDA draft/verification/retrieval times,
working-cache size, per-round rejection position, root KL(target || draft),
retrieval Jaccard similarity, and absolute/normalized selected chunk positions.

```bash
cd specextend
CUDA_VISIBLE_DEVICES=0 python baseline_profiler.py \
  --engine classic \
  --source_file data/govreport/govreport_16K.jsonl \
  --lengths 1024 2048 4096 8192 16384 \
  --samples 20 \
  --max_gen_len 256 \
  --tree_nodes 64 \
  --seed 42 \
  --base_model_path /path/to/vicuna-7b-v1.5-16k \
  --draft_model_path /path/to/vicuna-68m \
  --output /path/to/classic_baselines.json
```

The completed Classic and EAGLE raw records, paired bootstrap reports, and
four-baseline figures are under
`results/rtx4090_reproduction/reliable_baseline/`. EAGLE uses
`lmsys/vicuna-7b-v1.5-16k` with
`jycha-98/EAGLE-vicuna-7b-v1.5-16k`; access to the gated EAGLE repository was
authorized before the reliable run.
