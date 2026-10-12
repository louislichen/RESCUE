# SpecExtend BridgeKV experiments

This directory is a source snapshot of the current SpecExtend/BridgeKV research implementation used for the RTX 4090 experiments. The base implementation is derived from [jycha98/SpecExtend](https://github.com/jycha98/SpecExtend); the local additions and modifications implement the BridgeKV experiments described below.

The implementation includes:

- attention-guided Draft working-cache retrieval;
- Target-to-Draft KV mapping with Native, K-only, V-only, and K+V policies;
- post-RoPE attention-geometry diagnostics and oracle QK-rescue controls;
- real Draft-V projection bypass with full-shape semantic-shadow validation;
- Native-V retention sweeps at 4K and 8K contexts;
- analysis and plotting programs for acceptance length, end-to-end throughput, latency, and mechanism diagnostics.

## Layout

- `specextend/attn_bridge_experiment.py`: main BridgeKV experiment runner and policy definitions.
- `specextend/classic/attn_bridge.py`: mapper, retrieval, geometry, and Bridge state implementation.
- `specextend/classic/modeling_llama_kv_draft.py`: Draft attention and V-projection bypass implementation.
- `specextend/native_v_retention_sweep.py`: 0%--100% Native-V retention sweep.
- `specextend/baseline_profiler.py`: stage-level CUDA timing and system metrics.
- `analyze_kv_geometry.py`: V-only versus K+V and QK-rescue analysis.
- `analyze_bridge_compute_bypass.py`: semantic-shadow and compute-bypass analysis.
- `analyze_native_v_retention_sweep.py`: Native-V retention curve analysis.
- `design/attnbridge_method_and_exactness.md`: method and exactness notes.
- `design/bridgekv_exact_speculative_decoding.drawio`: editable method diagram.

## Environment

Install the original SpecExtend dependencies:

```bash
cd exp/specextend_bridgekv/specextend
pip install -r requirements.txt
```

Model checkpoints, mapper checkpoints, GovReport inputs, and raw experiment outputs are intentionally not committed. Pass their locations through the runner arguments. The 4090 sweep launcher takes all machine-specific locations from environment variables:

```bash
SPECEXTEND_BASE_MODEL=/path/to/target-model \
SPECEXTEND_DRAFT_MODEL=/path/to/draft-model \
SPECEXTEND_MAPPER_CHECKPOINT=/path/to/mapper.pt \
SPECEXTEND_SOURCE_FILE=/path/to/govreport.jsonl \
./run_native_v_retention_sweep_4090.sh 0 12 8
```

`SPECEXTEND_REPO_DIR`, `SPECEXTEND_PYTHON`, and `SPECEXTEND_SWEEP_OUTPUT` can additionally override the package directory, Python executable, and output directory.

Useful entry points:

```bash
python attn_bridge_experiment.py --help
python native_v_retention_sweep.py --help
python baseline_profiler.py --help
```

Analysis programs live one directory above the package and accept raw JSON bundles through their command-line interfaces.

## Provenance and scope

- Upstream SpecExtend base commit: [`a307ace`](https://github.com/jycha98/SpecExtend/commit/a307ace6960a8bcf39f919b2d8fdcac587fa444a).
- Snapshot date: 2026-10-12.
- This directory contains code and design documentation only. It excludes datasets, model weights, mapper checkpoints, raw results, logs, caches, and credentials.
- `UPSTREAM_SPECEXTEND_README.md` preserves the upstream project documentation; `REPRODUCTION.md` records the local reproduction workflow and findings.
