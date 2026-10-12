#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU SAMPLE_START SAMPLE_COUNT" >&2
  exit 2
fi

physical_gpu="$1"
sample_start="$2"
sample_count="$3"
sample_stop=$((sample_start + sample_count - 1))

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${SPECEXTEND_REPO_DIR:-$script_dir/specextend}"
python_bin="${SPECEXTEND_PYTHON:-python3}"
output_dir="${SPECEXTEND_SWEEP_OUTPUT:-results/native_v_retention_sweep}"
base_model_path="${SPECEXTEND_BASE_MODEL:?Set SPECEXTEND_BASE_MODEL to the target model path}"
draft_model_path="${SPECEXTEND_DRAFT_MODEL:?Set SPECEXTEND_DRAFT_MODEL to the draft model path}"
checkpoint="${SPECEXTEND_MAPPER_CHECKPOINT:?Set SPECEXTEND_MAPPER_CHECKPOINT to the mapper checkpoint path}"
source_file="${SPECEXTEND_SOURCE_FILE:?Set SPECEXTEND_SOURCE_FILE to the evaluation JSONL path}"

cd "$repo_dir"
mkdir -p "$output_dir"
export CUDA_VISIBLE_DEVICES="$physical_gpu"

output_file=$(printf \
  "%s/native_v_retention_docs%02d_%02d.json" \
  "$output_dir" "$sample_start" "$sample_stop")

exec "$python_bin" -u native_v_retention_sweep.py \
  --base_model_path "$base_model_path" \
  --draft_model_path "$draft_model_path" \
  --device_map auto \
  --checkpoint "$checkpoint" \
  --source_file "$source_file" \
  --sample_start "$sample_start" \
  --samples "$sample_count" \
  --lengths 4096 8192 \
  --retention_percentages 0 12.5 25 37.5 50 62.5 75 87.5 100 \
  --max_gen_len 128 \
  --tree_nodes 64 \
  --chunk_size 32 \
  --cache_top_k 32 \
  --retrieve_every 4 \
  --seed 542 \
  --timing_repeats 3 \
  --warmup_gen_len 128 \
  --output "$output_file"
