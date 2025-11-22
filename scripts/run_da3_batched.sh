#!/usr/bin/env bash
set -euo pipefail
# Usage:
# ./run_da3_batched.sh /mnt/c/data/da3_input /mnt/c/data/da3_output 10 336 da3-small mini_npz
INPUT_ROOT=${1:-/mnt/c/data/da3_input}
OUTPUT_ROOT=${2:-/mnt/c/data/da3_output}
BATCH_SIZE=${3:-10}
PROCESS_RES=${4:-336}
MODEL=${5:-da3-small}
EXPORT=${6:-mini_npz}

mkdir -p "$OUTPUT_ROOT"

# For each name folder in input, run process_batched.py
for indir in "$INPUT_ROOT"/*; do
  [ -d "$indir" ] || continue
  name=$(basename "$indir")
  outdir="$OUTPUT_ROOT/$name"
  mkdir -p "$outdir"

  echo "Running DA3 for $name -> $outdir"
  python process_batched.py \
    --input-dir "$indir" \
    --output-dir "$outdir" \
    --batch-size "$BATCH_SIZE" \
    --process-res "$PROCESS_RES" \
    --model "$MODEL" \
    --export "$EXPORT"

  echo "Finished $name"
done

echo "All done."
