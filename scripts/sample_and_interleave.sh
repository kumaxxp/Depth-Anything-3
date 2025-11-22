#!/usr/bin/env bash
set -euo pipefail
# Usage:
# ./sample_and_interleave.sh /mnt/c/data/camera /mnt/c/data/da3_input 5
# This will scan for pairs like <name>_cam0 and <name>_cam1 under the root,
# sample every STEP frames from cam0, and copy cam0/cam1 frames into an
# interleaved folder per-name under the output root.

ROOT=${1:-/mnt/c/data/camera}
OUT_ROOT=${2:-/mnt/c/data/da3_input}
STEP=${3:-5}

mkdir -p "$OUT_ROOT"

echo "root: $ROOT"
echo "out root: $OUT_ROOT"
echo "step: $STEP"

shopt -s nullglob

for cam0dir in "$ROOT"/*_cam0; do
  base=$(basename "$cam0dir" | sed 's/_cam0$//')
  cam1dir="$ROOT/${base}_cam1"
  if [ ! -d "$cam1dir" ]; then
    echo "Skipping $base: no corresponding _cam1 directory" >&2
    continue
  fi

  outdir="$OUT_ROOT/$base"
  mkdir -p "$outdir"
  echo "Processing $base -> $outdir"

    # if images are under an "xy" subfolder, prefer that; otherwise use top-level
    if [ -d "$cam0dir/xy" ]; then
      img0dir="$cam0dir/xy"
    else
      img0dir="$cam0dir"
    fi
    if [ -d "$cam1dir/xy" ]; then
      img1dir="$cam1dir/xy"
    else
      img1dir="$cam1dir"
    fi

    # collect only regular image files (case-insensitive), list basenames sorted
    mapfile -t cam0files < <(find "$img0dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) -printf '%f\n' | sort)

    seqnum=0
    for ((i=0;i<${#cam0files[@]};i+=STEP)); do
      fa=${cam0files[i]}
      fb=${fa}

      printf -v seq "%06d" "$seqnum"
      if [ -f "$img0dir/$fa" ]; then
        cp -- "$img0dir/$fa" "$outdir/${seq}_A.jpg"
      else
        echo "Warning: missing source $img0dir/$fa" >&2
      fi
      seqnum=$((seqnum+1))

      if [ -f "$img1dir/$fb" ]; then
        printf -v seq "%06d" "$seqnum"
        cp -- "$img1dir/$fb" "$outdir/${seq}_B.jpg"
        seqnum=$((seqnum+1))
      else
        echo "Warning: $img1dir/$fb not found" >&2
      fi
    done
done

echo "Done. Interleaved inputs under $OUT_ROOT"
