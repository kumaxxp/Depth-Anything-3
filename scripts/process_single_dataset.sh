#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/mnt/c/data/camera}
OUT_ROOT=${2:-/mnt/c/data/da3_input}
NAME=${3}
STEP=${4:-5}

if [ -z "$NAME" ]; then
    echo "Usage: $0 <ROOT> <OUT_ROOT> <NAME> [STEP]"
    exit 1
fi

echo "Processing single dataset: $NAME"
echo "Root: $ROOT"
echo "Out: $OUT_ROOT"
echo "Step: $STEP"

cam0dir="$ROOT/${NAME}_cam0"
cam1dir="$ROOT/${NAME}_cam1"

if [ ! -d "$cam0dir" ]; then
    echo "Error: $cam0dir not found"
    exit 1
fi
if [ ! -d "$cam1dir" ]; then
    echo "Error: $cam1dir not found"
    exit 1
fi

outdir="$OUT_ROOT/$NAME"
mkdir -p "$outdir"

# Check for xy subfolder
if [ -d "$cam0dir/xy" ]; then img0dir="$cam0dir/xy"; else img0dir="$cam0dir"; fi
if [ -d "$cam1dir/xy" ]; then img1dir="$cam1dir/xy"; else img1dir="$cam1dir"; fi

echo "Source 0: $img0dir"
echo "Source 1: $img1dir"

# Collect files
mapfile -t cam0files < <(find "$img0dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) -printf '%f\n' | sort)

echo "Found ${#cam0files[@]} frames in cam0"

seqnum=0
for ((i=0;i<${#cam0files[@]};i+=STEP)); do
    fa=${cam0files[i]}
    fb=${fa}

    printf -v seq "%06d" "$seqnum"
    
    # Copy cam0
    if [ -f "$img0dir/$fa" ]; then
        cp -- "$img0dir/$fa" "$outdir/${seq}_A.jpg"
    else
        echo "Warning: missing source $img0dir/$fa" >&2
    fi
    seqnum=$((seqnum+1))

    # Copy cam1
    if [ -f "$img1dir/$fb" ]; then
        printf -v seq "%06d" "$seqnum"
        cp -- "$img1dir/$fb" "$outdir/${seq}_B.jpg"
        seqnum=$((seqnum+1))
    else
        echo "Warning: $img1dir/$fb not found" >&2
    fi
done

echo "Done. Output in $outdir"
