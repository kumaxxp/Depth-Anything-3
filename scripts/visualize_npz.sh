#!/usr/bin/env bash
set -euo pipefail
# Usage:
# ./visualize_npz.sh /mnt/c/data/da3_output
OUTROOT=${1:-/mnt/c/data/da3_output}

for d in "$OUTROOT"/*; do
  [ -d "$d" ] || continue
  for npz in "$d"/*.npz; do
    [ -f "$npz" ] || continue
    echo "Visualizing $npz"
    python view_npz.py "$npz" || echo "view_npz failed for $npz"
  done
done
echo "Visualization done."
