#!/usr/bin/env bash
# Run the full M101-style processing pipeline on a stacked FITS.
#
# Usage:
#   scripts/run_pipeline.sh <input.fit> [version-label]
#
# The final image is always written under a UNIQUE, versioned name
# (output/<input>_<label>.{tif,png}); if no label is given a timestamp is used,
# so runs never overwrite each other. Intermediate stage files land in work/.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "No .venv found. Run 'make setup' (or see README.md) first." >&2
  exit 1
fi

IN="${1:?usage: run_pipeline.sh <input.fit> [version-label]}"
# Resolve to an absolute path (we cd into scripts/ below, so relative paths break).
case "$IN" in /*) ;; *) IN="$PWD/$IN" ;; esac
LABEL="${2:-$(date +%Y%m%d-%H%M%S)}"
NAME="$(basename "${IN%.*}" | tr ' ' '_')"

WORK="$ROOT/work"; mkdir -p "$WORK" "$ROOT/output"
OUTBASE="$ROOT/output/${NAME}_${LABEL}"

cd "$HERE"
echo ">> 01 crop";       "$PY" 01_crop.py       "$IN"               "$WORK/01_crop.fit"
echo ">> 02 background"; "$PY" 02_background.py "$WORK/01_crop.fit" "$WORK/02_bg.fit"
echo ">> 03 color";      "$PY" 03_color.py      "$WORK/02_bg.fit"   "$WORK/03_color.fit"
echo ">> 04 stretch";    "$PY" 04_stretch.py    "$WORK/03_color.fit" "$WORK/04_stretch.fit"
echo ">> 05 finish";     "$PY" 05_finish.py     "$WORK/04_stretch.fit" "$OUTBASE"
echo ">> done -> output/${NAME}_${LABEL}.{tif,png}"
