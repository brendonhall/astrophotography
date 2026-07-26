#!/usr/bin/env bash
# ccdenoise.sh — AI denoise an image with the native Cosmic Clarity CLI.
#
# Wraps the standalone Cosmic Clarity denoise install so it works on arbitrary
# input/output paths instead of its fixed input/ and output/ folders. This runs
# Cosmic Clarity in a normal arm64 venv (GPU via MPS), bypassing the SASpro
# frozen-app PyTorch bug. See README "Cosmic Clarity denoise" for the install.
#
# Usage:
#   tools/ccdenoise.sh <input-image> <output-image> [strength] [mode]
#     strength   0..1                          (default 0.5)
#     mode       luminance | full | separate   (default luminance)
#                'full' also denoises chroma (color) noise.
#
# The input must be a STRETCHED (nonlinear) image — tif/fits/png. The CLI has no
# linear flag. Never overwrites an existing output (keep-image-versions rule).
#
# Override the install location with COSMIC_CLARITY_DIR (default ~/CosmicClarity).

set -euo pipefail

CC_DIR="${COSMIC_CLARITY_DIR:-$HOME/CosmicClarity}"
CC_PY="$CC_DIR/.venv/bin/python"
CC_SCRIPT="$CC_DIR/setiastrocosmicclarity_denoisemac.py"

IN="${1:?Usage: ccdenoise.sh <input> <output> [strength] [mode]}"
OUT="${2:?Usage: ccdenoise.sh <input> <output> [strength] [mode]}"
STRENGTH="${3:-0.5}"
MODE="${4:-luminance}"

[[ -f "$IN" ]]        || { echo "error: input '$IN' not found" >&2; exit 1; }
[[ -e "$OUT" ]]       && { echo "error: output '$OUT' exists — choose another name" >&2; exit 1; }
[[ -x "$CC_PY" ]]     || { echo "error: Cosmic Clarity venv not at $CC_PY (set COSMIC_CLARITY_DIR)" >&2; exit 1; }
[[ -f "$CC_SCRIPT" ]] || { echo "error: denoise script not at $CC_SCRIPT" >&2; exit 1; }

# Cosmic Clarity reads a FIXED input/ and writes output/ next to its script.
# Stage a clean copy there (this clears any prior staging).
STAGE_IN="$CC_DIR/input"; STAGE_OUT="$CC_DIR/output"
mkdir -p "$STAGE_IN" "$STAGE_OUT"
rm -f "$STAGE_IN"/* "$STAGE_OUT"/*
base="$(basename "$IN")"
cp "$IN" "$STAGE_IN/$base"

echo ">> Cosmic Clarity denoise: strength=$STRENGTH mode=$MODE  ($base)"
( cd "$CC_DIR" && "$CC_PY" "$CC_SCRIPT" --denoise_strength "$STRENGTH" --denoise_mode "$MODE" ) \
  2>&1 | grep -iE "saved|error|traceback|device|final bit" || true

# Result is "<stem>_denoised.<ext>" in the stage output; glob so we don't guess ext.
stem="${base%.*}"
RESULT="$(ls "$STAGE_OUT/${stem}_denoised".* 2>/dev/null | head -1 || true)"
if [[ -n "$RESULT" && -f "$RESULT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  cp "$RESULT" "$OUT"
  echo ">> Wrote: $OUT"
else
  echo "error: no denoised result produced — check the log above" >&2
  exit 1
fi
