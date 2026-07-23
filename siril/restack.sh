#!/usr/bin/env bash
# restack.sh — headless re-stack of SeeStar subframes via Siril.
#
# Registers and sigma-clip-stacks a folder of SeeStar "save each frame" subs
# into a single linear FITS in data/, ready to feed the pipeline (`make run`).
#
# Usage:
#   ./siril/restack.sh [--cfa] <sub-folder> [output-name]
#
#   <sub-folder>   folder containing the individual *.fit subframes
#   output-name    basename for the result (default: a timestamp)
#   --cfa          subs are raw Bayer (single-channel); debayer during convert.
#                  Omit for already-color subs (the usual SeeStar case).
#
# Examples:
#   ./siril/restack.sh "data/M 101-sub" M101_restacked
#   ./siril/restack.sh --cfa "data/M 101-sub"

set -euo pipefail

SIRIL="/Applications/Siril.app/Contents/MacOS/siril-cli"
SCRIPT="siril/restack_seestar.ssf"

if [[ "${1:-}" == "--cfa" ]]; then
  SCRIPT="siril/restack_seestar_cfa.ssf"
  shift
fi

SUBDIR="${1:?Usage: ./siril/restack.sh [--cfa] <sub-folder> [output-name]}"
NAME="${2:-restacked_$(date +%Y%m%d-%H%M%S)}"

[[ -d "$SUBDIR" ]]   || { echo "error: '$SUBDIR' is not a directory" >&2; exit 1; }
[[ -x "$SIRIL" ]]    || { echo "error: siril-cli not found at $SIRIL" >&2; exit 1; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/data/${NAME}.fit"
# Never overwrite an existing result (keep-image-versions rule).
[[ -e "$OUT" ]] && { echo "error: $OUT already exists — choose another output-name" >&2; exit 1; }

echo ">> Restacking subs in: $SUBDIR"
echo ">> Using script:       $SCRIPT"
"$SIRIL" -d "$SUBDIR" -s "$REPO/$SCRIPT"

# The .ssf writes restacked.fit into the sub-folder; move it into data/.
if [[ -f "$SUBDIR/restacked.fit" ]]; then
  mv "$SUBDIR/restacked.fit" "$OUT"
  echo ">> Wrote: $OUT"
  echo ">> Inspect it:  make inspect FITS=\"$OUT\""
  echo ">> Then run:    make run FITS=\"$OUT\" V=restack"
else
  echo "error: Siril did not produce restacked.fit — check the log above" >&2
  exit 1
fi
