#!/usr/bin/env bash
# Run the pipeline via the flow executor (built-in linear/starless graphs).
#
# Usage: scripts/run_pipeline.sh <input.fit> [version-label] [--starless]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "No .venv found. Run 'make setup' (or see README.md) first." >&2
  exit 1
fi

STARLESS=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --starless) STARLESS=1 ;;
    *) ARGS+=("$a") ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

IN="${1:?usage: run_pipeline.sh <input.fit> [version-label] [--starless]}"
case "$IN" in /*) ;; *) IN="$PWD/$IN" ;; esac
LABEL="${2:-$(date +%Y%m%d-%H%M%S)}"

FLOW=linear
[[ "$STARLESS" == "1" ]] && FLOW=starless

cd "$ROOT"                      # so output/ and work/ resolve at repo root
echo ">> flow run --builtin $FLOW"
PYTHONPATH="$HERE" exec "$PY" -m flow run --builtin "$FLOW" --input "$IN" --label "$LABEL"
