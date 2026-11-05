#!/usr/bin/env bash
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/." && pwd)"
if [[ "${1:-}" != --* && -n "${1:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$1"
    shift
fi
K=3
if [[ "${1:-}" != --* && -n "${1:-}" ]]; then
    K="$1"
    shift
fi
exec "${PYTHON:-python}" "$ROOT/run.py" --dataset new --experiment full --k "$K" "$@"

