#!/usr/bin/env bash
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$HERE/calculate.py" --dataset old --gold_file "$HERE/OldDataSet/test.gold" --result_file "$HERE/OldDataSet/test.output" --output_file "$HERE/OldDataSet/per_type.csv" "$@"

