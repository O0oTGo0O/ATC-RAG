#!/usr/bin/env bash
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$HERE/calculate.py" --dataset new --gold_file "$HERE/NewDataSet/test.gold" --result_file "$HERE/NewDataSet/test.output" --output_file "$HERE/NewDataSet/per_type.csv" "$@"

