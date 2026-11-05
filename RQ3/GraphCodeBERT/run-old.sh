#!/usr/bin/env bash
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
NUMBER="${2:-3}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
MODEL="${MODEL_NAME_OR_PATH:-microsoft/graphcodebert-base}"
OUT="${OUTPUT_DIR:-$HERE/saved_models/OldDataSet_$NUMBER}"
"${PYTHON:-python}" "$HERE/graphcodebert_main.py" \
    --do_train --do_eval --model_name_or_path "$MODEL" \
    --dataset old --output_dir "$OUT" \
    --max_source_length 512 --max_target_length 64 --query_length 256 \
    --beam_size 1 --train_batch_size 8 --eval_batch_size 8 \
    --learning_rate 5e-5 --gradient_accumulation_steps 4 \
    --num_train_epochs 20 --passage_number "$NUMBER" --GPU_ids "$GPU" "$@"
"${PYTHON:-python}" "$HERE/graphcodebert_main.py" \
    --do_test --model_name_or_path "$MODEL" --dataset old --output_dir "$OUT" \
    --max_source_length 512 --max_target_length 64 --query_length 256 \
    --beam_size 1 --test_batch_size 4 --GPU_ids "$GPU" "$@"
