#!/bin/bash
# DER++ with DEDUCE (official code, paper reproduction)
# Expected: CIL ~44.4, TIL ~79.8, BWT ~-28.2 (10 tasks, seq-cifar100, seed=0)
# Paper Table 1 reports: CIL ~39.8, TIL ~78.4

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/DEDUCE_code"
export PYTHONPATH="$SCRIPT_DIR/DEDUCE_code:$PYTHONPATH"

mkdir -p ../logs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -u utils/main.py \
  --model derpp \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --load_best_args \
  --seed 0 \
  2>&1 | tee ../logs/deduce_seed0.txt
