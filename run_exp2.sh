#!/bin/bash
# Exp 2: DEDUCE with Fisher Multi-task LUM (weak selectivity, c=0.001)
# Expected: CIL 36.96, TIL 76.36, BWT -53.5 (10 tasks, seq-cifar100, seed=0)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/DEDUCE_code"
export PYTHONPATH="$SCRIPT_DIR/DEDUCE_code:$PYTHONPATH"

mkdir -p ../logs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -u utils/main.py \
  --model exp_fisher_multitask \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --lr 0.03 \
  --alpha 0.1 \
  --beta 0.5 \
  --batch_size 32 \
  --minibatch_size 32 \
  --n_epochs 50 \
  --seed 0 \
  2>&1 | tee ../logs/exp2_seed0.txt
