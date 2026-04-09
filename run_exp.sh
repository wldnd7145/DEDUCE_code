#!/bin/bash
# Custom DEDUCE variants (full 10 tasks, seq-cifar100, seed=0)
#
# Available models:
#   exp_a_lum_proj            : DEDUCE LUM + gradient projection at learning step
#                               (CIL 43.9, TIL 80.1, BWT -28.2)
#   exp_fisher_multitask      : Replace LUM with Fisher-weighted multi-task update
#                               (weak selectivity c=0.001)
#                               (CIL ~39.6, TIL ~80.2, BWT ~-45.3)
#   exp_fisher_multitask_strong: Same with strong selectivity c=0.1
#                               (CIL ~37.3, TIL ~77.4, BWT ~-54.4)
#   exp_a_no_ewc              : Exp A without EWC penalty
#   exp_a_conflict_mask       : Exp A with per-parameter conflict mask
#   exp_g_conflict_mask       : DEDUCE LUM with conflict mask instead of Fisher
#   exp_b_contrastive         : Replace LUM with contrastive loss
#   exp_c_dropout_gum         : Replace GUM neuron reinit with stochastic dropout
#   exp_cumul_conflict        : Cumulative conflict score weighting
#   exp_layerwise_conflict    : Layer-wise conflict intensity
#
# Usage:
#   ./run_exp.sh exp_a_lum_proj           # default GPU 0
#   CUDA_VISIBLE_DEVICES=1 ./run_exp.sh exp_fisher_multitask
#   ./run_exp.sh exp_a_lum_proj 4         # stop after 4 tasks

set -e
MODEL=${1:?"Usage: $0 <model_name> [stop_after]"}
STOP_AFTER=${2:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/DEDUCE_code"
export PYTHONPATH="$SCRIPT_DIR/DEDUCE_code:$PYTHONPATH"

mkdir -p ../logs

EXTRA=""
if [ -n "$STOP_AFTER" ]; then
  EXTRA="--stop_after $STOP_AFTER"
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -u utils/main.py \
  --model "$MODEL" \
  --dataset seq-cifar100 \
  --buffer_size 500 \
  --lr 0.03 \
  --alpha 0.1 \
  --beta 0.5 \
  --batch_size 32 \
  --minibatch_size 32 \
  --n_epochs 50 \
  --seed 0 \
  $EXTRA \
  2>&1 | tee "../logs/${MODEL}_seed0.txt"
