#!/usr/bin/env bash
# Direct-residual encoder experiment on the same 1K split, followed by holdout evaluation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

HEARTFFD_PYTHON="${HEARTFFD_PYTHON:-/home/yilong/.conda/envs/4dheart/bin/python}"
RUN_OUTPUT="${RUN_OUTPUT:-outputs/neural_ffd_1k_direct_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$RUN_OUTPUT"

"$HEARTFFD_PYTHON" scripts/prepare_neural_ffd_1k.py | tee "$RUN_OUTPUT/split.log"
"$HEARTFFD_PYTHON" -u train_neural_ffd.py \
  --config configs/neural_ffd_1k_direct.yaml \
  --device cuda:0 --exact \
  --train-workers 4 --train-prefetch 2 \
  --eval-workers 4 --eval-cache-cases 32 --val-cases 32 \
  --output "$RUN_OUTPUT" 2>&1 | tee "$RUN_OUTPUT/train.log"

# Test is not read during training or epoch validation. Evaluate final checkpoint.
for SPLIT in val test; do
  "$HEARTFFD_PYTHON" -u train_neural_ffd.py \
    --config configs/neural_ffd_1k_direct.yaml \
    --device cuda:0 --evaluate "$RUN_OUTPUT/last.pth" \
    --split "$SPLIT" --val-cases 100 --exact \
    --eval-workers 4 --eval-cache-cases 32 \
    --output "$RUN_OUTPUT/final_$SPLIT" \
    2>&1 | tee "$RUN_OUTPUT/final_$SPLIT.log"
done
