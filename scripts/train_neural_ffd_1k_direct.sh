#!/usr/bin/env bash
# Direct-residual encoder experiment on the same 1K split, followed by holdout evaluation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

HEARTFFD_PYTHON="${HEARTFFD_PYTHON:-/home/yilong/.conda/envs/4dheart/bin/python}"
TRAIN_PROFILE="${TRAIN_PROFILE:-diagnostic}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
case "$TRAIN_PROFILE" in
  diagnostic|long) ;;
  *) echo "TRAIN_PROFILE must be diagnostic or long" >&2; exit 2 ;;
esac
RUN_OUTPUT="${RUN_OUTPUT:-outputs/neural_ffd_1k_direct_${TRAIN_PROFILE}_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$RUN_OUTPUT"
if [[ -z "$RESUME_CHECKPOINT" && ( -e "$RUN_OUTPUT/last.pth" || -e "$RUN_OUTPUT/history.jsonl" ) ]]; then
  echo "Fresh training requires a new RUN_OUTPUT directory: $RUN_OUTPUT" >&2
  exit 2
fi

TRAIN_CONFIG="configs/neural_ffd_1k_direct.yaml"
VAL_CASES=32
RESUME_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_CONFIG="$RUN_OUTPUT/config.yaml"
  if [[ ! -f "$TRAIN_CONFIG" || ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "Resume requires the saved config.yaml and checkpoint" >&2
    exit 2
  fi
  VAL_CASES="$("$HEARTFFD_PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["training"]["validation_cases"])' "$TRAIN_CONFIG")"
  RESUME_ARGS=(--resume "$RESUME_CHECKPOINT")
elif [[ "$TRAIN_PROFILE" == long ]]; then
  TRAIN_CONFIG="$RUN_OUTPUT/config.yaml"
  VAL_CASES=100
  "$HEARTFFD_PYTHON" - "$TRAIN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path('configs/neural_ffd_1k_direct.yaml').read_text())
config['training'].update(epochs=180, shape_epochs=30, motion_epochs=30,
                          validation_cases=100, joint_min_learning_rate=1e-5,
                          save_best=True, output=str(Path(sys.argv[1]).parent))
Path(sys.argv[1]).write_text(yaml.safe_dump(config, sort_keys=False))
PY
fi
echo "Training profile=$TRAIN_PROFILE | physical GPU=$CUDA_VISIBLE_DEVICES | output=$RUN_OUTPUT"

"$HEARTFFD_PYTHON" scripts/prepare_neural_ffd_1k.py | tee "$RUN_OUTPUT/split.log"
"$HEARTFFD_PYTHON" -u train_neural_ffd.py \
  --config "$TRAIN_CONFIG" \
  --device cuda:0 --exact \
  --train-workers 4 --train-prefetch 2 \
  --eval-workers 4 --eval-cache-cases "$VAL_CASES" --val-cases "$VAL_CASES" \
  --output "$RUN_OUTPUT" "${RESUME_ARGS[@]}" 2>&1 | tee -a "$RUN_OUTPUT/train.log"

# Select on validation only. TEST is read after all training is complete.
EVAL_CHECKPOINT="$RUN_OUTPUT/last.pth"
if [[ "$TRAIN_PROFILE" == long ]]; then
  EVAL_CHECKPOINT="$RUN_OUTPUT/best_joint.pth"
fi
for SPLIT in val test; do
  "$HEARTFFD_PYTHON" -u train_neural_ffd.py \
    --config "$TRAIN_CONFIG" \
    --device cuda:0 --evaluate "$EVAL_CHECKPOINT" \
    --split "$SPLIT" --val-cases 100 --exact \
    --eval-workers 4 --eval-cache-cases 100 \
    --output "$RUN_OUTPUT/final_$SPLIT" \
    2>&1 | tee "$RUN_OUTPUT/final_$SPLIT.log"
done
