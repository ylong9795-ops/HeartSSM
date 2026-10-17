#!/usr/bin/env bash
# Complete 1K random-init population training; run the pinned source snapshot.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
HEARTFFD_PYTHON="${HEARTFFD_PYTHON:-/home/yilong/.conda/envs/4dheart/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Override inherited NAS scratch before ANY Python/spawned worker starts.
# The same environment fix also applies when resuming an older source snapshot.
export HEARTFFD_TMPDIR="${HEARTFFD_TMPDIR:-/tmp/heartssm-${UID}}"
mkdir -p -- "$HEARTFFD_TMPDIR"
export TMPDIR="$HEARTFFD_TMPDIR" TEMP="$HEARTFFD_TMPDIR" TMP="$HEARTFFD_TMPDIR"
export PYTHONPATH="${GEOMETRY_SITE:-$PWD/outputs/audit_result/real_tto_motion_locked_20261016/_geometry_site}${PYTHONPATH:+:$PYTHONPATH}"
RUN_OUTPUT="${RUN_OUTPUT:-outputs/population_ffd_1k_fromscratch_$(date +%Y%m%d_%H%M%S)}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
CNN_CHUNK_SIZE="${CNN_CHUNK_SIZE:-144}"
GEOMETRY_FINAL="${GEOMETRY_FINAL:-1}"
GEOMETRY_WORKERS="${GEOMETRY_WORKERS:-8}"
case "$GEOMETRY_FINAL" in 0|1) ;; *) echo 'GEOMETRY_FINAL must be 0 or 1' >&2; exit 2 ;; esac
RUN_OUTPUT="$("$HEARTFFD_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$RUN_OUTPUT")"
ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  RESUME_CHECKPOINT="$("$HEARTFFD_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$RESUME_CHECKPOINT")"
  [[ -f "$RUN_OUTPUT/config.yaml" && -f "$RESUME_CHECKPOINT" && -f "$RUN_OUTPUT/source/train_population_ffd.py" ]] || {
    echo 'Resume needs the original RUN_OUTPUT, source snapshot and checkpoint' >&2; exit 2;
  }
  ARGS=(--resume "$RESUME_CHECKPOINT")
else
  if [[ -d "$RUN_OUTPUT" && -n "$(ls -A "$RUN_OUTPUT")" ]]; then
    echo "Fresh training requires a new RUN_OUTPUT: $RUN_OUTPUT" >&2; exit 2
  fi
  "$HEARTFFD_PYTHON" - "$PWD" "$RUN_OUTPUT" "${BATCH_SIZE:-4}" <<'PY'
import hashlib, json, os, shutil, sys
from pathlib import Path
import yaml
root, output = Path(sys.argv[1]), Path(sys.argv[2])
snapshot = output/'source'
snapshot.mkdir(parents=True)
for folder in ('net','utils','data','tto','scripts','tests'):
    for path in (root/folder).rglob('*.py'):
        target = snapshot/path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path,target)
for path in root.glob('*.py'):
    shutil.copy2(path,snapshot/path.name)
shutil.copytree(root/'configs',snapshot/'configs')
for folder in ('synthesis_data','synthetic_bundle_20261009'):
    (snapshot/folder).symlink_to(root/folder, target_is_directory=True)
config = yaml.safe_load((root/'configs/neural_ffd_1k_refiner.yaml').read_text())
config['training'].update(batch_size=int(sys.argv[3]),output=str(output))
if config['training']['batch_size'] < 1: raise ValueError('Positive BATCH_SIZE required')
rendered = yaml.safe_dump(config,sort_keys=False)
(output/'config.yaml').write_text(rendered)
(snapshot/'configs/population_run.yaml').write_text(rendered)
hashes = {str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in snapshot.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json')}
(output/'launch.json').write_text(json.dumps(dict(
    initialization='random; no inherited network checkpoint',physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
    logical_device='cuda:0',split=dict(train=800,val=100,test=100),
    schedule=dict(shape=30,motion=30,joint=120),source_sha256=hashes,
    training_observation_cache=False),indent=2)+'\n')
PY
fi
SOURCE="$RUN_OUTPUT/source"
echo "Base+Refiner from scratch | GPU=$CUDA_VISIBLE_DEVICES | output=$RUN_OUTPUT | tmp=$TMPDIR"
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  echo "Prepared source snapshot/config only; training has not started."
  exit 0
fi
"$HEARTFFD_PYTHON" "$SOURCE/scripts/prepare_neural_ffd_1k.py" | tee -a "$RUN_OUTPUT/split.log"
cd -- "$SOURCE"
"$HEARTFFD_PYTHON" -u train_population_ffd.py \
  --config configs/population_run.yaml --output "$RUN_OUTPUT" --device cuda:0 \
  --memory-profile throughput --cnn-chunk-size "$CNN_CHUNK_SIZE" \
  --train-workers 4 --train-prefetch 2 --eval-workers 4 --eval-cache-cases 100 \
  --val-cases 100 --exact "${ARGS[@]}" 2>&1 | tee -a "$RUN_OUTPUT/train.log"
# The full run has finished. Choose on VAL tails; only now read TEST samples.
CHECKPOINT="$RUN_OUTPUT/best_tail.pth"
for SPLIT in val test; do
  "$HEARTFFD_PYTHON" -u train_population_ffd.py \
    --config configs/population_run.yaml --evaluate "$CHECKPOINT" --split "$SPLIT" \
    --device cuda:0 --val-cases 100 --exact --eval-workers 4 --eval-cache-cases 100 \
    --output "$RUN_OUTPUT/final_$SPLIT" 2>&1 | tee "$RUN_OUTPUT/final_$SPLIT.log"
  if [[ "$GEOMETRY_FINAL" == 1 ]]; then
    "$HEARTFFD_PYTHON" -u scripts/audit_population_ffd_geometry.py \
      --checkpoint "$CHECKPOINT" --split "$SPLIT" --device cuda:0 \
      --workers "$GEOMETRY_WORKERS" --output "$RUN_OUTPUT/final_$SPLIT/geometry" \
      2>&1 | tee "$RUN_OUTPUT/final_${SPLIT}_geometry.log"
  fi
done
