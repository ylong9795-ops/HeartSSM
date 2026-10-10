# HeartSSM
Learning a dynamic four-chamber shape model of the human heart for 95,695 UK Biobank participants

Neural HeartFFDNet adds three-view 2D mask/SDF sequences, shape/motion latents,
learned 3D features, physical multiscale FFD and four-chamber 4D supervision.
The original volume-based pipeline remains available unchanged.

```bash
python train_neural_ffd.py --config configs/neural_ffd.yaml
python -m unittest discover -s tests -v
python train_neural_ffd.py --evaluate outputs/neural_ffd/last.pth --exact
```

`--exact` renders the predicted mesh using each sample's original pose (R, t, s),
image size and source label renderer, then compares directly with the saved
original native labels. Evaluation performs no image warp or resampling.
The first GT check rejects any saved-label pixel mismatch; subsequent epochs
reuse verified GT only while its source, pose, topology and renderer match.
This requires the original `synthetic_bundle_20261009/data_generation/core.py`;
the differentiable training losses are unchanged.

Training uses tqdm progress bars and prints one compact summary after each
epoch's validation: mean training loss, validation shape/motion RMSE (mm),
and original-label Dice (`n/a` without `--exact`). Per-step loss details remain
in `history.jsonl`; epoch summaries are appended to `epoch_history.jsonl` and
full reports are archived under `validation/epoch_0001.json`, etc.

Use `--train-workers 4 --train-prefetch 2` to overlap source reading and SDF
preparation with training. CUDA training uses pinned memory and nonblocking
copies. Evaluation defaults to `--eval-workers 4 --eval-cache-cases 32`: bounded
host input caches, cached verified GT/boundaries/surface samples, and spawned
original-renderer workers overlap CPU evaluation with GPU inference. Caches
are rebuilt after restart; `--eval-workers 0` runs cached evaluation inline.
Worker/cache options do not change the checkpoint's model/training configuration.
Epoch records include reader wait and validation cache/timing measurements.

For a fresh diagnostic run on the existing **1K** cohort and physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_neural_ffd_1k_direct.sh
```

This single launcher prepares reproducible case-level **800 train / 100
validation / 100 test** manifests without modifying source NPZ, QC, poses or
labels. It uses `configs/neural_ffd_1k_direct.yaml` and
`configs/synthetic_data_1k.yaml`: 30 epochs (3 shape, 5 motion, 22 joint), four
training readers and four evaluation workers. Each epoch evaluates the same
32 validation cases; the final checkpoint is evaluated on all 100 validation
and 100 test cases. The split is a diagnostic holdout; this cohort also has a
historical 900/100 pilot split. Every launch starts a fresh model and timestamped
output directory. Override `RUN_OUTPUT` or `HEARTFFD_PYTHON` if needed.
The CLI defaults to the validated batched slice backend with 16 cells per group.
The source-file preparation prints a progress bar before training begins.
Outputs include `epoch_history.jsonl`, `history.jsonl`, `last.pth`,
`final_val/evaluation.json` and `final_test/evaluation.json`.

`configs/neural_ffd_1k_direct.yaml` enables `feature_transfer: direct_residual_v1`.
The existing layer2/layer3 projected CNN maps are each pooled to 2x2, flattened
and concatenated within each view, then across A2C/A3C/A4C in canonical order.
The resulting 3072D descriptor is projected to 128D and added to each frame
representation. The reference-frame direct feature also passes through a
bias-free 128-to-64 projection added after the shape head, covering the global
and all four regional shape codes. Missing/unannotated views are zeroed after
CNN processing; a missing reference frame contributes zero direct residual.
Both residual gains default to 1 and are fixed configuration values; setting
one gain to 0 supports ablation. `direct_pool_side: 4` supports a larger grid
comparison. Both branches reuse one CNN forward and run on the model device.

This adapts DeepMesh's per-view, multiscale concatenation from
[network_motion.py](https://github.com/ImperialCollegeLondon/DeepMesh/blob/4753673e56c0f35b5c920c2f52a534e1a60ff909/network_motion.py),
using spatial pooling instead of its upsampling and 3D feature conversion.
It supplies a direct frame-feature path into the existing temporal model;
[Laumer's echo_dhb.py](https://github.com/laumerf/4DHeartModel/blob/main/source/models/echo_dhb.py)
provides a CNN-flatten-to-temporal-model precedent. These sources motivate
the transfer path, but do not guarantee reconstruction accuracy or prevent
collapse. Test patient sensitivity and small-cohort fitting before judging it.
The FFD decoder, supervision and original-label evaluation use existing rules.
The read-only `EncoderDiagnostics` class is kept in `net/multiview_encoder.py`.
Validation records `encoder_diagnostics` in each evaluation JSON and in
`epoch_history.jsonl`: pooled RMS of attention/direct frame branches, time
embedding, main/direct Shape branches and final Shape latent, residual-to-main
ratios, observed-reference counts and between-case Shape latent RMS deviation.
Direct RMS includes its configured gain; absent frames are excluded from frame
RMS and absent references contribute zero direct Shape residual. These read-only
hooks reuse existing forwards and reduce on the model device, transferring only
summary scalars at the end of validation. Terminal epoch output remains compact.
The legacy mode remains the default for old configs/checkpoints. Start this
variant in a fresh output directory; do not resume a legacy training checkpoint
with the direct-mode config. Strict loading records the mode in encoder metadata.

For joint-stage acceleration, pass `--slice-backend batched --slice-cell-chunk 16`
to `train_neural_ffd.py`. This groups observed sections by chamber, packs their
intersection segments and batches nearest-segment distance/parity computations
on GPU. Every valid frame/view/chamber and the original stride, physical grid,
soft-boundary width, Dice/BCE/boundary weights and exact evaluator are retained.
Empty padded segments are excluded from both distance search and parity.
The CLI defaults to `batched`; use `--slice-backend serial` for comparison. These are execution options, so
an existing checkpoint can resume with the identical YAML, optimizer, split and
sample order; float32 reduction/matrix multiplication order can differ slightly.
Per-step `slice_execution` and epoch `training_performance` record the backend.
Network gradient checks and loss logging each transfer one batch summary.
Use a larger cell chunk (e.g. 32) only after checking timing and GPU memory.

`scripts/check_neural_ffd.py --benchmark-joint` compares the two executions on TRAIN,
reports loss/vertex-gradient differences, CUDA-synchronized forward/backward
times and peak allocated memory, and never updates the checkpoint or reads TEST:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/check_neural_ffd.py --benchmark-joint \
  --config configs/neural_ffd_1k_direct.yaml --checkpoint PATH/last.pth \
  --device cuda:0 --cases 2 --repeats 3 --output outputs/archive/joint_benchmark.json
```

`outputs/` retains active experiments and `data_check/`. Historical baseline
checkpoints/configs and consolidated engineering reports are in `outputs/archive/`.
Completed one-step smoke checkpoint directories were removed after preserving
summaries, gradient/timing evidence and test logs. Grouped-section regression
tests are included in `tests/test_neural_ffd.py`; the parallel input/evaluation
runtime remains a separate module because it owns process lifetimes and caches.

See [implementation and validation](docs/neural_ffd.rst),
[public-source review](docs/neural_ffd_sources.json), and
[configuration](configs/neural_ffd.yaml). Architecture checks do not establish
trained reconstruction accuracy; formal population training remains to be run.
