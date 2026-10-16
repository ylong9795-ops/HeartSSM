# NeuralFFDNet-based Sparse 2D Echocardiography to 4D Heart Mesh Reconstruction

The sole active reconstruction pipeline is **NeuralFFDNet, physical FFD,
Native slice evaluation, TTO and geometry safety**. Three-view A2C/A3C/A4C
Mask/SDF sequences produce Shape64 / Motion32 latents and fixed-topology
four-chamber meshes over 24 phases. Progressive physical B-spline FFD applies
Coarse → Medium → Fine reference deformations, followed by recurrent motion.

G1/G2 Geometry-aware networks, Stage S/S2 networks and their dedicated training
code/configurations were retired on 2026-10-16. Their committed source is
recoverable from the local reference `backup/main-before-g1-cleanup-20261016`;
historical results, checkpoints and experiment source snapshots remain in
place. This cleanup does not change NeuralFFDNet inference or FFD mathematics.
Shape solver design is a separate next step; no new architecture is introduced.

The original CMR pipeline (`train_ffd.py`, `net/ffdnet.py`, `net/unet.py`,
`heartssm_train.py`) remains for a separate cleanup and is outside the active
sparse echocardiography pipeline. The original HeartSSM work is
*Learning a dynamic four-chamber shape model of the human heart for 95,695 UK
Biobank participants*.

Use the project's existing Python environment for the commands below:

```bash
export PATH="/home/yilong/.conda/envs/4dheart/bin:$PATH"
```

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
`utils/slice_evaluation.py` pins the original `section_mask` function recorded
by the existing 1K generation plan (core SHA256 `4bd081a4...`). Its
`mesh.section` endpoint merging must not be replaced by the later bundled
topological-endpoint renderer: rare sections differ by a pixel. Prediction and
GT verification use the same original function; labels and differentiable
training losses are unchanged. Reports record the generator and function hashes.

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

For a longer fresh run on physical GPU 7, use the same launcher:

```bash
CUDA_VISIBLE_DEVICES=7 TRAIN_PROFILE=long \
RUN_OUTPUT="outputs/neural_ffd_1k_long_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)" \
bash scripts/train_neural_ffd_1k_direct.sh
```

The long profile writes its effective configuration to the new output folder,
using the same 800/100/100 split and architecture. It allocates **30 shape,
30 shape+motion, and 120 joint epochs**. Shape supervision remains active in
motion and joint; the reference path is not frozen. Adam uses 1e-4 during both
pretraining stages; joint cosine decay starts at 1e-4 and ends at 1e-5.
The rate is derived from the epoch, including on resume. All 100 validation
cases are evaluated every epoch and cached with a 100-case host cache.
Epoch terminal output retains the existing full-cycle Dice format; reference
frame Dice and learning rate are recorded in JSON only.

Long runs retain `last.pth`, `best_shape.pth`, `best_motion.pth`, `best_joint.pth`
and `best_dice.pth`, plus their validation scores in `best_metrics.json`.
The shape checkpoint minimizes reference RMSE. Motion and joint checkpoints
minimize `sqrt(reference_RMSE^2 + relative_motion_RMSE^2)`, so cancellation of
shape and motion errors in the predicted trajectory cannot improve this score.
The separate joint Dice checkpoint maximizes exact full-cycle Dice. Final
100-case validation and test evaluation use **best_joint.pth**, selected only
on validation geometry. A larger epoch budget does not establish convergence;
inspect the separate reference/motion validation trends. The launcher rejects
output folders that already contain training history or a final checkpoint.
No additional source config or launcher files are needed for this profile.
To resume a long run in its existing folder, set `TRAIN_PROFILE=long`,
`RUN_OUTPUT` to that folder and `RESUME_CHECKPOINT` to its `last.pth`.
The launcher reuses the saved config, appends the terminal log, and still
evaluates the validation-selected joint checkpoint after training completes.

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

Historical benchmark evidence is retained in experiment outputs. The auxiliary
benchmark runner was removed in an earlier cleanup.

`outputs/` retains active experiments and `data_check/`. Historical baseline
checkpoints/configs and consolidated engineering reports are in `outputs/archive/`.
Completed one-step smoke checkpoint directories were removed after preserving
summaries, gradient/timing evidence and test logs. The current test suite is
`tests/test_real_tto.py`, `tests/test_slice_supervision.py` and
`tests/test_neural_ffd_losses.py`; older auxiliary test runners were retired separately.
The parallel input/evaluation runtime owns process lifetimes and caches.

See the [configuration](configs/neural_ffd.yaml) and
[project status and historical evidence](PROJECT_STATUS_HANDOFF.md).
Existing full-cycle baseline results do not establish that all Shape/Motion
accuracy and continuous safety requirements are met.

## TTO and geometry safety

The [TTO implementation](REAL_TTO_IMPLEMENTATION.md) uses the frozen
NeuralFFDNet prior, Native observations, per-patient pose/Shape fitting and
locked motion fields. `tto/geometry_guard.py` retains `SourceAVCapPolicy` from
`utils/geometry_pair_policy.py`, actual mesh checks from
`utils/geometry_v2_checks.py`, and the pinned bundle geometry kernels for
discrete surfaces and continuous inter-frame checks. Missing backends remain
`UNVERIFIED` and reject admission. No G1 control certificate is applied to
NeuralFFDNet.

Dependencies are recorded in [tto/requirements.txt](tto/requirements.txt).
On this server, the existing isolated geometry packages can be enabled when
running the tests:

```bash
PYTHONPATH=outputs/real_tto_motion_locked_20261016/_geometry_site \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m unittest discover -s tests -v
```

Tests and imports verify only their exercised contracts; they do not certify
all meshes, patients, historical checkpoints or continuous neural deformations.

## Pose contract

The Encoder reads Mask/SDF, view/chamber availability and synchronized cycle
phases to produce Shape/Motion latents. Probe Pose, physical pixel grids and
GT meshes are used by supervision and evaluation, without an Encoder geometry
embedding or Pose-specific optimizer group.

Both main configs select `slice.target_mode: paired_soft_geometry_v1`:

\[
M_{t,v}^{GT}=\mathcal S(V_t^{GT},P_{t,v}),\qquad
\widehat M_{t,v}=\mathcal S(\widehat V_t,P_{t,v}).
\]

GT and predicted meshes use the same true Probe Pose, physical coordinate
system, pixel grid and differentiable slice operator. GT soft targets are
detached; slice gradients propagate through the predicted mesh to the network.
`pose_gt.probe_t` is converted to mm using the asset normalization; canonical
pixel spacing is in mm. `utils/slice_supervision.py`, `utils/slice_evaluation.py`
and `data/synthetic_4d.py` retain their existing geometry contracts. Joint
training retains Pose and pixel spacing in its batches; shape/motion stages
need only image observations and 3D supervision.

Hard Dice continues to use the original Probe R/t/s, source-native image grid
and original label renderer against saved native labels. GT native rerendering
must match those labels pixel for pixel. Soft GT targets never replace native
labels or change Hard Dice. `binary_native_v1` remains available for historical
saved configurations; resuming a run uses its identical saved config.

Both main configs now additionally select `supervision_grid: native_grid` and
`boundary_objective: paired_boundary_distance_v1`. Soft GT and prediction use
the source pixel centres and measured y/x spacing. The boundary term measures
symmetric distances between actual GT/predicted section curves in mm, with
arc-length samples at most one native pixel apart and a smooth zero at matching
curves. This avoids using a rasterised binary boundary as a force on exact GT.
Observed curve pieces are restricted to the native FOV. Missing predicted
sections receive a differentiable distance from observed GT boundary points to
the predicted triangle surface; matched empty cells remain explicit zero cells.

`MeshSliceSupervisor` returns `cell_indices[C,4]` (patient, phase slot, view,
chamber), differentiable `cell_losses[C]` and `cell_terms` vectors including
`boundary_mm` and `recovery_mm`. Case IDs, actual phases and source frame indices
are included when supplied. Consumers can inspect or differentiate individual
cells; `total` remains the configured aggregate for the existing trainer.
Joint collation retains native labels/spacing. No constrained Joint optimiser
or Shape/Motion error-budget protection is implemented by this change.

The new objective is explicitly versioned in the main configs. Historical
configs without `boundary_objective` retain `legacy_probability_v1` behaviour;
old saved experiments must not be described as using the new supervision.
The current suite includes `tests/test_slice_supervision.py` for stationary GT,
physical boundary directions, finite parameter steps and missing-section recovery.

Both main configs also select `loss.objective_mode: grouped_gt_boundary_v1`.
`NeuralFFDLoss` returns differentiable `L_S`, `L_M`, `L_O`, plus `total`:

| Objective | Active Joint components |
| --- | --- |
| `L_S` | Reference Shape + all-phase GT vertex Huber + GT edge/Laplacian/normal supervision + Shape control regularity |
| `L_M` | Relative GT displacement + cyclic velocity (including last→first) + GT volume + Motion control regularity |
| `L_O` | Equal-chamber mean of physical `boundary_mm` mean + TopK mean |

Each observed chamber's tail contains `ceil(0.25 * cell_count)` cells, at least
one, with tail coefficient 1.0. These are explicit starting settings, not tuned
accuracy thresholds. Unobserved chambers are reported and excluded from the
chamber average. This aggregation consumes `cell_terms.boundary_mm` directly,
independently of the slicer's Dice/BCE weights or aggregate `total`.

The per-cell residual is explicitly decomposed into `curve_mm`, `recovery_mm`
and `empty_target_mm`; their sum is `boundary_mm`. A matched Native empty cell
has zero residual. For an empty GT section with an observable spurious prediction,
a smooth half-space violation moves it toward the GT side of the plane, with
margin `min(native_half_pixel, GT_clearance)`. This avoids a sparse-pixel
occupancy surrogate losing its gradient. Observation-kind flags distinguish
curve matches, recovery, empty targets, matched empty and invisible GT boundaries.

The primary evaluation goal is **Native Exact Dice >= 0.98 for every valid
Cell**. Native boundary ASSD <= 0.5 mm and HD95 <= 1.0 mm are diagnostic goals,
not automatic checkpoint rejection rules. Exact reports include per-cell
values/flags, failing counts and the all-cell Dice result. Empty/empty has
Dice=1 and boundary diagnostics marked inapplicable; a single absent boundary
fails the boundary diagnostic. The frozen renderer is unchanged.

`observation_boundary_threshold_mm` stays `null`: uncalibrated numeric proxy
thresholds are rejected, including a direct 0.5 mm proxy cutoff. A future proxy
threshold requires TRAIN-only qualifying cells (Native Dice >= .98 and ASSD
<= .5 mm), separate view/chamber calibration and validated proxy ranking.
No proxy-threshold violation loss is active. TopK denotes largest residuals,
not Native Exact failures.

Curve quadrature is independent of Soft `pixel_stride=4`. Main configs cap the
arc-length step at 0.5 mm, tightening to 0.25 mm after 75% of Joint epochs;
actual steps never exceed native spacing. The epoch-derived schedule is
deterministic on resume and adds no model forward. A finer surrogate still
does not guarantee every Cell's discrete Exact Dice improves in one update.

Shape/Motion groups include their component weights; the combined objective is
`L_S + L_M + weights.slice * L_O`. Coarse, Medium, Fine0 and regional Shape
regularity belong to Shape; Fine1…23 and regional Motion regularity belong to
Motion. Their sum preserves the previous control regularizer's scale, and the
diagnostic aggregate `terms.control` has zero additional active weight.
All vertices, including unobserved regions, retain full GT supervision.
No template attraction or Unknown-vertex freeze is added.

The trainer logs each objective, per-chamber boundary mean/tail and GT-relative
motion amplitude/direction diagnostics, using the existing single forward and
backward. Historical configs default to `legacy_v1`; pre-Joint Shape/Motion
stage activation is retained. All-phase position loss is newly active in Joint.
Soft error-budget penalties and Cell subsampling remain separate work.

The two `neural_ffd_1k_known_pose*.yaml` research configs and Encoder Pose branch
were retired. Their source remains in Git history at `1999287c`; existing
experiment artifacts remain in `outputs/known_pose_regional_v1/`. Reproduce or
evaluate Pose-aware checkpoints with that revision: the current model does not
load their removed geometry weights. Mask/SDF-only checkpoints retain strict
loading. Fresh runs use the current main configs and paired slice targets.

`regional_motion`, its physical domains, losses, diagnostics and audited
initialization remain unchanged. Its reconstruction benefit will be assessed
separately after this cleanup. The [historical research report](docs/neural_ffd.rst)
records the earlier experiments and commands for their corresponding revisions.
