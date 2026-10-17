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

## Physical observation residual features

`utils/observation_residual.py` connects true mesh-plane sections, physical
arc-length quadrature and the observed model-grid SDF. It returns packed world
positions and in-plane correction vectors in mm, signed distances, raw SDF
gradients/norms, FOV validity, boolean confidence and arc-length weights, with
patient/frame/view/chamber indices and actual cycle phases.

```python
from utils.observation_residual import ObservationResidualExtractor

extractor = ObservationResidualExtractor(chamber_ranges, faces, norm_to_mm)
residual = extractor(vertices_base_mm, observations, pose=observations["pose_gt"])
# Real inference supplies an estimated pose using the same probe_R/probe_t contract.
```

Pass the model grid's `pixel_spacing` together with `sdf_units="mm"` and
`MaskTransform(sdf_normalize=False)`. Pose is explicit; `probe_t` is multiplied
once by `norm_to_mm`. Missing sections are flagged with empty point ranges;
empty/full masks, flat gradients and out-of-FOV samples supply no confident
direction. Raster EDT can have pixel bias even for a GT rerender, so this
feature extractor does not replace exact geometric supervision or acceptance.
It contains no learned parameters, FFD aggregation or Refiner training path.

Implementation mechanisms were checked against DeepMesh's
[physical coordinate mapping and contour supervision](https://github.com/ImperialCollegeLondon/DeepMesh/blob/main/train_motion.py)
and [projection helper](https://github.com/ImperialCollegeLondon/DeepMesh/blob/main/utils.py).
Its probabilistic vertex projection is different from the true-section SDF
operator implemented here; no DeepMesh code is copied.

`utils/ffd_observation_splat.py` maps these hints to the original Fine lattice
through the SAME `bspline_support_physical` kernel as FFD evaluation. Each
point votes to 64 nodes using `index_add_`; physical arc length and confidence
weight both the vector numerator and support denominator. Four chamber blocks
remain separate in `[B,T,16,Gx,Gy,Gz]` (LV, LA, RV, RA; each dx, dy, dz,
support). For 24 phases and the original Fine grid this is
`[B,24,16,24,24,32]`; 16 is a feature channel count, not a Motion latent size.
Vectors are clipped by physical norm; invalid/nonfinite/unsupported samples
have no vote. Support is accumulated evidence, not a probability.

```python
from utils.ffd_observation_splat import build_missing_section_recovery, splat_observation_residuals

recovery = build_missing_section_recovery(
    vertices_base_mm, observations, residual, pose=observations["pose_gt"],
    chamber_ranges=chamber_ranges, faces=faces, norm_to_mm=norm_to_mm)
packed = splat_observation_residuals(
    residual, model.control_origins_mm[2], model.control_spacings_mm[2], model.grid_shapes[2],
    batch_size=vertices_base_mm.shape[0], frames=vertices_base_mm.shape[1], recovery=recovery)
features = packed["features"]
```

Recovery uses observed model-grid SDF zero crossings and the current predicted
chamber surface. Detached nearest-surface anchors receive low-confidence
vectors toward the observed boundary, including normal-plane capture. It uses
no GT vertices or patient optimisation. Missing observed sections require this
explicit recovery result; `cell_needs_recovery` and `cell_recovery_available`
distinguish absent evidence from a usable capture hint. Recovery probes use
approximate edge weights with orientation bias, rather than true-curve arc
length. These averaged features are not an analytic inverse FFD solution.
The shared Refiner and composed forward path are implemented below; cohort
training and full geometric acceptance are separate from these module tests.

The accumulation/normalisation mechanism was checked against
[PyTorch3D points_to_volumes.py](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/points_to_volumes.py);
its trilinear voxel kernel is replaced by HeartSSM's existing physical cubic
B-spline support. PyTorch3D is not added as a dependency.

## Shared observation-driven FFD Refiner

`net/joint_ffd_refiner.py` provides `ObservationFFDRefiner` (the shared network)
and `JointFFDRefiner` (the complete composed forward path). The default network
has **15,150 parameters**, Shape64/Motion32 conditions and hidden width 24.
A grouped four-chamber input stem precedes learned fusion, two depthwise 3D
residual blocks and a circular temporal Conv1d at each control node. It keeps
the original Fine resolution and predicts one common world displacement field
for the fixed-topology four-chamber mesh. No patient IDs, patient parameters
or per-patient optimisation enter the network.

```python
from net.joint_ffd_refiner import JointFFDRefiner

# base_model is an existing NeuralFFDNet with its checkpoint already loaded.
joint = JointFFDRefiner(base_model)
output = joint(observations, pose=observations["pose_gt"])
vertices_final_mm = output["vertices_4d"]
controls = output["refinement"]
# controls["controls_res_mm"]: [B,24,24,24,32,3] for the original Fine setup.
```

The wrapper reads the base model's Fine origin, spacing and shape; it executes
base prediction -> physical section residual -> chamber-separated splat ->
shared Refiner -> `bspline_ffd_physical` at the CURRENT base vertices.
Real inference must explicitly supply an estimated pose with the same units
as the synthetic pose interface. No GT vertex fields are read by refinement.
Zero output heads make the initial final mesh exactly equal to the base mesh.
A patient with zero observation support receives zero correction even after
training. The caller chooses whether to freeze base weights; gradients through
the base and the ordinary observation path remain connected.

The dynamic head predicts uniform periodic cubic B-spline coefficients in
time. For raw static field S and evaluated dynamic field D(t), the outputs are
`C(t)=cap(S+D(t)-D(0))`, `C_static=cap(S)` and
`C_dynamic(t)=C(t)-C_static`. Thus reference dynamic correction is exactly zero,
and the TOTAL control vector norm is smoothly limited to the configurable
default 3 mm. The relative dynamic field alone can exceed that limit.
`cap(v)=v/sqrt(1+sum((v/M)**2))` retains an identity derivative at zero, so a
zero-initialised head can learn. The first update reaches the output heads;
subsequent updates reach the feature network.

`evaluate_residual_controls` evaluates returned `raw_static_mm` and
`dynamic_coefficients_mm` at arbitrary cycle phases, with periodic values,
velocities and accelerations. Default output phases are `0,...,(T-1)/T`;
the last sampled phase is not an extra duplicate endpoint. Circular input
modelling requires a full uniformly sampled cycle; missing observations retain
their phase slots with zero support. Spatial/temporal processing uses bounded
frame/node chunks. Residual controls have continuous-time evaluation, while the
existing base decoder remains discrete: to refine a queried time, the caller
must supply base vertices at that same time to `apply_residual_ffd`.

The output retains the base `control_grids` for existing loss interfaces;
refinement controls live separately under `refinement`. Physical spatial
gradient, cyclic velocity and cyclic acceleration penalties are returned under
`control_regularization` for explicit weighting by a training objective.
The original training CLI remains a Base-model entry. The separate
`train_joint_refiner.py` entry below trains and reloads the shared composition.
A 3 mm control
bound or small penalties do not certify positive Jacobians or mesh quality;
Native Exact, GT Shape/Motion and complete composed-mesh geometry acceptance
remain necessary before judging a trained model.

Implementation mechanisms were reviewed against
[MedNeXt blocks](https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/network_architecture/mednextv1/blocks.py),
[Time-Series-Library circular embedding](https://github.com/thuml/Time-Series-Library/blob/main/layers/Embed.py)
and [DiT output initialisation](https://github.com/facebookresearch/DiT/blob/main/models.py).
Their full architectures are not imported.

## GPU memory and Joint throughput

The launcher now defaults to `MEMORY_PROFILE=throughput`. Direct CLI use is
explicit: append `--memory-profile throughput`. It spends more GPU activation
memory and larger work batches while preserving the active grouped geometric
objective and all observed cells:

| Execution control | Original | Throughput |
|---|---:|---:|
| CNN view/frame chunk | 6 | 72 |
| CNN activation recomputation | on | off |
| Slice cell chunk | 16 | 64 |
| Nearest-distance query chunk | 256 | 1024 |
| Paired slice work | geometric boundary + unused probability maps | batched geometric boundary only |

`grouped_gt_boundary_v1` consumes per-cell physical boundary distances, not
probability Dice/BCE. The throughput path therefore skips those unused raster
maps, batches physical clipping/quadrature/distances and omits unnecessary
ray-parity calculations for the squared boundary distance. Missing predicted
sections and empty-target clearance retain the original semantics. This path
is rejected for legacy objectives that use probability losses. Skipped
probability diagnostics are explicitly marked, not presented as measured zero
errors. Native Exact evaluation, sample spacing and observation membership
are unchanged.

Refiner observations now compute SDF gradients and `grid_sample` across the
active cells together. Its default spatial chunk is 24 frames, temporal node
chunk is the complete original Fine lattice (18,432 nodes), observation section
chunk is 64 cells, splat chunk is 65,536 points and FFD chunk is 8 frames. These
larger chunks use more temporary memory, with no per-patient parameter fitting
or second full raster fitting objective.

```bash
# Same optimisation batch as the saved configuration; larger GPU work chunks.
python train_neural_ffd.py --config YOUR_CONFIG --resume YOUR_CHECKPOINT \
  --memory-profile throughput --exact

# Optional, explicit new experiment with more cases per optimiser update.
BATCH_SIZE=4 CNN_CHUNK_SIZE=144 MEMORY_PROFILE=throughput \
  bash scripts/train_neural_ffd_1k_direct.sh
```

Use `--cnn-chunk-size`, `--slice-cell-chunk`, `--slice-query-chunk` or
`--checkpoint-cnn` to override individual settings. `--batch-size` is separate:
it changes optimiser updates per epoch and batchwise tail reductions, so it
is recorded in the resume identity when different from the saved config.
An old checkpoint's strict resume cannot silently change that identity; use
an explicit new initialisation experiment when changing batch size. Execution
chunks do not mutate checkpoint architecture/config dictionaries. Effective
settings are saved to `execution.json` and per-step history. Already running
Python processes acquire the new settings only on restart/resume.

`--profile-steps 5` produces a bounded CPU/CUDA Chrome trace and operator table,
with named base forward, slice, GT/geometry loss, backward and optimiser ranges;
the Refiner has separate physical-observation, recovery, splat, network and FFD
ranges. Profiling adds overhead and is disabled by default.

`scripts/benchmark_joint_training.py` compares the same loaded weights and
TRAIN case batch across execution profiles, checks loss/gradient differences,
and records phase timing and CUDA peak memory without optimizer updates.
`--with-refiner` includes the composed observation path; optional
`--monitor-gpu-index` samples device-level utilisation (including other jobs).
`scripts/benchmark_joint_throughput.py` isolates geometric slice execution on
synthetic true sections. Neither benchmark claims full-epoch speed.

Measured reports are under `outputs/joint_throughput_20261017/`. In one real
TRAIN-case GPU-4 short benchmark, base Joint forward/loss/backward decreased
from 3.323 s to 0.346 s while peak allocated memory increased from 1.97 GB to
3.51 GB. Including the zero-initialised Refiner, throughput took 0.493 s and
5.65 GB. Both comparisons passed the recorded numerical parity gates. These
are warm fixed-case medians, not cohort epoch times or reconstruction accuracy.
With the Refiner, batch 4 and CNN chunk 144 took 1.240 s per four-case step
(0.310 s/case), with 22.15 GB allocated / 25.08 GB reserved peak memory.
Device-level utilisation samples were mostly 94--100% after warmup (83.1%
mean including warmup, potentially including other processes). Its paired
four-case loss difference was 2.38e-7 and gradient difference relative to the
global gradient peak was 1.03e-5. This larger-batch result is an engineering
throughput test, not evidence of identical convergence to batch 1.

## Train Base + Refiner from scratch on the complete 1K split

```bash
CUDA_VISIBLE_DEVICES=7 RESUME_CHECKPOINT= BATCH_SIZE=4 CNN_CHUNK_SIZE=144 \
  bash scripts/train_population_ffd_1k.sh
```

This is the full population architecture, not the frozen-Base pilot below.
`train_population_ffd.py` reuses the original data/training/validation loop via
`PopulationWorkflow`: random Base initialization, Shape30, Motion30 and
Joint120 epochs. Shape/Motion train Base; Joint updates **both** Base and the
shared Refiner, including physical residual extraction in the graph. Inputs
are recomputed from the current Base each update; there is no frozen prediction
or observation-feature cache. Both networks belong to the same Adam from the
start, so the stage transition does not omit Refiner parameters. Joint adds
the residual spatial/velocity/acceleration regularizers to its final-mesh loss.

The launcher defaults to physical GPU7 (logical cuda:0), batch4, CNN chunk144,
no CNN recomputation and the validated batched geometric objective. It pins a
copy of the code/config under a fresh `outputs/population_ffd_1k_fromscratch_*`
directory; fixed source assets/manifests are shared by symlink. Existing runs
are not overwritten. `launch.json` records source hashes and initialization.
The split remains TRAIN800/VAL100/TEST100. Every epoch evaluates all100 VAL
patients, including Native Exact; Joint also reports the same Base and final
predictions with chamber/right-heart tails. TEST is evaluated after training.

Checkpoints embed both networks, complete Adam, RNG, stage, epoch and sample
offset. Resume with the original output directory and source snapshot:

```bash
CUDA_VISIBLE_DEVICES=7 RUN_OUTPUT=/absolute/path/to/run \
  RESUME_CHECKPOINT=/absolute/path/to/run/last.pth \
  bash scripts/train_population_ffd_1k.sh
```

Keep the saved batch/schedule when resuming. The new entry rejects inherited
Base-only/frozen-pilot checkpoints. It saves `best_joint`, `best_dice` and
`best_tail` separately; `best_tail` prioritizes fewer VAL cells below0.98,
then higher minimum Dice. Final VAL/TEST use this VAL-selected checkpoint.
They include Base/final Native and GT comparisons. By default, complete final
geometry audits check every selected split case: all24 saved surfaces, source
AV-cap policy and all linear vertex chords including23->0; composed Jacobians
are sampled separately. This does not certify the entire continuous spatial
field or arbitrary neural trajectories. Existing CGAL/IPC dependencies are
exposed through `GEOMETRY_SITE`; `GEOMETRY_WORKERS=8` is the default. Setting
`GEOMETRY_FINAL=0` explicitly skips those final audits, which leaves geometry
acceptance incomplete. `PREPARE_ONLY=1` only creates the source/config snapshot.

Multiprocessing scratch defaults to local `/tmp/heartssm-$UID`, overriding
inherited `TMPDIR`/`TEMP`/`TMP` that may point to NAS. An explicit
`HEARTFFD_TMPDIR` can select another writable local disk. This avoids NFS
`.nfs*` busy-file errors at worker shutdown; checkpoints remain in the output
directory. The launcher also sets this environment when resuming an older
source snapshot. Already-running processes retain their original environment;
there is no automatic restart or deletion of live IPC files.

## Start a shared Refiner Joint from an existing epoch

`train_joint_refiner.py` starts a **new experiment** from an existing original
`best_joint.pth`; it does not resume the old Adam or modify its run directory.
The first-stage scope is explicit: Base stays frozen/eval and all patients
share one trainable Refiner. No patient-specific controls are optimized.
The original GT Shape/Motion, mesh edge/normal/Laplacian and paired physical
boundary objective supervise the final mesh. Residual spatial gradients,
cyclic velocity and acceleration are additionally included in the total loss.

```bash
CUDA_VISIBLE_DEVICES=7 /home/yilong/.conda/envs/4dheart/bin/python -u train_joint_refiner.py \
  --initialize PATH_TO_ORIGINAL_BEST_JOINT.pth \
  --output outputs/NEW_REFINER_EXPERIMENT \
  --device cuda:0 --train-cases 16 --val-cases 8 --steps 200 --batch-size 4
```

The source checkpoint supplies the original configuration, geometry and data
identity. Fixed first TRAIN/VAL manifest members, source hashes, inherited
epoch/update count, new update budget and regularization are saved before
learning. The fixed endpoint is compared with its frozen Base; validation
does not tune the learning rate, stopping time or checkpoint selection. TEST
is not read. This is a bounded learning diagnostic, not cohort acceptance.

Frozen Base predictions and physical observation features are cached once;
the shared Refiner, original physical FFD and active supervised objective run
on every update. Cached and online inference use the same methods. Caching
must not be reused if Base is later unfrozen or observations/Pose change.

`last.pth` embeds the full Base, Refiner configuration/weights, new Adam, RNG
and sampler state. `--resume PATH --output RUN_DIRECTORY` restores the saved
experiment settings and remaining updates. `--evaluate PATH --output NEW_DIR`
reloads the trained model and repeats its fixed TRAIN/VAL comparison. These
checkpoints load with `utils.joint_refiner_training.load_refiner_checkpoint`,
not the original Base-only loader; inference still requires explicit Pose.

Reports include original Native Exact cells (GT re-render verified), GT
Shape/Motion, each chamber and right-heart minimum/P1/P5/Dice-failure counts,
and worst identities. Sampled Jacobians replay the actual complete Base plus
residual FFD composition. Full selected-case geometry audits reuse the pinned
source AV-cap policy, CGAL and IPC for saved surfaces and all linear vertex
chords including phase23->0. Sampled Jacobians and chord checks do not certify
the whole spatial field or arbitrary continuous neural trajectories. Missing
geometry backends remain UNVERIFIED. The existing isolated native packages
can be exposed through:

```bash
export PYTHONPATH=outputs/audit_result/real_tto_motion_locked_20261016/_geometry_site
```

`protocol.json`, `history.jsonl`, `reload_verification.json`,
`train_comparison.json`, `val_comparison.json`, `*_native_cells.json`,
`geometry.json` and `decision.json` retain the experiment evidence. Geometry
failures are reported even when Dice improves; no failed cases are discarded.

The first fixed TRAIN16/VAL8 experiment initialized epoch75/step60000 and
completed 200 shared Refiner updates with Base tensors unchanged. VAL Native
Dice improved from 0.9756 to 0.9829; cells below0.98 fell from907 to394 and
right-heart failures from339 to186. All24 selected cases passed saved-mesh
and inter-frame chord audits. The minimum VAL Dice remained0.9388 and no VAL
case passed every cell, so this is improvement evidence, not tail acceptance.
See [the complete bounded report](outputs/joint_refiner_train16_val8_20261017_v2/final_report.rst).

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


## TTO scaffold

The current TTO entry is `python -m tto.run` with `configs/tto.yaml`.
It supports native-observation preflight and exact frozen-model initialization.
Global/chamber alignment and Shape/Motion/joint optimization are explicit pending
contracts. Chamber translations only generate initial candidates.
See [the TTO architecture, cleanup record and commands](tto/README.md).
The former TTO implementation reports describe retired protocols.
