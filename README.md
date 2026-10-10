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

For a diagnostic run on the existing **1K** cohort, use:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/train_neural_ffd_1k.sh
```

The script prepares reproducible case-level **800 train / 100 validation / 100
test** manifests without modifying source NPZ files, QC, poses or labels.
It uses `configs/neural_ffd_1k.yaml`: 30 epochs (3 shape, 5 motion, 22 joint),
the original model/loss settings, four training readers and four evaluation
workers. Each epoch evaluates the same 32 validation cases; after training,
the final checkpoint is evaluated on all 100 validation and 100 test cases.
The test holdout is for this fresh diagnostic run; this cohort also has a
historical 900/100 pilot split. Start with a fresh model/output directory;
the full-cohort checkpoint has a different data/config identity.
Override `RUN_OUTPUT` or `HEARTFFD_PYTHON` through environment variables if needed.
Outputs include `epoch_history.jsonl`, `history.jsonl`, `last.pth`,
`final_val/evaluation.json` and `final_test/evaluation.json`; evaluation JSON
contains aggregate, case, chamber, view and phase-level measurements and timings.

See [implementation and validation](docs/neural_ffd.rst),
[public-source review](docs/neural_ffd_sources.json), and
[configuration](configs/neural_ffd.yaml). Architecture checks do not establish
trained reconstruction accuracy; formal population training remains to be run.
