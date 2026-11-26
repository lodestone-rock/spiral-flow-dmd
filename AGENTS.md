# Agent entrypoint

This repository is a self-contained eight-spiral toy study: train a
class-conditional CFG flow-matching teacher, distill it into a 4-step
DMD/DMD2 generator, and evaluate frozen clockwise interpolation between class
labels. No external dataset or diffusion framework is required.

All commands below are run from the repository root (the directory containing
this file), not from the parent research repository.

## Environment

- Python 3.10+ with PyTorch, NumPy, Matplotlib (`requirements.txt`).
- CUDA is recommended. `--device cuda` raises if CUDA is unavailable; there is
  no silent CPU fallback. CPU works functionally but is much slower.
- The recorded runs used PyTorch 2.11.0+cu128, NumPy 2.2.6, Matplotlib 3.10.9,
  Python 3.10.10 on an RTX 5090.
- `runs/` contains generated checkpoints/archives and is gitignored. Do not
  commit `.pt`, `.npz`, or `runs/`.

## Verify before changing anything

```bash
python -m unittest discover -s tests -v
```

Expected: **41 tests pass** (fast, CPU-only: 19 pipeline + 17 PDD + 5 class-prior).

## Fast smoke run (about 1–2 minutes)

Use fresh run names; every stage refuses to overwrite an existing output
folder or checkpoint.

```bash
python train_flow.py all --device cuda --run-name smoke_teacher \
    --train-steps 50 --batch-size 256 --log-every 25 --eval-samples 128 --steps 32
python dmd.py all --device cuda --run-name smoke_dmd --teacher-run smoke_teacher \
    --gen-updates 50 --batch-size 256 --pairs 4096 --eval-samples 128 \
    --teacher-steps 8 --log-every 25 --probe-every 50 --scales 2
python pdd.py all --device cuda --run-name smoke_pdd --teacher-run smoke_teacher \
    --updates 50 --batch-size 256 --eval-samples 128 --teacher-steps 8 \
    --log-every 25 --probe-every 50 --probe-samples 64 --scales 2
python interpolate.py --device cuda --teacher-run smoke_teacher \
    --eval-samples 128 --steps 32 --scales 1 --output-name smoke_interpolation
```

Smoke outputs are intentionally low quality; use them only to check plumbing.

## Full reproduction (about 30–40 minutes on an RTX 5090)

```bash
./run_all.sh cuda teacher dmd pdd
```

This runs, in order:

1. `tests/` (41 unit tests)
2. `train_flow.py all --run-name teacher` (~1 minute)
3. `dmd.py all --run-name dmd --teacher-run teacher` (~25–30 minutes)
4. `pdd.py all --run-name pdd --teacher-run teacher --scales 2 --updates 50000`
   (~2–5 minutes for the toy 50k-update recipe)
5. `interpolate.py --teacher-run teacher --scales 1 2` (~1 minute, **no retraining**)

For a random class-imbalanced dataset, set the prior on the teacher and give the
students fresh run names:

```bash
CLASS_ALPHA=0.5 CLASS_SEED=4 ./run_all.sh cuda teacher_imb dmd_imb pdd_imb
```

or stage by stage:

```bash
python train_flow.py all --device cuda --run-name teacher_imb \
    --class-alpha 0.5 --class-seed 4
python dmd.py all --device cuda --run-name dmd_imb --teacher-run teacher_imb
python pdd.py all --device cuda --run-name pdd_imb --teacher-run teacher_imb \
    --scales 2 --updates 200000 --dmd-run-name dmd_imb \
    --controls-from dmd_imb/evaluation_s40_n4096_seed1234
python interpolate.py --device cuda --teacher-run teacher_imb --scales 1 2
```

`--class-alpha 0` (default) is the balanced control. With `alpha > 0`, the
teacher draws one prior from `Dirichlet(alpha)` using `--class-seed`, stores it
as `class_probs` in `config.json`, and every student inherits it. `--class-floor`
clamps rare classes before renormalization. Evaluation always writes both
`average` (uniform over classes) and `weighted_average` (under the run's class
prior); summaries and figures use the weighted value when present.

To run stages individually:

```bash
python train_flow.py all --device cuda --run-name teacher
python dmd.py all --device cuda --run-name dmd --teacher-run teacher
python pdd.py all --device cuda --run-name pdd --teacher-run teacher --scales 2 --updates 50000
python interpolate.py --device cuda --teacher-run teacher --scales 1 2
```

## What each file does

- `core.py` — spiral geometry/data, random class priors, conditional CFG U-MLP,
  linear flow path, Heun sampler, metrics (uniform and class-weighted), plotting
  helpers.
- `train_flow.py` — teacher flow matching (10% class dropout) and its sample
  grid. `--class-alpha/--class-seed/--class-floor` draw the dataset's class
  prior.
- `dmd.py` — DMD1/DMD2 distillation. Arms are
  `{dmd1,dmd2} × {bsim,bptt} × w{2,4}`.
  - `bptt` = backprop through **all four** rollout steps.
  - `bsim` = backward simulation, gradient through **one** step only.
  - DMD1 = DM loss + λ=0.25 regression anchor; DMD2 = DM loss only, TTUR 5:1.
  - There is **no GAN loss** and **no EMA** in this implementation.
- `pdd.py` — Parallel decoding distillation in `s=1−t`.
  - Widens the teacher's final linear to `N=64` interval heads; all heads read
    the same block-start feature.
  - Training draws a block start from the eight student-grid nodes, rolls the
    student's detached heads to `K=1` sampled head, and regresses it onto the
    guided teacher's midpoint RK interval mean.
  - No critic, GAN, JVP, finite differences, or BPTT.
  - Inference fuses a block's heads into one linear; one checkpoint serves
    NFE 1/2/4/8. `--data-free`, `--shift`, and Euler `N=128, K=2` are ablations.
  - **Time convention:** the paper is `t=0` noise → `t=1` data; this repo is
    the reverse. `pdd.py` predicts `u_s = −v_t`, initializes heads from
    `−teacher.output`, and uses positive `s` steps. Do not feed the teacher
    `s` directly.
- `interpolate.py` — frozen clockwise class sweep `g=7→0`, shared initial
  noise, rotated geometric references. Reads an existing teacher checkpoint.
- `run_all.sh` — end-to-end runner.
- `tests/test_pipeline.py` — fast unit tests.
- `figures/`, `results/` — curated recorded outputs.
- `runs/` — generated artifacts (gitignored).
- `README.md` — methods, recorded tables, embedded figures, citation.

## Where results go

Teacher: `runs/<teacher-run>/`
- `checkpoint.pt`, `config.json`, `history.json`
- `teacher_samples.png`, `teacher_metrics.json`

DMD: `runs/<dmd-run>/`
- `<arm>/generator.pt`, `<arm>/history.json`, `<arm>/training_curves.png`
- `evaluation_s40_n4096_seed1234/<arm>/trajectory_progression.png`
- `comparison_w2.png`, `comparison_w4.png`
- `evaluation_*/summary.json`

Interpolation: `runs/<teacher-run>/interpolation_s<steps>_n<n>_seed<seed>/`
- `metrics.json`
- `clockwise_all_intervals_w{1,2}.png`
- `clockwise_4_to_3_w*.png`
- `clockwise_paths_w*.png`

PDD: `runs/<pdd-run>/`
- `<arm>/generator.pt`, `<arm>/history.json`, `<arm>/training_curves.png`
- `evaluation_s256_n4096_seed1234/summary.json` (SW1/HDR50, diversity, RMSE to
  the teacher grid and Heun-256 under shared noise)
- `evaluation_*/<arm>_samples.npz` (matched-noise arrays)
- `pdd_samples_w2.png`, `pdd_nfe_sweep_w2.png`, `pdd_vs_dmd_w2.png`
- curated numbers: `results/pdd_summary.json`

For a class-imbalanced run, the same artifacts are produced, but every metric
block also carries `weighted_average` and the summaries use it. Curated
balanced-vs-imbalanced numbers live in `results/imbalanced_summary.json`.

### Recorded imbalanced rerun (class seed 4)

Prior: `Dirichlet(0.5)`, floor 0.02, values
`[0.2298, 0.0352, 0.0847, 0.1271, 0.1581, 0.0196, 0.3259, 0.0196]`.
Runs: `teacher_imb`, `dmd_imb`, `pdd_imb`, plus `pdd_imb_{w1,shift2,euler128,datafree}`.

- Teacher `w=2`: weighted SW1 0.08994 (uniform 0.09342), weighted HDR50 34.6%
  (uniform 40.6%), assignment 99.85%. The balanced teacher was 0.08147 / 74.7%.
- Teacher per-class HDR50 is uneven: `g6` 6.1% and `g3` 7.2% versus `g0/g1/g2`
  at 78/77/64%. This is a class-specific `w=2` CFG effect, not a pipeline bug.
- PDD 4-NFE: SW1 0.09200 (uniform 0.09449), teacher-256 SW1 0.08982,
  RMSE→teacher 0.02468, diversity 0.2407 (teacher 0.2435).
- DMD ranking flips: balanced best is `dmd2_bsim_w2` (0.06976), imbalanced
  weighted best is `dmd1_bptt_w2` (0.07902); `dmd2_bsim_w2` degrades to 0.10391
  weighted / 0.11867 uniform.
- Interpolation stays smooth: `w=2` −313.2°, `w=1` −315.9°, 28/28 clockwise
  steps in both; mean |step| 11.2°. Unseen SW1/HDR50 are 0.0847/38.7% (`w=2`)
  and 0.0202/28.0% (`w=1`), essentially equal to the trained classes
  (0.0868/40.6% and 0.0171/28.3%). The unseen conditions interpolate the
  teacher's uneven per-class quality, not an extra interpolation error.

Use `--dmd-run-name dmd_imb --controls-from dmd_imb/evaluation_s40_n4096_seed1234`
for PDD so noise and prior match the DMD baseline. The re-evaluation commands for
the imbalanced teacher/DMD are `train_flow.py evaluate --run-name teacher_imb`
and `dmd.py evaluate --run-name dmd_imb --teacher-run teacher_imb`.

## Repo invariants

- Do not overwrite recorded outputs. Pass a fresh `--run-name`,
  `--output-name`, or delete only scratch runs you created.
- Keep the extraction faithful: the recorded metrics reproduce the original
  `experiments/toy_2d_diffusion` tables. If you change training code, say so in
  the README rather than silently updating numbers.
- `interpolate.py` is inference-only; it does not need a new teacher.
- Intermediate interpolation conditions were never trained. Rotated references
  are a geometric continuation, not ground truth; the scalar encoding has no
  periodic `0→7` wrap.
- In PDD, keep the `s=1−t` bridge explicit: student velocities are `u_s=−v_t`,
  head init is `−teacher.output`, and all `s` steps are positive. Training and
  sampling must both start at `s=0` (noise).
- Report PDD against the **teacher trajectory** as well as the exact target.
  DMD can beat the guided teacher on SW1 by partially undoing CFG; PDD instead
  matches the guided field, so a higher SW1 with much lower teacher RMSE is the
  expected result, not a failure.
- The class prior is dataset-level: draw it once in `train_flow.py`, store it in
  `config.json`, and let DMD/PDD inherit it. Do not resample a different prior
  per stage. For imbalanced runs, report the class-weighted average as the
  primary number and keep the uniform average alongside it; SW1 can look fine
  while HDR50 collapses under a skewed CFG marginal.
- Update `README.md` and `figures/` when adding user-facing results, and rebuild
  `experiments/spiral_flow_dmd.zip` if a shareable archive is requested.
