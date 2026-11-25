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

Expected: **19 tests pass** (fast, CPU-only).

## Fast smoke run (about 1–2 minutes)

Use fresh run names; every stage refuses to overwrite an existing output
folder or checkpoint.

```bash
python train_flow.py all --device cuda --run-name smoke_teacher \
    --train-steps 50 --batch-size 256 --log-every 25 --eval-samples 128 --steps 32
python dmd.py all --device cuda --run-name smoke_dmd --teacher-run smoke_teacher \
    --gen-updates 50 --batch-size 256 --pairs 4096 --eval-samples 128 \
    --teacher-steps 8 --log-every 25 --probe-every 50 --scales 2
python interpolate.py --device cuda --teacher-run smoke_teacher \
    --eval-samples 128 --steps 32 --scales 1 --output-name smoke_interpolation
```

Smoke outputs are intentionally low quality; use them only to check plumbing.

## Full reproduction (about 30–40 minutes on an RTX 5090)

```bash
./run_all.sh cuda teacher dmd
```

This runs, in order:

1. `tests/` (19 unit tests)
2. `train_flow.py all --run-name teacher` (~1 minute)
3. `dmd.py all --run-name dmd --teacher-run teacher` (~25–30 minutes)
4. `interpolate.py --teacher-run teacher --scales 1 2` (~1 minute, **no retraining**)

To run stages individually:

```bash
python train_flow.py all --device cuda --run-name teacher
python dmd.py all --device cuda --run-name dmd --teacher-run teacher
python interpolate.py --device cuda --teacher-run teacher --scales 1 2
```

## What each file does

- `core.py` — spiral geometry/data, conditional CFG U-MLP, linear flow path,
  Heun sampler, metrics, plotting helpers.
- `train_flow.py` — teacher flow matching (10% class dropout) and its sample grid.
- `dmd.py` — DMD1/DMD2 distillation. Arms are
  `{dmd1,dmd2} × {bsim,bptt} × w{2,4}`.
  - `bptt` = backprop through **all four** rollout steps.
  - `bsim` = backward simulation, gradient through **one** step only.
  - DMD1 = DM loss + λ=0.25 regression anchor; DMD2 = DM loss only, TTUR 5:1.
  - There is **no GAN loss** and **no EMA** in this implementation.
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
- Update `README.md` and `figures/` when adding user-facing results, and rebuild
  `experiments/spiral_flow_dmd.zip` if a shareable archive is requested.
