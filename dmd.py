"""DMD / DMD2 distribution-matching distillation of the CFG 8-spiral toy teacher.

Distills the class-conditional CFG teacher (runs/teacher, FullCFGMLP, 10%
dropout) into a 4-step generator with the DMD family of losses:

  DMD  (2311.18828): approximate forward-KL gradient = difference of two scores
      (frozen guided teacher "real" score minus online "fake" critic), plus a
      regression loss on pregenerated teacher ODE pairs (lambda = 0.25).
  DMD2 (2405.14867): no regression; two time-scale update rule (5 fake updates
      per generator update); multi-step via backward simulation (rollout steps
      < k under no_grad, one grad-enabled step per iteration). --modes bptt
      instead backpropagates through the whole 4-step chain (DMD2 avoids this;
      we run both because toy scale makes the ablation cheap).

Linear-path identities used throughout (t: 0 data, 1 noise; v targets noise-data):
  x_t = (1-t) x0 + t eps;   x0_hat = x_t - t v(x_t, t);
  eps_hat(x_t, t) = x_t + (1-t) v(x_t, t);   score s = -eps_hat/t.
Guided real score: s_real from depth_velocity(teacher, ., ., w, c) (DMD's fixed-w
CFG variant). Generator gradient coefficient (stop-grad): (v_real - v_fake)/t
 = s_fake - s_real, so descending E[w_t (1-t) <d, x0_hat>] moves x0_hat toward
 the real score. Paper weighting: w_t = (sigma_t^2/alpha_t) * CS / ||x0_real -
 x0_hat||_1 with sigma_t = t, alpha_t = 1 - t. NOTE (found empirically): the
 paper's L1 normalizer sums over ~1e6 image dimensions and never vanishes; over
 2 toy coordinates it vanishes as the student matches the guided teacher, which
 biases the cloud outward and (with TTUR) destabilizes training. The default
 weight mode is therefore "kl" (w_t = 1, the exact estimand); "paper" remains
 available with --weight paper --l1-floor ... for comparison. Gradient norms
 are clipped at 10 (the DMD paper's own stabilization).

Schedule: DMD2's 4-step fractions {0.999, 0.749, 0.499, 0.249}; each step
denoises to x0_hat then renoises (1-t')x0_hat + t' eps' (consistency style);
the final x0_hat is the output. Teacher trajectory row: Heun-40 snapshots at
steps 10/20/30/40 (t = 0.75/0.5/0.25/0.0).

From this folder:
  python dmd.py all --device cuda --run-name dmd
  python dmd.py all --device cuda --run-name dmd_smoke --gen-updates 50 \
      --batch-size 256 --pairs 4096 --eval-samples 128 --log-every 25 \
      --probe-every 50     # quick smoke
Artifacts live in runs/<run-name>/.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from core import (FullCFGMLP, Geometry, annotate_outside, average_classes,
                  conditional_metrics, depth_velocity, describe_samples,
                  distribution_metrics, encode_groups, fingerprint, freeze_teacher,
                  make_generator, mixture_diagnostics, sample_data, sample_depth,
                  save_json, setup_device, state_digest, target_view_bounds)

ROOT = Path(__file__).resolve().parent
DMD2_SCHEDULE = (0.999, 0.749, 0.499, 0.249)
METHODS = ("dmd1", "dmd2")
MODES = ("bsim", "bptt")
WEIGHT_MODES = ("paper", "kl", "sigma_over_alpha")
DISPLAY_STRIDE = 4
OVERLAY_STRIDE = 2


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

def uniform_times(n: int, generator: torch.Generator, device: str | torch.device,
                  low: float, high: float) -> torch.Tensor:
    if not 0 <= low < high <= 1:
        raise ValueError("Uniform time range must satisfy 0 <= low < high <= 1")
    return torch.rand((n, 1), device=device, generator=generator) * (high - low) + low


def denoise(x: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Linear-path clean estimate x0_hat = x_t - t v."""
    return x - t * v


def score_from_velocity(x_t: torch.Tensor, t: torch.Tensor,
                        v: torch.Tensor) -> torch.Tensor:
    """Exact linear-path score of the noised marginal given the FM velocity.

    Along a trajectory eps = x0 + v, and x_t = x0 + t v, so the MSE-optimal
    noise estimate is eps_hat = x_t + (1 - t) v(x_t, t) and the Tweedie score
    is s = -eps_hat/t."""
    return -(x_t + (1 - t) * v) / t


def validate_schedule(schedule) -> tuple[float, ...]:
    values = tuple(float(t) for t in schedule)
    if not values or not all(math.isfinite(t) and 0 < t <= 1 for t in values):
        raise ValueError("Schedule timesteps must lie in (0, 1]")
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("Schedule timesteps must be strictly decreasing")
    return values


def renoise(x0: torch.Tensor, t_next: float, generator: torch.Generator | None) -> torch.Tensor:
    """Consistency-style renoise: x' = (1-t') x0 + t' eps'."""
    eps = torch.randn(x0.shape, device=x0.device) if generator is None \
        else torch.randn(x0.shape, device=x0.device, generator=generator)
    return (1 - t_next) * x0 + t_next * eps


def rollout(student: nn.Module, z: torch.Tensor, c: torch.Tensor,
            schedule: tuple[float, ...], generator: torch.Generator | None,
            grad_step: int | str | None = None):
    """DMD2 denoise/renoise rollout over `schedule`.

    grad_step: None -> every step without grad (inference/eval);
               int k -> steps j != k without grad, step k with grad (bsim);
               "all" -> every step with grad (bptt; renoise keeps the graph).
    Returns (x0_per_step list, renoised_states list [(t, x)], final_x0).
    Renoised states are returned detached for plotting only.
    """
    schedule = validate_schedule(schedule)
    if not isinstance(grad_step, (int, str, type(None))) or (
            isinstance(grad_step, int) and not 0 <= grad_step < len(schedule)):
        raise ValueError("grad_step must be None, an index into the schedule, or 'all'")
    with_grad_all = grad_step == "all"
    x, x0s, states = z, [], []
    for j, t_value in enumerate(schedule):
        t = x.new_full((len(x), 1), t_value)
        needs_grad = with_grad_all or grad_step == j
        if needs_grad:
            x0 = denoise(x, t, student(x, t, True, c))
        else:
            with torch.no_grad():
                x0 = denoise(x, t, student(x, t, True, c))
        x0s.append(x0)
        if j + 1 < len(schedule):
            source = x0 if needs_grad else x0.detach()
            new_x = renoise(source, schedule[j + 1], generator)
            states.append((t_next := schedule[j + 1], new_x.detach()))
            x = new_x
        else:
            states.append((0.0, x0.detach()))
    return x0s, states, x0s[-1]


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def dm_per_sample(x0: torch.Tensor, x_t: torch.Tensor, tau: torch.Tensor,
                  v_real: torch.Tensor, v_fake: torch.Tensor,
                  weight_mode: str, cs: float = 1.0,
                  l1_floor: float = 0.05) -> torch.Tensor:
    """Per-sample distribution-matching loss (stop-grad coefficient).

    Scores follow from the velocity via eps_hat = x_t + (1-tau) v, s =
    -eps_hat/tau; d = s_fake - s_real. Descending E[w_t (1-tau) <d, x0>]
    minimizes the approximate forward KL (the factor (1-tau) is d x_t/d x0).
    `v_real` and `v_fake` must be detached; the graph flows through `x0` only.
    Paper weight: w_t = (sigma_t^2/alpha_t) CS / ||x0_real - x0||_1 with
    sigma_t = tau, alpha_t = 1 - tau; "kl" is the plain estimator (w_t = 1).
    l1_floor: the paper's L1 sums over ~1e6 image dimensions and never
    vanishes; over 2 toy coordinates it does as the student matches the
    guided teacher, so the weight is floored to keep the game bounded.
    """
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"Unknown weight mode {weight_mode!r}")
    s_real = score_from_velocity(x_t, tau, v_real)
    s_fake = score_from_velocity(x_t, tau, v_fake)
    d = s_fake - s_real
    if weight_mode == "paper":
        x0_real = denoise(x_t, tau, v_real)
        l1 = (x0_real - x0).abs().sum(-1, keepdim=True).clamp_min(l1_floor)
        w_t = tau.square() / (1 - tau) * cs / l1
    elif weight_mode == "kl":
        w_t = torch.ones_like(tau)
    else:  # DreamFusion-style sigma/alpha, reported worse in the paper
        w_t = tau / (1 - tau)
    return (w_t * (1 - tau) * (d * x0).sum(-1, keepdim=True)).squeeze(-1)


def dm_loss(x0: torch.Tensor, teacher: nn.Module, fake: nn.Module, c: torch.Tensor,
            tau: torch.Tensor, eps: torch.Tensor, scale: float,
            weight_mode: str, cs: float = 1.0, l1_floor: float = 0.05) -> torch.Tensor:
    """Full DM loss: no_grad guided-teacher and fake velocities at noised x0."""
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Freeze/eval the teacher before distribution matching")
    with torch.no_grad():
        x_t = (1 - tau) * x0 + tau * eps
        v_real = depth_velocity(teacher, x_t, tau, scale, c)
        v_fake = fake(x_t, tau, True, c)
    return dm_per_sample(x0, x_t, tau, v_real, v_fake, weight_mode, cs, l1_floor).mean()


def fake_critic_loss(fake: nn.Module, x0: torch.Tensor, c: torch.Tensor,
                     tau: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Denoising MSE of the fake critic on noised generator outputs.

    The linear-path velocity target is eps - x0; x0 is stopped."""
    x_t = (1 - tau) * x0.detach() + tau * eps
    return (fake(x_t, tau, True, c) - (eps - x0.detach())).square().mean()


def regression_loss(x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """DMD1 regression anchor: MSE between the generator output and the
    pregenerated deterministic teacher ODE output for the same z."""
    return (x0 - y).square().mean()


# --------------------------------------------------------------------------- #
# Teacher handling
# --------------------------------------------------------------------------- #

def validate_teacher_checkpoint(cp, args, seed: int) -> dict:
    cfg = cp["config"]
    arguments = cfg["arguments"]
    if arguments.get("guidance") != "cfg" or not arguments.get("class_conditional", False):
        raise ValueError("Teacher must be the class-conditional CFG checkpoint")
    if cfg["seed"] != seed:
        raise ValueError("Teacher checkpoint seed mismatch")
    if arguments["width"] != args.width or arguments["levels"] != args.levels:
        raise ValueError("Teacher width/levels mismatch")
    if cfg["geometry"] != asdict(Geometry()):
        raise ValueError("Teacher geometry must be the default eight-spiral")
    return cfg


def load_teacher(folder: Path, device: str | torch.device, args):
    """Load the frozen teacher from the run-local byte-identical copy."""
    cp = torch.load(folder / "teacher.pt", map_location=device, weights_only=True)
    cfg = validate_teacher_checkpoint(cp, args, args.teacher_seed)
    teacher = FullCFGMLP(cfg["arguments"]["width"], cfg["arguments"]["levels"]).to(device)
    teacher.load_state_dict(cp["model"])
    return freeze_teacher(teacher), cfg


def source_teacher(args, folder: Path) -> dict:
    """Validate and byte-copy the external CFG teacher into the run folder."""
    source = ROOT / "runs" / args.teacher_run / f"seed{args.teacher_seed}"
    if not source.exists():  # train_flow.py writes runs/<name>/ directly
        source = ROOT / "runs" / args.teacher_run
    if not source.exists():
        raise FileNotFoundError(source)
    original = fingerprint(source / "checkpoint.pt")
    if (folder / "teacher.pt").exists():
        raise FileExistsError(folder / "teacher.pt")
    shutil.copyfile(source / "checkpoint.pt", folder / "teacher.pt")
    if fingerprint(folder / "teacher.pt")["sha256"] != original["sha256"]:
        raise AssertionError("Reused teacher must be a byte-identical checkpoint copy")
    record = {"source_run": args.teacher_run, "source_seed": args.teacher_seed,
              "checkpoint_sha256": original["sha256"],
              "config_sha256": fingerprint(source / "config.json")["sha256"]}
    save_json(folder / "teacher_source.json", record)
    return record


def initialize_copy(teacher: FullCFGMLP, width: int, levels: int,
                    device: str | torch.device) -> FullCFGMLP:
    """Fresh FullCFGMLP with an exact, independently stored copy of the teacher."""
    model = FullCFGMLP(width, levels).to(device)
    model.load_state_dict({k: v.detach().clone() for k, v in teacher.state_dict().items()})
    model.requires_grad_(True)
    if state_digest(model) != state_digest(teacher):
        raise AssertionError("Teacher copy must be exact")
    return model


# --------------------------------------------------------------------------- #
# DMD1 regression pairs
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def generate_pairs(teacher: nn.Module, n: int, scale: float, steps: int,
                   geometry: Geometry, device: str | torch.device, seed: int,
                   chunk: int = 2048) -> dict:
    """Pregenerate (z, y) with y = guided teacher Heun output from z (DMD1)."""
    if n < 1 or steps < 1:
        raise ValueError("Pair count and solver steps must be positive")
    gen = make_generator(device, seed)
    zs, ys, cs = [], [], []
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        z = torch.randn((m, 2), device=device, generator=gen)
        _, ids, _ = sample_data(m, geometry, gen, device)
        c = encode_groups(ids)
        y = sample_depth(teacher, z, steps, scale, c)
        zs.append(z.cpu().numpy())
        ys.append(y.cpu().numpy())
        cs.append(ids.cpu().numpy())
    return {"z": np.concatenate(zs), "y": np.concatenate(ys), "groups": np.concatenate(cs),
            "scale": scale, "steps": steps, "count": n}


def save_pairs(folder: Path, pairs: dict) -> None:
    np.savez_compressed(folder / "pairs.npz", z=pairs["z"], y=pairs["y"],
                        groups=pairs["groups"], scale=np.float64(pairs["scale"]),
                        steps=np.int64(pairs["steps"]), count=np.int64(pairs["count"]))


def load_pairs(folder: Path, device: str | torch.device, scale: float, steps: int,
               count: int) -> dict:
    with np.load(folder / "pairs.npz", allow_pickle=False) as archive:
        if int(archive["count"]) != count or float(archive["scale"]) != scale \
                or int(archive["steps"]) != steps:
            raise ValueError("Pairs archive metadata mismatch")
        pairs = {k: torch.from_numpy(archive[k]).to(device) for k in ("z", "y", "groups")}
    if len(pairs["z"]) != count or pairs["z"].shape != pairs["y"].shape \
            or pairs["groups"].shape != (count,):
        raise ValueError("Pairs archive arrays inconsistent")
    return pairs


def sample_pairs(pairs: dict, index: torch.Tensor):
    groups = pairs["groups"][index]
    return pairs["z"][index], encode_groups(groups), groups


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def arm_folder(root: Path, method: str, mode: str, scale: float) -> Path:
    return root / f"{method}_{mode}_w{scale:g}"


@torch.inference_mode()
def probe_sw1(student: nn.Module, schedule: tuple[float, ...], geometry: Geometry,
              probe_noise: torch.Tensor, probe_targets: np.ndarray,
              directions: np.ndarray, rng_seed: int) -> float:
    """Cheap 4-step sample-quality probe: average per-class SW1 vs fixed targets.

    Renoise draws come from a freshly seeded generator so every probe is
    directly comparable across training steps."""
    gen = make_generator(probe_noise.device, rng_seed)
    c = encode_groups(torch.arange(8, device=probe_noise.device)
                      .repeat_interleave(len(probe_noise)))
    x0s, _, _ = rollout(student, probe_noise.repeat(8, 1), c, schedule, generator=gen)
    values = []
    for g in range(8):
        sample = x0s[-1].reshape(8, -1, 2)[g].cpu().double().numpy()
        values.append(distribution_metrics(sample, probe_targets[g], g, geometry,
                                           directions)["sliced_wasserstein_1"])
    return float(np.mean(values))


def train_arm(args, folder: Path, device: torch.device, seed: int, method: str,
              mode: str, scale: float, geometry: Geometry) -> dict:
    """Train one DMD arm; returns the final history record."""
    folder.mkdir(parents=True, exist_ok=False)
    schedule = validate_schedule(args.schedule)
    teacher_record = source_teacher(args, folder)
    teacher, teacher_cfg = load_teacher(folder, device, args)
    teacher_digest = state_digest(teacher)

    student = initialize_copy(teacher, args.width, args.levels, device)
    fake = initialize_copy(teacher, args.width, args.levels, device)
    initial_digest = state_digest(student)
    if initial_digest != teacher_digest:
        raise AssertionError("Generator must start as an exact teacher copy")

    pairs = None
    if method == "dmd1":
        raw = generate_pairs(teacher, args.pairs, scale, args.pair_steps, geometry,
                             device, seed + 101, chunk=args.batch_size)
        save_pairs(folder, raw)
        pairs = load_pairs(folder, device, scale, args.pair_steps, args.pairs)
        pairs["z"] = pairs["z"].float()
        pairs["y"] = pairs["y"].float()

    opt_gen = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    opt_fake = torch.optim.AdamW(fake.parameters(), lr=args.lr, weight_decay=0.0)
    gen_rng = make_generator(device, seed + 100)
    fake_ratio = 5 if method == "dmd2" else 1

    probe_gen = make_generator(device, seed + 103)
    probe_noise = torch.randn((64, 2), device=device, generator=probe_gen)
    probe_targets = np.stack([sample_data(64, geometry, probe_gen, device, group=g)[0]
                              .cpu().double().numpy() for g in range(8)])
    del probe_gen
    probe_dirs = np.random.default_rng(seed + 105).normal(size=(32, 2))
    probe_dirs /= np.linalg.norm(probe_dirs, axis=1, keepdims=True)

    config = arm_config(args, method, mode, scale, seed, teacher_cfg, teacher_record,
                        schedule, {"fake_ratio": fake_ratio,
                                   "generator_initial_state_sha256": initial_digest,
                                   "fake_initial_state_sha256": state_digest(fake),
                                   "pairs": None if pairs is None else
                                   {"count": args.pairs, "steps": args.pair_steps,
                                    "scale": scale}})
    save_json(folder / "config.json", config)

    history = [{"step": 0, "probe_sw1": probe_sw1(student, schedule, geometry, probe_noise,
                                                  probe_targets, probe_dirs, seed + 106)}]
    running = {}
    start = time.perf_counter()
    for step in range(1, args.gen_updates + 1):
        # ---- generator step -------------------------------------------------
        z = torch.randn((args.batch_size, 2), device=device, generator=gen_rng)
        ids = torch.randint(8, (args.batch_size,), device=device, generator=gen_rng)
        c = encode_groups(ids)
        if method == "dmd2":
            k = int(torch.randint(len(schedule), (1,), device=device,
                                  generator=gen_rng).item())
            gs = "all" if mode == "bptt" else k
        else:
            gs = "all" if mode == "bptt" else len(schedule) - 1
        x0s, _, final_x0 = rollout(student, z, c, schedule, gen_rng, grad_step=gs)
        if method == "dmd2":
            dm_target = final_x0 if mode == "bptt" else x0s[k]
        else:
            dm_target = final_x0
        tau = uniform_times(args.batch_size, gen_rng, device, args.dm_low, args.dm_high)
        eps = torch.randn((args.batch_size, 2), device=device, generator=gen_rng)
        dm = dm_loss(dm_target, teacher, fake, c, tau, eps, scale, args.weight,
                     args.cs, args.l1_floor)
        losses = {"dm_loss": dm.detach()}
        if method == "dmd1":
            index = torch.randint(args.pairs, (args.batch_size,), device=device,
                                  generator=gen_rng)
            z_ref, c_ref, _ = sample_pairs(pairs, index)
            x0s_ref, _, final_ref = rollout(student, z_ref, c_ref, schedule, gen_rng,
                                            grad_step=gs)
            reg = regression_loss(final_ref, pairs["y"][index])
            losses["regression_loss"] = reg.detach()
        loss_gen = dm if method == "dmd2" else dm + args.regression_lambda * reg
        opt_gen.zero_grad(set_to_none=True)
        loss_gen.backward()
        gen_grad = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip,
                                                  error_if_nonfinite=True)
        opt_gen.step()
        # ---- fake critic updates (TTUR 5:1 for DMD2, 1:1 for DMD1) ----------
        if method == "dmd2":
            fake_bank, bank_c = dm_target.detach(), c
        else:
            fake_bank = torch.cat((final_x0.detach(), final_ref.detach()))
            bank_c = torch.cat((c, c_ref))
        fake_mse = None
        for _ in range(fake_ratio):
            tau_f = uniform_times(len(fake_bank), gen_rng, device, args.dm_low, args.dm_high)
            eps_f = torch.randn(fake_bank.shape, device=device, generator=gen_rng)
            fl = fake_critic_loss(fake, fake_bank, bank_c, tau_f, eps_f)
            opt_fake.zero_grad(set_to_none=True)
            fl.backward()
            fake_grad = torch.nn.utils.clip_grad_norm_(fake.parameters(), args.grad_clip,
                                                       error_if_nonfinite=True)
            opt_fake.step()
            fake_mse = fl.detach()
        if state_digest(teacher) != teacher_digest:
            raise AssertionError("Teacher changed during training")
        # ---- bookkeeping ----------------------------------------------------
        for key, value in (("dm_loss", losses["dm_loss"]),
                           ("regression_loss", losses.get("regression_loss")),
                           ("fake_mse", fake_mse)):
            if value is not None:
                running[key] = running.get(key, 0.0) + float(value)
        if step % args.log_every == 0 or step == args.gen_updates:
            record = {"step": step,
                      "elapsed_seconds": time.perf_counter() - start,
                      "gen_grad_norm": float(gen_grad), "fake_grad_norm": float(fake_grad)}
            for key in ("dm_loss", "regression_loss", "fake_mse"):
                if key in running:
                    record[key] = running[key] / (step % args.log_every or args.log_every)
            if step % args.probe_every == 0 or step == args.gen_updates:
                record["probe_sw1"] = probe_sw1(student, schedule, geometry, probe_noise,
                                                probe_targets, probe_dirs, seed + 106)
            history.append(record)
            save_json(folder / "history.json", history)
            extra = f" reg={record['regression_loss']:.4f}" if "regression_loss" in record else ""
            probe = f" probe_sw1={record['probe_sw1']:.4f}" if "probe_sw1" in record else ""
            print(f"{method}_{mode} w={scale:g} seed={seed} step={step}/{args.gen_updates} "
                  f"dm={record.get('dm_loss', float('nan')):.4f}{extra} "
                  f"fake={record.get('fake_mse', float('nan')):.4f}{probe}"
                  f" elapsed={record['elapsed_seconds']:.1f}s", flush=True)
            running = {}
    if any(p.grad is not None for p in teacher.parameters()):
        raise AssertionError("Teacher received gradients during training")
    save_json(folder / "teacher_integrity.json",
              {"state_sha256": state_digest(teacher), "unchanged": True,
               "gradients_absent": True,
               "checkpoint_sha256": teacher_record["checkpoint_sha256"]})
    torch.save({"model": student.state_dict(), "optimizer": opt_gen.state_dict(),
                "rng": gen_rng.get_state(), "seed": seed, "role": "generator",
                "method": method, "mode": mode, "scale": scale,
                "initial_state_sha256": initial_digest,
                "teacher_state_sha256": teacher_digest,
                "teacher_checkpoint_sha256": teacher_record["checkpoint_sha256"],
                "config": config}, folder / "generator.pt")
    torch.save({"model": fake.state_dict(), "optimizer": opt_fake.state_dict(),
                "rng": gen_rng.get_state(), "seed": seed, "role": "fake_critic",
                "method": method, "mode": mode, "scale": scale,
                "initial_state_sha256": initial_digest,
                "teacher_state_sha256": teacher_digest,
                "teacher_checkpoint_sha256": teacher_record["checkpoint_sha256"],
                "config": config}, folder / "fake.pt")
    training_curves(folder, history, method, mode, scale, seed)
    return history[-1]


def arm_config(args, method: str, mode: str, scale: float, seed: int, teacher_cfg,
               teacher_record: dict, schedule: tuple[float, ...], extra: dict) -> dict:
    return {
        "method": method, "mode": mode, "guidance_scale": scale, "seed": seed,
        "schedule": list(schedule), "generator_updates": args.gen_updates,
        "batch_size": args.batch_size, "lr": args.lr, "weight_mode": args.weight,
        "cs": args.cs, "l1_floor": args.l1_floor, "grad_clip": args.grad_clip,
        "dm_time_range": [args.dm_low, args.dm_high],
        "regression_lambda": args.regression_lambda if method == "dmd1" else None,
        "mode_note": {
            "bsim": "backward simulation: rollout steps < k run without grad, one grad "
                    "step per iteration (DMD2); for DMD1 the last step carries the grad",
            "bptt": "all four denoise steps carry gradient (BPTT through the sampler); "
                    "DM loss on the final output"}[mode],
        "architecture": "FullCFGMLP width 128 levels 2; conditional route only, c=group/7",
        "teacher": {"run": args.teacher_run, "seed": args.teacher_seed,
                    "checkpoint_sha256": teacher_record["checkpoint_sha256"]},
        "geometry": asdict(Geometry()),
        **extra,
    }


def training_curves(folder: Path, history: list, method: str, mode: str,
                    scale: float, seed: int) -> None:
    keys = [k for k in ("dm_loss", "regression_loss", "fake_mse", "probe_sw1")
            if any(k in h for h in history)]
    fig, axs = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6), squeeze=False)
    for ax, key in zip(axs[0], keys):
        steps = [h["step"] for h in history if key in h]
        values = [h[key] for h in history if key in h]
        ax.plot(steps, values, marker=".", markersize=3)
        ax.set(title=key, xlabel="Generator updates", ylabel=key)
        if key == "probe_sw1":
            ax.set_yscale("log")
    fig.suptitle(f"DMD {method} {mode} w={scale:g}, seed {seed}")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(folder / "training_curves.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def heun_snapshots(model: nn.Module, noise: torch.Tensor, steps: int, scale: float,
                   c: torch.Tensor, capture: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """Heun sampler keeping intermediate states; final equals sample_depth output."""
    if steps < 1 or not capture or max(capture) > steps or min(capture) < 1 \
            or len(set(capture)) != len(capture):
        raise ValueError("Capture steps must be distinct values within 1..steps")
    x = noise.clone()
    snapshots = {}
    finite = torch.ones((), dtype=torch.bool, device=x.device)
    dt = -1.0 / steps
    for i in range(steps):
        t = x.new_full((len(x), 1), 1 - i / steps)
        s = x.new_full((len(x), 1), 1 - (i + 1) / steps)
        v = depth_velocity(model, x, t, scale, c)
        predicted = x + dt * v
        x = x + dt / 2 * (v + depth_velocity(model, predicted, s, scale, c))
        finite = torch.isfinite(x).all() & torch.isfinite(predicted).all()
        if i + 1 in capture:
            snapshots[i + 1] = x.clone()
    if not finite:
        raise FloatingPointError("Nonfinite guided trajectory")
    return snapshots


@torch.inference_mode()
def student_trajectory(student: nn.Module, noise: torch.Tensor, c: torch.Tensor,
                       schedule: tuple[float, ...], generator: torch.Generator):
    """Inference rollout keeping every renoised state plus the final x0."""
    x0s, states, final = rollout(student, noise, c, schedule, generator, grad_step=None)
    return x0s, states, final


def build_controls(args, geometry: Geometry, evaluation: Path) -> dict:
    """Shared evaluation controls: matched noise, per-class targets, calibration."""
    path = evaluation / "controls.npz"
    if path.exists():
        return load_controls(path)
    cpu_gen = make_generator("cpu", args.eval_seed)
    noise = torch.randn((args.eval_samples, 2), generator=cpu_gen).numpy()
    refs = np.stack([sample_data(args.eval_samples, geometry, cpu_gen, group=g)[0]
                     .double().numpy() for g in range(8)])
    calibration = sample_data(100000, geometry, cpu_gen)[0].double().numpy()
    logs, _ = mixture_diagnostics(calibration, geometry)
    thresholds = {"50": float(np.quantile(logs, .5)), "90": float(np.quantile(logs, .1))}
    directions = np.random.default_rng(args.eval_seed + 2).normal(size=(64, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    overlay = sample_data(args.overlay_samples, geometry,
                          make_generator("cpu", args.overlay_seed))[0].double().numpy()
    bounds = target_view_bounds(refs.reshape(-1, 2))
    np.savez_compressed(path, initial_noise=noise, refs=refs, calibration=calibration,
                        directions=directions, overlay=overlay,
                        bounds=np.asarray(bounds), eval_seed=np.int64(args.eval_seed))
    return {"initial_noise": noise, "refs": refs, "directions": directions,
            "overlay": overlay, "bounds": bounds, "thresholds": thresholds}


def load_controls(path: Path) -> dict:
    geometry = Geometry()
    with np.load(path, allow_pickle=False) as archive:
        controls = {k: archive[k] for k in archive.files}
    calibration = controls.pop("calibration")
    logs, _ = mixture_diagnostics(calibration, geometry)
    controls["thresholds"] = {"50": float(np.quantile(logs, .5)),
                              "90": float(np.quantile(logs, .1))}
    controls["bounds"] = tuple(controls["bounds"].tolist())
    return controls


def measure_grid(x: np.ndarray, refs: np.ndarray, geometry: Geometry,
                 directions: np.ndarray) -> dict:
    """Per-class conditional metrics for an (8, n, 2) sample grid."""
    rows = [conditional_metrics(x[g], refs[g], g, geometry, directions) for g in range(8)]
    return {"by_class": rows, "average": average_classes(rows)}


def evaluate_arm(args, folder: Path, evaluation: Path, controls: dict, seed: int,
                 device: torch.device, method: str, mode: str, scale: float,
                 geometry: Geometry, schedule: tuple[float, ...]) -> dict:
    out = evaluation / f"{method}_{mode}_w{scale:g}"
    out.mkdir(exist_ok=False)
    student_cp = torch.load(folder / "generator.pt", map_location=device, weights_only=True)
    if student_cp["method"] != method or student_cp["mode"] != mode \
            or student_cp["scale"] != scale or student_cp["seed"] != seed:
        raise ValueError("Generator checkpoint metadata mismatch")
    student = FullCFGMLP(args.width, args.levels).to(device)
    student.load_state_dict(student_cp["model"])
    student.eval()
    teacher, _ = load_teacher(folder, device, args)
    teacher_digest = state_digest(teacher)
    if student_cp["teacher_state_sha256"] != teacher_digest \
            or student_cp["initial_state_sha256"] != teacher_digest:
        raise ValueError("Generator not bound to this run's frozen teacher")
    digest = {"generator_state_sha256": state_digest(student),
              "teacher_state_sha256": teacher_digest,
              "teacher_checkpoint_sha256": fingerprint(folder / "teacher.pt")["sha256"]}
    if digest["generator_state_sha256"] == teacher_digest:
        raise AssertionError("Generator must have moved away from its initialization")

    n = args.eval_samples
    noise = torch.from_numpy(controls["initial_noise"]).to(device)
    c = encode_groups(torch.arange(8, device=device).repeat_interleave(n))
    capture = tuple(args.teacher_steps * k // 4 for k in (1, 2, 3, 4))

    teacher_snap = heun_snapshots(teacher, noise.repeat(8, 1), args.teacher_steps,
                                  scale, c, capture)
    teacher_fine = sample_depth(teacher, noise.repeat(8, 1), 256, scale, c)
    teacher_w1 = sample_depth(teacher, noise.repeat(8, 1), 256, 1.0, c)
    x0s, states, final = student_trajectory(student, noise.repeat(8, 1), c, schedule,
                                            make_generator(device, args.eval_seed + 10))
    _, _, final_redraw = student_trajectory(student, noise.repeat(8, 1), c, schedule,
                                            make_generator(device, args.eval_seed + 11))

    def grid(tensor: torch.Tensor) -> np.ndarray:
        return tensor.reshape(8, n, 2).cpu().double().numpy()

    teacher_states = np.stack([grid(teacher_snap[k]) for k in capture])
    student_states = np.stack([grid(state) for _, state in states])
    teacher_fine_g, teacher_w1_g, final_g = grid(teacher_fine), grid(teacher_w1), grid(final)
    refs, directions = controls["refs"], controls["directions"]
    coarse40 = teacher_states[-1]
    records = {
        "seed": seed, "method": method, "mode": mode, "scale": scale,
        "schedule": list(schedule), "capture_steps": list(capture), **digest,
        "student": {"metrics": measure_grid(final_g, refs, geometry, directions)},
        "teacher_heun40": {"metrics": measure_grid(coarse40, refs, geometry, directions)},
        "teacher_heun256": {"metrics": measure_grid(teacher_fine_g, refs, geometry,
                                                    directions)},
        "teacher_w1_heun256_context": {"metrics": measure_grid(teacher_w1_g, refs, geometry,
                                                               directions)},
        "target_control": {"metrics": measure_grid(refs, refs, geometry, directions)},
        "solver_checks": {},
        "renoise_redraw_rmse": float(np.sqrt(((final_g - grid(final_redraw)) ** 2).mean())),
        "caveat": "Student renoise states use fresh noise per step (DMD2 inference); "
                  "teacher and student rows share the initial noise but not the per-step "
                  "renoise draws.",
    }
    for name in ("student", "teacher_heun40", "teacher_heun256"):
        average = records[name]["metrics"]["average"]
        print(f"eval {method}_{mode} w={scale:g} {name}: "
              f"SW1={average['sliced_wasserstein_1']:.5f} "
              f"HDR50={average['hdr_50_mass']:.3%} "
              f"assign={average['requested_class_assignment_fraction']:.3%}", flush=True)
    check = min(256, n)
    cc = encode_groups(torch.arange(8, device=device).repeat_interleave(check))
    fine80 = sample_depth(teacher, noise[:check].repeat(8, 1), 2 * args.teacher_steps,
                          scale, cc).reshape(8, check, 2).cpu().double().numpy()
    records["solver_checks"]["teacher_heun40_vs_80"] = {
        "rmse": float(np.sqrt(((fine80 - coarse40[:, :check]) ** 2).mean()))}
    records["mixture"] = {
        "student": describe_samples(final_g.reshape(-1, 2), refs.reshape(-1, 2), geometry,
                                    directions, controls["thresholds"]),
        "teacher_heun256": describe_samples(teacher_fine_g.reshape(-1, 2),
                                            refs.reshape(-1, 2), geometry, directions,
                                            controls["thresholds"])}
    np.savez_compressed(out / "samples.npz",
                        initial_noise=noise.repeat(8, 1).reshape(8, n, 2).cpu().double()
                        .numpy(),
                        teacher_states=teacher_states, student_states=student_states,
                        teacher_final=coarse40, teacher_fine=teacher_fine_g,
                        teacher_w1=teacher_w1_g, student_final=final_g,
                        schedule=np.asarray(schedule), capture=np.asarray(capture),
                        w=np.float64(scale))
    save_json(out / "metrics.json", records)
    trajectory_figure(out / "trajectory_progression.png", teacher_states, student_states,
                      capture, schedule, controls, scale, method, mode, seed)
    return records


def trajectory_figure(path: Path, teacher_states: np.ndarray, student_states: np.ndarray,
                      capture: tuple[int, ...], schedule: tuple[float, ...],
                      controls: dict, scale: float, method: str, mode: str,
                      seed: int) -> None:
    colors = plt.get_cmap("tab10")(np.arange(8))
    overlay = controls["overlay"][::OVERLAY_STRIDE]
    lo, hi = controls["bounds"]
    fig, axs = plt.subplots(2, 4, figsize=(17.5, 8.8), squeeze=False)
    teacher_t = tuple(1 - k / max(capture) for k in capture)
    titles = [f"Teacher Heun-{max(capture)} · step {k} (t={t:g})"
              for k, t in zip(capture, teacher_t)]
    student_titles = [f"Student · after step {j+1} (t={schedule[j+1]:g})"
                      for j in range(len(schedule) - 1)] + ["Student · final x0 (t=0)"]
    for row, (states, titles, label) in enumerate(
            ((teacher_states, titles, "Teacher"), (student_states, student_titles, "Student"))):
        for col, (state, title) in enumerate(zip(states, titles)):
            ax = axs[row, col]
            ax.scatter(*controls["overlay"][::OVERLAY_STRIDE].T, s=1, alpha=.10,
                       color="lightgray", zorder=1)
            for g in range(8):
                ax.scatter(*state[g, ::DISPLAY_STRIDE].T, s=2, alpha=.5,
                           color=colors[g], zorder=2)
            annotate_outside(ax, state, controls["bounds"])
            ax.set(title=title, xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal",
                   xlabel="x", ylabel="y" if col == 0 else "")
            if row == 0 and col == 0:
                ax.scatter([], [], s=8, color="gray", alpha=.4, label="Target (overlay)")
                ax.scatter([], [], s=8, color=colors[0], alpha=.7, label="Samples (by class)")
                ax.legend(fontsize=6, loc="upper right", markerscale=1.5)
    fig.suptitle(f"DMD {method}/{mode} · distilled w={scale:g} vs guided teacher · seed {seed}"
                 f" · gray underlay = exact 8-spiral target", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def comparison_figure(path: Path, columns: list, controls: dict, scale: float,
                      seed: int) -> None:
    """Final outputs of all arms vs teachers, target overlay in every panel."""
    colors = plt.get_cmap("tab10")(np.arange(8))
    overlay = controls["overlay"][::OVERLAY_STRIDE]
    lo, hi = controls["bounds"]
    fig, axs = plt.subplots(2, len(columns), figsize=(3.1 * len(columns), 7.6),
                            squeeze=False)
    view = {"bounds": controls["bounds"], "outside_counts": {}}
    for col, (label, x) in enumerate(columns):
        for row in range(2):
            ax = axs[row, col]
            ax.scatter(*overlay.T, s=1, alpha=.10, color="lightgray", zorder=1)
            points = x if row == 0 else x[4]
            if row == 0:
                for g in range(8):
                    ax.scatter(*points[g, ::DISPLAY_STRIDE].T, s=1.5, alpha=.5,
                               color=colors[g], zorder=2)
            else:
                ax.scatter(*points[::DISPLAY_STRIDE].T, s=2, alpha=.5, color="darkorange",
                           zorder=2)
            view["outside_counts"][f"{label}_{'mixture' if row == 0 else 'class4'}"] = \
                annotate_outside(ax, points, controls["bounds"])
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal",
                   xlabel="x", ylabel="y" if col == 0 else "")
            if row == 0:
                ax.set(title=label)
    fig.suptitle(f"Guidance w={scale:g} · seed {seed} · final outputs · gray = exact target"
                 f" · top: 8-class mixture, bottom: class-4 close-up", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    save_json(path.with_name(path.stem + "_view.json"), view)


def comparison_figures(args, root: Path, evaluation: Path, controls: dict,
                       arms: list) -> None:
    """Final outputs of all arms vs both teachers, one figure per guidance scale."""
    for scale in args.scales:
        scale_arms = [(m, mode) for m, mode, s in arms if s == scale]
        columns = [("Target", controls["refs"])]
        first = evaluation / f"{scale_arms[0][0]}_{scale_arms[0][1]}_w{scale:g}" / "samples.npz"
        with np.load(first, allow_pickle=False) as archive:
            columns.append((f"Teacher Heun-256 w={scale:g}", archive["teacher_fine"]))
            columns.append((f"Teacher Heun-{args.teacher_steps} w={scale:g}",
                            archive["teacher_final"]))
        for m, mode in scale_arms:
            with np.load(evaluation / f"{m}_{mode}_w{scale:g}" / "samples.npz",
                         allow_pickle=False) as archive:
                columns.append((f"{m}/{mode} (student)", archive["student_final"]))
        path = root / f"comparison_w{scale:g}.png"
        comparison_figure(path, columns, controls, scale, args.seed)
        print(f"wrote {path}", flush=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def summarize(args, evaluation: Path, records: list, arms: list) -> dict:
    summary = {"config": {k: vars(args)[k] for k in vars(args)},
               "arms": [f"{m}_{mode}_w{w:g}" for m, mode, w in arms],
               "per_arm": {}, "caveats": [
        "Single training seed (42); DMD1 is 1-step in the paper and is extended to the "
        "4-step schedule with losses on the final output; bsim mode carries gradient "
        "through the last step only.",
        "Regression pairs use the guided teacher Heun-256 sampler; the trajectory figure "
        "row shows Heun-40 for progression visualization.",
        "DM evaluation and fake-critic times use U(0.02, 0.98) (paper: U(0.02T, 0.98T) for "
        "DM, U(0,1) for the critic; the toy avoids the t->0 score singularity).",
        "No EMA, no GAN loss (DMD2 additions kept off); guidance w>1 means the distilled "
        "distribution is the guided (biased) one, not the exact target."]}
    for record in records:
        name = f"{record['method']}_{record['mode']}_w{record['scale']:g}"
        summary["per_arm"][name] = {
            "student_sw1": record["student"]["metrics"]["average"]["sliced_wasserstein_1"],
            "teacher256_sw1":
                record["teacher_heun256"]["metrics"]["average"]["sliced_wasserstein_1"],
            "student_hdr50": record["student"]["metrics"]["average"]["hdr_50_mass"],
            "student_inverse_mean_error":
                record["student"]["metrics"]["average"]["inverse_mean_error"],
            "renoise_redraw_rmse": record["renoise_redraw_rmse"],
            "solver_teacher40_vs_80":
                record["solver_checks"]["teacher_heun40_vs_80"]["rmse"]}
    save_json(evaluation / "summary.json", summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("all", "train", "evaluate", "plots"), nargs="?",
                   default="all")
    p.add_argument("--run-name", default="dmd")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher-run", default="teacher")
    p.add_argument("--teacher-seed", type=int, default=42)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--levels", type=int, default=2)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--scales", type=float, nargs="+", default=[2.0, 4.0])
    p.add_argument("--schedule", type=float, nargs="+", default=list(DMD2_SCHEDULE))
    p.add_argument("--gen-updates", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--regression-lambda", type=float, default=0.25)
    p.add_argument("--pairs", type=int, default=65536)
    p.add_argument("--pair-steps", type=int, default=256)
    p.add_argument("--weight", choices=WEIGHT_MODES, default="kl")
    p.add_argument("--cs", type=float, default=1.0)
    p.add_argument("--l1-floor", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--dm-low", type=float, default=0.02)
    p.add_argument("--dm-high", type=float, default=0.98)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--probe-every", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=4096)
    p.add_argument("--eval-seed", type=int, default=1234)
    p.add_argument("--teacher-steps", type=int, default=40)
    p.add_argument("--overlay-samples", type=int, default=32000)
    p.add_argument("--overlay-seed", type=int, default=20260926)
    args = p.parse_args(argv)
    if args.dm_low >= args.dm_high or not 0 <= args.dm_low or args.dm_high > 1:
        raise SystemExit("DM time range must satisfy 0 <= low < high <= 1")
    if args.regression_lambda < 0:
        raise SystemExit("Regression lambda must be nonnegative")
    if args.grad_clip <= 0 or args.l1_floor <= 0 or args.cs <= 0:
        raise SystemExit("grad-clip, l1-floor and cs must be positive")
    device = setup_device(args.device)
    geometry = Geometry()
    root = ROOT / "runs" / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    arms = [(m, mode, w) for m in args.methods for mode in args.modes for w in args.scales]
    schedule = validate_schedule(args.schedule)

    if args.stage in ("all", "train"):
        for method, mode, scale in arms:
            print(f"=== training {method}_{mode} w={scale:g} ===", flush=True)
            train_arm(args, arm_folder(root, method, mode, scale), device, args.seed,
                      method, mode, scale, geometry)

    if args.stage in ("all", "evaluate", "plots"):
        evaluation = root / f"evaluation_s{args.teacher_steps}_n{args.eval_samples}" \
            f"_seed{args.eval_seed}"
        if args.stage in ("all", "evaluate"):
            evaluation.mkdir(exist_ok=True)
        controls = build_controls(args, geometry, evaluation)
        if args.stage in ("all", "evaluate"):
            records = []
            for method, mode, scale in arms:
                records.append(evaluate_arm(args, arm_folder(root, method, mode, scale),
                                            evaluation, controls, args.seed, device,
                                            method, mode, scale, geometry, schedule))
            summarize(args, evaluation, records, arms)
        comparison_figures(args, root, evaluation, controls, arms)


if __name__ == "__main__":
    main()
