"""Parallel Decoding Distillation (PDD) on the eight-arm spiral.

PDD distills the frozen CFG teacher into a student that predicts the mean
velocity of several consecutive intervals in one backbone evaluation. The
teacher's final linear layer is widened to ``N`` heads (one per grid interval);
at inference a block's heads are fused into a single linear layer, so a 4-NFE
sample is four backbone passes. Training is an MSE to a one/two-step
Runge-Kutta estimate of the guided teacher's interval mean velocity, evaluated
at states from the student's own detached parallel rollout. No critic, GAN,
JVP, or BPTT.

Time convention: the paper uses ``t=0`` noise -> ``t=1`` data. Our spiral uses
``t=0`` data -> ``t=1`` noise, so this file works in ``s = 1 - t`` (noise ->
data). The student predicts ``u_s = dx/ds = -v_t``; heads are initialized from
``-teacher.output`` and all steps are positive.

From this folder:
  python pdd.py train --device cuda --run-name pdd --scales 1 2 4
  python pdd.py evaluate --device cuda --run-name pdd
  python pdd.py all --device cuda --run-name pdd --scales 1 2 4

Artifacts: runs/<run-name>/<arm>/{config.json, history.json, generator.pt,
training_curves.png}; runs/<run-name>/evaluation_*/{metrics.json, summary.json,
figures}.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core import (GROUPS, FullCFGMLP, Geometry, depth_velocity, describe_samples,
                  encode_groups, make_generator, measure_grid, mixture_diagnostics,
                  sample_data, sample_depth, save_json, setup_device, state_digest,
                  target_view_bounds)

ROOT = Path(__file__).resolve().parent
METHOD = "pdd"
DISPLAY_STRIDE = 2
OVERLAY_STRIDE = 2


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #

@dataclass
class PDDGrid:
    """Noise-to-data grid in ``s = 1 - t`` with ``s=0`` noise and ``s=1`` data."""

    steps: int = 64
    block_min: int = 8
    block_max: int = 64
    shift: float = 1.0
    device: str | torch.device = "cpu"

    def __post_init__(self):
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.block_min < 1 or self.block_max < self.block_min:
            raise ValueError("Need 1 <= block_min <= block_max")
        if self.steps % self.block_min != 0:
            raise ValueError("steps must be divisible by block_min")
        if self.block_max > self.steps:
            raise ValueError("block_max must not exceed steps")
        if not (np.isfinite(self.shift) and self.shift > 0):
            raise ValueError("shift must be finite and positive")
        u = torch.linspace(0, 1, self.steps + 1, dtype=torch.float64,
                           device=self.device)
        if self.shift == 1:
            s = u
        else:
            s = (u / self.shift) / (1 + (1 / self.shift - 1) * u)
        s[0], s[-1] = 0.0, 1.0
        if not torch.all(s[1:] > s[:-1]):
            raise ValueError("shift produced a non-monotone grid")
        self.s = s.float()
        self.t = 1 - self.s
        self.delta = self.s[1:] - self.s[:-1]
        self.starts = torch.arange(0, self.steps, self.block_min,
                                   device=self.device)
        if len(self.starts) < 1:
            raise ValueError("empty block-start set")

    @property
    def num_starts(self) -> int:
        return len(self.starts)

    def clone_for(self, device: str | torch.device) -> "PDDGrid":
        return PDDGrid(self.steps, self.block_min, self.block_max, self.shift, device)


# --------------------------------------------------------------------------- #
# Student
# --------------------------------------------------------------------------- #

class PDDStudent(nn.Module):
    """Teacher backbone + N interval heads; head k predicts u_s over [s_k,s_{k+1}]."""

    def __init__(self, teacher: FullCFGMLP, steps: int):
        super().__init__()
        if steps < 2:
            raise ValueError("steps must be at least 2")
        self.steps = steps
        self.width = teacher.output.in_features
        self.stem = copy.deepcopy(teacher.stem)
        self.middle = copy.deepcopy(teacher.middle)
        self.fuse = copy.deepcopy(teacher.fuse)
        self.head = nn.Linear(self.width, teacher.output.out_features * steps,
                              device=teacher.output.weight.device,
                              dtype=teacher.output.weight.dtype)
        with torch.no_grad():
            # u_s = -v_t, and each head starts as the teacher's readout.
            weight = -teacher.output.weight.detach()
            bias = -teacher.output.bias.detach()
            self.head.weight.copy_(weight.repeat(steps, 1))
            self.head.bias.copy_(bias.repeat(steps))

    def features(self, x: torch.Tensor, t: torch.Tensor,
                 c: torch.Tensor) -> torch.Tensor:
        if c is None or c.shape != t.shape:
            raise ValueError("PDD requires c with the same shape as t")
        return torch.cat((x, c, t), dim=-1)

    def backbone(self, x: torch.Tensor, t: torch.Tensor,
                 c: torch.Tensor) -> torch.Tensor:
        early = self.stem(self.features(x, t, c))
        deep = self.middle(early)
        return self.fuse(torch.cat((early, deep), dim=-1))

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        """Return all interval heads as [batch, steps, 2]."""
        hidden = self.backbone(x, t, c)
        return self.head(hidden).view(len(x), self.steps, 2)

    def fused_velocity(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor,
                       start: int, end: int, weights: torch.Tensor) -> torch.Tensor:
        """Block mean velocity from the fused linear head (paper Eq. 14-15)."""
        if not 0 <= start < end <= self.steps:
            raise ValueError("Invalid block bounds")
        if weights.shape != (end - start,):
            raise ValueError("weights must have one entry per interval")
        hidden = self.backbone(x, t, c)
        weight = self.head.weight.view(self.steps, 2, self.width)
        bias = self.head.bias.view(self.steps, 2)
        local = weights.to(device=hidden.device, dtype=hidden.dtype)
        fused_weight = (local[:, None, None] * weight[start:end]).sum(0)
        fused_bias = (local[:, None] * bias[start:end]).sum(0)
        return F.linear(hidden, fused_weight, fused_bias)


# --------------------------------------------------------------------------- #
# Grid operations
# --------------------------------------------------------------------------- #

def sample_heads(start: torch.Tensor, grid: PDDGrid, count: int,
                 generator: torch.Generator) -> torch.Tensor:
    """Sample `count` unique absolute interval heads inside each block."""
    if count < 1 or count > grid.block_min:
        raise ValueError("heads must lie in 1..block_min")
    batch = len(start)
    index = torch.arange(grid.steps, device=start.device)
    stop = torch.clamp(start + grid.block_max, max=grid.steps)
    mask = (index[None, :] >= start[:, None]) & (index[None, :] < stop[:, None])
    if int(mask.sum(1).min()) < count:
        raise AssertionError("A sampled block has fewer heads than requested")
    scores = torch.rand((batch, grid.steps), device=start.device, generator=generator)
    scores = scores.masked_fill(~mask, -1.0)
    return scores.topk(count, dim=1).indices


def advance_heads(heads: torch.Tensor, start: torch.Tensor, k: torch.Tensor,
                  delta: torch.Tensor) -> torch.Tensor:
    """Displacement from the block start to each sampled head (detached).

    ``prefix[m] = sum_{j<m} delta_j heads_j``; the displacement to k is
    ``prefix[k] - prefix[start]``. k == start gives exactly zero.
    """
    if heads.ndim != 3 or heads.shape[1] != len(delta):
        raise ValueError("heads must be [batch, steps, 2]")
    increments = heads.detach() * delta[None, :, None]
    cumulative = torch.cumsum(increments, dim=1)
    prefix = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative), dim=1)
    batch = torch.arange(heads.shape[0], device=heads.device)
    return prefix[batch[:, None], k] - prefix[batch, start][:, None, :]


@torch.no_grad()
def teacher_interval_velocity(teacher: nn.Module, x: torch.Tensor, s: torch.Tensor,
                              delta: torch.Tensor, c: torch.Tensor,
                              scale: float, rk: str) -> torch.Tensor:
    """Guided teacher mean velocity in s over each sampled interval.

    ``s``, ``delta`` are [batch, K]; ``c`` is [batch, 1]; returns [batch, K, 2].
    Euler uses one teacher call; midpoint uses two. Inputs are in s, converted
    to our flow time as ``t = 1 - s``.
    """
    if rk not in ("euler", "midpoint"):
        raise ValueError("rk must be euler or midpoint")
    shape = x.shape
    if x.ndim != 3 or shape[-1] != 2 or s.shape != shape[:2] or delta.shape != shape[:2]:
        raise ValueError("Inconsistent target shapes")
    batch, count, _ = shape
    flat_x = x.reshape(-1, 2)
    flat_s = s.reshape(-1, 1)
    flat_delta = delta.reshape(-1, 1)
    flat_c = c[:, None, :].expand(-1, count, -1).reshape(-1, 1)
    first = -depth_velocity(teacher, flat_x, 1 - flat_s, scale, flat_c)
    if rk == "euler":
        return first.reshape(batch, count, 2)
    mid = flat_x + .5 * flat_delta * first
    mid_s = flat_s + .5 * flat_delta
    second = -depth_velocity(teacher, mid, 1 - mid_s, scale, flat_c)
    return second.reshape(batch, count, 2)


@torch.inference_mode()
def sample_pdd(student: PDDStudent, noise: torch.Tensor, c: torch.Tensor,
               grid: PDDGrid, nfe: int, fused: bool = True) -> torch.Tensor:
    """Deterministic block integration from s=0 (noise) to s=1 (data)."""
    if nfe < 1 or grid.steps % nfe != 0:
        raise ValueError("nfe must divide the grid size")
    block = grid.steps // nfe
    x = noise
    for start in range(0, grid.steps, block):
        end = start + block
        span = grid.s[end] - grid.s[start]
        weights = grid.delta[start:end] / span
        t = grid.t[start].expand(len(x), 1)
        if fused:
            velocity = student.fused_velocity(x, t, c, start, end, weights)
        else:
            heads = student(x, t, c)
            velocity = (heads[:, start:end] * grid.delta[start:end][None, :, None]
                        ).sum(1) / span
        x = x + span * velocity
    return x


@torch.inference_mode()
def sample_teacher_grid(teacher: nn.Module, noise: torch.Tensor, c: torch.Tensor,
                        grid: PDDGrid, scale: float, rk: str) -> torch.Tensor:
    """Teacher integrated on the same fine grid and RK rule (reference path)."""
    x = noise
    for j in range(grid.steps):
        span = grid.delta[j]
        s0 = grid.s[j].expand(len(x), 1)
        first = -depth_velocity(teacher, x, 1 - s0, scale, c)
        if rk == "euler":
            x = x + span * first
        else:
            mid = x + .5 * span * first
            second = -depth_velocity(teacher, mid, 1 - (s0 + .5 * span), scale, c)
            x = x + span * second
    return x


# --------------------------------------------------------------------------- #
# Data-free carry (Algorithm 3)
# --------------------------------------------------------------------------- #

class DataFreeCarry:
    """Carry one student trajectory across updates; advance by block_min, stop-grad."""

    def __init__(self, batch: int, grid: PDDGrid, device: torch.device,
                 generator: torch.Generator):
        self.batch, self.grid, self.device, self.generator = batch, grid, device, generator
        self.x: torch.Tensor | None = None
        self.c: torch.Tensor | None = None
        self.position: int | None = None

    def reset(self) -> None:
        self.x = torch.randn((self.batch, 2), device=self.device,
                             generator=self.generator)
        ids = torch.randint(GROUPS, (self.batch,), device=self.device,
                            generator=self.generator)
        self.c = encode_groups(ids)
        self.position = 0

    def current(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.x is None or self.position is None or self.position >= self.grid.steps:
            self.reset()
        start = torch.full((self.batch,), self.position, device=self.device,
                           dtype=torch.long)
        return self.x, self.c, start

    @torch.no_grad()
    def advance(self, heads: torch.Tensor) -> None:
        start = self.position
        end = start + self.grid.block_min
        displacement = (heads.detach()[:, start:end]
                        * self.grid.delta[start:end][None, :, None]).sum(1)
        self.x = self.x + displacement
        self.position = end


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def load_teacher(folder: Path, device: torch.device, width: int, levels: int):
    checkpoint = torch.load(folder / "checkpoint.pt", map_location=device,
                            weights_only=True)
    config = checkpoint["config"]
    arguments = config["arguments"]
    if arguments.get("guidance") != "cfg" or not arguments.get("class_conditional"):
        raise SystemExit("PDD needs the class-conditional CFG teacher")
    if arguments.get("width") != width or arguments.get("levels") != levels:
        raise SystemExit("Teacher width/levels do not match the requested architecture")
    teacher = FullCFGMLP(width, levels).to(device)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher, checkpoint


def train_step(args, student: PDDStudent, teacher: nn.Module, grid: PDDGrid,
               optimizer: torch.optim.Optimizer, generator: torch.Generator,
               geometry: Geometry, carry: DataFreeCarry | None):
    batch = args.batch_size
    if carry is None:
        data, ids, _ = sample_data(batch, geometry, generator, args.device)
        c = encode_groups(ids)
        noise = torch.randn(data.shape, device=args.device, generator=generator)
        slot = torch.randint(grid.num_starts, (batch,), device=args.device,
                             generator=generator)
        start = grid.starts[slot]
        s_n = grid.s[start][:, None]
        x_n = (1 - s_n) * noise + s_n * data
    else:
        x_n, c, start = carry.current()

    t_n = grid.t[start][:, None]
    heads = student(x_n, t_n, c)
    k = sample_heads(start, grid, args.heads, generator)
    x_k = x_n[:, None, :] + advance_heads(heads, start, k, grid.delta)
    target = teacher_interval_velocity(teacher, x_k, grid.s[k], grid.delta[k],
                                       c, args.scale, args.rk)
    index = torch.arange(batch, device=args.device)[:, None]
    prediction = heads[index, k]
    loss = (prediction - target).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip,
                                               error_if_nonfinite=True)
    optimizer.step()
    if carry is not None:
        carry.advance(heads)
    return loss.detach(), grad_norm.detach()


@torch.inference_mode()
def probe_metrics(student: PDDStudent, grid: PDDGrid, noise: torch.Tensor,
                  c: torch.Tensor, refs: np.ndarray, directions: np.ndarray,
                  geometry: Geometry, nfe: int) -> dict:
    samples = sample_pdd(student, noise, c, grid, nfe)
    samples = samples.reshape(GROUPS, -1, 2).cpu().double().numpy()
    return measure_grid(samples, refs, geometry, directions)["average"]


def train_arm(args, folder: Path, teacher: nn.Module, teacher_checkpoint: dict,
              teacher_folder: Path, device: torch.device, geometry: Geometry,
              scale: float) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    grid = PDDGrid(args.steps, args.block_min, args.block_max, args.shift, device)
    student = PDDStudent(teacher, args.steps).to(device)
    # The backbone must start as the teacher; only the head bank is new.
    for name in ("stem", "middle", "fuse"):
        left = getattr(student, name).state_dict()
        right = getattr(teacher, name).state_dict()
        if left.keys() != right.keys() or any(
                not torch.equal(left[key], right[key]) for key in left):
            raise AssertionError(f"Student {name} is not the teacher backbone")
    initial_head = student.head.weight.detach().cpu().clone()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    generator = make_generator(device, args.seed + 100)
    carry = DataFreeCarry(args.batch_size, grid, device, generator) \
        if args.data_free else None

    probe_generator = make_generator(device, args.seed + 103)
    probe_noise = torch.randn((args.probe_samples * GROUPS, 2), device=device,
                              generator=probe_generator)
    probe_c = encode_groups(torch.arange(GROUPS, device=device)
                            .repeat_interleave(args.probe_samples))
    cpu_generator = make_generator("cpu", args.seed + 104)
    probe_refs = np.stack([sample_data(args.probe_samples, geometry, cpu_generator,
                                       group=g)[0].double().numpy()
                           for g in range(GROUPS)])
    probe_directions = np.random.default_rng(args.seed + 105).normal(size=(64, 2))
    probe_directions /= np.linalg.norm(probe_directions, axis=1, keepdims=True)
    del probe_generator, cpu_generator

    teacher_digest = state_digest(teacher)
    config = {
        "method": METHOD, "rk": args.rk, "scale": scale, "seed": args.seed,
        "width": args.width, "levels": args.levels, "heads": args.heads,
        "data_free": args.data_free, "updates": args.updates,
        "batch_size": args.batch_size, "lr": args.lr, "grad_clip": args.grad_clip,
        "grid": {"steps": args.steps, "block_min": args.block_min,
                 "block_max": args.block_max, "shift": args.shift},
        "teacher_run": args.teacher_run,
        "teacher_checkpoint_sha256": teacher_checkpoint.get("sha256"),
        "teacher_state_sha256": teacher_digest,
        "initial_head_sha256": initial_head.numpy().tobytes().hex()[:64],
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    save_json(folder / "config.json", config)

    probe_nfe = 4 if args.steps % 4 == 0 else args.steps
    history = [{"step": 0, "probe": probe_metrics(student, grid, probe_noise, probe_c,
                                                  probe_refs, probe_directions,
                                                  geometry, probe_nfe)}]
    running, count = 0.0, 0
    began = time.perf_counter()
    for step in range(1, args.updates + 1):
        loss, grad_norm = train_step(args, student, teacher, grid, optimizer, generator,
                                     geometry, carry)
        running += float(loss)
        count += 1
        if step % args.log_every == 0 or step == args.updates:
            record = {"step": step, "train_mse": running / count,
                      "grad_norm": float(grad_norm),
                      "elapsed_seconds": time.perf_counter() - began}
            if step % args.probe_every == 0 or step == args.updates:
                record["probe"] = probe_metrics(student, grid, probe_noise, probe_c,
                                                probe_refs, probe_directions,
                                                geometry, probe_nfe)
            history.append(record)
            save_json(folder / "history.json", history)
            probe = record.get("probe", {}).get("sliced_wasserstein_1")
            shown = "" if probe is None else f" probe_sw1={probe:.5f}"
            print(f"{args.rk} w={scale:g} seed={args.seed} step={step}/{args.updates} "
                  f"mse={record['train_mse']:.6f}{shown} "
                  f"elapsed={record['elapsed_seconds']:.1f}s", flush=True)
            running, count = 0.0, 0

    if any(p.grad is not None for p in teacher.parameters()):
        raise AssertionError("Teacher received gradients")
    if state_digest(teacher) != teacher_digest:
        raise AssertionError("Teacher changed during training")
    torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": generator.get_state(), "seed": args.seed, "role": "pdd_student",
                "method": METHOD, "rk": args.rk, "scale": scale,
                "teacher_state_sha256": teacher_digest,
                "teacher_checkpoint_sha256": teacher_checkpoint.get("sha256"),
                "config": config}, folder / "generator.pt")
    training_curves(folder / "training_curves.png", history, args.rk, scale, args.seed)
    return history[-1]


def training_curves(path: Path, history: list, rk: str, scale: float,
                    seed: int) -> None:
    steps = [row["step"] for row in history]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    axs[0].plot(steps, [row.get("train_mse", np.nan) for row in history], ".-")
    axs[0].set(xlabel="Updates", ylabel="PD loss (MSE)", title="Training loss")
    probe = [(row["step"], row["probe"]["sliced_wasserstein_1"])
             for row in history if "probe" in row]
    if probe:
        axs[1].plot(*zip(*probe), ".-")
    axs[1].set(xlabel="Updates", ylabel="Probe SW1 (NFE=4)",
               title="Probe distribution error")
    fig.suptitle(f"PDD {rk} · w={scale:g} · seed {seed}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def load_controls(path: Path) -> dict:
    geometry = Geometry()
    with np.load(path, allow_pickle=False) as archive:
        controls = {key: archive[key] for key in archive.files}
    calibration = controls.pop("calibration")
    logs, _ = mixture_diagnostics(calibration, geometry)
    controls["thresholds"] = {"50": float(np.quantile(logs, .5)),
                              "90": float(np.quantile(logs, .1))}
    controls["bounds"] = tuple(controls["bounds"].tolist())
    return controls


def build_controls(args, geometry: Geometry, evaluation: Path) -> dict:
    path = evaluation / "controls.npz"

    def compatible(controls: dict) -> bool:
        return (controls["initial_noise"].shape == (args.eval_samples, 2)
                and controls["refs"].shape == (GROUPS, args.eval_samples, 2))

    if path.exists():
        controls = load_controls(path)
        if compatible(controls):
            return controls
        print(f"Ignoring {path}: sample-count mismatch", flush=True)
    candidate = ROOT / "runs" / args.controls_from / "controls.npz"
    if args.controls_from and candidate.exists():
        controls = load_controls(candidate)
        if compatible(controls):
            print(f"Using matched DMD controls: {candidate}", flush=True)
            return controls
        print(f"Ignoring {candidate}: sample-count mismatch; building fresh controls",
              flush=True)
    cpu_generator = make_generator("cpu", args.eval_seed)
    noise = torch.randn((args.eval_samples, 2), generator=cpu_generator).numpy()
    refs = np.stack([sample_data(args.eval_samples, geometry, cpu_generator, group=g)[0]
                     .double().numpy() for g in range(GROUPS)])
    calibration = sample_data(100000, geometry, cpu_generator)[0].double().numpy()
    logs, _ = mixture_diagnostics(calibration, geometry)
    thresholds = {"50": float(np.quantile(logs, .5)),
                  "90": float(np.quantile(logs, .1))}
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


def diversity_metrics(samples: np.ndarray, limit: int = 512) -> dict:
    """Mean pairwise distance per requested class (subset for O(n^2))."""
    per_class = []
    for group in range(GROUPS):
        local = samples[group]
        if len(local) > limit:
            local = local[:limit]
        distances = np.linalg.norm(local[:, None, :] - local[None, :, :], axis=-1)
        upper = np.triu_indices(len(local), 1)
        per_class.append(float(distances[upper].mean()))
    return {"mean_pairwise_distance_by_class": per_class,
            "mean_pairwise_distance": float(np.mean(per_class))}


def distance_metrics(samples: np.ndarray, reference: np.ndarray) -> dict:
    """Per-sample trajectory agreement under shared initial noise."""
    difference = np.asarray(samples, dtype=np.float64) - np.asarray(reference,
                                                                     dtype=np.float64)
    flat = difference.reshape(-1, difference.shape[-1])
    return {"rmse": float(np.sqrt(np.mean(difference ** 2))),
            "mean_l2": float(np.linalg.norm(flat, axis=-1).mean())}


def sample_grid_figure(path: Path, panels: list, controls: dict, title: str,
                       shape=(2, 3)) -> None:
    rows, columns = shape
    if rows * columns < len(panels):
        raise ValueError("Panel grid is too small")
    colors = plt.get_cmap("tab10")(np.arange(GROUPS))
    lo, hi = controls["bounds"]
    overlay = controls["overlay"]
    fig, axs = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.3 * rows),
                            squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (name, data, metric) in zip(axs.ravel(), panels):
        ax.axis("on")
        for group in range(GROUPS):
            ax.scatter(*data[group, ::DISPLAY_STRIDE].T, s=1.4, alpha=.55,
                       color=colors[group], zorder=2)
        ax.scatter(*overlay[::OVERLAY_STRIDE].T, s=.7, alpha=.08, color="lightgray",
                   zorder=1)
        ax.set(title=name, xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal",
               xlabel="x", ylabel="y" if ax is axs[0, 0] else "")
        if metric is not None:
            ax.text(.03, .03, f"SW1 {metric:.4f}", transform=ax.transAxes, fontsize=8,
                    bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def nfe_figure(path: Path, rows: list, teacher: dict, baseline_rmse: float,
               scale: float) -> None:
    nfes = [row["nfe"] for row in rows]
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.2))
    series = (("sliced_wasserstein_1", "SW1 vs exact target"),
              ("hdr_50_mass", "HDR50 mass"),
              ("inverse_mean_error", "Inverse mean error"))
    for ax, (key, label) in zip(axs, series):
        ax.plot(nfes, [row["student"]["average"][key] for row in rows], ".-",
                label="PDD")
        ax.plot(nfes, [row["teacher_euler"]["average"][key] for row in rows], "s--",
                alpha=.6, label="Teacher Euler, same NFE")
        if key in teacher:
            ax.axhline(teacher[key], linestyle=":", color="gray",
                       label=f"Teacher-256 {teacher[key]:.4f}")
        ax.set(xlabel="NFE", ylabel=label, xticks=nfes,
               xticklabels=[str(n) for n in nfes])
        ax.legend(fontsize=8)
    ax = axs[3]
    ax.plot(nfes, [row["rmse_teacher_grid"] for row in rows], ".-",
            label="PDD vs teacher grid")
    ax.axhline(baseline_rmse, linestyle=":", color="gray",
               label=f"Teacher grid vs Heun-256 {baseline_rmse:.4f}")
    ax.set(xlabel="NFE", ylabel="RMSE under shared noise",
           xticks=nfes, xticklabels=[str(n) for n in nfes])
    ax.legend(fontsize=8)
    fig.suptitle(f"PDD NFE sweep · w={scale:g} · deterministic ODE, no CFG at inference",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def comparison_figure(path: Path, panels: list, controls: dict, title: str) -> None:
    sample_grid_figure(path, panels, controls, title, shape=(1, 4))


def discover_arms(root: Path) -> list[Path]:
    return [path for path in sorted(root.iterdir())
            if path.is_dir() and (path / "generator.pt").exists()]


def evaluate(args, root: Path, teacher: nn.Module, device: torch.device,
             geometry: Geometry) -> dict:
    evaluation = root / f"evaluation_s{args.teacher_steps}_n{args.eval_samples}" \
        f"_seed{args.eval_seed}"
    evaluation.mkdir(parents=True, exist_ok=True)
    controls = build_controls(args, geometry, evaluation)
    refs, directions = controls["refs"], controls["directions"]
    n = args.eval_samples
    noise = torch.from_numpy(controls["initial_noise"]).to(device)
    noise8 = noise.repeat(GROUPS, 1)
    c = encode_groups(torch.arange(GROUPS, device=device).repeat_interleave(n))

    arms = discover_arms(root)
    if not arms:
        raise SystemExit(f"No trained PDD arms under {root}")
    records = {"controls_from": args.controls_from, "eval_seed": args.eval_seed,
               "eval_samples": n, "teacher_steps": args.teacher_steps, "arms": {}}
    teacher_cache: dict = {}
    sample_banks: dict = {}
    rows_for_summary = []

    for arm in arms:
        config = __import__("json").loads((arm / "config.json").read_text())
        steps = config["grid"]["steps"]
        grid = PDDGrid(steps, config["grid"]["block_min"],
                       config["grid"]["block_max"], config["grid"]["shift"], device)
        student = PDDStudent(teacher, steps).to(device)
        checkpoint = torch.load(arm / "generator.pt", map_location=device,
                                weights_only=True)
        if checkpoint["config"]["grid"] != config["grid"]:
            raise ValueError(f"{arm.name}: checkpoint/config grid mismatch")
        student.load_state_dict(checkpoint["model"])
        student.eval()
        scale, rk = config["scale"], config["rk"]
        arm_name = arm.name
        print(f"evaluating {arm_name} (w={scale:g}, {rk}, N={steps})", flush=True)

        scale_key = (scale, rk, steps, config["grid"]["block_min"],
                     config["grid"]["block_max"], config["grid"]["shift"])
        if scale_key not in teacher_cache:
            teacher_fine = sample_depth(teacher, noise8, args.teacher_steps, scale, c)
            teacher_grid = sample_teacher_grid(teacher, noise8, c, grid, scale, rk)
            teacher_fine_np = teacher_fine.reshape(GROUPS, n, 2).cpu().double().numpy()
            teacher_grid_np = teacher_grid.reshape(GROUPS, n, 2).cpu().double().numpy()
            teacher_cache[scale_key] = {
                "teacher_256": teacher_fine_np,
                "teacher_grid": teacher_grid_np,
                "teacher_256_metrics": measure_grid(teacher_fine_np, refs, geometry,
                                                    directions),
                "teacher_grid_metrics": measure_grid(teacher_grid_np, refs, geometry,
                                                     directions),
                "teacher_256_diversity": diversity_metrics(teacher_fine_np),
                "teacher_grid_diversity": diversity_metrics(teacher_grid_np)}
        cache = teacher_cache[scale_key]

        sample_bank = {}
        arm_record = {"config": config,
                      "teacher_256": cache["teacher_256_metrics"],
                      "teacher_grid": cache["teacher_grid_metrics"],
                      "teacher_256_diversity": cache["teacher_256_diversity"],
                      "teacher_grid_diversity": cache["teacher_grid_diversity"],
                      "grid_rmse": distance_metrics(cache["teacher_grid"],
                                                    cache["teacher_256"]),
                      "nfe": {}}
        for nfe in args.eval_nfe:
            if steps % nfe != 0:
                continue
            samples = sample_pdd(student, noise8, c, grid, nfe)
            samples_np = samples.reshape(GROUPS, n, 2).cpu().double().numpy()
            teacher_nfe = sample_depth(teacher, noise8, nfe, scale, c)
            teacher_nfe_np = teacher_nfe.reshape(GROUPS, n, 2).cpu().double().numpy()
            teacher_euler = sample_teacher_grid(
                teacher, noise8, c, PDDGrid(nfe, 1, nfe, 1.0, device), scale,
                "euler")
            teacher_euler_np = teacher_euler.reshape(GROUPS, n, 2).cpu().double().numpy()
            student_metrics = measure_grid(samples_np, refs, geometry, directions)
            teacher_nfe_metrics = measure_grid(teacher_nfe_np, refs, geometry,
                                               directions)
            teacher_euler_metrics = measure_grid(teacher_euler_np, refs, geometry,
                                                 directions)
            sample_bank[str(nfe)] = samples_np
            arm_record["nfe"][str(nfe)] = {
                "student": student_metrics,
                "teacher_nfe": teacher_nfe_metrics,
                "teacher_euler": teacher_euler_metrics,
                "diversity": diversity_metrics(samples_np),
                "teacher_nfe_diversity": diversity_metrics(teacher_nfe_np),
                "teacher_euler_diversity": diversity_metrics(teacher_euler_np),
                "to_teacher_grid": distance_metrics(samples_np,
                                                    cache["teacher_grid"]),
                "to_teacher_256": distance_metrics(samples_np, cache["teacher_256"]),
                "mixture": describe_samples(samples_np.reshape(-1, 2),
                                            refs.reshape(-1, 2), geometry,
                                            directions, controls["thresholds"])}
            average = student_metrics["average"]
            row = {"arm": arm_name, "scale": scale, "rk": rk, "nfe": nfe,
                   "student_sw1": average["sliced_wasserstein_1"],
                   "student_hdr50": average["hdr_50_mass"],
                   "student_center": average["inverse_mean_error"],
                   "teacher_nfe_sw1": teacher_nfe_metrics["average"][
                       "sliced_wasserstein_1"],
                   "teacher_euler_sw1": teacher_euler_metrics["average"][
                       "sliced_wasserstein_1"],
                   "teacher_256_sw1": cache["teacher_256_metrics"]["average"][
                       "sliced_wasserstein_1"],
                   "teacher_grid_sw1": cache["teacher_grid_metrics"]["average"][
                       "sliced_wasserstein_1"],
                   "rmse_teacher_grid": arm_record["nfe"][str(nfe)][
                       "to_teacher_grid"]["rmse"],
                   "rmse_teacher_256": arm_record["nfe"][str(nfe)][
                       "to_teacher_256"]["rmse"],
                   "mean_l2_teacher_grid": arm_record["nfe"][str(nfe)][
                       "to_teacher_grid"]["mean_l2"],
                   "teacher_grid_rmse": arm_record["grid_rmse"]["rmse"],
                   "diversity": arm_record["nfe"][str(nfe)]["diversity"][
                       "mean_pairwise_distance"],
                   "teacher_grid_diversity": arm_record["teacher_grid_diversity"][
                       "mean_pairwise_distance"],
                   "teacher_256_diversity": arm_record["teacher_256_diversity"][
                       "mean_pairwise_distance"]}
            rows_for_summary.append(row)
            print(f"  NFE={nfe}: SW1={row['student_sw1']:.5f} "
                  f"HDR50={row['student_hdr50']:.3%} "
                  f"euler={row['teacher_euler_sw1']:.5f} "
                  f"heun={row['teacher_nfe_sw1']:.5f} "
                  f"rmse@grid={row['rmse_teacher_grid']:.5f} "
                  f"div={row['diversity']:.4f}", flush=True)
        records["arms"][arm_name] = arm_record
        sample_banks[arm_name] = sample_bank
        np.savez_compressed(evaluation / f"{arm_name}_samples.npz",
                            noise=controls["initial_noise"], refs=refs,
                            c=c.cpu().numpy(), teacher_grid=cache["teacher_grid"],
                            teacher_256=cache["teacher_256"],
                            **{f"student_{key}": value
                               for key, value in sample_bank.items()})
    save_json(evaluation / "summary.json", rows_for_summary)

    dmd_root = next(iter(sorted(
        (ROOT / "runs" / "dmd").glob(f"evaluation_s*_n{n}_seed{args.eval_seed}"))),
        None)
    if dmd_root is not None and dmd_root.exists():
        baselines = {}
        for path in sorted(dmd_root.glob("*/samples.npz")):
            with np.load(path, allow_pickle=False) as archive:
                final = archive["student_final"]
                teacher = archive["teacher_fine"]
            metrics = measure_grid(final, refs, geometry, directions)
            baselines[path.parent.name] = {
                "student_sw1": metrics["average"]["sliced_wasserstein_1"],
                "student_hdr50": metrics["average"]["hdr_50_mass"],
                "student_center": metrics["average"]["inverse_mean_error"],
                "diversity": diversity_metrics(final),
                "to_teacher": distance_metrics(final, teacher)}
        records["dmd_baselines"] = baselines

    # Figures: one per scale, using the midpoint arm and NFE 1/2/4/8.
    for scale in sorted({row["scale"] for row in rows_for_summary}):
        candidates = [row for row in rows_for_summary
                      if row["scale"] == scale and row["rk"] == "midpoint"]
        if not candidates:
            candidates = [row for row in rows_for_summary if row["scale"] == scale]
        if not candidates:
            continue
        arm_name = candidates[0]["arm"]
        arm_record = records["arms"][arm_name]
        bank = sample_banks[arm_name]
        rk = arm_record["config"]["rk"]
        first_key = next(key for key in teacher_cache
                         if key[0] == scale and key[1] == rk)
        cache = teacher_cache[first_key]
        steps = arm_record["config"]["grid"]["steps"]
        panels = [
            ("Target", refs, None),
            (f"Teacher Heun-{args.teacher_steps}", cache["teacher_256"],
             arm_record["teacher_256"]["average"]["sliced_wasserstein_1"]),
            (f"Teacher grid ({arm_record['config']['rk']}, N={steps})",
             cache["teacher_grid"],
             arm_record["teacher_grid"]["average"]["sliced_wasserstein_1"])]
        display_nfes = [nfe for nfe in (1, 2, 4, 8) if str(nfe) in arm_record["nfe"]]
        for nfe in display_nfes:
            panels.append((f"PDD NFE={nfe}", bank[str(nfe)],
                           arm_record["nfe"][str(nfe)]["student"]["average"][
                               "sliced_wasserstein_1"]))
        sample_grid_figure(root / f"pdd_samples_w{scale:g}.png", panels, controls,
                           f"PDD · w={scale:g} · teacher and NFE sweep",
                           shape=(2, 4))
        nfe_rows = [{"nfe": int(key), "student": value["student"],
                     "teacher_nfe": value["teacher_nfe"],
                     "teacher_euler": value["teacher_euler"],
                     "rmse_teacher_grid": value["to_teacher_grid"]["rmse"]}
                    for key, value in sorted(arm_record["nfe"].items(),
                                             key=lambda item: int(item[0]))]
        nfe_figure(root / f"pdd_nfe_sweep_w{scale:g}.png", nfe_rows,
                   arm_record["teacher_256"]["average"],
                   arm_record["grid_rmse"]["rmse"], scale)

    # DMD comparison for w=2 if the recorded archive exists.
    dmd_path = (dmd_root / "dmd2_bsim_w2" / "samples.npz"
                if dmd_root is not None else None)
    pdd_arm = next((row for row in rows_for_summary
                    if row["scale"] == 2 and row["rk"] == "midpoint"), None)
    if dmd_path is not None and dmd_path.exists() and pdd_arm is not None:
        with np.load(dmd_path, allow_pickle=False) as archive:
            dmd_final = archive["student_final"]
            dmd_teacher = archive["teacher_fine"]
        arm_record = records["arms"][pdd_arm["arm"]]
        bank = sample_banks[pdd_arm["arm"]]
        pdd_nfe = next((nfe for nfe in (4, 2, 8, 1) if str(nfe) in bank), None)
        if pdd_nfe is not None:
            dmd_metrics = measure_grid(dmd_final, refs, geometry, directions)
            cache = next(value for key, value in teacher_cache.items()
                         if key[0] == pdd_arm["scale"] and key[1] == "midpoint")
            records["dmd2_bsim_w2"] = {
                "student_sw1": dmd_metrics["average"]["sliced_wasserstein_1"],
                "diversity": diversity_metrics(dmd_final),
                "to_teacher_256": distance_metrics(dmd_final,
                                                   cache["teacher_256"])}
            comparison_figure(
                root / "pdd_vs_dmd_w2.png",
                [("Target", refs, None),
                 ("Teacher Heun-256", dmd_teacher,
                  arm_record["teacher_256"]["average"]["sliced_wasserstein_1"]),
                 ("DMD2 bsim w=2", dmd_final,
                  dmd_metrics["average"]["sliced_wasserstein_1"]),
                 (f"PDD NFE={pdd_nfe}", bank[str(pdd_nfe)],
                  arm_record["nfe"][str(pdd_nfe)]["student"]["average"][
                      "sliced_wasserstein_1"])],
                controls, "Spiral 4-NFE comparison · DMD2 vs PDD (matched noise)")
    save_json(evaluation / "metrics.json", records)
    return {"evaluation": str(evaluation), "summary": rows_for_summary}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def arm_name(args) -> str:
    name = f"pdd_{args.rk}_w{args.scale:g}"
    if args.heads != 1:
        name += f"_k{args.heads}"
    if args.lr != 0.001:
        name += f"_lr{args.lr:g}"
    if args.data_free:
        name += "_df"
    if args.shift != 1:
        name += f"_shift{args.shift:g}"
    if args.steps != 64:
        name += f"_n{args.steps}"
    if args.block_min != 8:
        name += f"_l{args.block_min}"
    return name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "all"), nargs="?",
                        default="all")
    parser.add_argument("--run-name", default="pdd")
    parser.add_argument("--teacher-run", default="teacher")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-min", type=int, default=8)
    parser.add_argument("--block-max", type=int, default=64)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--rk", choices=("euler", "midpoint"), default="midpoint")
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--data-free", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--probe-every", type=int, default=1000)
    parser.add_argument("--probe-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--eval-nfe", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--teacher-steps", type=int, default=256)
    parser.add_argument("--overlay-samples", type=int, default=32000)
    parser.add_argument("--overlay-seed", type=int, default=20260926)
    parser.add_argument("--controls-from",
                        default="dmd/evaluation_s40_n4096_seed1234")
    args = parser.parse_args(argv)
    if args.heads < 1 or args.heads > args.block_min:
        raise SystemExit("--heads must lie in 1..block-min")
    if args.updates < 1 or args.batch_size < 1:
        raise SystemExit("updates and batch-size must be positive")
    device = setup_device(args.device)
    geometry = Geometry()
    root = ROOT / "runs" / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    teacher_folder = ROOT / "runs" / args.teacher_run
    teacher, teacher_checkpoint = load_teacher(teacher_folder, device, args.width,
                                               args.levels)
    from core import fingerprint
    teacher_checkpoint["sha256"] = fingerprint(teacher_folder / "checkpoint.pt")["sha256"]

    if args.stage in ("train", "all"):
        for scale in args.scales:
            local = copy.deepcopy(args)
            local.scale = scale
            name = arm_name(local)
            print(f"=== training {name} ===", flush=True)
            train_arm(local, root / name, teacher, teacher_checkpoint, teacher_folder,
                      device, geometry, scale)
    if args.stage in ("evaluate", "all"):
        result = evaluate(args, root, teacher, device, geometry)
        print(f"Evaluation: {result['evaluation']}", flush=True)


if __name__ == "__main__":
    main()
