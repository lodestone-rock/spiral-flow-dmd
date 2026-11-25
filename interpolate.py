"""Frozen clockwise class-condition interpolation for the spiral CFG teacher.

The teacher only ever saw integer groups g=0..7 (c=g/7). This stage freezes it
and sweeps the scalar condition through 29 fractional values in *decreasing*
order, g=7, 6.75, ..., 0, which is clockwise in this dataset. The same initial
noise is used in every panel, so the figure isolates the learned condition map.

Intermediate rotations are a hypothetical geometric continuation (rotate the
group-0 target), NOT training ground truth: there are no examples at fractional
groups, and there is no periodic wrap through g=0 -> 7 in the scalar encoding.

No retraining: this reads runs/<teacher-run>/checkpoint.pt.

From this folder:
  python interpolate.py --device cuda --teacher-run teacher --scales 1 2

Artifacts: runs/<teacher-run>/interpolation_s<steps>_n<n>_seed<seed>/
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import (FullCFGMLP, Geometry, annotate_outside, distribution_metrics,
                  make_generator, sample_data, sample_depth, save_json, setup_device,
                  standardized_coordinates, target_view_bounds)

ROOT = Path(__file__).resolve().parent


def clockwise_grid(subdivisions: int) -> np.ndarray:
    """29 conditions for the default 4 subdivisions, ordered 7 -> 0 (clockwise)."""
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    return np.linspace(7.0, 0.0, 7 * subdivisions + 1)


def rotate_numpy(x: np.ndarray, angle: float) -> np.ndarray:
    """Same convention as core.rotate for float64 NumPy arrays (CCW for angle>0)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(x) @ np.array([[c, s], [-s, c]])


def reference_sweep(base: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.stack([rotate_numpy(base, float(g) * math.pi / 4) for g in groups])


@torch.inference_mode()
def sample_conditions(model, noise: torch.Tensor, groups: np.ndarray, steps: int,
                      scale: float, chunk: int = 8) -> np.ndarray:
    """Sample each fractional condition with identical initial noise."""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    result = []
    for start in range(0, len(groups), chunk):
        selected = groups[start:start + chunk]
        c = noise.new_tensor(selected / 7).repeat_interleave(len(noise)).reshape(-1, 1)
        x = sample_depth(model, noise.repeat(len(selected), 1), steps, scale, c)
        result.append(x.reshape(len(selected), len(noise), 2).cpu().double().numpy())
    return np.concatenate(result)


def condition_metrics(x: np.ndarray, reference0: np.ndarray, group: float,
                      geometry: Geometry, directions: np.ndarray) -> dict:
    """Canonicalize by the inverse rotation, then score against the group-0 target."""
    canonical = rotate_numpy(x, -float(group) * math.pi / 4)
    row = distribution_metrics(canonical, reference0, 0, geometry, directions)
    z = standardized_coordinates(torch.from_numpy(canonical),
                                 torch.zeros(len(canonical), dtype=torch.long),
                                 geometry).numpy()
    zr = standardized_coordinates(torch.from_numpy(reference0),
                                  torch.zeros(len(reference0), dtype=torch.long),
                                  geometry).numpy()
    return {"sliced_wasserstein_1": row["sliced_wasserstein_1"],
            "hdr_50_mass": row["hdr_50_mass"],
            "inverse_center_norm": float(np.linalg.norm(z.mean(0))),
            "inverse_variance_ratio": float(z.var(0).sum() / zr.var(0).sum())}


def analyze(samples: np.ndarray, reference0: np.ndarray, groups: np.ndarray,
            geometry: Geometry, directions: np.ndarray) -> dict:
    rows = [condition_metrics(x, reference0, g, geometry, directions)
            for x, g in zip(samples, groups)]
    trained = np.isclose(groups, np.round(groups))
    means = samples.mean(1)
    angles = np.unwrap(np.arctan2(means[:, 1], means[:, 0]))
    angular_steps = np.diff(angles)
    reference = reference_sweep(reference0, groups)
    return {
        "by_condition": rows,
        "means": means.tolist(),
        "reference_means": reference.mean(1).tolist(),
        "covariance_traces": [float(np.trace(np.cov(x.T))) for x in samples],
        "trained": {k: float(np.mean([r[k] for r, f in zip(rows, trained) if f]))
                    for k in rows[0]},
        "unseen": {k: float(np.mean([r[k] for r, f in zip(rows, trained) if not f]))
                   for k in rows[0]},
        "mean_angle_steps_degrees": np.rad2deg(angular_steps).tolist(),
        "counterclockwise_mean_step_count": int((angular_steps > 0).sum()),
        "total_mean_rotation_degrees": float(np.rad2deg(angles[-1] - angles[0])),
        "minimum_mean_radius": float(np.linalg.norm(means, axis=1).min()),
        "paired_adjacent_rms_displacement":
            np.sqrt(np.mean(np.sum(np.diff(samples, axis=0) ** 2, axis=-1), axis=1)).tolist(),
        "geometric_reference_mean_error":
            np.linalg.norm(means - reference.mean(1), axis=-1).tolist(),
        "caveat": "Only integer groups were trained. Rotated intermediate targets are "
                  "a hypothetical geometric continuation, not ground truth. No "
                  "periodic 7-to-0 scalar interval exists.",
    }


def plot_all_intervals(path: Path, samples: np.ndarray, reference: np.ndarray,
                       groups: np.ndarray, scale: float, seed: int,
                       subdivisions: int) -> None:
    bounds = target_view_bounds(reference)
    lo, hi = bounds
    fig, axs = plt.subplots(7, subdivisions + 1,
                            figsize=(3.1 * (subdivisions + 1), 19), squeeze=False)
    for interval in range(7):
        for j in range(subdivisions + 1):
            index = interval * subdivisions + j
            ax = axs[interval, j]
            ax.scatter(*reference[index].T, s=1, alpha=.20, color="gray", zorder=1)
            ax.scatter(*samples[index, ::2].T, s=1.5, alpha=.35, color="tab:blue", zorder=2)
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal",
                   xlabel="x", ylabel="y" if j == 0 else "",
                   title=f"g={groups[index]:g}, c={groups[index] / 7:.4f}"
                         + (" · trained" if float(groups[index]).is_integer() else " · unseen"))
            annotate_outside(ax, samples[index], bounds)
    axs[0, 0].scatter([], [], s=8, color="gray", alpha=.4, label="Geometric reference")
    axs[0, 0].scatter([], [], s=8, color="tab:blue", alpha=.7, label="Generated")
    axs[0, 0].legend(fontsize=6, loc="upper right")
    fig.suptitle(f"Frozen clockwise condition interpolation 7 → 0 · CFG w={scale:g} · "
                 f"seed {seed}\nColumns go clockwise within each interval; same initial "
                 f"noise everywhere; gray is a hypothetical geometric continuation",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .965))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_interval(path: Path, samples: np.ndarray, reference: np.ndarray,
                  groups: np.ndarray, scale: float, seed: int,
                  start: float = 4.0, stop: float = 3.0) -> None:
    mask = (groups <= start) & (groups >= stop)
    selected = np.flatnonzero(mask)
    bounds = target_view_bounds(reference[selected])
    lo, hi = bounds
    fig, axs = plt.subplots(1, len(selected), figsize=(3.5 * len(selected), 4.2),
                            squeeze=False)
    for ax, index in zip(axs[0], selected):
        ax.scatter(*reference[index].T, s=1, alpha=.20, color="gray", zorder=1)
        ax.scatter(*samples[index, ::2].T, s=1.8, alpha=.40, color="tab:orange", zorder=2)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal",
               xlabel="x", ylabel="y",
               title=f"g={groups[index]:g}, c={groups[index] / 7:.4f}"
                     + (" · trained" if float(groups[index]).is_integer() else " · unseen"))
        annotate_outside(ax, samples[index], bounds)
    axs[0, 0].scatter([], [], s=8, color="gray", alpha=.4, label="Geometric reference")
    axs[0, 0].scatter([], [], s=8, color="tab:orange", alpha=.7, label="Generated")
    axs[0, 0].legend(fontsize=6, loc="upper right")
    fig.suptitle(f"Group {start:g} → {stop:g}: unseen conditions · frozen CFG w={scale:g} · "
                 f"seed {seed}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_paths(path: Path, samples: np.ndarray, reference: np.ndarray,
               groups: np.ndarray, scale: float, seed: int,
               start: float = 4.0, stop: float = 3.0) -> None:
    mask = (groups <= start) & (groups >= stop)
    selected = np.flatnonzero(mask)
    means = samples.mean(1)
    reference_means = reference.mean(1)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    axs[0].plot(*reference_means.T, "--", color="gray", label="Geometric reference mean")
    axs[0].plot(*means.T, ".-", color="tab:blue", label="Generated mean")
    for index in range(0, len(groups), 4):
        axs[0].annotate(f"{groups[index]:g}", means[index], xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
    axs[0].set(xlim=(-.65, .65), ylim=(-.65, .65), aspect="equal",
               title="Mean path across decreasing g", xlabel="x", ylabel="y")
    axs[0].legend(fontsize=8)
    axs[1].plot(groups, [float(np.trace(np.cov(x.T))) for x in samples], ".-")
    axs[1].axhline(float(np.trace(np.cov(reference[0].T))), linestyle="--",
                   color="gray", label="Reference")
    axs[1].invert_xaxis()
    axs[1].set(xlabel="Group g (clockwise →)", ylabel="Covariance trace",
               title="Spread across conditions")
    axs[1].legend(fontsize=8)
    n_show = min(48, samples.shape[1])
    for i in range(n_show):
        axs[2].plot(*samples[selected, i].transpose(1, 0), color="tab:blue",
                    alpha=.30, linewidth=.8)
    axs[2].scatter(*samples[selected[0], :n_show].T, s=14, color="gray",
                   label=f"g={start:g} start")
    axs[2].scatter(*samples[selected[-1], :n_show].T, s=14, color="tab:blue",
                   label=f"g={stop:g} end")
    axs[2].set(aspect="equal", title=f"Matched-noise paths: {start:g} → {stop:g}",
               xlabel="x", ylabel="y")
    axs[2].legend(fontsize=8)
    fig.suptitle(f"Clockwise interpolation diagnostics · CFG w={scale:g} · seed {seed}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run", default="teacher")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--subdivisions", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args(argv)

    device = setup_device(args.device)
    geometry = Geometry()
    folder = ROOT / "runs" / args.teacher_run
    checkpoint = torch.load(folder / "checkpoint.pt", map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    arguments = cfg["arguments"]
    if arguments.get("guidance") != "cfg" or not arguments.get("class_conditional", False):
        raise SystemExit("Interpolation requires the class-conditional CFG teacher")
    model = FullCFGMLP(arguments["width"], arguments["levels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    destination = folder / (args.output_name or
                            f"interpolation_s{args.steps}_n{args.eval_samples}"
                            f"_seed{args.seed}")
    destination.mkdir(parents=True, exist_ok=False)

    groups = clockwise_grid(args.subdivisions)
    gen = make_generator(device, args.seed + 200)
    noise = torch.randn((args.eval_samples, 2), device=device, generator=gen)
    reference0 = sample_data(args.eval_samples, geometry,
                             make_generator("cpu", args.seed + 201), group=0)[0]
    reference = reference_sweep(reference0.double().numpy(), groups)
    directions = np.random.default_rng(args.seed + 202).normal(size=(64, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    metrics = {"seed": args.seed, "steps": args.steps,
               "samples_per_condition": args.eval_samples,
               "teacher_run": args.teacher_run,
               "teacher_step": checkpoint.get("step"),
               "subdivisions": args.subdivisions,
               "fractional_group_ids": groups.tolist(),
               "conditions": (groups / 7).tolist(),
               "scales": {}}
    for scale in args.scales:
        samples = sample_conditions(model, noise, groups, args.steps, scale, args.chunk)
        tag = f"w{scale:g}"
        np.savez_compressed(destination / f"samples_{tag}.npz",
                            samples=samples, reference=reference, conditions=groups / 7,
                            initial_noise=noise.cpu().double().numpy())
        metrics["scales"][f"{scale:g}"] = analyze(samples, reference0.double().numpy(),
                                                  groups, geometry, directions)
        plot_all_intervals(destination / f"clockwise_all_intervals_{tag}.png",
                           samples, reference, groups, scale, args.seed, args.subdivisions)
        plot_interval(destination / f"clockwise_4_to_3_{tag}.png",
                      samples, reference, groups, scale, args.seed)
        plot_paths(destination / f"clockwise_paths_{tag}.png",
                   samples, reference, groups, scale, args.seed)
        result = metrics["scales"][f"{scale:g}"]
        print(f"w={scale:g}: total mean rotation "
              f"{result['total_mean_rotation_degrees']:.1f}° "
              f"(clockwise steps {len(result['mean_angle_steps_degrees']) - result['counterclockwise_mean_step_count']}"
              f"/{len(result['mean_angle_steps_degrees'])}), "
              f"unseen SW1 {result['unseen']['sliced_wasserstein_1']:.5f}, "
              f"unseen HDR50 {result['unseen']['hdr_50_mass']:.1%}", flush=True)
    save_json(destination / "metrics.json", metrics)
    print(f"Interpolation saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
