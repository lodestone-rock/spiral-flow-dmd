"""Shared primitives for the eight-spiral flow-matching + DMD experiments.

Extracted from the larger ``experiments/toy_2d_diffusion`` codebase so this
folder is self-contained: geometry/data, the conditional CFG U-MLP, linear
flow-path sampling, and the evaluation metrics used by ``train_flow.py`` and
``dmd.py``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

GROUPS = 8


# --------------------------------------------------------------------------- #
# Spiral geometry and data
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Geometry:
    radius: float = 0.55
    radial_std: float = 0.10
    tangential_std: float = 0.035
    twist: float = 6.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Geometry parameters must be finite")
        if self.radius < 0 or min(self.radial_std, self.tangential_std) <= 0:
            raise ValueError("Radius must be nonnegative and standard deviations positive")


def rotate(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    c, s = angle.cos(), angle.sin()
    return torch.stack((c * x[..., 0] - s * x[..., 1],
                        s * x[..., 0] + c * x[..., 1]), dim=-1)


def twist_points(x: torch.Tensor, strength: float) -> torch.Tensor:
    return rotate(x, strength * x.square().sum(dim=-1))


def validate_groups(groups: torch.Tensor) -> None:
    if groups.dtype not in (torch.int32, torch.int64):
        raise ValueError("Group IDs must be integer tensors")
    if ((groups < 0) | (groups > GROUPS - 1)).any():
        raise ValueError(f"Group IDs must be in 0..{GROUPS - 1}")


def encode_groups(groups: torch.Tensor) -> torch.Tensor:
    validate_groups(groups)
    return groups.to(torch.float32).unsqueeze(-1) / (GROUPS - 1)


def sample_data(n: int, geometry: Geometry, generator: torch.Generator,
                device: str | torch.device = "cpu", group: int | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if n <= 0:
        raise ValueError("Sample count must be positive")
    if group is not None and (type(group) is not int or not 0 <= group < GROUPS):
        raise ValueError(f"Group must be an integer in 0..{GROUPS - 1}")
    groups = (torch.randint(GROUPS, (n,), device=device, generator=generator)
              if group is None else torch.full((n,), group, device=device, dtype=torch.long))
    local = torch.randn((n, 2), device=device, generator=generator)
    local = local * local.new_tensor([geometry.radial_std, geometry.tangential_std])
    local[:, 0] += geometry.radius
    base = rotate(local, groups * (2 * math.pi / GROUPS))
    return twist_points(base, geometry.twist), groups, base


def standardized_coordinates(x: torch.Tensor, groups: torch.Tensor,
                             geometry: Geometry) -> torch.Tensor:
    base = twist_points(x, -geometry.twist)
    local = rotate(base, -groups * (2 * math.pi / GROUPS))
    return (local - local.new_tensor([geometry.radius, 0])) / local.new_tensor(
        [geometry.radial_std, geometry.tangential_std])


def log_density(x: torch.Tensor, groups: torch.Tensor, geometry: Geometry) -> torch.Tensor:
    """Exact conditional log density; the twist has unit Jacobian determinant."""
    z = standardized_coordinates(x, groups, geometry)
    return -0.5 * z.square().sum(-1) - math.log(
        2 * math.pi * geometry.radial_std * geometry.tangential_std)


def sample_times(n: int, generator: torch.Generator, device: str | torch.device,
                 mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Logit mean must be finite and standard deviation positive")
    return (mu + sigma * torch.randn((n, 1), device=device, generator=generator)).sigmoid()


def flow_path(data: torch.Tensor, noise: torch.Tensor,
              t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear path x_t=(1-t)data+t*noise from t=1 (noise) to t=0 (data)."""
    return (1 - t) * data + t * noise, noise - data


# --------------------------------------------------------------------------- #
# Conditional CFG U-MLP (same architecture as the original teacher)
# --------------------------------------------------------------------------- #

class NestedMiddle(nn.Module):
    """At every nested level concatenate the early feature with the deeper result."""

    def __init__(self, width: int, levels: int):
        super().__init__()
        if levels < 0 or width < 1:
            raise ValueError("Invalid width/levels")
        if levels == 0:
            self.body = nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                      nn.Linear(width, width), nn.SiLU())
        else:
            self.pre = nn.Sequential(nn.Linear(width, width), nn.SiLU())
            self.middle = NestedMiddle(width, levels - 1)
            self.fuse = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        self.levels = levels

    def forward(self, h):
        if self.levels == 0:
            return self.body(h)
        early = self.pre(h)
        return self.fuse(torch.cat((early, self.middle(early)), dim=-1))


class FullCFGMLP(nn.Module):
    """4-input velocity MLP with class conditioning (c=group/7, c=-1 negative).

    Both CFG branches always execute the complete U-shaped middle; the boolean
    ``full`` argument only substitutes the negative class token.
    """

    def __init__(self, width: int = 128, levels: int = 2):
        super().__init__()
        self.width, self.levels = width, levels
        self.stem = nn.Sequential(nn.Linear(4, width), nn.SiLU())
        self.middle = NestedMiddle(width, levels)
        self.fuse = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        self.output = nn.Linear(width, 2)

    def features(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c is None or c.shape != t.shape:
            raise ValueError("CFG requires c with the same shape as t")
        return torch.cat((x, c, t), dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, full: bool = True,
                c: torch.Tensor | None = None) -> torch.Tensor:
        if type(full) is not bool or c is None:
            raise ValueError("CFG requires a boolean positive/negative selector and a class tensor")
        c = c if full else -torch.ones_like(c)
        early = self.stem(self.features(x, t, c))
        deep = self.middle(early)
        return self.output(self.fuse(torch.cat((early, deep), dim=-1)))

    def pair(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        """Return (negative, positive) with both branches fully executed."""
        return self.forward(torch.cat((x, x)), torch.cat((t, t)),
                            True, torch.cat((-torch.ones_like(c), c))).chunk(2)


def depth_velocity(model: nn.Module, x: torch.Tensor, t: torch.Tensor, scale: float,
                   c: torch.Tensor | None = None) -> torch.Tensor:
    """Ordinary CFG: v_uncond + scale * (v_cond - v_uncond)."""
    kwargs = {} if c is None else {"c": c}
    if scale == 0:
        return model(x, t, full=False, **kwargs)
    if scale == 1:
        return model(x, t, full=True, **kwargs)
    negative, positive = model.pair(x, t, **kwargs)
    return negative + scale * (positive - negative)


@torch.inference_mode()
def sample_depth(model: nn.Module, noise: torch.Tensor, steps: int, scale: float,
                 c: torch.Tensor | None = None) -> torch.Tensor:
    """Heun integration from t=1 to t=0 with frozen CFG scale."""
    if steps < 1 or not math.isfinite(scale):
        raise ValueError("Steps must be positive and scale finite")
    x = noise.clone()
    finite = torch.ones((), dtype=torch.bool, device=x.device)
    dt = -1 / steps
    for i in range(steps):
        t = x.new_full((len(x), 1), 1 - i / steps)
        s = x.new_full((len(x), 1), 1 - (i + 1) / steps)
        v = depth_velocity(model, x, t, scale, c)
        predicted = x + dt * v
        x = x + dt / 2 * (v + depth_velocity(model, predicted, s, scale, c))
        finite &= torch.isfinite(x).all() & torch.isfinite(predicted).all()
    if not finite:
        raise FloatingPointError("Nonfinite guided trajectory")
    return x


def freeze_teacher(model: nn.Module) -> nn.Module:
    model.eval().requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    return model


def state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

CONDITIONAL_KEYS = ("sliced_wasserstein_1", "hdr_50_mass", "hdr_90_mass",
                    "mean_displacement", "median_displacement", "covariance_trace_ratio",
                    "inverse_mean_error", "inverse_mean_norm", "inverse_variance_ratio",
                    "requested_class_assignment_fraction")


def geometric_median(x: np.ndarray) -> np.ndarray:
    y = np.median(x, axis=0)
    for _ in range(200):
        distance = np.maximum(np.linalg.norm(x - y, axis=1), 1e-10)
        updated = (x / distance[:, None]).sum(0) / (1 / distance).sum()
        if np.linalg.norm(updated - y) < 1e-8:
            return updated
        y = updated
    return y


def distribution_metrics(samples: np.ndarray, target: np.ndarray, group: int,
                         geometry: Geometry, directions: np.ndarray) -> dict:
    projections = np.sort(samples @ directions.T, axis=0)
    target_projections = np.sort(target @ directions.T, axis=0)
    cov = np.cov(samples.T)
    target_cov = np.cov(target.T)
    points = torch.from_numpy(samples)
    ids = torch.full((len(points),), group, dtype=torch.long)
    mahal_sq = standardized_coordinates(points, ids, geometry).square().sum(-1).numpy()
    # For a 2D normal, ||z||^2 ~ chi^2(2), CDF 1-exp(-q/2).
    return {"sliced_wasserstein_1": float(np.abs(projections - target_projections).mean()),
            "mean": samples.mean(0).tolist(),
            "geometric_median": geometric_median(samples).tolist(),
            "mean_displacement": float(np.linalg.norm(samples.mean(0) - target.mean(0))),
            "median_displacement": float(np.linalg.norm(geometric_median(samples)
                                                        - geometric_median(target))),
            "covariance": cov.tolist(),
            "covariance_frobenius_error": float(np.linalg.norm(cov - target_cov)),
            "covariance_trace_ratio": float(np.trace(cov) / np.trace(target_cov)),
            "hdr_50_mass": float((mahal_sq <= -2 * math.log(0.5)).mean()),
            "hdr_90_mass": float((mahal_sq <= -2 * math.log(0.1)).mean()),
            "mean_target_log_density": float(log_density(points, ids, geometry).mean()),
            "radius_99": float(np.quantile(np.linalg.norm(samples, axis=1), 0.99))}


def mixture_diagnostics(x, geometry: Geometry):
    points = torch.as_tensor(x, dtype=torch.float64)
    logs = torch.stack([log_density(points, torch.full((len(points),), g,
                                                       dtype=torch.long), geometry)
                        for g in range(GROUPS)], dim=-1)
    return (logs.logsumexp(-1) - math.log(GROUPS)).numpy(), logs.argmax(-1).numpy()


def sw1(x, ref, directions) -> float:
    if len(x) != len(ref):
        raise ValueError("SW1 reference and sample counts must match")
    return float(np.abs(np.sort(x @ directions.T, axis=0)
                        - np.sort(ref @ directions.T, axis=0)).mean())


def conditional_metrics(x, ref, group, geometry: Geometry, directions) -> dict:
    """No reassignment/filtering: every sample is scored against its requested class."""
    row = distribution_metrics(x, ref, group, geometry, directions)
    ids = torch.full((len(x),), group, dtype=torch.long)
    z = standardized_coordinates(torch.from_numpy(x), ids, geometry).numpy()
    zr = standardized_coordinates(torch.from_numpy(ref), ids, geometry).numpy()
    _, assigned = mixture_diagnostics(x, geometry)
    row.update({"group": group, "count": len(x), "inverse_mean": z.mean(0).tolist(),
                "inverse_variances": z.var(0).tolist(),
                "inverse_reference_mean": zr.mean(0).tolist(),
                "inverse_mean_norm": float(np.linalg.norm(z.mean(0))),
                "inverse_mean_error": float(np.linalg.norm(z.mean(0) - zr.mean(0))),
                "inverse_variance_ratio": float(z.var(0).sum() / zr.var(0).sum()),
                "requested_class_assignment_fraction": float((assigned == group).mean())})
    return row


def average_classes(rows) -> dict:
    return {key: float(np.mean([r[key] for r in rows])) for key in CONDITIONAL_KEYS}


def measure_grid(x: np.ndarray, refs: np.ndarray, geometry: Geometry,
                 directions: np.ndarray) -> dict:
    """Per-class conditional metrics for an (8, n, 2) sample grid."""
    rows = [conditional_metrics(x[g], refs[g], g, geometry, directions)
            for g in range(GROUPS)]
    return {"by_class": rows, "average": average_classes(rows)}


def describe_samples(x, ref, geometry: Geometry, directions, thresholds) -> dict:
    logp, ids = mixture_diagnostics(x, geometry)
    _, ref_ids = mixture_diagnostics(ref, geometry)
    groups = []
    for g in range(GROUPS):
        local, local_ref = x[ids == g], ref[ref_ids == g]
        row = {"group": g, "count": len(local), "fraction": len(local) / len(x)}
        if len(local) >= 3 and len(local_ref) >= 3:
            cov = np.cov(local.T)
            mean, median = local.mean(0), geometric_median(local)
            z = standardized_coordinates(
                torch.from_numpy(local),
                torch.full((len(local),), g, dtype=torch.long), geometry).numpy()
            z_ref = standardized_coordinates(
                torch.from_numpy(local_ref),
                torch.full((len(local_ref),), g, dtype=torch.long), geometry).numpy()
            row.update({"inverse_gaussian_reference_mean": z_ref.mean(0).tolist(),
                        "inverse_gaussian_mean_error": float(np.linalg.norm(z.mean(0)
                                                                            - z_ref.mean(0))),
                        "inverse_gaussian_variance_ratio": float(z.var(0).sum()
                                                                 / z_ref.var(0).sum()),
                        "mean": mean.tolist(), "median": median.tolist(),
                        "covariance_trace": float(np.trace(cov)),
                        "covariance_trace_ratio": float(np.trace(cov)
                                                        / np.trace(np.cov(local_ref.T))),
                        "mean_error": float(np.linalg.norm(mean - local_ref.mean(0))),
                        "median_error": float(np.linalg.norm(median
                                                             - geometric_median(local_ref))),
                        "inverse_gaussian_mean": z.mean(0).tolist(),
                        "inverse_gaussian_variances": z.var(0).tolist()})
        groups.append(row)
    valid = [g for g in groups if "mean_error" in g]
    averages = {key: float(np.mean([g[key] for g in valid])) if len(valid) == GROUPS else None
                for key in ("covariance_trace_ratio", "mean_error", "median_error",
                            "inverse_gaussian_mean_error", "inverse_gaussian_variance_ratio")}
    return {"sw1": sw1(x, ref, directions),
            "hdr50_mass": float((logp >= thresholds["50"]).mean()),
            "hdr90_mass": float((logp >= thresholds["90"]).mean()),
            "mean_log_target_density": float(logp.mean()),
            "mean": x.mean(0).tolist(),
            "global_covariance_trace": float(np.trace(np.cov(x.T))),
            "radius_quantiles_50_90_99": np.quantile(np.linalg.norm(x, axis=1),
                                                     [.5, .9, .99]).tolist(),
            "outside_plot_fraction": float((np.abs(x).max(1) > 1.2).mean()),
            "max_group_fraction_error": float(max(abs(g["fraction"] - 1 / GROUPS)
                                                  for g in groups)),
            "valid_local_groups": len(valid), "local_averages": averages, "groups": groups}


def target_view_bounds(target, quantile: float = .005, padding: float = .08):
    """Square, target-only window; outlier counts must accompany cropped plots."""
    lo, hi = np.quantile(target.reshape(-1, 2), [quantile, 1 - quantile], axis=0)
    center = (lo + hi) / 2
    half = max(float((hi - lo).max()) / 2, .01) * (1 + 2 * padding)
    return (center - half).tolist(), (center + half).tolist()


def annotate_outside(ax, x, bounds) -> int:
    lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
    points = x.reshape(-1, 2)
    count = int(((points < lo) | (points > hi)).any(axis=1).sum())
    ax.text(.03, .97, f"Outside view: {count / len(points):.2%} ({count})",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
    return count


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def make_generator(device: str | torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def setup_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    device = torch.device(name)
    torch.set_num_threads(4)
    print(f"torch={torch.__version__} device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    return device
