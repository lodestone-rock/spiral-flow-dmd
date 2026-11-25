"""Train the conditional CFG flow-matching teacher on the eight-arm spiral.

Stage 1 of the standalone pipeline. The model is the same 4-input U-MLP used by
the original experiments: ``[x, y, c, t]`` with ``c = group/7`` and ``c = -1``
for the unconditional (negative) branch, trained with linear flow matching,
logit-normal times, and 10% class dropout.

From this folder:
  python train_flow.py train --device cuda --run-name teacher
  python train_flow.py evaluate --device cuda --run-name teacher

Artifacts land in ``runs/<run-name>/``: checkpoint.pt, config.json, history.json,
training_loss.png, teacher_samples.png, teacher_metrics.json.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import (FullCFGMLP, GROUPS, Geometry, describe_samples, encode_groups,
                  flow_path, make_generator, measure_grid, mixture_diagnostics,
                  sample_data, sample_depth, sample_times, save_json, setup_device,
                  target_view_bounds)

ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def validate(model, geometry: Geometry, device, seed: int, n: int = 8192) -> dict:
    gen = make_generator(device, seed)
    data, ids, _ = sample_data(n, geometry, gen, device)
    c = encode_groups(ids)
    noise = torch.randn(data.shape, device=device, generator=gen)
    t = sample_times(n, gen, device)
    x, target = flow_path(data, noise, t)
    negative, positive = model.pair(x, t, c)
    result = {"route_velocity_rms_difference": (positive - negative).square().mean().sqrt().item()}
    for name, prediction in (("unconditional", negative), ("conditional", positive)):
        error = (prediction - target).square().mean(-1)
        result[name] = {"mse": error.mean().item(), "time_bins": []}
        for i in range(10):
            mask = (t[:, 0] >= i / 10) & (t[:, 0] < (i + 1) / 10)
            result[name]["time_bins"].append({
                "lower": i / 10, "count": int(mask.sum()),
                "mse": error[mask].mean().item() if mask.any() else None})
    return result


def train(args, folder: Path, device: torch.device, geometry: Geometry) -> None:
    if (folder / "checkpoint.pt").exists():
        raise FileExistsError(f"Refusing to overwrite {folder / 'checkpoint.pt'}")
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = FullCFGMLP(args.width, args.levels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    gen = make_generator(device, args.seed + 1)

    # Fixed-count IID split: same recipe as the original CFG teacher.
    full_count = round(args.batch_size * (1 - args.class_dropout))
    weak_count = args.batch_size - full_count
    config = {
        "arguments": {**vars(args), "guidance": "cfg", "class_conditional": True},
        "seed": args.seed, "geometry": asdict(geometry),
        "architecture": "stem 4->W; full NestedMiddle(levels) on BOTH branches; "
                        "concat(stem,middle)->W->2",
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "route_examples_per_update": {"conditional": full_count,
                                      "unconditional": weak_count},
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    save_json(folder / "config.json", config)

    history = [{"step": 0, "validation": validate(model, geometry, device, args.seed + 2)}]
    start = time.perf_counter()
    running = torch.zeros(2, device=device)
    for step in range(1, args.train_steps + 1):
        data, ids, _ = sample_data(args.batch_size, geometry, gen, device)
        c = encode_groups(ids)
        noise = torch.randn(data.shape, device=device, generator=gen)
        t = sample_times(len(data), gen, device)
        x, target = flow_path(data, noise, t)
        optimizer.zero_grad(set_to_none=True)
        full_loss = (model(x[:full_count], t[:full_count], True, c[:full_count])
                     - target[:full_count]).square().mean()
        weak_loss = (model(x[full_count:], t[full_count:], False, c[full_count:])
                     - target[full_count:]).square().mean()
        loss = (full_count * full_loss + weak_count * weak_loss) / args.batch_size
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                                    error_if_nonfinite=True)
        optimizer.step()
        running += torch.stack((weak_loss.detach(), full_loss.detach()))
        if step % args.log_every == 0 or step == args.train_steps:
            count = step % args.log_every or args.log_every
            val = validate(model, geometry, device, args.seed + 2)
            record = {"step": step,
                      "train_unconditional_mse": (running[0] / count).item(),
                      "train_conditional_mse": (running[1] / count).item(),
                      "grad_norm": grad_norm.item(),
                      "elapsed_seconds": time.perf_counter() - start,
                      "validation": val}
            history.append(record)
            running.zero_()
            save_json(folder / "history.json", history)
            print(f"seed={args.seed} step={step}/{args.train_steps} "
                  f"u={val['unconditional']['mse']:.6f} "
                  f"c={val['conditional']['mse']:.6f} "
                  f"elapsed={record['elapsed_seconds']:.1f}s", flush=True)

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "rng": gen.get_state()}, folder / "checkpoint.pt")
    save_json(folder / "history.json", history)

    fig, ax = plt.subplots(figsize=(8, 4))
    for route in ("unconditional", "conditional"):
        ax.plot([h["step"] for h in history],
                [h["validation"][route]["mse"] for h in history], label=route)
    ax.set(xlabel="Optimizer updates", ylabel="Held-out velocity MSE per coordinate",
           title=f"Spiral CFG flow matching, seed {args.seed}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "training_loss.png", dpi=150)
    plt.close(fig)
    print(f"Saved checkpoint: {folder / 'checkpoint.pt'}", flush=True)


@torch.inference_mode()
def evaluate(args, folder: Path, device: torch.device, geometry: Geometry) -> dict:
    checkpoint = torch.load(folder / "checkpoint.pt", map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    model = FullCFGMLP(cfg["arguments"]["width"],
                                         cfg["arguments"]["levels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    n = args.eval_samples
    gen = make_generator(device, args.seed + 10)
    noise = torch.randn((n * GROUPS, 2), device=device, generator=gen)
    groups = torch.arange(GROUPS, device=device).repeat_interleave(n)
    c = encode_groups(groups)
    cpu_gen = make_generator("cpu", args.seed + 11)
    refs = np.stack([sample_data(n, geometry, cpu_gen, group=g)[0].double().numpy()
                     for g in range(GROUPS)])
    calibration = sample_data(100000, geometry, cpu_gen)[0].double().numpy()
    overlay = sample_data(40000, geometry, make_generator("cpu", args.seed + 12))[0] \
        .double().numpy()
    logs, _ = mixture_diagnostics(calibration, geometry)
    thresholds = {"50": float(np.quantile(logs, .5)), "90": float(np.quantile(logs, .1))}
    directions = np.random.default_rng(args.seed + 13).normal(size=(64, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    scales = (0.0, 1.0, 2.0, 4.0)
    samples = {scale: sample_depth(model, noise, args.steps, scale, c)
               .reshape(GROUPS, n, 2).cpu().double().numpy() for scale in scales}
    metrics = {"seed": args.seed, "steps": args.steps, "samples_per_class": n,
               "scales": {}}
    for scale in scales:
        metrics["scales"][f"{scale:g}"] = {
            "vs_target": measure_grid(samples[scale], refs, geometry, directions),
            "mixture": describe_samples(samples[scale].reshape(-1, 2),
                                        refs.reshape(-1, 2), geometry, directions,
                                        thresholds)}
    save_json(folder / "teacher_metrics.json", metrics)

    colors = plt.get_cmap("tab10")(np.arange(GROUPS))
    lo, hi = target_view_bounds(refs.reshape(-1, 2))
    panels = [("Target", refs, "lightgray")] + [
        (f"CFG w={scale:g}", samples[scale], None) for scale in scales]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.7),
                            squeeze=False)
    for ax, (title, data, color) in zip(axs[0], panels):
        for g in range(GROUPS):
            ax.scatter(*data[g, ::2].T, s=1.4, alpha=.55,
                       color=color or colors[g], zorder=2)
        ax.scatter(*overlay[::6].T, s=.7, alpha=.08, color="lightgray", zorder=1)
        ax.set(title=title, xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
               aspect="equal", xlabel="x", ylabel="y" if ax is axs[0, 0] else "")
    for scale in scales:
        row = metrics["scales"][f"{scale:g}"]["vs_target"]["average"]
        axs[0][list(scales).index(scale) + 1].text(
            .03, .03, f"SW1 {row['sliced_wasserstein_1']:.4f}", transform=axs[0][
                list(scales).index(scale) + 1].transAxes, fontsize=8,
            bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
    fig.suptitle(f"Spiral CFG flow-matching teacher · seed {args.seed} · "
                 f"Heun-{args.steps} · near-origin spiral underlay", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(folder / "teacher_samples.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {folder / 'teacher_samples.png'}", flush=True)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "all"), nargs="?",
                        default="all")
    parser.add_argument("--run-name", default="teacher")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--class-dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=256)
    args = parser.parse_args(argv)
    if not 0 <= args.class_dropout < 1:
        raise SystemExit("class-dropout must lie in [0,1)")
    device = setup_device(args.device)
    geometry = Geometry()
    folder = ROOT / "runs" / args.run_name
    if args.stage in ("train", "all"):
        train(args, folder, device, geometry)
    if args.stage in ("evaluate", "all"):
        evaluate(args, folder, device, geometry)


if __name__ == "__main__":
    main()
