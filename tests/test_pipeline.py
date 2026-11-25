"""Focused unit tests for the standalone spiral flow-matching + DMD pipeline.

Run from the parent of this folder:
  python -m unittest discover -s experiments/spiral_flow_dmd/tests -v
or from inside the folder:
  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from core import (FullCFGMLP, Geometry, encode_groups, log_density, sample_data,
                  state_digest, standardized_coordinates)
from dmd import (DMD2_SCHEDULE, WEIGHT_MODES, denoise, dm_per_sample,
                 heun_snapshots, initialize_copy, renoise, rollout,
                 score_from_velocity, uniform_times, validate_schedule,
                 validate_teacher_checkpoint)


class TinyVelocity(nn.Module):
    """Minimal velocity model with the house (x, t, full, c) call signature."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x, t, full=True, c=None):
        return self.net(torch.cat((x, c, t), dim=-1))


def make_args(**overrides):
    defaults = dict(run_name="dmd_test", device="cpu", seed=42,
                    teacher_run="teacher", teacher_seed=42, width=128, levels=2,
                    gen_updates=2, batch_size=64, lr=1e-3, regression_lambda=0.25,
                    pairs=256, pair_steps=4, weight="kl", cs=1.0, dm_low=0.05,
                    dm_high=0.9, log_every=1, probe_every=1, grad_clip=10.0,
                    l1_floor=0.05, schedule=list(DMD2_SCHEDULE))
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class DataTests(unittest.TestCase):
    def test_geometry_validates(self):
        with self.assertRaises(ValueError):
            Geometry(radial_std=0)
        with self.assertRaises(ValueError):
            Geometry(radius=float("nan"))

    def test_groups_roundtrip(self):
        g = torch.arange(8)
        c = encode_groups(g)
        torch.testing.assert_close(c[:, 0], g.float() / 7)
        with self.assertRaises(ValueError):
            encode_groups(torch.tensor([8]))

    def test_standardized_coordinates_and_density(self):
        geometry = Geometry()
        gen = torch.Generator().manual_seed(0)
        x, ids, _ = sample_data(2048, geometry, gen, group=3)
        z = standardized_coordinates(x, ids, geometry)
        self.assertTrue(torch.isfinite(z).all())
        # Inverse map is exact for the class-local coordinates.
        self.assertEqual(z.shape, (2048, 2))
        logp = log_density(x, ids, geometry)
        expected = -0.5 * (z * z).sum(-1) - np.log(
            2 * np.pi * geometry.radial_std * geometry.tangential_std)
        torch.testing.assert_close(logp, torch.as_tensor(expected), rtol=1e-6, atol=1e-7)
        self.assertGreater(float(logp.mean()), -2.0)


class DMDPrimitiveTests(unittest.TestCase):
    def test_denoiser_recovers_clean_point(self):
        x0, eps = torch.randn(5, 2), torch.randn(5, 2)
        t = torch.rand(5, 1) * 0.9 + 0.05
        x_t = (1 - t) * x0 + t * eps
        torch.testing.assert_close(denoise(x_t, t, eps - x0), x0)

    def test_score_relation(self):
        torch.manual_seed(0)
        eps = torch.randn(64, 2)
        t = torch.full((64, 1), 0.3)
        torch.testing.assert_close(score_from_velocity(t * eps, t, eps), -eps / t,
                                   rtol=1e-6, atol=1e-7)
        v = torch.randn(64, 2)
        x_t = torch.randn(64, 2)
        torch.testing.assert_close(score_from_velocity(x_t, t, v),
                                   -(x_t + 0.7 * v) / t, rtol=1e-6, atol=1e-7)

    def test_schedule_validation(self):
        self.assertEqual(validate_schedule(DMD2_SCHEDULE), DMD2_SCHEDULE)
        for bad in ((0.5, 0.5), (0.5, 0.6), (1.2, 0.5), (0.9, 0.0), ()):
            with self.assertRaises(ValueError):
                validate_schedule(bad)

    def test_uniform_times_bounds(self):
        t = uniform_times(20000, torch.Generator().manual_seed(0), "cpu", 0.02, 0.98)
        self.assertTrue(bool((t >= 0.02).all()) and bool((t <= 0.98).all()))
        with self.assertRaises(ValueError):
            uniform_times(4, torch.Generator().manual_seed(7), "cpu", 0.9, 0.1)

    def test_renoise_moments(self):
        x = renoise(torch.ones(200000, 2), 0.4, torch.Generator().manual_seed(7))
        self.assertAlmostEqual(float(x.mean()), 0.6, delta=0.01)
        self.assertAlmostEqual(float(x.std()), 0.4, delta=0.01)

    def test_kl_weight_closed_form(self):
        x0, tau = torch.randn(6, 2), torch.full((6, 1), 0.4)
        eps, v_real, v_fake = torch.randn(6, 2), torch.randn(6, 2), torch.randn(6, 2)
        x_t = 0.6 * x0 + 0.4 * eps
        expected = (0.6 * (score_from_velocity(x_t, tau, v_fake)
                           - score_from_velocity(x_t, tau, v_real)) * x0).sum(-1)
        torch.testing.assert_close(dm_per_sample(x0, x_t, tau, v_real, v_fake, "kl"),
                                   expected)

    def test_weight_modes(self):
        self.assertEqual(WEIGHT_MODES, ("paper", "kl", "sigma_over_alpha"))
        x0, tau = torch.randn(4, 2), torch.full((4, 1), 0.5)
        x_t = torch.randn(4, 2)
        with self.assertRaises(ValueError):
            dm_per_sample(x0, x_t, tau, x0, x0, "bogus")

    def test_gradient_zero_when_fake_equals_real(self):
        student, critic = TinyVelocity(), TinyVelocity()
        z = torch.randn(4096, 2)
        tau = torch.rand((4096, 1)) * 0.9 + 0.05
        eps = torch.randn(4096, 2)
        c = torch.zeros(4096, 1)
        x0 = student(z, tau, True, c)
        with torch.no_grad():
            v = critic((1 - tau) * x0.detach() + tau * eps, tau, True, c)
        x_t = (1 - tau) * x0 + tau * eps
        loss = dm_per_sample(x0, x_t, tau, v, v, "kl").mean()
        grads = torch.autograd.grad(loss, list(student.parameters())
                                    + list(critic.parameters()), allow_unused=True)
        self.assertTrue(all(g is None or float(g.abs().max()) < 1e-9 for g in grads))


class RolloutTests(unittest.TestCase):
    SCHEDULE = (0.9, 0.6, 0.3, 0.1)

    def test_grad_masks(self):
        student = TinyVelocity()
        z, c = torch.randn(8, 2), torch.full((8, 1), 0.5)
        x0s, _, _ = rollout(student, z, c, self.SCHEDULE, None, grad_step=1)
        self.assertEqual([x.requires_grad for x in x0s],
                         [False, True, False, False])
        x0s, _, final = rollout(student, z, c, self.SCHEDULE, None, grad_step="all")
        self.assertTrue(all(x.requires_grad for x in x0s))
        self.assertTrue(any(g is not None for g in
                            torch.autograd.grad(final.sum(), list(student.parameters()),
                                                allow_unused=True)))

    def test_inference_states(self):
        student = TinyVelocity()
        z, c = torch.randn(12, 2), torch.zeros(12, 1)
        x0s, states, final = rollout(student, z, c, self.SCHEDULE, None)
        self.assertEqual((len(x0s), len(states)), (4, 4))
        self.assertFalse(any(x.requires_grad for x in x0s))
        self.assertEqual(states[0][0], 0.6)
        self.assertEqual(states[-1][0], 0.0)
        torch.testing.assert_close(states[-1][1], final, rtol=1e-6, atol=1e-7)

    def test_bad_grad_step(self):
        with self.assertRaises(ValueError):
            rollout(TinyVelocity(), torch.randn(4, 2), torch.zeros(4, 1),
                    (0.9, 0.5), None, grad_step=5)


class HeunTests(unittest.TestCase):
    def test_final_equals_sample_depth(self):
        from core import sample_depth
        torch.manual_seed(0)
        teacher = FullCFGMLP(16, 1).eval()
        noise = torch.randn(16, 2)
        c = encode_groups(torch.arange(8).repeat_interleave(2))
        snaps = heun_snapshots(teacher, noise, 40, 2.0, c, (10, 20, 30, 40))
        reference = sample_depth(teacher, noise, 40, 2.0, c)
        torch.testing.assert_close(snaps[40], reference, rtol=1e-6, atol=1e-7)


class CopyAndValidationTests(unittest.TestCase):
    def test_copy_independent_and_exact(self):
        teacher = FullCFGMLP(8, 1)
        student = initialize_copy(teacher, 8, 1, "cpu")
        self.assertEqual(state_digest(student), state_digest(teacher))
        for a, b in zip(student.parameters(), teacher.parameters()):
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
        with torch.no_grad():
            next(teacher.parameters()).add_(1.0)
        self.assertNotEqual(state_digest(student), state_digest(teacher))

    def test_validate_teacher_checkpoint(self):
        args = make_args()
        good = {"config": {"arguments": {"guidance": "cfg", "class_conditional": True,
                                         "width": 128, "levels": 2},
                           "seed": 42, "geometry": asdict(Geometry())}}
        validate_teacher_checkpoint(good, args, 42)
        bad = {"config": {**good["config"],
                          "arguments": {**good["config"]["arguments"],
                                        "guidance": "depth"}}}
        with self.assertRaises(ValueError):
            validate_teacher_checkpoint(bad, args, 42)


if __name__ == "__main__":
    unittest.main()
