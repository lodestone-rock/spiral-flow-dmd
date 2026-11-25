"""Unit tests for Parallel Decoding Distillation on the spiral.

Run from the parent of this folder:
  python -m unittest discover -s experiments/spiral_flow_dmd/tests -v
or from inside the folder:
  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from core import FullCFGMLP, Geometry, depth_velocity, state_digest
from pdd import (DataFreeCarry, PDDGrid, PDDStudent, advance_heads, sample_heads,
                 sample_pdd, sample_teacher_grid, teacher_interval_velocity,
                 train_step)


def tiny_teacher(width: int = 16, levels: int = 1, seed: int = 0) -> FullCFGMLP:
    torch.manual_seed(seed)
    return FullCFGMLP(width, levels)


def make_args(**overrides):
    defaults = dict(batch_size=8, device="cpu", heads=1, scale=2.0,
                    rk="midpoint", grad_clip=10.0, data_free=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class GridTests(unittest.TestCase):
    def test_uniform_grid_endpoints_and_weights(self):
        grid = PDDGrid(steps=64, block_min=8, block_max=64, shift=1.0, device="cpu")
        self.assertEqual(grid.s[0].item(), 0.0)
        self.assertEqual(grid.s[-1].item(), 1.0)
        self.assertEqual(grid.t[0].item(), 1.0)
        self.assertTrue(bool((grid.delta > 0).all()))
        self.assertAlmostEqual(float(grid.delta.sum()), 1.0, places=5)
        self.assertEqual(grid.starts.tolist(), list(range(0, 64, 8)))
        self.assertEqual(grid.num_starts, 8)

    def test_shift_concentrates_intervals_near_noise(self):
        uniform = PDDGrid(16, 2, 16, 1.0, "cpu")
        shifted = PDDGrid(16, 2, 16, 2.0, "cpu")
        self.assertLess(float(shifted.delta[0]), float(uniform.delta[0]))
        self.assertGreater(float(shifted.delta[-1]), float(uniform.delta[-1]))
        self.assertAlmostEqual(float(shifted.s[-1].item()), 1.0, places=6)

    def test_grid_validation(self):
        with self.assertRaises(ValueError):
            PDDGrid(steps=0, block_min=1, block_max=1)
        with self.assertRaises(ValueError):
            PDDGrid(steps=64, block_min=7, block_max=64)
        with self.assertRaises(ValueError):
            PDDGrid(steps=64, block_min=8, block_max=128)
        with self.assertRaises(ValueError):
            PDDGrid(steps=64, block_min=8, block_max=64, shift=0.0)


class StudentTests(unittest.TestCase):
    def setUp(self):
        self.teacher = tiny_teacher()
        self.student = PDDStudent(self.teacher, 64)
        self.grid = PDDGrid(64, 8, 64, 1.0, "cpu")
        self.x = torch.randn(5, 2)
        self.c = torch.rand(5, 1)

    def test_backbone_and_head_initialization(self):
        for name in ("stem", "middle", "fuse"):
            left, right = getattr(self.student, name), getattr(self.teacher, name)
            for key, value in left.state_dict().items():
                torch.testing.assert_close(value, right.state_dict()[key])
        weight = self.student.head.weight.view(64, 2, -1)
        bias = self.student.head.bias.view(64, 2)
        expected = -self.teacher.output.weight.detach()
        expected_bias = -self.teacher.output.bias.detach()
        for head in range(64):
            torch.testing.assert_close(weight[head], expected)
            torch.testing.assert_close(bias[head], expected_bias)

    def test_fused_head_matches_materialized_heads(self):
        for start, end in ((0, 8), (8, 16), (24, 48), (0, 64)):
            span = float(self.grid.s[end] - self.grid.s[start])
            weights = self.grid.delta[start:end] / span
            t = self.grid.t[start].expand(len(self.x), 1)
            fused = self.student.fused_velocity(self.x, t, self.c, start, end, weights)
            heads = self.student(self.x, t, self.c)
            manual = (heads[:, start:end]
                      * self.grid.delta[start:end][None, :, None]).sum(1) / span
            torch.testing.assert_close(fused, manual, rtol=1e-5, atol=1e-6)

    def test_fused_head_is_differentiable(self):
        weights = self.grid.delta[:8] / float(self.grid.s[8])
        t = self.grid.t[0].expand(len(self.x), 1)
        out = self.student.fused_velocity(self.x, t, self.c, 0, 8, weights)
        out.sum().backward()
        self.assertIsNotNone(self.student.head.weight.grad)
        self.assertGreater(float(self.student.head.weight.grad.abs().sum()), 0.0)

    def test_sample_heads_unique_and_in_block(self):
        generator = torch.Generator().manual_seed(0)
        start = self.grid.starts.repeat_interleave(3)
        heads = sample_heads(start, self.grid, 3, generator)
        self.assertEqual(heads.shape, (len(start), 3))
        for row, s in zip(heads.tolist(), start.tolist()):
            self.assertEqual(len(set(row)), 3)
            self.assertTrue(all(s <= head < min(s + self.grid.block_max, 64) for head in row))
        with self.assertRaises(ValueError):
            sample_heads(start, self.grid, 0, generator)
        with self.assertRaises(ValueError):
            sample_heads(start, self.grid, 9, generator)

    def test_advance_heads_matches_prefix_sum(self):
        generator = torch.Generator().manual_seed(1)
        heads = torch.randn(4, 64, 2, generator=generator, requires_grad=True)
        start = torch.tensor([0, 8, 24, 56])
        k = torch.tensor([0, 10, 47, 63])
        displacement = advance_heads(heads, start, k[:, None], self.grid.delta)
        self.assertEqual(displacement.shape, (4, 1, 2))
        for row in range(4):
            if k[row] == start[row]:
                torch.testing.assert_close(displacement[row, 0],
                                           torch.zeros(2), rtol=0, atol=0)
                continue
            expected = sum(self.grid.delta[j] * heads[row, j].detach()
                           for j in range(int(start[row]), int(k[row])))
            torch.testing.assert_close(displacement[row, 0], expected)
        self.assertFalse(displacement.requires_grad)


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.teacher = tiny_teacher(seed=3)
        self.grid = PDDGrid(64, 8, 64, 1.0, "cpu")
        generator = torch.Generator().manual_seed(2)
        self.x = torch.randn(4, 3, 2, generator=generator)
        self.s = torch.rand(4, 3, generator=generator)
        self.delta = torch.full((4, 3), 1 / 64)
        self.c = torch.rand(4, 1, generator=generator)

    def test_euler_target_is_negative_teacher_velocity_at_start(self):
        got = teacher_interval_velocity(self.teacher, self.x, self.s, self.delta,
                                        self.c, 2.0, "euler")
        flat_c = self.c[:, None, :].expand(-1, 3, -1).reshape(-1, 1)
        expected = -depth_velocity(self.teacher, self.x.reshape(-1, 2),
                                   (1 - self.s).reshape(-1, 1), 2.0, flat_c)
        torch.testing.assert_close(got, expected.reshape(4, 3, 2))

    def test_midpoint_target_uses_second_stage(self):
        first = teacher_interval_velocity(self.teacher, self.x, self.s, self.delta,
                                          self.c, 2.0, "euler")
        mid_x = self.x + 0.5 * self.delta[:, :, None] * first
        mid_s = self.s + 0.5 * self.delta
        got = teacher_interval_velocity(self.teacher, self.x, self.s, self.delta,
                                        self.c, 2.0, "midpoint")
        flat_c = self.c[:, None, :].expand(-1, 3, -1).reshape(-1, 1)
        expected = -depth_velocity(self.teacher, mid_x.reshape(-1, 2),
                                   (1 - mid_s).reshape(-1, 1), 2.0, flat_c)
        torch.testing.assert_close(got, expected.reshape(4, 3, 2))
        self.assertFalse(got.requires_grad)

    def test_constant_teacher_has_zero_loss(self):
        with torch.no_grad():
            self.teacher.output.weight.zero_()
            self.teacher.output.bias.fill_(0.25)
        student = PDDStudent(self.teacher, 64)
        x = torch.randn(6, 2)
        c = torch.rand(6, 1)
        t = torch.zeros(6, 1)
        heads = student(x, t, c)
        target = teacher_interval_velocity(
            self.teacher, x[:, None, :].expand(6, 4, 2), torch.zeros(6, 4),
            torch.full((6, 4), 1 / 64), c, 2.0, "midpoint")
        torch.testing.assert_close(heads[:, :4], target)
        torch.testing.assert_close(heads[:, :4], heads[:, :1].expand(-1, 4, -1))

    def test_invalid_solver_rejected(self):
        with self.assertRaises(ValueError):
            teacher_interval_velocity(self.teacher, self.x, self.s, self.delta,
                                      self.c, 2.0, "rk4")


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.teacher = tiny_teacher(seed=5)
        self.student = PDDStudent(self.teacher, 64)
        self.grid = PDDGrid(64, 8, 64, 1.0, "cpu")
        self.noise = torch.randn(16, 2)
        self.c = torch.rand(16, 1)
    def manual_sample(self, nfe: int) -> torch.Tensor:
        block = 64 // nfe
        x = self.noise
        for start in range(0, 64, block):
            end = start + block
            span = self.grid.s[end] - self.grid.s[start]
            t = self.grid.t[start].expand(len(x), 1)
            heads = self.student(x, t, self.c)
            mean = (heads[:, start:end]
                    * self.grid.delta[start:end][None, :, None]).sum(1) / span
            x = x + span * mean
        return x

    def test_fused_sampling_matches_materialized_rollout(self):
        for nfe in (1, 2, 4, 8):
            fused = sample_pdd(self.student, self.noise, self.c, self.grid, nfe,
                               fused=True)
            materialized = sample_pdd(self.student, self.noise, self.c, self.grid,
                                      nfe, fused=False)
            torch.testing.assert_close(fused, materialized, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(fused, self.manual_sample(nfe),
                                       rtol=1e-5, atol=1e-6)

    def test_sampling_validation(self):
        with self.assertRaises(ValueError):
            sample_pdd(self.student, self.noise, self.c, self.grid, 0)
        with self.assertRaises(ValueError):
            sample_pdd(self.student, self.noise, self.c, self.grid, 3)

    def test_constant_teacher_has_closed_form_trajectory(self):
        constant = torch.tensor([0.25, -0.4])
        with torch.no_grad():
            self.teacher.output.weight.zero_()
            self.teacher.output.bias.copy_(constant)
        student = PDDStudent(self.teacher, 64)
        expected = self.noise - constant
        for nfe in (1, 2, 4, 8, 16, 32, 64):
            got = sample_pdd(student, self.noise, self.c, self.grid, nfe)
            torch.testing.assert_close(got, expected, rtol=0, atol=1e-5)
        for rk in ("euler", "midpoint"):
            got = sample_teacher_grid(self.teacher, self.noise, self.c, self.grid,
                                      2.0, rk)
            torch.testing.assert_close(got, expected, rtol=0, atol=1e-5)


class TrainingTests(unittest.TestCase):
    def test_train_step_updates_student_only(self):
        teacher = tiny_teacher(seed=7)
        digest = state_digest(teacher)
        grid = PDDGrid(16, 4, 16, 1.0, "cpu")
        student = PDDStudent(teacher, 16)
        before = {name: value.detach().clone()
                  for name, value in student.named_parameters()}
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        generator = torch.Generator().manual_seed(8)
        args = make_args()
        losses = []
        for _ in range(3):
            loss, grad_norm = train_step(args, student, teacher, grid, optimizer,
                                         generator, Geometry(), None)
            losses.append(float(loss))
            self.assertTrue(np.isfinite(float(loss)))
            self.assertTrue(np.isfinite(float(grad_norm)))
        self.assertTrue(all(np.isfinite(losses)))
        self.assertEqual(state_digest(teacher), digest)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        changed = sum(int(not torch.equal(before[name], value))
                      for name, value in student.named_parameters())
        self.assertGreater(changed, 0)

    def test_data_free_carry_advances_and_resets(self):
        grid = PDDGrid(16, 4, 16, 1.0, "cpu")
        generator = torch.Generator().manual_seed(9)
        carry = DataFreeCarry(4, grid, torch.device("cpu"), generator)
        x, c, start = carry.current()
        self.assertEqual(start.tolist(), [0, 0, 0, 0])
        for position in range(0, 16, 4):
            x, c, start = carry.current()
            self.assertEqual(start.tolist(), [position] * 4)
            heads = torch.randn(4, 16, 2, generator=generator)
            expected = x + (heads[:, position:position + 4]
                            * grid.delta[position:position + 4][None, :, None]).sum(1)
            carry.advance(heads)
            torch.testing.assert_close(carry.x, expected)
        x, c, start = carry.current()
        self.assertEqual(start.tolist(), [0, 0, 0, 0])
        self.assertIsNotNone(carry.x)


if __name__ == "__main__":
    unittest.main()
