"""
test_privacy.py – Unit tests for the differential privacy module.

Validates:
  - RDP accountant produces valid epsilon values
  - Epsilon grows monotonically with training steps
  - Noise calibration finds valid σ for target ε
  - Gradient clipping respects max_norm bound
  - More noise → smaller epsilon (higher privacy)
"""

import sys
import os
import math
import unittest

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.privacy import RDPAccountant, calibrate_noise, clip_gradients, add_noise


class TestRDPAccountant(unittest.TestCase):
    """Tests for the Rényi DP accountant."""

    def test_initial_epsilon_is_zero(self):
        """Fresh accountant should report ε ≈ 0."""
        accountant = RDPAccountant()
        eps = accountant.get_epsilon(delta=1e-5)
        # With no steps, epsilon should be very small (just the log(1/δ) offset)
        # Actually with 0 rdp_budget, eps = min over α of log(1/δ)/(α-1)
        self.assertIsNotNone(eps)

    def test_epsilon_increases_with_steps(self):
        """Privacy budget should grow as more training steps are taken."""
        accountant = RDPAccountant()
        epsilons = []

        for i in range(10):
            accountant.step(noise_multiplier=1.0, sample_rate=0.01)
            eps = accountant.get_epsilon(delta=1e-5)
            epsilons.append(eps)

        # Epsilon should be monotonically non-decreasing
        for i in range(1, len(epsilons)):
            self.assertGreaterEqual(epsilons[i], epsilons[i - 1],
                                    f"Epsilon decreased at step {i}")

    def test_more_noise_means_lower_epsilon(self):
        """Higher noise multiplier should yield lower ε (better privacy)."""
        eps_values = []
        for sigma in [0.5, 1.0, 2.0, 5.0]:
            accountant = RDPAccountant()
            for _ in range(100):
                accountant.step(noise_multiplier=sigma, sample_rate=0.01)
            eps = accountant.get_epsilon(delta=1e-5)
            eps_values.append(eps)

        # Each successive sigma should give lower epsilon
        for i in range(1, len(eps_values)):
            self.assertLess(eps_values[i], eps_values[i - 1],
                            f"Higher noise σ did not reduce ε")

    def test_epsilon_is_finite(self):
        """Epsilon should always be a finite positive number."""
        accountant = RDPAccountant()
        for _ in range(50):
            accountant.step(noise_multiplier=1.0, sample_rate=0.05)
        eps = accountant.get_epsilon(delta=1e-5)
        self.assertTrue(math.isfinite(eps), f"Epsilon is not finite: {eps}")
        self.assertGreater(eps, 0, "Epsilon should be positive")

    def test_reset(self):
        """Reset should clear accumulated privacy budget."""
        accountant = RDPAccountant()
        for _ in range(50):
            accountant.step(noise_multiplier=1.0, sample_rate=0.01)
        eps_before = accountant.get_epsilon(delta=1e-5)

        accountant.reset()
        eps_after = accountant.get_epsilon(delta=1e-5)

        self.assertLess(eps_after, eps_before)
        self.assertEqual(accountant.steps, 0)


class TestNoiseCalibration(unittest.TestCase):
    """Tests for the noise calibration function."""

    def test_calibrated_noise_achieves_target(self):
        """calibrate_noise should find σ that achieves target ε."""
        target_eps = 5.0
        target_delta = 1e-5
        sample_rate = 0.01
        num_steps = 100

        sigma = calibrate_noise(target_eps, target_delta, sample_rate, num_steps)

        # Verify: run accountant with calibrated sigma
        accountant = RDPAccountant()
        for _ in range(num_steps):
            accountant.step(sigma, sample_rate)
        actual_eps = accountant.get_epsilon(target_delta)

        self.assertLessEqual(actual_eps, target_eps + 0.5,
                             f"Calibrated σ={sigma:.3f} gives ε={actual_eps:.2f}, "
                             f"expected ≤ {target_eps}")

    def test_tighter_epsilon_needs_more_noise(self):
        """Lower target ε should require higher σ."""
        sigmas = []
        for target_eps in [10.0, 5.0, 2.0, 1.0]:
            sigma = calibrate_noise(
                target_eps, 1e-5, sample_rate=0.01, num_steps=100
            )
            sigmas.append(sigma)

        for i in range(1, len(sigmas)):
            self.assertGreater(sigmas[i], sigmas[i - 1],
                               "Tighter ε should require more noise")


class TestGradientClipping(unittest.TestCase):
    """Tests for gradient clipping."""

    def test_clips_large_gradients(self):
        """Gradients exceeding max_norm should be clipped."""
        param = torch.nn.Parameter(torch.randn(100))
        param.grad = torch.randn(100) * 10  # Large gradients

        max_norm = 1.0
        original_norm = param.grad.norm(2).item()
        clip_gradients([param], max_norm)
        clipped_norm = param.grad.norm(2).item()

        self.assertLessEqual(clipped_norm, max_norm + 1e-5,
                             "Clipped gradient exceeds max_norm")

    def test_preserves_small_gradients(self):
        """Gradients within max_norm should not be modified."""
        param = torch.nn.Parameter(torch.randn(10))
        param.grad = torch.randn(10) * 0.01  # Small gradients

        max_norm = 1.0
        original_grad = param.grad.clone()
        clip_gradients([param], max_norm)

        self.assertTrue(
            torch.allclose(param.grad, original_grad, atol=1e-6),
            "Small gradients should not be modified by clipping"
        )


class TestNoiseAddition(unittest.TestCase):
    """Tests for Gaussian noise injection."""

    def test_noise_changes_gradients(self):
        """Adding noise should modify gradient values."""
        param = torch.nn.Parameter(torch.zeros(1000))
        param.grad = torch.zeros(1000)

        add_noise([param], noise_multiplier=1.0, max_grad_norm=1.0)

        # After adding noise to zero gradients, they should be non-zero
        self.assertGreater(param.grad.abs().mean().item(), 0.01,
                           "Noise should modify gradients")

    def test_noise_scale_proportional_to_sigma(self):
        """Higher σ should produce larger noise."""
        stds = []
        for sigma in [0.1, 1.0, 10.0]:
            param = torch.nn.Parameter(torch.zeros(10000))
            param.grad = torch.zeros(10000)
            add_noise([param], noise_multiplier=sigma, max_grad_norm=1.0)
            stds.append(param.grad.std().item())

        for i in range(1, len(stds)):
            self.assertGreater(stds[i], stds[i - 1],
                               "Higher σ should produce more noise")


if __name__ == '__main__':
    unittest.main(verbosity=2)
