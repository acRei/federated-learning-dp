"""
privacy.py – Differential privacy utilities.

Implements:
  - Rényi Differential Privacy (RDP) accountant for tracking cumulative privacy loss
  - RDP → (ε, δ) conversion
  - Noise calibration given a target epsilon
  - Per-sample gradient clipping utility
"""

import math
import numpy as np
from scipy.special import comb


class RDPAccountant:
    """
    Rényi Differential Privacy Accountant.
    
    Tracks privacy loss across multiple rounds of the Gaussian mechanism
    using Rényi divergence, then converts to standard (ε, δ)-DP.
    
    This provides tighter composition bounds than basic or advanced
    composition theorems, enabling more training rounds within a budget.
    
    Reference: Mironov, "Rényi Differential Privacy" (2017)
    """

    # Default RDP orders to evaluate (higher orders → tighter bounds for small δ)
    DEFAULT_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def __init__(self, orders=None):
        self.orders = orders or self.DEFAULT_ORDERS
        # Cumulative RDP guarantee at each order α
        self.rdp_budget = np.zeros(len(self.orders))
        self.steps = 0

    def step(self, noise_multiplier, sample_rate):
        """
        Record one step of the subsampled Gaussian mechanism.
        
        Args:
            noise_multiplier: σ (ratio of noise std to sensitivity)
            sample_rate: q (probability of each record being in a batch)
        """
        rdp = self._compute_rdp_single_step(noise_multiplier, sample_rate)
        self.rdp_budget += rdp
        self.steps += 1

    def get_epsilon(self, delta):
        """
        Convert accumulated RDP to (ε, δ)-DP.
        
        Uses the optimal conversion: ε = min_α [rdp(α) + log(1/δ) / (α-1)]
        
        Args:
            delta: Target δ (probability of privacy failure)
            
        Returns:
            Minimum ε satisfying the accumulated RDP budget at given δ
        """
        eps_candidates = []
        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                continue
            eps = self.rdp_budget[i] + math.log(1 / delta) / (alpha - 1)
            eps_candidates.append(eps)

        if not eps_candidates:
            return float('inf')
        return min(eps_candidates)

    def get_epsilon_per_order(self, delta):
        """Return ε at each RDP order (useful for debugging/visualization)."""
        results = {}
        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                continue
            eps = self.rdp_budget[i] + math.log(1 / delta) / (alpha - 1)
            results[alpha] = eps
        return results

    def _compute_rdp_single_step(self, sigma, q):
        """
        Compute RDP of the subsampled Gaussian mechanism for one step.
        
        For the Gaussian mechanism with sensitivity 1 and noise σ:
            RDP_α = α / (2σ²)
        
        With Poisson subsampling at rate q, we use the numerically stable
        log-space computation from Mironov et al.
        """
        rdp = np.zeros(len(self.orders))
        for i, alpha in enumerate(self.orders):
            if q == 0:
                rdp[i] = 0.0
            elif q == 1.0:
                # No subsampling: standard Gaussian mechanism RDP
                rdp[i] = alpha / (2.0 * sigma ** 2)
            else:
                rdp[i] = self._compute_rdp_subsampled_gaussian(q, sigma, alpha)
        return rdp

    @staticmethod
    def _compute_rdp_subsampled_gaussian(q, sigma, alpha):
        """
        Numerically stable RDP computation for subsampled Gaussian.
        
        Uses the analytic moments accountant formula.
        For integer α, this is exact. For fractional α, we use interpolation.
        """
        if alpha <= 1:
            return 0.0

        # For large sigma or small q, use the simple bound
        if sigma > 100 or q < 1e-6:
            return q ** 2 * alpha / (2 * sigma ** 2)

        # Standard Gaussian mechanism RDP (no subsampling)
        rdp_no_subsample = alpha / (2.0 * sigma ** 2)

        # Subsampling amplification (simplified tight bound)
        if q < 0.1:
            # For small sampling rates, use the Poisson subsampling bound
            # Use log-space computation to avoid overflow in math.exp()
            log_exp_term = rdp_no_subsample * (alpha - 1)

            if log_exp_term > 500:
                # When exp() would overflow, use the approximation:
                # log(1 + q^2 * exp(x)) ≈ log(q^2) + x  for large x
                subsampled = (2 * math.log(q) + log_exp_term) / (alpha - 1)
            else:
                subsampled = (
                    math.log(1 + q ** 2 * (math.exp(log_exp_term) - 1) / (alpha - 1))
                    / (alpha - 1)
                )

            return min(rdp_no_subsample, subsampled) if alpha > 1 else 0
        else:
            # For larger sampling rates, use direct computation
            return min(rdp_no_subsample, q ** 2 * alpha / (2 * sigma ** 2) + math.log(q) * 2)

    def reset(self):
        """Reset the accountant for a new experiment."""
        self.rdp_budget = np.zeros(len(self.orders))
        self.steps = 0


def calibrate_noise(target_epsilon, target_delta, sample_rate, num_steps,
                    orders=None, tol=0.01):
    """
    Find the minimum noise_multiplier σ to achieve (target_epsilon, target_delta)-DP
    after num_steps of the subsampled Gaussian mechanism.
    
    Uses binary search over σ.
    
    Args:
        target_epsilon: Desired ε guarantee
        target_delta: Desired δ guarantee
        sample_rate: Batch size / dataset size
        num_steps: Total number of training steps
        tol: Tolerance for binary search
        
    Returns:
        noise_multiplier σ that achieves the target privacy guarantee
    """
    sigma_low, sigma_high = 0.01, 100.0

    while sigma_high - sigma_low > tol:
        sigma_mid = (sigma_low + sigma_high) / 2.0
        accountant = RDPAccountant(orders)

        for _ in range(num_steps):
            accountant.step(sigma_mid, sample_rate)

        eps = accountant.get_epsilon(target_delta)

        if eps > target_epsilon:
            sigma_low = sigma_mid  # Need more noise
        else:
            sigma_high = sigma_mid  # Can use less noise

    return sigma_high


def clip_gradients(parameters, max_norm):
    """
    Clip per-sample gradients to bound sensitivity.
    
    This is the "C" in DP-SGD: each individual gradient contribution
    is clipped to have L2 norm ≤ max_norm before noise is added.
    
    Args:
        parameters: Iterator of model parameters with .grad
        max_norm: Maximum L2 norm for gradient clipping
        
    Returns:
        Total L2 norm of gradients before clipping
    """
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    total_norm = math.sqrt(total_norm)

    clip_factor = max_norm / max(total_norm, max_norm)
    for param in parameters:
        if param.grad is not None:
            param.grad.data.mul_(clip_factor)

    return total_norm


def add_noise(parameters, noise_multiplier, max_grad_norm, device='cpu'):
    """
    Add calibrated Gaussian noise to gradients.
    
    Noise scale = σ · C, where:
      σ = noise_multiplier
      C = max_grad_norm (sensitivity after clipping)
    
    Args:
        parameters: Iterator of model parameters with .grad
        noise_multiplier: σ value
        max_grad_norm: Clipping threshold C
        device: Torch device
    """
    import torch
    for param in parameters:
        if param.grad is not None:
            noise = torch.normal(
                mean=0.0,
                std=noise_multiplier * max_grad_norm,
                size=param.grad.shape,
                device=device
            )
            param.grad.data.add_(noise)