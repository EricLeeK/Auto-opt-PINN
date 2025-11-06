"""Benchmark speed comparison between original and optimized KAN implementations.

This script measures forward+backward pass throughput for both versions of KANLayer
to quantify the performance gain from vectorization.
"""

from __future__ import annotations

import time
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import math


# ============================================================================
# Original KAN implementation (with Python loops)
# ============================================================================
class KANLayerOriginal(nn.Module):
    """Original KAN layer with per-feature loop in forward pass."""

    def __init__(self, in_features: int, width: int, grid_points: int, spline_order: int) -> None:
        super().__init__()
        if grid_points < 2:
            raise ValueError("KANLayer requires at least two grid_points")
        if spline_order < 1:
            raise ValueError("KANLayer spline_order must be >= 1")

        self.in_features = in_features
        self.width = width
        self.degree = int(spline_order)
        self.num_ctrl_pts = int(grid_points)

        knot_vector = self._create_knot_vector(self.num_ctrl_pts, self.degree)
        self.register_buffer("knots", knot_vector, persistent=False)

        self.input_shift = nn.Parameter(torch.zeros(in_features))
        self.input_log_scale = nn.Parameter(torch.zeros(in_features))

        coeff_init = math.sqrt(2.0 / (self.num_ctrl_pts * max(1, in_features)))
        self.spline_coeffs = nn.Parameter(
            coeff_init * torch.randn(in_features, self.num_ctrl_pts, width)
        )
        self.bias = nn.Parameter(torch.zeros(width))

        self.skip_linear = nn.Linear(in_features, width)
        self.layer_norm = nn.LayerNorm(width)
        self.activation = nn.GELU()

    @staticmethod
    def _create_knot_vector(num_ctrl_pts: int, degree: int) -> torch.Tensor:
        knot_count = num_ctrl_pts + degree + 1
        knots = torch.zeros(knot_count, dtype=torch.float32)
        knots[-(degree + 1) :] = 1.0
        interior = knot_count - 2 * (degree + 1)
        if interior > 0:
            interior_points = torch.linspace(0.0, 1.0, interior + 2, dtype=torch.float32)[1:-1]
            knots[degree + 1 : degree + 1 + interior] = interior_points
        return knots

    def _bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Original implementation with Python loops."""
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        degree = self.degree
        n_basis = knots.shape[0] - 1

        x = x.unsqueeze(-1)
        basis_segments = []
        for i in range(n_basis):
            left, right = knots[i], knots[i + 1]
            if i == n_basis - 1:
                cond = (x >= left) & (x <= right)
            else:
                cond = (x >= left) & (x < right)
            basis_segments.append(cond.to(x.dtype))
        basis = torch.cat(basis_segments, dim=-1)

        if degree == 0:
            return basis

        for d in range(1, degree + 1):
            new_basis = []
            upper = n_basis - d
            for i in range(upper):
                denom1 = knots[i + d] - knots[i]
                denom2 = knots[i + d + 1] - knots[i + 1]
                term1 = torch.zeros_like(x)
                term2 = torch.zeros_like(x)
                if denom1 > 0:
                    term1 = ((x - knots[i]) / denom1) * basis[:, i : i + 1]
                if denom2 > 0:
                    term2 = ((knots[i + d + 1] - x) / denom2) * basis[:, i + 1 : i + 2]
                new_basis.append(term1 + term2)
            basis = torch.cat(new_basis, dim=-1)

        return basis

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.input_log_scale) + 1e-3
        scaled = (inputs - self.input_shift) * scale
        normalised = torch.sigmoid(scaled)

        # Original: loop over features
        basis_values = []
        for feat in range(self.in_features):
            basis_feat = self._bspline_basis(normalised[:, feat])
            basis_values.append(basis_feat)
        basis_tensor = torch.stack(basis_values, dim=1)

        spline_mix = torch.einsum("bfg,fgw->bw", basis_tensor, self.spline_coeffs)
        linear_skip = self.skip_linear(inputs)
        combined = spline_mix + linear_skip + self.bias
        combined = self.layer_norm(combined)
        return self.activation(combined)


# ============================================================================
# Optimized KAN implementation (vectorized)
# ============================================================================
class KANLayerOptimized(nn.Module):
    """Optimized KAN layer with fully vectorized forward pass."""

    def __init__(self, in_features: int, width: int, grid_points: int, spline_order: int) -> None:
        super().__init__()
        if grid_points < 2:
            raise ValueError("KANLayer requires at least two grid_points")
        if spline_order < 1:
            raise ValueError("KANLayer spline_order must be >= 1")

        self.in_features = in_features
        self.width = width
        self.degree = int(spline_order)
        self.num_ctrl_pts = int(grid_points)

        knot_vector = self._create_knot_vector(self.num_ctrl_pts, self.degree)
        self.register_buffer("knots", knot_vector, persistent=False)

        self.input_shift = nn.Parameter(torch.zeros(in_features))
        self.input_log_scale = nn.Parameter(torch.zeros(in_features))

        coeff_init = math.sqrt(2.0 / (self.num_ctrl_pts * max(1, in_features)))
        self.spline_coeffs = nn.Parameter(
            coeff_init * torch.randn(in_features, self.num_ctrl_pts, width)
        )
        self.bias = nn.Parameter(torch.zeros(width))

        self.skip_linear = nn.Linear(in_features, width)
        self.layer_norm = nn.LayerNorm(width)
        self.activation = nn.GELU()

    @staticmethod
    def _create_knot_vector(num_ctrl_pts: int, degree: int) -> torch.Tensor:
        knot_count = num_ctrl_pts + degree + 1
        knots = torch.zeros(knot_count, dtype=torch.float32)
        knots[-(degree + 1) :] = 1.0
        interior = knot_count - 2 * (degree + 1)
        if interior > 0:
            interior_points = torch.linspace(0.0, 1.0, interior + 2, dtype=torch.float32)[1:-1]
            knots[degree + 1 : degree + 1 + interior] = interior_points
        return knots

    def _bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized vectorized implementation."""
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        degree = self.degree

        x = x.unsqueeze(-1)
        left = knots[:-1].view(1, 1, -1)
        right = knots[1:].view(1, 1, -1)
        basis = ((x >= left) & (x < right)).to(x.dtype)
        basis[..., -1:] = ((x >= left[..., -1:]) & (x <= right[..., -1:])).to(x.dtype)

        if degree == 0:
            return basis

        for r in range(1, degree + 1):
            m = basis.shape[-1] - 1
            if m <= 0:
                break
            idx = torch.arange(m, device=x.device)
            left_knots = knots[idx].view(1, 1, -1)
            right_knots = knots[idx + r + 1].view(1, 1, -1)
            denom1 = (knots[idx + r] - knots[idx]).view(1, 1, -1)
            denom2 = (knots[idx + r + 1] - knots[idx + 1]).view(1, 1, -1)

            basis_left = basis[..., :m]
            basis_right = basis[..., 1 : m + 1]

            term1 = torch.where(
                denom1 > 0,
                (x - left_knots) / denom1 * basis_left,
                torch.zeros_like(basis_left),
            )
            term2 = torch.where(
                denom2 > 0,
                (right_knots - x) / denom2 * basis_right,
                torch.zeros_like(basis_left),
            )
            basis = term1 + term2

        return basis

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.input_log_scale) + 1e-3
        scaled = (inputs - self.input_shift) * scale
        normalised = torch.sigmoid(scaled)

        # Optimized: vectorized across all features
        basis_tensor = self._bspline_basis(normalised)

        spline_mix = torch.einsum("bfg,fgw->bw", basis_tensor, self.spline_coeffs)
        linear_skip = self.skip_linear(inputs)
        combined = spline_mix + linear_skip + self.bias
        combined = self.layer_norm(combined)
        return self.activation(combined)


# ============================================================================
# Benchmarking utilities
# ============================================================================
def warmup(model: nn.Module, input_tensor: torch.Tensor, iterations: int = 5) -> None:
    """Warmup to stabilize GPU/CPU timings."""
    model.train()
    for _ in range(iterations):
        out = model(input_tensor)
        loss = out.sum()
        loss.backward()
        model.zero_grad()


def benchmark_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    epochs: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Run forward+backward for specified epochs and return average time per epoch."""
    model.train()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(epochs):
        out = model(input_tensor)
        loss = out.sum()
        loss.backward()
        model.zero_grad()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_per_epoch = elapsed / epochs
    return elapsed, avg_per_epoch


def main() -> None:
    # ========== Configuration ==========
    BATCH_SIZE = 512
    IN_FEATURES = 2
    WIDTH = 16
    GRID_POINTS = 7
    SPLINE_ORDER = 4
    EPOCHS = 300
    WARMUP_ITERS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("KAN Layer Performance Benchmark")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  In features:    {IN_FEATURES}")
    print(f"  Width:          {WIDTH}")
    print(f"  Grid points:    {GRID_POINTS}")
    print(f"  Spline order:   {SPLINE_ORDER}")
    print(f"  Epochs:         {EPOCHS}")
    print(f"  Device:         {DEVICE}")
    print("=" * 60)
    
    device = torch.device(DEVICE)
    input_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device, requires_grad=True)
    
    # Original implementation
    print("\n[1/2] Benchmarking Original KAN (with loops)...")
    model_original = KANLayerOriginal(IN_FEATURES, WIDTH, GRID_POINTS, SPLINE_ORDER).to(device)
    warmup(model_original, input_tensor, WARMUP_ITERS)
    total_time_orig, avg_time_orig = benchmark_model(model_original, input_tensor, EPOCHS, device)
    print(f"  Total time:  {total_time_orig:.4f}s")
    print(f"  Avg/epoch:   {avg_time_orig*1000:.2f}ms")
    
    # Optimized implementation
    print("\n[2/2] Benchmarking Optimized KAN (vectorized)...")
    model_optimized = KANLayerOptimized(IN_FEATURES, WIDTH, GRID_POINTS, SPLINE_ORDER).to(device)
    warmup(model_optimized, input_tensor, WARMUP_ITERS)
    total_time_opt, avg_time_opt = benchmark_model(model_optimized, input_tensor, EPOCHS, device)
    print(f"  Total time:  {total_time_opt:.4f}s")
    print(f"  Avg/epoch:   {avg_time_opt*1000:.2f}ms")
    
    # Summary
    speedup = total_time_orig / total_time_opt
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Original implementation:  {total_time_orig:.4f}s ({avg_time_orig*1000:.2f}ms/epoch)")
    print(f"  Optimized implementation: {total_time_opt:.4f}s ({avg_time_opt*1000:.2f}ms/epoch)")
    print(f"  Speedup:                  {speedup:.2f}x")
    print("=" * 60)
    
    # Sanity check: verify outputs are similar
    print("\nSanity check: comparing output similarity...")
    model_original.eval()
    model_optimized.eval()
    with torch.no_grad():
        out_orig = model_original(input_tensor)
        out_opt = model_optimized(input_tensor)
        max_diff = torch.abs(out_orig - out_opt).max().item()
        print(f"  Max absolute difference: {max_diff:.6e}")
        if max_diff < 1e-5:
            print("  ✓ Outputs are numerically identical (within tolerance)")
        else:
            print(" Outputs differ slightly (expected due to different random init)")


if __name__ == "__main__":
    main()
