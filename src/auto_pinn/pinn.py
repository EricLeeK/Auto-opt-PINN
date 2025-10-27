"""Hybrid PINN model construction and bespoke layer implementations."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .gene import Gene, LayerGene, LayerType


class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network spline layer.

    This implementation follows the spirit of KANs by expanding each input feature
    with a trainable B-spline basis and mixing the resulting coefficients. The layer
    carries learnable affine normalisation per input dimension, spline coefficients,
    and a residual linear skip so that the network can emulate both KAN-style basis
    aggregation and plain linear layers when beneficial.
    """

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

        # Learnable affine parameters to map inputs into the spline domain (0, 1)
        self.input_shift = nn.Parameter(torch.zeros(in_features))
        self.input_log_scale = nn.Parameter(torch.zeros(in_features))

        # Spline coefficient tensor: one set per input dimension.
        coeff_init = math.sqrt(2.0 / (self.num_ctrl_pts * max(1, in_features)))
        self.spline_coeffs = nn.Parameter(
            coeff_init * torch.randn(in_features, self.num_ctrl_pts, width)
        )
        self.bias = nn.Parameter(torch.zeros(width))

        # Residual linear skip helps when spline expansion is unnecessary.
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


class AttentionLayer(nn.Module):
    """Feature attention block with head-wise gating for vector inputs."""

    def __init__(self, in_features: int, embed_dim: int, heads: int) -> None:
        super().__init__()
        if embed_dim % heads != 0:
            raise ValueError("embed_dim must be divisible by heads")
        head_dim = embed_dim // heads
        self.heads = heads
        self.head_dim = head_dim
        self.query = nn.Linear(in_features, embed_dim)
        self.key = nn.Linear(in_features, embed_dim)
        self.value = nn.Linear(in_features, embed_dim)
        self.skip = nn.Linear(in_features, embed_dim)
        self.out_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        q = self.query(inputs).view(batch, self.heads, self.head_dim)
        k = self.key(inputs).view(batch, self.heads, self.head_dim)
        v = self.value(inputs).view(batch, self.heads, self.head_dim)
        attn_logits = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)
        weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)
        context = (weights * v).reshape(batch, -1)
        skip = torch.tanh(self.skip(inputs))
        projected = self.out_proj(context + skip)
        return projected


def _build_layer(gene: LayerGene, in_features: int) -> Tuple[nn.Module, int]:
    if gene.layer_type == LayerType.DNN:
        units = gene.params["units"]
        block = nn.Sequential(nn.Linear(in_features, units), nn.Tanh())
        return block, units
    if gene.layer_type == LayerType.KAN:
        width = gene.params["width"]
        grid = gene.params["grid_points"]
        order = gene.params["spline_order"]
        return KANLayer(in_features, width, grid, order), width
    embed = gene.params["embed_dim"]
    heads = gene.params["heads"]
    return AttentionLayer(in_features, embed, heads), embed


class HybridPINN(nn.Module):
    """Sequential hybrid PINN assembled from a gene description."""

    def __init__(self, genes: Gene, input_dim: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        for layer_gene in genes:
            block, current_dim = _build_layer(layer_gene, current_dim)
            layers.append(block)
        layers.append(nn.Linear(current_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
