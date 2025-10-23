"""Hybrid PINN model construction and bespoke layer implementations."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import nn

from .gene import Gene, LayerGene, LayerType


class KANLayer(nn.Module):
    """Light-weight kernel adaptive network layer.

    The layer maps the input to a set of learnable radial basis activations and then
    mixes them linearly, producing a non-linear feature expansion reminiscent of the
    KAN family while staying numerically stable for automatic differentiation.
    """

    def __init__(self, in_features: int, width: int, grid_points: int, spline_order: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.width = width
        self.grid_points = grid_points
        self.spline_order = spline_order
        self.centers = nn.Parameter(torch.randn(grid_points, in_features))
        self.log_scales = nn.Parameter(torch.zeros(grid_points))
        self.mixer = nn.Linear(grid_points, width)
        self.post = nn.Sequential(nn.LayerNorm(width), nn.Tanh())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        diff = inputs.unsqueeze(1) - self.centers.unsqueeze(0)
        scaled = torch.exp(-torch.sum(diff * diff, dim=-1) * torch.exp(self.log_scales))
        expanded = self.mixer(scaled)
        return self.post(expanded)


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
