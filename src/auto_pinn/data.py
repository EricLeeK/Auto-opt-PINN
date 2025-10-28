"""Sampling utilities for Burgers' equation PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .config import DomainConfig, TrainingConfig


@dataclass
class TrainingBatch:
    """Container aggregating training tensors for the PINN."""

    collocation: torch.Tensor
    boundary_inputs: torch.Tensor
    boundary_targets: torch.Tensor
    initial_inputs: torch.Tensor
    initial_targets: torch.Tensor


def _burgers_initial_condition(x: torch.Tensor) -> torch.Tensor:
    return -torch.sin(torch.pi * x)


def _burgers_boundary_condition(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    left = torch.zeros_like(t)
    right = torch.zeros_like(t)
    return left, right


def _sample_uniform(count: int, bounds: Tuple[float, float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    low, high = bounds
    return low + (high - low) * torch.rand((count, 1), device=device, dtype=dtype)


def generate_training_batch(domain: DomainConfig, training: TrainingConfig, device: torch.device, dtype: torch.dtype) -> TrainingBatch:
    x_coll = _sample_uniform(training.collocation_points, domain.x_bounds, device, dtype)
    t_coll = _sample_uniform(training.collocation_points, domain.t_bounds, device, dtype)
    collocation = torch.cat([x_coll, t_coll], dim=1)

    t_bc = _sample_uniform(training.boundary_points, domain.t_bounds, device, dtype)
    left_vals, right_vals = _burgers_boundary_condition(t_bc)
    left_inputs = torch.cat([torch.full_like(t_bc, domain.x_bounds[0]), t_bc], dim=1)
    right_inputs = torch.cat([torch.full_like(t_bc, domain.x_bounds[1]), t_bc], dim=1)
    boundary_inputs = torch.cat([left_inputs, right_inputs], dim=0)
    boundary_targets = torch.cat([left_vals, right_vals], dim=0)

    x_init = _sample_uniform(training.initial_points, domain.x_bounds, device, dtype)
    t_init = torch.zeros_like(x_init)
    initial_inputs = torch.cat([x_init, t_init], dim=1)
    initial_targets = _burgers_initial_condition(x_init)

    return TrainingBatch(
        collocation=collocation,
        boundary_inputs=boundary_inputs,
        boundary_targets=boundary_targets,
        initial_inputs=initial_inputs,
        initial_targets=initial_targets,
    )




# generate_training_batch()  ← 主函数
#     │
#     ├─ 配置项读取 (DomainConfig, TrainingConfig)
#     │
#     ├─ 三类采样点生成
#     │   ├─ Collocation Points (PDE 内部点)
#     │   ├─ Boundary Points (边界条件点)
#     │   └─ Initial Points (初始条件点)
#     │
#     └─ 返回 TrainingBatch (数据容器)

# 要点	说明
# 核心职责	为 PINN 生成符合 Burgers' 方程的训练数据
# 三类数据	Collocation（PDE）、Boundary（边界）、Initial（初值）
# 设计模式	使用 @dataclass 容器封装，类型安全
# 物理约束	通过采样点强制神经网络学习 PDE、BC、IC
# 可扩展性	可轻松替换采样策略（LHS、Sobol 序列等）