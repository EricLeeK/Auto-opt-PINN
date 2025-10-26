"""Sampling utilities for Burgers' equation PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
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


def _burgers_initial_condition(x: np.ndarray) -> np.ndarray:
    return -np.sin(np.pi * x)


def _burgers_boundary_condition(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    left = np.zeros_like(t)
    right = np.zeros_like(t)
    return left, right


def _sample_uniform(count: int, bounds: Tuple[float, float]) -> np.ndarray:
    return np.random.uniform(bounds[0], bounds[1], size=(count, 1))


def generate_training_batch(domain: DomainConfig, training: TrainingConfig, device: torch.device, dtype: torch.dtype) -> TrainingBatch:
    x_coll = _sample_uniform(training.collocation_points, domain.x_bounds)
    t_coll = _sample_uniform(training.collocation_points, domain.t_bounds)
    collocation = torch.tensor(np.hstack([x_coll, t_coll]), device=device, dtype=dtype)

    t_bc = _sample_uniform(training.boundary_points, domain.t_bounds)
    left_vals, right_vals = _burgers_boundary_condition(t_bc)
    left_inputs = torch.tensor(np.hstack([np.full_like(t_bc, domain.x_bounds[0]), t_bc]), device=device, dtype=dtype)
    right_inputs = torch.tensor(np.hstack([np.full_like(t_bc, domain.x_bounds[1]), t_bc]), device=device, dtype=dtype)
    boundary_inputs = torch.cat([left_inputs, right_inputs], dim=0)
    boundary_targets = torch.tensor(np.concatenate([left_vals, right_vals], axis=0), device=device, dtype=dtype)

    x_init = _sample_uniform(training.initial_points, domain.x_bounds)
    t_init = np.zeros_like(x_init)
    initial_inputs = torch.tensor(np.hstack([x_init, t_init]), device=device, dtype=dtype)
    initial_targets = torch.tensor(_burgers_initial_condition(x_init), device=device, dtype=dtype)

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