"""Configuration dataclasses used across the Auto-PINN project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class DomainConfig:
    """Physical domain and PDE constants."""

    x_bounds: Tuple[float, float] = (0.0, 1.0)
    t_bounds: Tuple[float, float] = (0.0, 1.0)
    viscosity: float = 0.01 / 3.141592653589793


@dataclass(frozen=True)
class SearchSpace:
    """Search space for layer-wise gene construction."""

    min_layers: int = 2
    max_layers: int = 5
    dnn_neurons: Sequence[int] = (16, 32, 64, 128)
    kan_width: Sequence[int] = (8, 16, 32, 64)
    kan_grid_points: Sequence[int] = (3, 5, 7)
    kan_spline_order: Sequence[int] = (2, 3)
    attn_embed_dim: Sequence[int] = (32, 64)
    attn_heads: Sequence[int] = (2, 4)


@dataclass(frozen=True)
class GAConfig:
    """Hyperparameters controlling the genetic search."""

    population_size: int = 10
    generations: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.3
    elite_count: int = 2
    tournament_size: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for the PINN training loop."""

    collocation_points: int = 512
    boundary_points: int = 128
    initial_points: int = 128
    epochs: int = 200
    learning_rate: float = 1e-3
    pde_weight: float = 1.0
    boundary_weight: float = 1.0
    initial_weight: float = 1.0
    device: str = "cpu"


@dataclass(frozen=True)
class RuntimeConfig:
    """Global runtime switches."""

    seed: int = 42
    dtype: str = "float32"
    log_every: int = 100


@dataclass(frozen=True)
class ProjectConfig:
    """Bundled configuration object for convenience."""

    domain: DomainConfig = field(default_factory=DomainConfig)
    search: SearchSpace = field(default_factory=SearchSpace)
    ga: GAConfig = field(default_factory=GAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


DEFAULT_CONFIG = ProjectConfig()
