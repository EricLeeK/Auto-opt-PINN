"""Configuration dataclasses used across the Auto-PINN project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class DomainConfig:
    """Physical domain and PDE constants."""

    x_bounds: Tuple[float, float] = (-1.0, 1.0)
    t_bounds: Tuple[float, float] = (0.0, 1.0)
    viscosity: float = 0.01 / 3.141592653589793


@dataclass(frozen=True)
class SearchSpace:     #基因搜索空间        
    """Search space for layer-wise gene construction."""

    min_layers: int = 3
    max_layers: int = 5
    dnn_neurons: Tuple[int, ...] = (16, 32, 64, 128)
    kan_width: Tuple[int, ...] = (8, 16, 24, 32)
    kan_grid_points: Tuple[int, ...] = (3, 5, 7)
    kan_spline_order: Tuple[int, ...] = (2, 3)
    attn_embed_dim: Tuple[int, ...] = (32, 48, 64)
    attn_heads: Tuple[int, ...] = (2, 4)


@dataclass(frozen=True)
class GAConfig:
    """Hyperparameters controlling the genetic search."""

    population_size: int = 8
    generations: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.4
    elite_count: int = 2
    tournament_size: int = 3
    deduplicate_population: bool = True
    resume_genes: Tuple[Tuple[dict, ...], ...] = ()  # Optional seeded genes for warm-starting the population
    resume_gene_files: Tuple[str, ...] = ()  # Optional JSON files providing additional seed genes


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for the PINN training loop."""

    collocation_points: int = 512
    boundary_points: int = 128
    initial_points: int = 128
    epochs: int = 25000
    learning_rate: float = 1e-3
    pde_weight: float = 1.0
    boundary_weight: float = 1.0
    initial_weight: float = 1.0
    device: str = "cuda"  # or "cpu"
    data_terms_require_grad: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    """Global runtime switches."""

    seed: int = 717
    dtype: str = "float32"
    log_every: int = 100
    cache_fitness: bool = True
    workers: int = 2
    checkpoint_dir: str = "checkpoints"
    save_every: int = 500  # Save checkpoint every N epochs
    gpu_devices: Tuple[str, ...] = ("cuda:0", "cuda:1")  # Optional explicit device list for multi-GPU execution
    gpu_concurrency: int = 1  # How many parallel workers can share a single GPU


@dataclass(frozen=True)
class ProjectConfig:
    """Bundled configuration object for convenience."""

    domain: DomainConfig = field(default_factory=DomainConfig)
    search: SearchSpace = field(default_factory=SearchSpace)
    ga: GAConfig = field(default_factory=GAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


DEFAULT_CONFIG = ProjectConfig()




# DataClass 的四大好处：
# 1. **类型检查** → 编译时发现错误，而非运行时崩溃
# 2. **不可变性** → 避免竞态条件和意外修改
# 3. **分组管理** → 配置清晰，避免参数爆炸
# 4. **易于测试** → 隔离性好，测试之间不互相污染

# 不使用会导致：
# - 类型错误难以发现
# - 并发 bug（竞态条件）
# - 参数传递混乱
# - 测试污染和不稳定