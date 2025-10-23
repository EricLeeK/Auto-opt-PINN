"""Core package for the Auto-PINN project."""

from .config import DEFAULT_CONFIG, ProjectConfig
from .gene import Gene, LayerGene, LayerType
from .genetic_algorithm import run_genetic_search
from .pinn import HybridPINN
from .trainer import PINNFitnessEvaluator

__all__ = [
	"DEFAULT_CONFIG",
	"Gene",
	"HybridPINN",
	"LayerGene",
	"LayerType",
	"PINNFitnessEvaluator",
	"ProjectConfig",
	"run_genetic_search",
]
