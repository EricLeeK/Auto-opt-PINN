"""Quick runtime benchmark comparing data-term gradients on vs off."""

from __future__ import annotations

import time
from dataclasses import replace

import torch

from auto_pinn.config import ProjectConfig
from auto_pinn.gene import Gene
from auto_pinn.genetic_algorithm import create_random_gene
from auto_pinn.trainer import PINNFitnessEvaluator


def _synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_config(require_data_grad: bool) -> ProjectConfig:
    base = ProjectConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training = replace(
        base.training,
        device=device,
        epochs=2,
        collocation_points=256,
        boundary_points=128,
        initial_points=128,
        data_terms_require_grad=require_data_grad,
    )
    return ProjectConfig(
        domain=base.domain,
        search=base.search,
        ga=base.ga,
        training=training,
        runtime=base.runtime,
    )


def _warmup(config: ProjectConfig, gene: Gene) -> None:
    evaluator = PINNFitnessEvaluator(config)
    evaluator._train_gene([layer.copy() for layer in gene])
    _synchronize_if_needed(config.training.device)


def _run_trial(config: ProjectConfig, gene: Gene) -> float:
    evaluator = PINNFitnessEvaluator(config)
    local_gene = [layer.copy() for layer in gene]
    _synchronize_if_needed(config.training.device)
    start = time.perf_counter()
    evaluator._train_gene(local_gene)
    _synchronize_if_needed(config.training.device)
    return time.perf_counter() - start


def main() -> None:
    reference_config = _make_config(require_data_grad=True)
    base_gene = create_random_gene(reference_config)
    repeats = 3
    durations = []
    labels = ["With data-term gradients", "No data-term gradients"]
    for flag, label in zip([True, False], labels):
        config = _make_config(require_data_grad=flag)
        _warmup(config, base_gene)
        runs = [_run_trial(config, base_gene) for _ in range(repeats)]
        avg_duration = sum(runs) / len(runs)
        durations.append(avg_duration)
        formatted_runs = ", ".join(f"{duration:.3f}s" for duration in runs)
        print(f"{label}: {formatted_runs} -> avg {avg_duration:.3f}s")
    if durations[1] > 0:
        print(f"Speed-up: {durations[0] / durations[1]:.2f}x")


if __name__ == "__main__":
    main()
