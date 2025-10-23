"""Entry point for running the Auto-PINN search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from auto_pinn.config import DEFAULT_CONFIG, ProjectConfig
from auto_pinn.genetic_algorithm import run_genetic_search
from auto_pinn.trainer import PINNFitnessEvaluator


def serialize_gene(gene: Any) -> Any:
    if isinstance(gene, list):
        return [serialize_gene(item) for item in gene]
    if hasattr(gene, "layer_type"):
        return {"layer_type": gene.layer_type.value, "params": gene.params}
    return gene


def run(config: ProjectConfig = DEFAULT_CONFIG) -> None:
    evaluator = PINNFitnessEvaluator(config)
    best_gene, best_fitness = run_genetic_search(evaluator, config)
    print("Genetic search completed.")
    print(f"Best fitness: {best_fitness:.6f}")
    print("Best gene:")
    for idx, layer in enumerate(best_gene):
        print(f"  Layer {idx + 1}: {layer.layer_type.value} {layer.params}")
    output = Path("search_results.json")
    payload = {"fitness": best_fitness, "gene": serialize_gene(best_gene)}
    output.write_text(json.dumps(payload, indent=2))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    run()
