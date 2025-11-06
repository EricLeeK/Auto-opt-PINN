"""Entry point for running the Auto-PINN search with seed genes.

This script allows you to warm-start the genetic algorithm with one or more
existing gene architectures, enabling continuation of the search from known
good solutions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Auto-PINN genetic search with optional seed genes."
    )
    parser.add_argument(
        "--seed-genes",
        type=str,
        nargs="+",
        default=[],
        help="Path(s) to JSON file(s) containing seed gene architectures to warm-start the population.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("search_results.json"),
        help="Where to save the best gene found.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Override population size (default from config).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override number of generations (default from config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs per individual (default from config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (cuda/cpu, default from config).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default from config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default from config).",
    )
    parser.add_argument(
        "--gpu-devices",
        type=str,
        nargs="+",
        default=None,
        help="GPU devices to use (e.g., cuda:0 cuda:1). If not specified, uses config default.",
    )
    parser.add_argument(
        "--gpu-concurrency",
        type=int,
        default=None,
        help="Number of parallel workers per GPU (default from config).",
    )
    return parser.parse_args()


def apply_overrides(config: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    """Apply command-line argument overrides to the config."""
    
    # GA config overrides
    ga_cfg = config.ga
    if args.population_size is not None:
        ga_cfg = replace(ga_cfg, population_size=args.population_size)
    if args.generations is not None:
        ga_cfg = replace(ga_cfg, generations=args.generations)
    
    # Add seed gene files
    if args.seed_genes:
        print(f"[Main] Seed genes specified: {args.seed_genes}")
        ga_cfg = replace(ga_cfg, resume_gene_files=tuple(args.seed_genes))
    
    # Training config overrides
    training_cfg = config.training
    if args.epochs is not None:
        training_cfg = replace(training_cfg, epochs=args.epochs)
    if args.device is not None:
        training_cfg = replace(training_cfg, device=args.device)
    
    # Runtime config overrides
    runtime_cfg = config.runtime
    if args.workers is not None:
        runtime_cfg = replace(runtime_cfg, workers=args.workers)
    if args.seed is not None:
        runtime_cfg = replace(runtime_cfg, seed=args.seed)
    if args.gpu_devices is not None:
        runtime_cfg = replace(runtime_cfg, gpu_devices=tuple(args.gpu_devices))
    if args.gpu_concurrency is not None:
        runtime_cfg = replace(runtime_cfg, gpu_concurrency=args.gpu_concurrency)
    
    # Build new config
    config = replace(
        config,
        ga=ga_cfg,
        training=training_cfg,
        runtime=runtime_cfg,
    )
    
    return config


def run(config: ProjectConfig, output_path: Path) -> None:
    """Run the genetic search with the given configuration."""
    
    print("\n" + "=" * 70)
    print("Auto-PINN Genetic Algorithm Configuration")
    print("=" * 70)
    print(f"Population size:    {config.ga.population_size}")
    print(f"Generations:        {config.ga.generations}")
    print(f"Training epochs:    {config.training.epochs}")
    print(f"Device:             {config.training.device}")
    print(f"Workers:            {config.runtime.workers}")
    print(f"GPU devices:        {config.runtime.gpu_devices}")
    print(f"GPU concurrency:    {config.runtime.gpu_concurrency}")
    print(f"Random seed:        {config.runtime.seed}")
    print(f"Seed gene files:    {config.ga.resume_gene_files or 'None'}")
    print("=" * 70 + "\n")
    
    evaluator = PINNFitnessEvaluator(config)
    best_gene, best_fitness = run_genetic_search(evaluator, config)
    
    print("\n" + "=" * 70)
    print("Genetic search completed!")
    print("=" * 70)
    print(f"Best fitness: {best_fitness:.6f}")
    print("Best gene architecture:")
    for idx, layer in enumerate(best_gene, start=1):
        params = ", ".join(f"{key}={value}" for key, value in layer.params.items())
        print(f"  Layer {idx}: {layer.layer_type.value}({params})")
    
    # Save results
    payload = {
        "fitness": best_fitness,
        "gene": serialize_gene(best_gene),
        "metadata": {
            "population_size": config.ga.population_size,
            "generations": config.ga.generations,
            "training_epochs": config.training.epochs,
            "seed": config.runtime.seed,
            "seed_genes_used": list(config.ga.resume_gene_files) if config.ga.resume_gene_files else [],
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    
    # Load base config and apply overrides
    config = apply_overrides(DEFAULT_CONFIG, args)
    
    # Run the search
    run(config, args.output)


if __name__ == "__main__":
    main()
