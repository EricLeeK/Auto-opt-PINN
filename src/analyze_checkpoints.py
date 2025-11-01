"""Analyse checkpoint trajectories by evaluating stored PINN states against the Burgers reference solution."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from auto_pinn.config import DEFAULT_CONFIG, ProjectConfig
from auto_pinn.gene import Gene, LayerGene, LayerType
from auto_pinn.pinn import HybridPINN
from run_best_gene import (
    ensure_device,
    evaluate_model,
    load_reference_solution,
    plot_comparison,
)

CHECKPOINT_REGEX = re.compile(r"checkpoint_epoch_(\d+)\.pt$", re.IGNORECASE)
DEFAULT_MAT_PATH = Path(__file__).resolve().parent / "burgers_shock.mat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a directory of checkpoint files and track PINN convergence.",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        required=True,
        help="Directory containing checkpoint_epoch_*.pt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoint_analysis"),
        help="Directory where analysis artefacts will be written.",
    )
    parser.add_argument(
        "--mat",
        type=Path,
        default=DEFAULT_MAT_PATH,
        help="Path to burgers_shock.mat reference solution.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override for evaluation (e.g. cuda:0 or cpu).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of checkpoints analysed (useful for smoke tests).",
    )
    parser.add_argument(
        "--save-grid",
        action="store_true",
        help="Persist the full prediction grid as .npz files alongside summary metrics.",
    )
    return parser.parse_args()


def _coerce_project_config(raw: Any) -> ProjectConfig:
    if isinstance(raw, ProjectConfig):
        return raw
    if isinstance(raw, dict):
        # Attempt to rebuild using dict data. Nested dataclasses can be restored with DEFAULT_CONFIG as template.
        base = asdict(DEFAULT_CONFIG)
        # Deep merge runtime/training overrides where available.
        for key in ("domain", "search", "ga", "training", "runtime"):
            if key in raw and isinstance(raw[key], dict):
                base[key].update(raw[key])
        return ProjectConfig(  # type: ignore[arg-type]
            domain=DEFAULT_CONFIG.domain.__class__(**base["domain"]),
            search=DEFAULT_CONFIG.search.__class__(**base["search"]),
            ga=DEFAULT_CONFIG.ga.__class__(**base["ga"]),
            training=DEFAULT_CONFIG.training.__class__(**base["training"]),
            runtime=DEFAULT_CONFIG.runtime.__class__(**base["runtime"]),
        )
    return DEFAULT_CONFIG


def _parse_gene_payload(raw_gene: Any) -> Gene:
    if isinstance(raw_gene, (list, tuple)):
        items = raw_gene
    elif isinstance(raw_gene, dict) and "gene" in raw_gene:
        items = raw_gene["gene"]
    else:
        raise ValueError("Gene payload must be a list of layer definitions or include a 'gene' key.")

    gene: Gene = []
    for idx, entry in enumerate(items):
        if isinstance(entry, LayerGene):
            gene.append(entry.copy())
            continue
        if not isinstance(entry, dict):
            raise TypeError(f"Layer definition at index {idx} must be a dict or LayerGene instance.")
        layer_type_value = entry.get("layer_type")
        if layer_type_value is None:
            raise KeyError(f"Missing 'layer_type' in layer definition at index {idx}")
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise TypeError(f"'params' for layer {idx} must be a mapping of parameter name to value")
        layer_type = LayerType(layer_type_value)
        processed_params = {str(k): int(v) for k, v in params.items()}
        gene.append(LayerGene(layer_type=layer_type, params=processed_params))
    if not gene:
        raise ValueError("Gene payload is empty")
    return gene


def _extract_epoch(path: Path, checkpoint_data: Dict[str, Any]) -> int:
    match = CHECKPOINT_REGEX.search(path.name)
    if match:
        return int(match.group(1))
    return int(checkpoint_data.get("epoch", 0))


def _load_state_dict(data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in data:
        return data["model_state_dict"]
    if "best_state" in data and data["best_state"] is not None:
        return data["best_state"]
    raise KeyError("Checkpoint does not contain a model_state_dict or best_state entry")


def analyse_checkpoint(
    checkpoint_path: Path,
    reference: Tuple[np.ndarray, np.ndarray, np.ndarray],
    device_override: Optional[str],
    output_dir: Path,
    save_grid: bool,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    gene_payload = checkpoint.get("gene")
    if gene_payload is None:
        raise KeyError(f"Checkpoint {checkpoint_path} is missing a 'gene' entry")
    gene = _parse_gene_payload(gene_payload)

    cfg = _coerce_project_config(checkpoint.get("config"))
    cfg = ensure_device(cfg, device_override)

    dtype = torch.float16 if cfg.runtime.dtype == "float16" else torch.float32
    device = torch.device(cfg.training.device)

    model = HybridPINN(gene).to(device).to(dtype)
    model.load_state_dict(_load_state_dict(checkpoint))
    model.eval()

    x_ref, t_ref, u_ref = reference
    prediction = evaluate_model(model, x_ref.astype(np.float64), t_ref.astype(np.float64), device, dtype)
    diff = prediction - u_ref

    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(u_ref))

    epoch = _extract_epoch(checkpoint_path, checkpoint)
    history = checkpoint.get("history")
    last_losses: Optional[Tuple[float, float, float, float]] = None
    if isinstance(history, Iterable):
        hist_list = list(history)
        if hist_list:
            last_losses = tuple(float(x) for x in hist_list[-1])  # total, pde, boundary, initial

    metrics = {
        "checkpoint": checkpoint_path.name,
        "epoch": epoch,
        "mse": mse,
        "mae": mae,
        "max_abs_error": max_abs,
        "relative_l2": rel_l2,
        "device": str(device),
        "dtype": str(dtype),
    }
    if last_losses is not None:
        metrics.update(
            {
                "train_total": last_losses[0],
                "train_pde": last_losses[1],
                "train_boundary": last_losses[2],
                "train_initial": last_losses[3],
            }
        )

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"comparison_epoch_{epoch:06d}.png"
    plot_comparison(x_ref, t_ref, u_ref, prediction, figure_path, show=False, relative_error=rel_l2)
    metrics["figure"] = str(figure_path.relative_to(output_dir))

    if save_grid:
        grid_dir = output_dir / "grids"
        grid_dir.mkdir(parents=True, exist_ok=True)
        grid_path = grid_dir / f"grid_epoch_{epoch:06d}.npz"
        np.savez_compressed(grid_path, prediction=prediction, reference=u_ref, diff=diff)
        metrics["grid"] = str(grid_path.relative_to(output_dir))

    metrics_path = output_dir / "metrics" / f"metrics_epoch_{epoch:06d}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    metrics["metrics_file"] = str(metrics_path.relative_to(output_dir))

    return metrics


def write_summary(metrics: List[Dict[str, Any]], output_dir: Path) -> None:
    summary_path = output_dir / "summary.csv"
    if not metrics:
        summary_path.write_text("checkpoint,epoch,mse,mae,max_abs_error,relative_l2\n")
        return

    fieldnames = [
        "checkpoint",
        "epoch",
        "mse",
        "mae",
        "max_abs_error",
        "relative_l2",
        "train_total",
        "train_pde",
        "train_boundary",
        "train_initial",
        "figure",
        "grid",
        "device",
        "dtype",
    ]

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(metrics, indent=2))

    final = metrics[-1]
    report_lines = [
        "Final Checkpoint Analysis",
        "==========================",
        f"Checkpoint: {final['checkpoint']}",
        f"Epoch: {final['epoch']}",
        "",
        "Metrics:",
        f"  MSE: {final['mse']:.6e}",
        f"  MAE: {final['mae']:.6e}",
        f"  Max |error|: {final['max_abs_error']:.6e}",
        f"  Relative L2: {final['relative_l2']:.6e}",
    ]
    if "train_total" in final:
        report_lines.extend(
            [
                "",
                "Latest Training Losses:",
                f"  Total: {final['train_total']:.6e}",
                f"  PDE: {final['train_pde']:.6e}",
                f"  Boundary: {final['train_boundary']:.6e}",
                f"  Initial: {final['train_initial']:.6e}",
            ]
        )
    report_lines.extend(
        [
            "",
            "Artefacts:",
            f"  Figure: {final.get('figure', 'N/A')}",
            f"  Metrics JSON: {final.get('metrics_file', 'N/A')}",
            f"  Grid: {final.get('grid', 'N/A')}",
        ]
    )

    report_path = output_dir / "final_report.txt"
    report_path.write_text("\n".join(report_lines))


def main() -> None:
    args = parse_args()

    checkpoint_dir = args.checkpoints
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = sorted(
        (path for path in checkpoint_dir.iterdir() if CHECKPOINT_REGEX.match(path.name)),
        key=lambda path: int(CHECKPOINT_REGEX.match(path.name).group(1)),  # type: ignore[union-attr]
    )
    if args.limit is not None:
        checkpoint_files = checkpoint_files[: args.limit]

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint_epoch_*.pt files found in {checkpoint_dir}")

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference_solution(args.mat)

    metrics: List[Dict[str, Any]] = []
    for idx, checkpoint_path in enumerate(checkpoint_files, start=1):
        print(f"[Analysis] Processing {checkpoint_path.name} ({idx}/{len(checkpoint_files)})")
        metrics.append(
            analyse_checkpoint(
                checkpoint_path,
                reference=reference,
                device_override=args.device,
                output_dir=output_dir,
                save_grid=args.save_grid,
            )
        )

    metrics.sort(key=lambda item: item["epoch"])
    write_summary(metrics, output_dir)
    print(f"[Analysis] Completed. Summary written to {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
