"""Train and visualise the best gene stored in ``search_results.json``.

This script loads the architecture discovered by the genetic search, retrains a
Hybrid PINN on the Burgers equation domain, and compares the resulting
prediction against the provided high-resolution numerical solution stored in
``src/burgers_shock.mat``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.optim import Adam

from auto_pinn.config import DEFAULT_CONFIG, ProjectConfig, RuntimeConfig, TrainingConfig
from auto_pinn.data import generate_training_batch
from auto_pinn.gene import Gene, LayerGene, LayerType
from auto_pinn.pinn import HybridPINN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the best Auto-PINN gene and visualise the result.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("search_results.json"),
        help="Path to the JSON file containing the best gene description.",
    )
    parser.add_argument(
        "--mat",
        type=Path,
        default=Path("src/burgers_shock.mat"),
        help="Reference Burgers solution stored as a MATLAB .mat file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("best_gene_solution.png"),
        help="Where to save the comparison figure.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs (falls back to config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run training on (defaults to config, falls back to CPU if unavailable).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Override logging frequency in epochs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the matplotlib figure interactively in addition to saving it.",
    )
    return parser.parse_args()


def load_gene(path: Path) -> Gene:
    payload = json.loads(path.read_text())
    if "gene" not in payload:
        raise KeyError(f"Missing 'gene' key in {path}")
    gene_layers = []
    for layer in payload["gene"]:
        layer_type = LayerType(layer["layer_type"])
        params: Dict[str, int] = {key: int(value) for key, value in layer["params"].items()}
        gene_layers.append(LayerGene(layer_type=layer_type, params=params))
    return gene_layers


def ensure_device(config: ProjectConfig, requested: str | None) -> ProjectConfig:
    training_cfg = config.training
    target_device = requested or training_cfg.device
    if target_device.startswith("cuda") and not torch.cuda.is_available():
        print("[Runner] CUDA not available; falling back to CPU.")
        target_device = "cpu"
    if target_device != training_cfg.device:
        training_cfg = replace(training_cfg, device=target_device)
        config = replace(config, training=training_cfg)
    return config


def override_runtime(config: ProjectConfig, epochs: int | None, log_every: int | None) -> ProjectConfig:
    training_cfg = config.training
    runtime_cfg = config.runtime
    updated = False
    if epochs is not None and epochs != training_cfg.epochs:
        training_cfg = replace(training_cfg, epochs=epochs)
        updated = True
    if log_every is not None and log_every != runtime_cfg.log_every:
        runtime_cfg = replace(runtime_cfg, log_every=log_every)
        updated = True
    if updated:
        config = replace(config, training=training_cfg, runtime=runtime_cfg)
    return config


def set_seeds(runtime: RuntimeConfig) -> None:
    import random

    random.seed(runtime.seed)
    np.random.seed(runtime.seed)
    torch.manual_seed(runtime.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runtime.seed)


def train_step(
    model: HybridPINN,
    optimizer: Adam,
    mse: nn.Module,
    batch,
    viscosity: float,
    training_cfg: TrainingConfig,
) -> Tuple[float, float, float, float]:
    optimizer.zero_grad(set_to_none=True)

    collocation = batch.collocation.clone().detach().requires_grad_(True)
    u_pred = model(collocation)
    grad_outputs = torch.ones_like(u_pred)
    grads = torch.autograd.grad(outputs=u_pred, inputs=collocation, grad_outputs=grad_outputs, create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]

    grad_outputs_x = torch.ones_like(u_x)
    u_xx = torch.autograd.grad(outputs=u_x, inputs=collocation, grad_outputs=grad_outputs_x, create_graph=True)[0][:, 0:1]

    pde_residual = u_t + u_pred * u_x - viscosity * u_xx
    pde_loss = mse(pde_residual, torch.zeros_like(pde_residual))

    if training_cfg.data_terms_require_grad:
        boundary_pred = model(batch.boundary_inputs)
        initial_pred = model(batch.initial_inputs)
    else:
        with torch.no_grad():
            boundary_pred = model(batch.boundary_inputs)
            initial_pred = model(batch.initial_inputs)

    boundary_loss = mse(boundary_pred, batch.boundary_targets)
    initial_loss = mse(initial_pred, batch.initial_targets)

    loss = (
        training_cfg.pde_weight * pde_loss
        + training_cfg.boundary_weight * boundary_loss
        + training_cfg.initial_weight * initial_loss
    )

    loss.backward()
    optimizer.step()

    return (
        float(loss.item()),
        float(pde_loss.item()),
        float(boundary_loss.item()),
        float(initial_loss.item()),
    )


def train_gene(config: ProjectConfig, gene: Gene) -> Tuple[HybridPINN, List[Tuple[float, float, float, float]], float]:
    set_seeds(config.runtime)
    device = torch.device(config.training.device)
    dtype = torch.float16 if config.runtime.dtype == "float16" else torch.float32

    model = HybridPINN(gene).to(device).to(dtype)
    optimizer = Adam(model.parameters(), lr=config.training.learning_rate)
    mse = nn.MSELoss()

    history: List[Tuple[float, float, float, float]] = []
    best_loss = math.inf
    best_components = (math.inf, math.inf, math.inf)
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(config.training.epochs):
        batch = generate_training_batch(config.domain, config.training, device, dtype)
        total, pde, boundary, initial = train_step(
            model,
            optimizer,
            mse,
            batch,
            viscosity=config.domain.viscosity,
            training_cfg=config.training,
        )
        history.append((total, pde, boundary, initial))
        if total < best_loss:
            best_loss = total
            best_components = (pde, boundary, initial)
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        should_log = (epoch + 1) % config.runtime.log_every == 0 or epoch == 0
        if should_log:
            print(
                f"[Runner] Epoch {epoch + 1}/{config.training.epochs} | Loss {total:.6f} "
                f"| PDE {pde:.6f} | Boundary {boundary:.6f} | Initial {initial:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(
        "[Runner] Best loss {:.6f} | PDE {:.6f} | Boundary {:.6f} | Initial {:.6f}".format(
            best_loss, *best_components
        )
    )
    model.eval()
    return model, history, best_loss


def load_reference_solution(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = loadmat(path)
    possible_keys = {key.lower(): key for key in data.keys()}
    try:
        x_key = possible_keys.get("x", "x")
        t_key = possible_keys.get("t", "t")
        u_key = possible_keys.get("usol", "usol")
        x = np.squeeze(data[x_key])
        t = np.squeeze(data[t_key])
        u = np.array(data[u_key], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"Could not find expected keys in {path}. Available: {list(data.keys())}") from exc
    if u.shape[0] != x.size and u.shape[0] == t.size:
        u = u.T  # align to (len(x), len(t))
    return x, t, u


def evaluate_model(model: HybridPINN, x: np.ndarray, t: np.ndarray, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    xx, tt = np.meshgrid(x, t, indexing="ij")
    grid = np.stack([xx.ravel(), tt.ravel()], axis=1)
    inputs = torch.from_numpy(grid).to(device=device, dtype=dtype)
    with torch.no_grad():
        prediction = model(inputs).view(xx.shape)
    return prediction.cpu().numpy()


def plot_comparison(
    x: np.ndarray,
    t: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    show: bool,
    relative_error: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    extent = [t.min(), t.max(), x.min(), x.max()]

    im0 = axes[0].imshow(reference, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title("Reference")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(prediction, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    axes[1].set_title("PINN Prediction")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    fig.colorbar(im1, ax=axes[1])

    diff = np.abs(prediction - reference)
    im2 = axes[2].imshow(diff, aspect="auto", origin="lower", extent=extent, cmap="magma")
    axes[2].set_title(f"Absolute Error (rel={relative_error:.3e})")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x")
    fig.colorbar(im2, ax=axes[2])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"[Runner] Saved comparison figure to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()

    gene = load_gene(args.results)

    config = ensure_device(DEFAULT_CONFIG, args.device)
    config = override_runtime(config, args.epochs, args.log_every)

    print("[Runner] Using device:", config.training.device)
    print("[Runner] Training epochs:", config.training.epochs)
    print("[Runner] Gene architecture:")
    for idx, layer in enumerate(gene, start=1):
        params = ", ".join(f"{key}={value}" for key, value in layer.params.items())
        print(f"  Layer {idx}: {layer.layer_type.value}({params})")

    model, history, best_loss = train_gene(config, gene)
    fitness = 1.0 / (best_loss + 1e-8)
    print(f"[Runner] Estimated fitness: {fitness:.6f}")

    x_ref, t_ref, u_ref = load_reference_solution(args.mat)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prediction = evaluate_model(model, x_ref.astype(np.float64), t_ref.astype(np.float64), device, dtype)

    rel_error = np.linalg.norm(prediction - u_ref) / np.linalg.norm(u_ref)
    print(f"[Runner] Relative L2 error vs reference: {rel_error:.6e}")

    plot_comparison(x_ref, t_ref, u_ref, prediction, args.output, args.show, rel_error)


if __name__ == "__main__":
    main()
