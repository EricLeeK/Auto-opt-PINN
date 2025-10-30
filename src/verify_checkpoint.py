"""Verify checkpoint integrity and perform a short resume test.

This utility validates that a checkpoint contains all artefacts required to
resume training for the hybrid PINN architecture. It also loads the checkpoint,

against the Burgers' reference solution to catch potential regression issues.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from auto_pinn.config import DEFAULT_CONFIG, ProjectConfig
from auto_pinn.data import generate_training_batch
from auto_pinn.gene import Gene
from auto_pinn.pinn import HybridPINN
from run_best_gene import (
    train_step,
    load_reference_solution,
    evaluate_model,
    plot_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify checkpoint integrity and resume capability")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint file to verify.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for the resume test (defaults to the checkpoint's training device).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Number of additional training steps to run for verification.",
    )
    parser.add_argument(
        "--mat",
        type=Path,
        default=Path("src/burgers_shock.mat"),
        help="Path to the reference Burgers solution (.mat file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the verification comparison plot (defaults next to the checkpoint).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the comparison figure interactively.",
    )
    return parser.parse_args()


def verify_checkpoint_structure(checkpoint: Dict[str, Any]) -> bool:
    required_keys = [
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "best_loss",
        "best_components",
        "history",
        "gene",
        "config",
    ]

    print("=" * 60)
    print("CHECKPOINT STRUCTURE VERIFICATION")
    print("=" * 60)

    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        print(f"❌ Missing keys: {missing}")
        print()
        return False

    print("✅ All required keys present:")
    for key in required_keys:
        print(f"   - {key}")
    print()
    return True


def verify_gene_structure(gene: Gene) -> bool:
    print("=" * 60)
    print("GENE ARCHITECTURE VERIFICATION")
    print("=" * 60)

    if not gene:
        print("❌ Gene is empty!")
        print()
        return False

    print(f"✅ Gene has {len(gene)} layers:")
    for idx, layer in enumerate(gene, start=1):
        print(f"   Layer {idx}: {layer.layer_type.value}")
        print(f"      Params: {layer.params}")
    print()
    return True


def verify_model_reconstruction(gene: Gene, state_dict: Dict[str, torch.Tensor]) -> bool:
    print("=" * 60)
    print("MODEL RECONSTRUCTION VERIFICATION")
    print("=" * 60)

    try:
        model = HybridPINN(gene)
        print("✅ Model successfully instantiated from gene")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {param_count:,}")
        model.load_state_dict(state_dict)
        print("✅ State dict successfully loaded")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Trainable parameters: {trainable_params:,}")
        if trainable_params != param_count:
            print(f"⚠️  Warning: {param_count - trainable_params} parameters are frozen")
        print()
        return True
    except Exception as exc:
        print(f"❌ Model reconstruction failed: {exc}")
        print()
        return False


def verify_optimizer_state(optimizer_state: Dict[str, Any]) -> None:
    print("=" * 60)
    print("OPTIMIZER STATE VERIFICATION")
    print("=" * 60)

    if "state" not in optimizer_state:
        print("❌ No optimizer state found")
        print()
        return

    param_groups = len(optimizer_state["state"])
    print(f"✅ Optimizer state entries: {param_groups}")
    if param_groups:
        first_key = next(iter(optimizer_state["state"]))
        first = optimizer_state["state"][first_key]
        if "exp_avg" in first and "exp_avg_sq" in first:
            print("   ✅ Adam momentum buffers detected (exp_avg, exp_avg_sq)")
        else:
            print("   ⚠️  Adam momentum buffers missing")
    print()


def verify_training_continuity(checkpoint: Dict[str, Any]) -> None:
    print("=" * 60)
    print("TRAINING CONTINUITY VERIFICATION")
    print("=" * 60)

    epoch = checkpoint.get("epoch", -1)
    best_loss = checkpoint.get("best_loss", float("inf"))
    best_components = checkpoint.get("best_components", (None, None, None))
    history = checkpoint.get("history", [])

    print(f"✅ Checkpoint from epoch: {epoch + 1}")
    print(f"✅ Best loss recorded: {best_loss:.6f}")
    pde, boundary, initial = best_components
    print(f"   - PDE loss: {pde:.6f}" if pde is not None else "   - PDE loss: N/A")
    print(f"   - Boundary loss: {boundary:.6f}" if boundary is not None else "   - Boundary loss: N/A")
    print(f"   - Initial loss: {initial:.6f}" if initial is not None else "   - Initial loss: N/A")
    print(f"✅ Training history length: {len(history)} epochs")
    if "best_state" in checkpoint:
        print("✅ Best model snapshot stored")
    else:
        print("⚠️  No dedicated best model snapshot (current weights will be used)")
    print()


def verify_parameter_integrity(state_dict: Dict[str, torch.Tensor]) -> None:
    print("=" * 60)
    print("PARAMETER INTEGRITY CHECK")
    print("=" * 60)

    has_nan = False
    has_inf = False
    for name, param in state_dict.items():
        if torch.isnan(param).any():
            print(f"❌ NaN detected in parameter: {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"❌ Inf detected in parameter: {name}")
            has_inf = True
    if not has_nan and not has_inf:
        print("✅ All parameters are finite")
    print()


def move_optimizer_state_to_device(optimizer: Adam, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def resume_and_plot(
    checkpoint: Dict[str, Any],
    checkpoint_path: Path,
    device_override: str | None,
    steps: int,
    mat_path: Path,
    output_path: Path | None,
    show: bool,
) -> None:
    print("=" * 60)
    print("RESUME TRAINING TEST")
    print("=" * 60)

    if steps <= 0:
        print("⚠️  Step count must be positive; skipping resume test.")
        print()
        return

    gene: Gene = checkpoint["gene"]
    config: ProjectConfig = checkpoint.get("config", DEFAULT_CONFIG)

    target_device = device_override or config.training.device
    if isinstance(target_device, str) and target_device.startswith("cuda") and not torch.cuda.is_available():
        print("[Verifier] CUDA requested but unavailable; falling back to CPU.")
        target_device = "cpu"
    device = torch.device(target_device)

    dtype = torch.float16 if config.runtime.dtype == "float16" else torch.float32

    model = HybridPINN(gene)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).to(dtype)

    optimizer = Adam(model.parameters(), lr=config.training.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)

    mse = nn.MSELoss()
    model.train()

    print(f"[Verifier] Running {steps} additional optimisation steps on {device}...")
    losses: list[float] = []
    for step in range(1, steps + 1):
        batch = generate_training_batch(config.domain, config.training, device, dtype)
        total, pde_loss, boundary_loss, initial_loss = train_step(
            model,
            optimizer,
            mse,
            batch,
            viscosity=config.domain.viscosity,
            training_cfg=config.training,
        )
        losses.append(total)
        print(
            f"   Step {step}/{steps} | Loss {total:.6f} | PDE {pde_loss:.6f} | "
            f"Boundary {boundary_loss:.6f} | Initial {initial_loss:.6f}"
        )
    if losses:
        print(f"[Verifier] Final loss after resume steps: {losses[-1]:.6f}")
    print()

    if not mat_path.exists():
        print(f"⚠️  Reference solution not found at {mat_path}; skipping visual comparison.")
        print()
        return

    model.eval()
    x_ref, t_ref, u_ref = load_reference_solution(mat_path)
    prediction = evaluate_model(
        model,
        x_ref.astype(np.float64),
        t_ref.astype(np.float64),
        device,
        dtype,
    )

    rel_error = np.linalg.norm(prediction - u_ref) / np.linalg.norm(u_ref)
    print(f"[Verifier] Relative L2 error vs reference: {rel_error:.6e}")

    if output_path is None:
        output_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_verification.png")

    plot_comparison(
        x_ref,
        t_ref,
        u_ref,
        prediction,
        output_path=output_path,
        show=show,
        relative_error=rel_error,
    )
    print(f"[Verifier] Verification plot saved to {output_path}\n")


def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"\nLoading checkpoint: {checkpoint_path}")
    print(f"File size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB\n")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        print(f"❌ Failed to load checkpoint: {exc}")
        return

    if not verify_checkpoint_structure(checkpoint):
        return

    gene = checkpoint.get("gene")
    if gene is None:
        print("❌ Checkpoint is missing the gene definition; aborting.")
        return

    verify_gene_structure(gene)

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is not None:
        verify_model_reconstruction(gene, state_dict)
        verify_parameter_integrity(state_dict)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        verify_optimizer_state(optimizer_state)

    verify_training_continuity(checkpoint)

    resume_and_plot(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        device_override=args.device,
        steps=args.steps,
        mat_path=args.mat,
        output_path=args.output,
        show=args.show,
    )

    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    resume_hint = (
        f"python run_best_gene.py --resume --checkpoint {checkpoint_path}"
        if checkpoint_path.is_file()
        else f"python run_best_gene.py --resume --checkpoint {checkpoint_path.parent}"
    )
    print(f"\n✅ Checkpoint verified. Resume command: {resume_hint}")


if __name__ == "__main__":
    main()
