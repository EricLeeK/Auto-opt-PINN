"""Training loop used to evaluate gene fitness via PINN optimisation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .config import ProjectConfig
from .data import TrainingBatch, generate_training_batch
from .gene import Gene
from .pinn import HybridPINN


@dataclass
class TrainingResult:
    gene: Gene
    fitness: float
    final_loss: float


class PINNFitnessEvaluator:
    """Callable fitness evaluator for the genetic algorithm."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.device = torch.device(config.training.device)
        self.dtype = torch.float16 if config.runtime.dtype == "float16" else torch.float32
        self._context = {
            "generation": 0,
            "total_generations": 0,
            "individual": 0,
            "population_size": 0,
        }

    def set_run_context(
        self,
        generation: int,
        total_generations: int,
        individual: int,
        population_size: int,
    ) -> None:
        self._context = {
            "generation": generation,
            "total_generations": total_generations,
            "individual": individual,
            "population_size": population_size,
        }

    def __call__(self, gene: Gene) -> float:
        try:
            result = self._train_gene(gene)
            return result.fitness
        except Exception:
            return 0.0

    def _train_gene(self, gene: Gene) -> TrainingResult:
        random.seed(self.config.runtime.seed)
        torch.manual_seed(self.config.runtime.seed)
        np.random.seed(self.config.runtime.seed)
        model = HybridPINN(gene).to(self.device).to(self.dtype)
        optimizer = Adam(model.parameters(), lr=self.config.training.learning_rate)
        mse = nn.MSELoss()

        gen = self._context.get("generation", 0)
        total_gen = self._context.get("total_generations", 0)
        ind = self._context.get("individual", 0)
        pop = self._context.get("population_size", 0)
        print(
            f"[Evaluator] Generation {gen}/{total_gen} | Individual {ind}/{pop} | Starting training with {len(gene)} layers"
        )

        final_loss = math.inf
        best_loss = math.inf
        best_components = (math.inf, math.inf, math.inf)
        for epoch in range(self.config.training.epochs):
            batch = self._sample_batch()
            final_loss, pde_loss, boundary_loss, initial_loss = self._train_step(model, optimizer, mse, batch)
            if final_loss < best_loss:
                best_loss = final_loss
                best_components = (pde_loss, boundary_loss, initial_loss)
            should_log = (epoch + 1) % self.config.runtime.log_every == 0 or epoch == 0
            if should_log:
                print(
                    f"[PINN] Generation {gen}/{total_gen} | Individual {ind}/{pop} | Epoch {epoch + 1}/{self.config.training.epochs}"
                    f" | Total {final_loss:.6f} | PDE {pde_loss:.6f} | Boundary {boundary_loss:.6f} | Initial {initial_loss:.6f}"
                )
        fitness = 1.0 / (best_loss + 1e-8)
        print(
            f"[Evaluator] Generation {gen}/{total_gen} | Individual {ind}/{pop} | Best total {best_loss:.6f}"
            f" | PDE {best_components[0]:.6f} | Boundary {best_components[1]:.6f} | Initial {best_components[2]:.6f}"
        )
        return TrainingResult(gene=gene, fitness=fitness, final_loss=best_loss)

    def _sample_batch(self) -> TrainingBatch:
        return generate_training_batch(self.config.domain, self.config.training, self.device, self.dtype)

    def _train_step(
        self,
        model: HybridPINN,
        optimizer: Adam,
        mse: nn.MSELoss,
        batch: TrainingBatch,
    ) -> Tuple[float, float, float, float]:
        optimizer.zero_grad(set_to_none=True)

        collocation = batch.collocation.clone().detach().requires_grad_(True)
        u_pred = model(collocation)
        grad_outputs = torch.ones_like(u_pred)
        grads = torch.autograd.grad(
            outputs=u_pred,
            inputs=collocation,
            grad_outputs=grad_outputs,
            create_graph=True,
        )[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        grad_outputs_x = torch.ones_like(u_x)
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=collocation,
            grad_outputs=grad_outputs_x,
            create_graph=True,
        )[0][:, 0:1]
        viscosity = self.config.domain.viscosity
        pde_residual = u_t + u_pred * u_x - viscosity * u_xx
        pde_loss = mse(pde_residual, torch.zeros_like(pde_residual))

        if self.config.training.data_terms_require_grad:
            boundary_pred = model(batch.boundary_inputs)
        else:
            with torch.no_grad():  # skips parameter gradients for faster but less constrained training
                boundary_pred = model(batch.boundary_inputs)
        boundary_loss = mse(boundary_pred, batch.boundary_targets)

        if self.config.training.data_terms_require_grad:
            initial_pred = model(batch.initial_inputs)
        else:
            with torch.no_grad():  # same trade-off applied to the initial condition term
                initial_pred = model(batch.initial_inputs)
        initial_loss = mse(initial_pred, batch.initial_targets)

        loss = (
            self.config.training.pde_weight * pde_loss
            + self.config.training.boundary_weight * boundary_loss
            + self.config.training.initial_weight * initial_loss
        )

        loss.backward()
        optimizer.step()

        return (
            float(loss.item()),
            float(pde_loss.item()),
            float(boundary_loss.item()),
            float(initial_loss.item()),
        )
