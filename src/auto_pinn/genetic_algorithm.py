"""Genetic algorithm utilities that explore the hybrid PINN search space."""

from __future__ import annotations

import math
import random
from typing import Callable, List, Sequence, Tuple

from .config import ProjectConfig
from .gene import Gene, LayerGene, LayerType

FitnessEvaluator = Callable[[Gene], float]


def _random_layer(config: ProjectConfig) -> LayerGene:
    choice = random.choice(list(LayerType))
    if choice == LayerType.DNN:
        neurons = random.choice(tuple(config.search.dnn_neurons))
        return LayerGene(layer_type=choice, params={"units": int(neurons)})
    if choice == LayerType.KAN:
        width = random.choice(tuple(config.search.kan_width))
        grid = random.choice(tuple(config.search.kan_grid_points))
        order = random.choice(tuple(config.search.kan_spline_order))
        return LayerGene(
            layer_type=choice,
            params={"width": int(width), "grid_points": int(grid), "spline_order": int(order)},
        )
    embed = random.choice(tuple(config.search.attn_embed_dim))
    heads = random.choice(tuple(config.search.attn_heads))
    heads = max(1, min(embed, heads))
    while embed % heads != 0:
        heads = max(1, heads - 1)
    return LayerGene(layer_type=choice, params={"embed_dim": int(embed), "heads": int(heads)})


def create_random_gene(config: ProjectConfig) -> Gene:
    layer_count = random.randint(config.search.min_layers, config.search.max_layers)
    return [_random_layer(config) for _ in range(layer_count)]


def initialize_population(config: ProjectConfig) -> List[Gene]:
    return [create_random_gene(config) for _ in range(config.ga.population_size)]


def tournament_selection(population: Sequence[Gene], fitness: Sequence[float], config: ProjectConfig) -> Gene:
    contenders = random.sample(range(len(population)), k=config.ga.tournament_size)
    best_idx = max(contenders, key=lambda idx: fitness[idx])
    return [layer.copy() for layer in population[best_idx]]


def crossover(parent_a: Gene, parent_b: Gene) -> Gene:
    if not parent_a or not parent_b:
        return [layer.copy() for layer in (parent_a or parent_b)]
    limit = min(len(parent_a), len(parent_b))
    if limit <= 1:
        return [layer.copy() for layer in parent_a]
    pivot = random.randint(1, limit - 1)
    child: Gene = []
    child.extend(layer.copy() for layer in parent_a[:pivot])
    child.extend(layer.copy() for layer in parent_b[pivot:])
    return child


def mutate(gene: Gene, config: ProjectConfig) -> Gene:
    if not gene:
        return create_random_gene(config)
    candidate = [layer.copy() for layer in gene]
    action = random.choice(["swap_type", "change_param", "add", "delete"])
    if action == "swap_type" and candidate:
        idx = random.randrange(len(candidate))
        candidate[idx] = _random_layer(config)
        return candidate
    if action == "change_param" and candidate:
        idx = random.randrange(len(candidate))
        layer = candidate[idx]
        if layer.layer_type == LayerType.DNN:
            layer.params["units"] = int(random.choice(tuple(config.search.dnn_neurons)))
        elif layer.layer_type == LayerType.KAN:
            param = random.choice(["width", "grid_points", "spline_order"])
            if param == "width":
                layer.params[param] = int(random.choice(tuple(config.search.kan_width)))
            elif param == "grid_points":
                layer.params[param] = int(random.choice(tuple(config.search.kan_grid_points)))
            else:
                layer.params[param] = int(random.choice(tuple(config.search.kan_spline_order)))
        else:
            embed = int(random.choice(tuple(config.search.attn_embed_dim)))
            heads = int(random.choice(tuple(config.search.attn_heads)))
            heads = max(1, min(embed, heads))
            while embed % heads != 0:
                heads = max(1, heads - 1)
            layer.params.update({"embed_dim": embed, "heads": heads})
        return candidate
    if action == "add" and len(candidate) < config.search.max_layers:
        pos = random.randint(0, len(candidate))
        candidate.insert(pos, _random_layer(config))
        return candidate
    if action == "delete" and len(candidate) > config.search.min_layers:
        idx = random.randrange(len(candidate))
        candidate.pop(idx)
        return candidate
    return candidate


def run_genetic_search(evaluator: FitnessEvaluator, config: ProjectConfig) -> Tuple[Gene, float]:
    population = initialize_population(config)
    best_gene: Gene = []
    best_fitness = -math.inf
    for generation in range(config.ga.generations):
        fitness_scores: List[float] = []
        for gene in population:
            score = evaluator(gene)
            fitness_scores.append(score)
            if score > best_fitness:
                best_fitness = score
                best_gene = [layer.copy() for layer in gene]
        elite_indices = sorted(range(len(population)), key=lambda idx: fitness_scores[idx], reverse=True)[
            : config.ga.elite_count
        ]
        new_population: List[Gene] = [[layer.copy() for layer in population[idx]] for idx in elite_indices]
        while len(new_population) < config.ga.population_size:
            parent_a = tournament_selection(population, fitness_scores, config)
            parent_b = tournament_selection(population, fitness_scores, config)
            if random.random() < config.ga.crossover_rate:
                child = crossover(parent_a, parent_b)
            else:
                child = parent_a
            if random.random() < config.ga.mutation_rate:
                child = mutate(child, config)
            new_population.append(child)
        population = new_population
    return best_gene, best_fitness
