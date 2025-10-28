"""Genetic algorithm utilities that explore the hybrid PINN search space."""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch.multiprocessing as mp

from .config import ProjectConfig
from .gene import Gene, GeneSignature, LayerGene, LayerType, gene_signature
from .trainer import PINNFitnessEvaluator

FitnessEvaluator = Callable[[Gene], float]


def _evaluate_gene_task(payload: Tuple[Gene, ProjectConfig, Dict[str, int]]) -> float:
    gene, config, context = payload
    evaluator = PINNFitnessEvaluator(config)
    if context:
        evaluator.set_run_context(
            generation=context["generation"],
            total_generations=context["total_generations"],
            individual=context["individual"],
            population_size=context["population_size"],
        )
    return evaluator(gene)


def _random_layer(config: ProjectConfig) -> LayerGene:
    choice = random.choice(list(LayerType))
    if choice == LayerType.DNN:
        neurons = random.choice(tuple(config.search.dnn_neurons))
        return LayerGene(layer_type=choice, params={"units": int(neurons)})
    if choice == LayerType.KAN:
        width = random.choice(tuple(config.search.kan_width))
        valid_pairs = [
            (grid, order)
            for grid in config.search.kan_grid_points
            for order in config.search.kan_spline_order
            if grid >= order + 1
        ]
        if not valid_pairs:
            raise ValueError("Search space does not contain any valid (grid_points, spline_order) pairs")
        grid, order = random.choice(valid_pairs)
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


def _format_gene_structure(gene: Gene) -> str:
    if not gene:
        return "  <empty gene>"
    lines = []
    for idx, layer in enumerate(gene, start=1):
        params = ", ".join(f"{key}={value}" for key, value in layer.params.items())
        lines.append(f"  Layer {idx}: {layer.layer_type.value}({params})")
    return "\n".join(lines)


def create_random_gene(config: ProjectConfig) -> Gene:
    layer_count = random.randint(config.search.min_layers, config.search.max_layers)
    return [_random_layer(config) for _ in range(layer_count)]


def initialize_population(config: ProjectConfig) -> List[Gene]:
    population: List[Gene] = []
    seen_signatures: Set[GeneSignature] = set()
    max_attempts = config.ga.population_size * 30
    attempts = 0
    while len(population) < config.ga.population_size:
        candidate = create_random_gene(config)
        if config.ga.deduplicate_population:
            signature = gene_signature(candidate)
            if signature in seen_signatures:
                attempts += 1
                if attempts < max_attempts:
                    continue
            else:
                seen_signatures.add(signature)
                attempts = 0
        population.append(candidate)
    return population


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
                valid_grids = [
                    g for g in config.search.kan_grid_points if g >= layer.params["spline_order"] + 1
                ]
                if not valid_grids:
                    valid_grids = [max(layer.params["spline_order"] + 1, min(config.search.kan_grid_points))]
                layer.params[param] = int(random.choice(tuple(valid_grids)))
            else:
                valid_orders = [
                    o for o in config.search.kan_spline_order if o + 1 <= layer.params["grid_points"]
                ]
                if not valid_orders:
                    valid_orders = [max(1, layer.params["grid_points"] - 1)]
                layer.params[param] = int(random.choice(tuple(valid_orders)))
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
    num_workers = max(1, config.runtime.workers)
    use_parallel = num_workers > 1
    supports_cache = all(
        hasattr(evaluator, attr)
        for attr in ("cache_key_for", "get_cached_fitness", "store_cached_fitness")
    )
    pool: Optional[mp.pool.Pool] = None
    try:
        if use_parallel:
            pool = mp.get_context("spawn").Pool(processes=num_workers)
        for generation in range(config.ga.generations):
            print(f"[GA] ==== Entering generation {generation + 1}/{config.ga.generations} ==== ")
            if use_parallel:
                fitness_scores: List[float] = [0.0] * len(population)
                pending_payloads: List[Tuple[Gene, ProjectConfig, Dict[str, int]]] = []
                pending_meta: List[Tuple[int, Optional[GeneSignature], Dict[str, int]]] = []
                for idx, gene in enumerate(population):
                    context = {
                        "generation": generation + 1,
                        "total_generations": config.ga.generations,
                        "individual": idx + 1,
                        "population_size": len(population),
                    }
                    print(
                        f"[GA] Generation {context['generation']}/{context['total_generations']} | Evaluating individual {context['individual']}/{context['population_size']}"
                    )
                    print("[GA] Gene structure:")
                    print(_format_gene_structure(gene))
                    cache_key = evaluator.cache_key_for(gene) if supports_cache else None
                    cached_score = (
                        evaluator.get_cached_fitness(cache_key) if supports_cache else None
                    )
                    if cached_score is not None:
                        if supports_cache:
                            print("[Evaluator] Using cached fitness for repeated gene")
                        fitness_scores[idx] = cached_score
                        if cached_score > best_fitness:
                            best_fitness = cached_score
                            best_gene = [layer.copy() for layer in gene]
                        print(
                            f"[GA] Generation {context['generation']}/{context['total_generations']} | Individual {context['individual']}/{context['population_size']}"
                            f" | Fitness {cached_score:.6f}"
                        )
                        continue
                    payload_gene = [layer.copy() for layer in gene]
                    context_copy = context.copy()
                    pending_payloads.append((payload_gene, config, context_copy))
                    pending_meta.append((idx, cache_key, context_copy))
                if pending_payloads and pool is not None:
                    results = pool.map(_evaluate_gene_task, pending_payloads)
                    for (idx, cache_key, context), score in zip(pending_meta, results):
                        fitness_scores[idx] = score
                        if supports_cache:
                            evaluator.store_cached_fitness(cache_key, score)
                        if score > best_fitness:
                            best_fitness = score
                            best_gene = [layer.copy() for layer in population[idx]]
                        print(
                            f"[GA] Generation {context['generation']}/{context['total_generations']} | Individual {context['individual']}/{context['population_size']}"
                            f" | Fitness {score:.6f}"
                        )
            else:
                fitness_scores = []
                for idx, gene in enumerate(population):
                    print(
                        f"[GA] Generation {generation + 1}/{config.ga.generations} | Evaluating individual {idx + 1}/{len(population)}"
                    )
                    if hasattr(evaluator, "set_run_context"):
                        evaluator.set_run_context(
                            generation=generation + 1,
                            total_generations=config.ga.generations,
                            individual=idx + 1,
                            population_size=len(population),
                        )
                    print("[GA] Gene structure:")
                    print(_format_gene_structure(gene))
                    score = evaluator(gene)
                    fitness_scores.append(score)
                    if score > best_fitness:
                        best_fitness = score
                        best_gene = [layer.copy() for layer in gene]
                    print(
                        f"[GA] Generation {generation + 1}/{config.ga.generations} | Individual {idx + 1}/{len(population)}"
                        f" | Fitness {score:.6f}"
                    )
            elite_indices = sorted(range(len(population)), key=lambda idx: fitness_scores[idx], reverse=True)[
                : config.ga.elite_count
            ]
            new_population: List[Gene] = [[layer.copy() for layer in population[idx]] for idx in elite_indices]
            new_seen: Set[GeneSignature] = set()
            if config.ga.deduplicate_population:
                for gene in new_population:
                    new_seen.add(gene_signature(gene))

            max_attempts = config.ga.population_size * 10
            while len(new_population) < config.ga.population_size:
                attempts = 0
                while True:
                    parent_a = tournament_selection(population, fitness_scores, config)
                    parent_b = tournament_selection(population, fitness_scores, config)
                    if random.random() < config.ga.crossover_rate:
                        child = crossover(parent_a, parent_b)
                    else:
                        child = parent_a
                    if random.random() < config.ga.mutation_rate:
                        child = mutate(child, config)
                    if not config.ga.deduplicate_population:
                        break
                    signature = gene_signature(child)
                    if signature not in new_seen or attempts >= max_attempts:
                        new_seen.add(signature)
                        break
                    attempts += 1
                new_population.append(child)
            population = new_population
            print(
                f"[GA] ---- Completed generation {generation + 1}/{config.ga.generations}; best fitness so far {best_fitness:.6f} ----"
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return best_gene, best_fitness



# | 阶段 | 代码 | 作用 | 关键点 |
# |------|------|------|--------|
# | **初始化** | `initialize_population()` | 创建随机种群 | 多样性 |
# | **评估** | `evaluator(gene)` | 计算适应度 | 最耗时 |
# | **记录最优** | `if score > best_fitness` | 全局追踪 | 深拷贝 |
# | **精英保留** | `sorted(...)[: elite_count]` | 防止退化 | 直接晋级 |
# | **选择** | `tournament_selection()` | 挑选父代 | 平衡性能与多样性 |
# | **交叉** | `crossover()` | 组合优势 | 单点交叉 |
# | **变异** | `mutate()` | 探索新架构 | 4 种操作 |
# | **更新** | `population = new_population` | 代际更新 | 完全替换 |