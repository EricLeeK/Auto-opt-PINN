"""Compare Auto-PINN architectures against hand-crafted baselines with matched parameter counts."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch

from auto_pinn.config import EVALUATION_TRAINING_EPOCHS, ProjectConfig, default_evaluation_config
from auto_pinn.gene import Gene, LayerGene, LayerType
from auto_pinn.pinn import HybridPINN
from run_best_gene import (
    ensure_device,
    evaluate_model,
    load_gene,
    load_reference_solution,
    override_runtime,
    plot_comparison,
    train_gene,
)


MODEL_TITLES = {
    "reference": "Auto-PINN Reference",
    "dnn": "Pure DNN",
    "kan": "Pure KAN",
    "attention": "Pure Attention",
}

DEFAULT_REFERENCE_RESULTS = Path("search_results.json")
DEFAULT_MAT_PATH = Path(__file__).resolve().parent / "Allen_Cahn.mat"


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare Auto-PINN architectures with matched parameter counts.")
    parser.add_argument("--reference-results", type=Path, default=DEFAULT_REFERENCE_RESULTS, help="JSON file containing the reference (best) gene.")
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT_PATH, help="Reference Allen-Cahn solution used for evaluation.")
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_runs"), help="Directory where comparison artefacts are written.")
    parser.add_argument("--device", type=str, default=None, help="Override training device (defaults to config).")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EVALUATION_TRAINING_EPOCHS,
        help=f"Training epochs for this comparison run (defaults to {EVALUATION_TRAINING_EPOCHS}).",
    )
    parser.add_argument("--log-every", type=int, default=None, help="Override logging interval.")
    parser.add_argument("--tolerance", type=float, default=0.10, help="Maximum relative parameter mismatch allowed (fraction).")
    parser.add_argument("--disable-auto-match", action="store_true", help="Do not rescale baselines to match parameters; only report differences.")
    parser.add_argument("--allow-mismatch", action="store_true", help="Continue training even if parameter counts exceed tolerance.")
    parser.add_argument("--skip-reference", action="store_true", help="Skip retraining the reference Auto-PINN gene.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained models against the reference solution.")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots when --evaluate is enabled.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively when --plot is used.")

    parser.add_argument("--skip-dnn", action="store_true", help="Skip the pure DNN baseline.")
    parser.add_argument("--dnn-depth", type=int, default=4, help="Hidden layer count for the DNN baseline.")
    parser.add_argument("--dnn-widths", type=int, nargs="*", default=None, help="Hidden units per DNN layer (repeat or match depth).")
    parser.add_argument("--dnn-base-width", type=int, default=64, help="Fallback hidden units when --dnn-widths is not provided.")

    parser.add_argument("--skip-kan", action="store_true", help="Skip the KAN baseline.")
    parser.add_argument("--kan-depth", type=int, default=3, help="Number of KAN layers.")
    parser.add_argument("--kan-widths", type=int, nargs="*", default=None, help="KAN widths per layer.")
    parser.add_argument("--kan-base-width", type=int, default=16, help="Fallback width when --kan-widths is absent.")
    parser.add_argument("--kan-grids", type=int, nargs="*", default=None, help="Grid points per KAN layer.")
    parser.add_argument("--kan-base-grid", type=int, default=5, help="Fallback grid points when --kan-grids is absent.")
    parser.add_argument("--kan-orders", type=int, nargs="*", default=None, help="Spline order per KAN layer.")
    parser.add_argument("--kan-base-order", type=int, default=2, help="Fallback spline order when --kan-orders is absent.")

    parser.add_argument("--skip-attention", action="store_true", help="Skip the attention-only baseline.")
    parser.add_argument("--attn-depth", type=int, default=3, help="Number of attention blocks.")
    parser.add_argument("--attn-embeds", type=int, nargs="*", default=None, help="Embedding dimension per attention block.")
    parser.add_argument("--attn-base-embed", type=int, default=64, help="Fallback embedding dimension when --attn-embeds is absent.")
    parser.add_argument("--attn-heads", type=int, nargs="*", default=None, help="Attention heads per block.")
    parser.add_argument("--attn-base-heads", type=int, default=2, help="Fallback attention heads when --attn-heads is absent.")
    parser.add_argument("--local-config", type=Path, default=None, help="Path to a JSON file overriding comparison settings (targets, widths, epochs, etc.).")
    return parser.parse_args()


def _consume_local_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.local_config is None:
        return args
    config_path = Path(args.local_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Local config not found: {config_path}")
    payload = json.loads(config_path.read_text())
    print(f"[Compare] Loaded local config from {config_path}")

    def set_if_present(key: str, attr: str | None = None, cast=None) -> None:
        if key in payload:
            value = payload[key]
            if cast is not None:
                value = cast(value)
            setattr(args, attr or key, value)

    set_if_present("epochs", cast=int)
    set_if_present("device")
    set_if_present("reference_results", cast=Path)
    set_if_present("mat", cast=Path)
    set_if_present("output_dir", cast=Path)
    set_if_present("tolerance", cast=float)
    set_if_present("disable_auto_match", cast=_coerce_bool)
    set_if_present("allow_mismatch", cast=_coerce_bool)
    set_if_present("evaluate", cast=_coerce_bool)
    set_if_present("plot", cast=_coerce_bool)
    set_if_present("show", cast=_coerce_bool)
    set_if_present("skip_reference", cast=_coerce_bool)

    targets = payload.get("targets")
    if targets is not None:
        if isinstance(targets, str):
            targets = [targets]
        targets_set = {str(item).lower() for item in targets}
        for name in ("dnn", "kan", "attention"):
            setattr(args, f"skip_{name}", name not in targets_set)

    def update_block(name: str, schema: List[Tuple[str, str, str]]) -> None:
        block = payload.get(name)
        if not block:
            return
        for key, attr, kind in schema:
            if key in block:
                value = block[key]
                if kind == "int":
                    value = int(value)
                elif kind == "float":
                    value = float(value)
                elif kind == "bool":
                    value = _coerce_bool(value)
                elif kind == "list-int":
                    if isinstance(value, (list, tuple)):
                        value = [int(item) for item in value]
                    else:
                        value = [int(value)]
                setattr(args, attr, value)

    update_block(
        "dnn",
        [
            ("depth", "dnn_depth", "int"),
            ("widths", "dnn_widths", "list-int"),
            ("base_width", "dnn_base_width", "int"),
        ],
    )
    update_block(
        "kan",
        [
            ("depth", "kan_depth", "int"),
            ("widths", "kan_widths", "list-int"),
            ("base_width", "kan_base_width", "int"),
            ("grids", "kan_grids", "list-int"),
            ("base_grid", "kan_base_grid", "int"),
            ("orders", "kan_orders", "list-int"),
            ("base_order", "kan_base_order", "int"),
        ],
    )
    update_block(
        "attention",
        [
            ("depth", "attn_depth", "int"),
            ("embeds", "attn_embeds", "list-int"),
            ("base_embed", "attn_base_embed", "int"),
            ("heads", "attn_heads", "list-int"),
            ("base_heads", "attn_base_heads", "int"),
        ],
    )
    return args


def expand_values(values: Iterable[int] | None, depth: int, default: int) -> List[int]:
    if depth <= 0:
        raise ValueError("Layer depth must be positive")
    if values is None:
        return [int(default)] * depth
    items = [int(v) for v in values]
    if len(items) == 1 and depth > 1:
        return [items[0]] * depth
    if len(items) != depth:
        raise ValueError(f"Expected {depth} values, got {len(items)}")
    return items


def build_dnn_gene(widths: List[int]) -> Gene:
    return [LayerGene(layer_type=LayerType.DNN, params={"units": int(max(1, w))}) for w in widths]


def build_kan_gene(widths: List[int], grids: List[int], orders: List[int]) -> Gene:
    gene: Gene = []
    for w, g, o in zip(widths, grids, orders):
        grid = max(o + 1, g)
        gene.append(
            LayerGene(
                layer_type=LayerType.KAN,
                params={"width": int(max(1, w)), "grid_points": int(grid), "spline_order": int(max(1, o))},
            )
        )
    return gene


def build_attention_gene(embeds: List[int], heads: List[int]) -> Gene:
    gene: Gene = []
    for embed, head in zip(embeds, heads):
        head = max(1, head)
        if embed < head:
            embed = head
        if embed % head != 0:
            embed += head - (embed % head)
        gene.append(
            LayerGene(
                layer_type=LayerType.ATTENTION,
                params={"embed_dim": int(embed), "heads": int(head)},
            )
        )
    return gene


def parameter_count(gene: Gene) -> int:
    model = HybridPINN(gene)
    return sum(int(param.numel()) for param in model.parameters())


def auto_scale_search(
    target_params: int,
    tolerance: float,
    build_gene: Callable[[float], Gene],
    initial_scale: float = 1.0,
    max_scale: float = 512.0,
    min_scale: float = 1e-3,
) -> Tuple[Gene, int, float]:
    cache: Dict[float, Tuple[Gene, int, float]] = {}

    def evaluate(scale: float) -> Tuple[Gene, int, float]:
        if scale not in cache:
            gene = build_gene(scale)
            count = parameter_count(gene)
            diff = abs(count - target_params) / target_params
            cache[scale] = (gene, count, diff)
        return cache[scale]

    best_gene, best_count, best_diff = evaluate(initial_scale)
    if best_diff <= tolerance:
        return best_gene, best_count, best_diff

    scale_low = initial_scale
    scale_high = initial_scale
    while True:
        gene_low, count_low, diff_low = evaluate(scale_low)
        if count_low <= target_params or scale_low <= min_scale:
            break
        scale_low *= 0.5
        if scale_low < min_scale:
            scale_low = min_scale
            break
    best_candidate = min((evaluate(scale_low), evaluate(scale_high)), key=lambda item: item[2])
    best_gene, best_count, best_diff = best_candidate

    while True:
        gene_high, count_high, diff_high = evaluate(scale_high)
        if count_high >= target_params or scale_high >= max_scale:
            break
        scale_high *= 2.0
        if scale_high > max_scale:
            scale_high = max_scale
            break
    best_candidate = min((evaluate(scale_low), evaluate(scale_high)), key=lambda item: item[2])
    best_gene, best_count, best_diff = best_candidate

    for _ in range(48):
        mid = 0.5 * (scale_low + scale_high)
        gene_mid, count_mid, diff_mid = evaluate(mid)
        if diff_mid < best_diff:
            best_gene, best_count, best_diff = gene_mid, count_mid, diff_mid
        if diff_mid <= tolerance:
            return gene_mid, count_mid, diff_mid
        if count_mid < target_params:
            scale_low = mid
        else:
            scale_high = mid
        if abs(scale_high - scale_low) <= 1e-4:
            break
    return best_gene, best_count, best_diff


def serialise_gene(gene: Gene) -> List[Dict[str, Dict[str, int]]]:
    return [
        {"layer_type": layer.layer_type.value, "params": {key: int(value) for key, value in layer.params.items()}}
        for layer in gene
    ]


def prepare_dnn(args: argparse.Namespace, target_params: int, tolerance: float) -> Tuple[Gene, int, float]:
    widths = expand_values(args.dnn_widths, args.dnn_depth, args.dnn_base_width)

    def builder(scale: float) -> Gene:
        scaled = [max(1, int(round(width * scale))) for width in widths]
        return build_dnn_gene(scaled)

    gene = builder(1.0)
    count = parameter_count(gene)
    diff = abs(count - target_params) / target_params
    if args.disable_auto_match or diff <= tolerance:
        return gene, count, diff
    return auto_scale_search(target_params, tolerance, builder)


def prepare_kan(args: argparse.Namespace, target_params: int, tolerance: float) -> Tuple[Gene, int, float]:
    widths = expand_values(args.kan_widths, args.kan_depth, args.kan_base_width)
    grids = expand_values(args.kan_grids, args.kan_depth, args.kan_base_grid)
    orders = expand_values(args.kan_orders, args.kan_depth, args.kan_base_order)

    def builder(scale: float) -> Gene:
        scaled_widths = [max(1, int(round(width * scale))) for width in widths]
        return build_kan_gene(scaled_widths, grids, orders)

    gene = builder(1.0)
    count = parameter_count(gene)
    diff = abs(count - target_params) / target_params
    if args.disable_auto_match or diff <= tolerance:
        return gene, count, diff
    return auto_scale_search(target_params, tolerance, builder)


def prepare_attention(args: argparse.Namespace, target_params: int, tolerance: float) -> Tuple[Gene, int, float]:
    embeds = expand_values(args.attn_embeds, args.attn_depth, args.attn_base_embed)
    heads = expand_values(args.attn_heads, args.attn_depth, args.attn_base_heads)

    def builder(scale: float) -> Gene:
        scaled_embeds = []
        for embed, head in zip(embeds, heads):
            scaled = max(head, int(round(embed * scale)))
            if scaled % head != 0:
                scaled += head - (scaled % head)
            scaled_embeds.append(scaled)
        return build_attention_gene(scaled_embeds, heads)

    gene = builder(1.0)
    count = parameter_count(gene)
    diff = abs(count - target_params) / target_params
    if args.disable_auto_match or diff <= tolerance:
        return gene, count, diff
    return auto_scale_search(target_params, tolerance, builder)


def save_history(path: Path, history: List[Tuple[float, float, float, float]]) -> None:
    if not history:
        return
    array = np.asarray(history, dtype=np.float64)
    header = "total,pde,boundary,initial"
    np.savetxt(path, array, delimiter=",", header=header, comments="")


def train_and_record(
    label: str,
    gene: Gene,
    config: ProjectConfig,
    output_dir: Path,
    mat_path: Path,
    evaluate: bool,
    plot: bool,
    show: bool,
) -> Dict[str, float]:
    run_dir = output_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    runtime = replace(config.runtime, checkpoint_dir=str(checkpoints))
    run_config = replace(config, runtime=runtime)

    model, history, best_loss, best_state = train_gene(run_config, gene)
    metrics: Dict[str, float] = {
        "best_loss": float(best_loss),
        "fitness": float(1.0 / (best_loss + 1e-8)),
    }

    save_history(run_dir / "training_history.csv", history)

    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "gene": serialise_gene(gene),
                "config": run_config,
                "best_loss": best_loss,
                "fitness": metrics["fitness"],
            },
            run_dir / "best_model.pt",
        )

    if evaluate:
        x_ref, t_ref, u_ref = load_reference_solution(mat_path)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        prediction = evaluate_model(model, x_ref.astype(np.float64), t_ref.astype(np.float64), device, dtype)
        rel_error = float(np.linalg.norm(prediction - u_ref) / np.linalg.norm(u_ref))
        metrics["relative_l2_error"] = rel_error
        if plot:
            plot_comparison(
                x_ref,
                t_ref,
                u_ref,
                prediction,
                run_dir / "comparison.png",
                show=show,
                relative_error=rel_error,
            )
    return metrics


def main() -> None:
    args = parse_args()
    args = _consume_local_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_gene = load_gene(args.reference_results)
    target_params = parameter_count(reference_gene)
    print(f"[Compare] Reference parameters: {target_params}")

    base_config = ensure_device(default_evaluation_config(), args.device)
    base_config = override_runtime(base_config, args.epochs, args.log_every, seed=None)

    summary: Dict[str, Dict[str, float]] = {}
    gene_records: Dict[str, Gene] = {}

    if not args.skip_reference:
        summary["reference"] = {"parameters": float(target_params), "ratio": 1.0, "diff": 0.0}
        gene_records["reference"] = reference_gene

    if not args.skip_dnn:
        gene, count, diff = prepare_dnn(args, target_params, args.tolerance)
        summary["dnn"] = {"parameters": float(count), "ratio": count / target_params, "diff": diff}
        gene_records["dnn"] = gene

    if not args.skip_kan:
        gene, count, diff = prepare_kan(args, target_params, args.tolerance)
        summary["kan"] = {"parameters": float(count), "ratio": count / target_params, "diff": diff}
        gene_records["kan"] = gene

    if not args.skip_attention:
        gene, count, diff = prepare_attention(args, target_params, args.tolerance)
        summary["attention"] = {"parameters": float(count), "ratio": count / target_params, "diff": diff}
        gene_records["attention"] = gene

    for label, stats in summary.items():
        friendly = MODEL_TITLES.get(label, label)
        mismatch = stats["diff"] > args.tolerance
        print(
            f"[Compare] {friendly}: params={stats['parameters']:.0f} ratio={stats['ratio']:.3f} diff={stats['diff'] * 100:.2f}%"
        )
        if mismatch and not args.allow_mismatch:
            print(f"[Compare] {friendly} exceeds tolerance; skipping training. Use --allow-mismatch to force training.")

    trained_summary: Dict[str, Dict[str, float]] = {}
    for label, gene in gene_records.items():
        stats = summary[label]
        if stats["diff"] > args.tolerance and not args.allow_mismatch:
            continue
        seed_offset = sum(ord(ch) for ch in label) % 9973
        runtime_seed = base_config.runtime.seed + seed_offset
        runtime = replace(base_config.runtime, seed=runtime_seed)
        run_config = replace(base_config, runtime=runtime)
        friendly = MODEL_TITLES.get(label, label)
        print(f"[Compare] ===== Training {friendly} (seed {runtime_seed}) =====")
        metrics = train_and_record(
            label,
            gene,
            run_config,
            args.output_dir,
            args.mat,
            evaluate=args.evaluate,
            plot=args.plot,
            show=args.show,
        )
        trained_summary[label] = {**stats, **metrics}
        (args.output_dir / label / "gene.json").write_text(json.dumps(serialise_gene(gene), indent=2))

    (args.output_dir / "summary.json").write_text(json.dumps(trained_summary, indent=2))
    print(f"[Compare] Wrote summary to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
