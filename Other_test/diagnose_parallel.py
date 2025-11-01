"""Utility script to sanity-check multiprocessing behaviour on the current system."""
# Please put the file back into the src folder before use.

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import platform
import sys
import time
from statistics import mean

import torch


def _probe_task(task_id: int, device_index: int, cpu_sleep: float, matmul_size: int) -> float:
    """Worker task that mixes CPU waiting with GPU matmul to expose scheduling issues."""

    pid = os.getpid()
    now = time.time()
    print(f"[Probe] pid={pid} task={task_id} start={now:.3f}s", flush=True)
    if cpu_sleep > 0:
        time.sleep(cpu_sleep)

    device: torch.device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device_index)
    else:
        device = torch.device("cpu")

    start = time.perf_counter()
    size = matmul_size
    a = torch.randn((size, size), device=device)
    b = torch.randn((size, size), device=device)
    for step in range(3):
        a = torch.matmul(a, b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time = time.time()
        print(
            f"[Probe] pid={pid} task={task_id} step={step} timestamp={step_time:.3f}s",
            flush=True,
        )
    duration = time.perf_counter() - start
    done = time.time()
    print(f"[Probe] pid={pid} task={task_id} done={done:.3f}s elapsed={duration:.3f}s", flush=True)
    return duration


def _heavy_pinn_task(task_id: int, device_index: int, epochs: int = 1000) -> float:
    """模拟重负载 PINN 训练任务，用于真实性能测试"""
    pid = os.getpid()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device_index)
        # 轻量预热，避免首次建立 CUDA 上下文的抖动/告警
        _ = torch.randn(8, 8, device=device) @ torch.randn(8, 8, device=device)
        torch.cuda.synchronize()
    else:
        device = torch.device("cpu")
    
    # 构建接近实际的深度神经网络
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    start = time.perf_counter()
    
    # 模拟指定数量的训练 epochs
    for epoch in range(epochs):
        # 模拟 CPU 数据准备阶段
        time.sleep(0.002)
        
        # GPU 密集计算：前向传播 + 梯度计算
        x = torch.randn(512, 2, device=device, requires_grad=True)
        u = model(x)
        
        # 计算一阶导数（PINN 特有）
        grad_u = torch.autograd.grad(
            outputs=u, 
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        # 计算二阶导数（PINN 特有）
        grad2_u = torch.autograd.grad(
            outputs=grad_u[:, 0].sum(),
            inputs=x,
            create_graph=True
        )[0]
        
        # 损失函数和反向传播
        loss = (u ** 2).mean() + (grad2_u ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # 定期打印进度
        if epoch % 50 == 0:
            print(f"[Heavy] pid={pid} task={task_id} epoch={epoch}/{epochs} loss={loss.item():.6f}", flush=True)
    
    duration = time.perf_counter() - start
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    print(f"[Heavy] pid={pid} task={task_id} completed in {duration:.3f}s", flush=True)
    return duration


def _log_environment_info(workers: int) -> None:
    print("=" * 72)
    print("Auto-PINN Parallel Diagnostics")
    print("=" * 72)
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Torch version  : {torch.__version__}")
    print(f"Platform       : {platform.platform()}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        print(f"CUDA device    : {props.name} (index {device})")
        print(f"Total memory   : {props.total_memory / 1024 ** 3:.2f} GiB")
    print(f"Requested workers: {workers}")
    print("Multiprocessing start methods available:")
    for method in mp.get_all_start_methods():
        print(f"  - {method}")
    try:
        current_method = mp.get_start_method(allow_none=True)
    except ValueError:
        current_method = None
    print(f"Current start method (before diagnostics): {current_method}")
    print("=" * 72)


def run_probe(workers: int, tasks: int, cpu_sleep: float, matmul_size: int) -> None:
    ctx = mp.get_context("spawn")
    _log_environment_info(workers)
    print("[Main] Launching diagnostic pool...")
    start = time.perf_counter()
    with ctx.Pool(processes=workers) as pool:
        payloads = [
            (task_id, 0, cpu_sleep, matmul_size)
            for task_id in range(tasks)
        ]
        print(
            f"[Main] Submitting {len(payloads)} tasks with spawn start method...",
            flush=True,
        )
        results = list(pool.starmap(_probe_task, payloads))
    total = time.perf_counter() - start
    print("=" * 72)
    print("Diagnostics summary")
    print(f"Tasks completed  : {len(results)}")
    print(f"Average task time: {mean(results):.3f}s")
    print(f"Total wall time  : {total:.3f}s")
    theoretical = mean(results) * (len(results) / max(1, workers))
    print(f"Ideal wall time (perfect overlap): {theoretical:.3f}s")
    print("If total wall time is close to ideal, workers are overlapping correctly.")
    print("Review timestamps above: overlapping start times indicate parallel execution.")
    print("=" * 72)


def benchmark_workers(max_workers: int = 8, tasks_per_worker: int = 2, epochs: int = 1000) -> None:
    """对比不同 Workers 数量的性能，找出最优配置
    
    关键指标说明:
    - 吞吐量加速比 = (workers=N 的任务数 / workers=N 的总耗时) / (workers=1 的任务数 / workers=1 的总耗时)
    - 简化后 = (N × tasks_per_worker / time_N) / (tasks_per_worker / time_1)
    - 再简化 = N × time_1 / time_N
    """
    print("\n" + "="*72)
    print("WORKERS SCALABILITY BENCHMARK (吞吐量测试)")
    print("="*72)
    print(f"测试范围: 1 到 {max_workers} workers")
    print(f"每个配置运行的任务数 = workers × {tasks_per_worker}")
    print(f"每个任务训练 {epochs} epochs")
    print(f"注意: Workers 越多，总任务数越多，这样才能测试吞吐量！")
    print("="*72 + "\n")
    
    results = {}  # {workers: (total_time, num_tasks, avg_task_time)}
    
    for workers in range(1, max_workers + 1):
        print(f"\n{'='*72}")
        print(f"Testing with {workers} workers...")
        print(f"{'='*72}")
        
        ctx = mp.get_context("spawn")
        num_tasks = workers * tasks_per_worker
        
        start = time.perf_counter()
        with ctx.Pool(processes=workers) as pool:
            task_durations = pool.starmap(
                _heavy_pinn_task,
                [(i, 0, epochs) for i in range(num_tasks)]
            )
        total_time = time.perf_counter() - start
        
        avg_task_time = mean(task_durations)
        
        # 计算吞吐量 (任务数/秒)
        throughput = num_tasks / total_time
        
        # 计算吞吐量加速比 (相对于 workers=1)
        if 1 in results:
            baseline_throughput = results[1][1] / results[1][0]  # 任务数 / 总时间
            throughput_speedup = throughput / baseline_throughput
            efficiency = throughput_speedup / workers * 100
        else:
            throughput_speedup = 1.0
            efficiency = 100.0
        
        results[workers] = (total_time, num_tasks, avg_task_time)
        
        print(f"\n{'='*72}")
        print(f"Workers={workers} 结果:")
        print(f"  完成任务数: {num_tasks} 个")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  吞吐量: {throughput:.3f} 任务/秒")
        print(f"  平均单任务时间: {avg_task_time:.2f}s")
        print(f"  吞吐量加速比: {throughput_speedup:.2f}x (相对于 workers=1)")
        print(f"  并行效率: {efficiency:.1f}%")
        print(f"{'='*72}")
    
    # 汇总表格
    print("\n" + "="*72)
    print("SUMMARY TABLE (吞吐量对比)")
    print("="*72)
    print(f"{'Workers':<10} {'任务数':<10} {'总耗时(s)':<12} {'吞吐量':<15} {'加速比':<10} {'并行效率':<12}")
    print("-"*72)
    
    baseline_time, baseline_tasks, _ = results[1]
    baseline_throughput = baseline_tasks / baseline_time
    
    for workers in sorted(results.keys()):
        total_time, num_tasks, _ = results[workers]
        throughput = num_tasks / total_time
        speedup = throughput / baseline_throughput
        efficiency = speedup / workers * 100
        
        print(f"{workers:<10} {num_tasks:<10} {total_time:<12.2f} {throughput:<15.3f} {speedup:<10.2f}x {efficiency:<12.1f}%")
    print("="*72)
    
    # 找出最优配置
    best_workers = max(results.keys(), key=lambda w: (results[w][1] / results[w][0]) / baseline_throughput)
    best_time, best_tasks, _ = results[best_workers]
    best_throughput = best_tasks / best_time
    best_speedup = best_throughput / baseline_throughput
    
    print(f"\n✅ 推荐配置: workers={best_workers}")
    print(f"   吞吐量加速比: {best_speedup:.2f}x")
    print(f"   并行效率: {best_speedup / best_workers * 100:.1f}%")
    print(f"   实际意义: 在遗传算法中，完成种群评估的速度提升 {best_speedup:.2f} 倍\n")
    
    # 性能建议
    print("="*72)
    print("性能建议:")
    print("="*72)
    
    if best_speedup < 1.2:
        print("⚠️  多进程收益很小（<20%），建议使用 workers=1")
        print("    原因: GPU 计算已饱和，或上下文切换开销过大")
        print("    建议: 专注于缓存和去重优化")
    elif best_workers <= 2:
        print(f"✅ workers={best_workers} 最优")
        print(f"    在遗传算法中可以实现 {best_speedup:.2f}x 的整体加速")
        print(f"    建议: 使用 config.runtime.workers={best_workers}")
    else:
        print(f"✅ workers={best_workers} 最优")
        print(f"    吞吐量提升 {best_speedup:.2f}x，并行效率 {best_speedup / best_workers * 100:.1f}%")
        print(f"    建议: 使用 config.runtime.workers={best_workers}")
    
    # 显示实际应用场景
    print(f"\n📊 实际应用示例 (假设种群大小=20):")
    print(f"   Workers=1: 完成一代评估需要 ~{baseline_time * 20 / baseline_tasks:.1f}秒")
    print(f"   Workers={best_workers}: 完成一代评估需要 ~{best_time * 20 / best_tasks:.1f}秒")
    savings = (baseline_time * 20 / baseline_tasks) - (best_time * 20 / best_tasks)
    print(f"   每代节省: ~{savings:.1f}秒")
    print(f"   15代节省: ~{savings * 15 / 60:.1f}分钟")
    
    # 警告信息
    if max_workers > best_workers:
        worst_workers = max_workers
        worst_time, worst_tasks, _ = results[worst_workers]
        worst_throughput = worst_tasks / worst_time
        worst_speedup = worst_throughput / baseline_throughput
        regression = (best_speedup - worst_speedup) / best_speedup * 100
        if regression > 10:
            print(f"\n⚠️  警告: workers={worst_workers} 比 workers={best_workers} 慢了 {regression:.1f}%")
            print(f"    原因: 过多进程导致调度开销超过并行收益")
    print("="*72 + "\n")


def benchmark_fixed_tasks(max_workers: int = 8, total_tasks: int = 20, epochs: int = 1000) -> None:
    """固定总任务数的基准测试：贴近 GA 一代评估真实墙钟时间。

    对于每个 workers 配置，都完成相同数量的任务（total_tasks），
    直接比较完成这些任务所需的总时间（墙钟时间）。
    """
    print("\n" + "="*72)
    print("FIXED-TOTAL-TASKS BENCHMARK (同任务量墙钟时间)")
    print("="*72)
    print(f"每个配置的总任务数: {total_tasks}")
    print(f"每个任务 epochs: {epochs}")
    print("="*72 + "\n")

    results = {}  # {workers: (total_time, avg_task_time)}
    baseline_time = None

    for workers in range(1, max_workers + 1):
        print(f"\n{'='*72}")
        print(f"Testing with {workers} workers...")
        print(f"{'='*72}")
        ctx = mp.get_context("spawn")
        start = time.perf_counter()
        with ctx.Pool(processes=workers) as pool:
            durations = pool.starmap(_heavy_pinn_task, [(i, 0, epochs) for i in range(total_tasks)])
        total_time = time.perf_counter() - start
        avg_task_time = mean(durations)
        results[workers] = (total_time, avg_task_time)
        if baseline_time is None:
            baseline_time = total_time
        speedup = baseline_time / total_time
        efficiency = speedup / workers * 100
        print(f"\nWorkers={workers} 结果:")
        print(f"  完成相同任务数: {total_tasks} 个")
        print(f"  总耗时(墙钟): {total_time:.2f}s")
        print(f"  平均单任务: {avg_task_time:.2f}s")
        print(f"  速度提升(相对W=1): {speedup:.2f}x")
        print(f"  并行效率: {efficiency:.1f}%")
        print(f"{'='*72}")

    print("\n" + "="*72)
    print("SUMMARY TABLE (固定任务数)")
    print("="*72)
    print(f"{'Workers':<10} {'总耗时(s)':<12} {'速度提升':<10} {'并行效率':<12}")
    print("-"*72)
    for w in sorted(results.keys()):
        total_time, _ = results[w]
        speedup = results[1][0] / total_time
        efficiency = speedup / w * 100
        print(f"{w:<10} {total_time:<12.2f} {speedup:<10.2f}x {efficiency:<12.1f}%")
    print("="*72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose multiprocessing behaviour for Auto-PINN.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes to launch.")
    parser.add_argument(
        "--tasks",
        type=int,
        default=4,
        help="Number of probe tasks to execute (>= workers to show overlap).",
    )
    parser.add_argument(
        "--cpu-sleep",
        type=float,
        default=0.25,
        help="Seconds of CPU-side sleep to inject before GPU work (simulates data prep).",
    )
    parser.add_argument(
        "--matmul-size",
        type=int,
        default=1024,
        help="Size of the square matrices used in each GPU matmul probe.",
    )
    parser.add_argument(
        "--benchmark-fixed",
        action="store_true",
        help="Run fixed-total-tasks benchmark (GA-like wall time).",
    )
    parser.add_argument(
        "--total-tasks",
        type=int,
        default=20,
        help="Total tasks to run in --benchmark-fixed mode (default: 20).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full workers scalability benchmark (1 to max-workers)."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum workers to test in benchmark mode (default: 8)."
    )
    parser.add_argument(
        "--tasks-per-worker",
        type=int,
        default=2,
        help="Number of tasks per worker in benchmark mode (default: 2)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs per task in benchmark (default: 1000)."
    )
    args = parser.parse_args()

    if args.benchmark_fixed:
        benchmark_fixed_tasks(
            max_workers=args.max_workers,
            total_tasks=args.total_tasks,
            epochs=args.epochs,
        )
    elif args.benchmark:
        # 运行完整基准测试
        benchmark_workers(
            max_workers=args.max_workers,
            tasks_per_worker=args.tasks_per_worker,
            epochs=args.epochs
        )
    else:
        # 运行简单诊断（轻量级探测）
        if args.workers < 1:
            parser.error("workers must be >= 1")
        if args.tasks < args.workers:
            parser.error("tasks must be >= workers to observe overlap")
        run_probe(args.workers, args.tasks, args.cpu_sleep, args.matmul_size)
