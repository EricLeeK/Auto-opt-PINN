"""
测试新旧KAN实现在真实gene配置上的100 epoch速度对比

使用search_results.json中的gene配置：
- Attention(48, 2) -> DNN(32) -> KAN(24, 5, 2) -> Attention(32, 4) -> Output

对比老KAN vs 新KAN的训练速度
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn

# 添加src目录到path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from auto_pinn.gene import LayerGene, LayerType, Gene
from auto_pinn.config import DomainConfig, TrainingConfig
from auto_pinn.data import generate_training_batch


# ============================================================================
# 导入新旧两个版本的KAN + Attention层
# ============================================================================
# 从auto_pinn.pinn导入新版本
from auto_pinn.pinn import KANLayer as KANLayerNew, AttentionLayer

# 老版本KAN（带循环）
import math
from torch.nn import functional as F

class KANLayerOld(nn.Module):
    """原始KAN实现，forward pass中有per-feature循环"""

    def __init__(self, in_features: int, width: int, grid_points: int, spline_order: int) -> None:
        super().__init__()
        if grid_points < 2:
            raise ValueError("KANLayer requires at least two grid_points")
        if spline_order < 1:
            raise ValueError("KANLayer spline_order must be >= 1")

        self.in_features = in_features
        self.width = width
        self.degree = int(spline_order)
        self.num_ctrl_pts = int(grid_points)

        knot_vector = self._create_knot_vector(self.num_ctrl_pts, self.degree)
        self.register_buffer("knots", knot_vector, persistent=False)

        self.input_shift = nn.Parameter(torch.zeros(in_features))
        self.input_log_scale = nn.Parameter(torch.zeros(in_features))

        coeff_init = math.sqrt(2.0 / (self.num_ctrl_pts * max(1, in_features)))
        self.spline_coeffs = nn.Parameter(
            coeff_init * torch.randn(in_features, self.num_ctrl_pts, width)
        )
        self.bias = nn.Parameter(torch.zeros(width))

        self.skip_linear = nn.Linear(in_features, width)
        self.layer_norm = nn.LayerNorm(width)
        self.activation = nn.GELU()

    @staticmethod
    def _create_knot_vector(num_ctrl_pts: int, degree: int) -> torch.Tensor:
        knot_count = num_ctrl_pts + degree + 1
        knots = torch.zeros(knot_count, dtype=torch.float32)
        knots[-(degree + 1) :] = 1.0
        interior = knot_count - 2 * (degree + 1)
        if interior > 0:
            interior_points = torch.linspace(0.0, 1.0, interior + 2, dtype=torch.float32)[1:-1]
            knots[degree + 1 : degree + 1 + interior] = interior_points
        return knots

    def _bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """原始实现，有Python循环"""
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        degree = self.degree
        n_basis = knots.shape[0] - 1

        x = x.unsqueeze(-1)
        basis_segments = []
        for i in range(n_basis):
            left, right = knots[i], knots[i + 1]
            if i == n_basis - 1:
                cond = (x >= left) & (x <= right)
            else:
                cond = (x >= left) & (x < right)
            basis_segments.append(cond.to(x.dtype))
        basis = torch.cat(basis_segments, dim=-1)

        if degree == 0:
            return basis

        for d in range(1, degree + 1):
            new_basis = []
            upper = n_basis - d
            for i in range(upper):
                denom1 = knots[i + d] - knots[i]
                denom2 = knots[i + d + 1] - knots[i + 1]
                term1 = torch.zeros_like(x)
                term2 = torch.zeros_like(x)
                if denom1 > 0:
                    term1 = ((x - knots[i]) / denom1) * basis[:, i : i + 1]
                if denom2 > 0:
                    term2 = ((knots[i + d + 1] - x) / denom2) * basis[:, i + 1 : i + 2]
                new_basis.append(term1 + term2)
            basis = torch.cat(new_basis, dim=-1)

        return basis

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.input_log_scale) + 1e-3
        scaled = (inputs - self.input_shift) * scale
        normalised = torch.sigmoid(scaled)

        # 老版本：循环处理每个特征
        basis_values = []
        for feat in range(self.in_features):
            basis_feat = self._bspline_basis(normalised[:, feat])
            basis_values.append(basis_feat)
        basis_tensor = torch.stack(basis_values, dim=1)

        spline_mix = torch.einsum("bfg,fgw->bw", basis_tensor, self.spline_coeffs)
        linear_skip = self.skip_linear(inputs)
        combined = spline_mix + linear_skip + self.bias
        combined = self.layer_norm(combined)
        return self.activation(combined)


# ============================================================================
# 构建混合PINN模型
# ============================================================================
def build_layer(gene: LayerGene, in_features: int, kan_class) -> Tuple[nn.Module, int]:
    """根据gene构建层，kan_class可以是KANLayerOld或KANLayerNew"""
    if gene.layer_type == LayerType.DNN:
        units = gene.params["units"]
        block = nn.Sequential(nn.Linear(in_features, units), nn.Tanh())
        return block, units
    elif gene.layer_type == LayerType.KAN:
        width = gene.params["width"]
        grid = gene.params["grid_points"]
        order = gene.params["spline_order"]
        return kan_class(in_features, width, grid, order), width
    else:  # Attention
        embed = gene.params["embed_dim"]
        heads = gene.params["heads"]
        return AttentionLayer(in_features, embed, heads), embed


class HybridPINN(nn.Module):
    """使用指定KAN类构建的混合PINN"""

    def __init__(self, genes: Gene, input_dim: int, kan_class):
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        for layer_gene in genes:
            block, current_dim = build_layer(layer_gene, current_dim, kan_class)
            layers.append(block)
        layers.append(nn.Linear(current_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


# ============================================================================
# Allen-Cahn 方程的损失计算
# ============================================================================
def compute_pde_loss(model, collocation, nu=1e-3):
    """计算Allen-Cahn PDE残差损失"""
    xt = collocation.clone().requires_grad_(True)
    u = model(xt)
    
    # 计算一阶导数
    grads = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    
    # 计算二阶导数
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    
    # Allen-Cahn PDE: u_t - nu*u_xx + 5*(u^3 - u) = 0
    residual = u_t - nu * u_xx + 5.0 * (u.pow(3) - u)
    
    return (residual ** 2).mean()


def compute_boundary_loss(model, boundary_inputs, boundary_targets):
    """计算边界条件损失"""
    u_pred = model(boundary_inputs)
    return ((u_pred - boundary_targets) ** 2).mean()


def compute_initial_loss(model, initial_inputs, initial_targets):
    """计算初始条件损失"""
    u_pred = model(initial_inputs)
    return ((u_pred - initial_targets) ** 2).mean()


def compute_total_loss(model, batch, nu=1e-3):
    """计算总损失"""
    pde_loss = compute_pde_loss(model, batch.collocation, nu)
    bc_loss = compute_boundary_loss(model, batch.boundary_inputs, batch.boundary_targets)
    ic_loss = compute_initial_loss(model, batch.initial_inputs, batch.initial_targets)
    
    total_loss = pde_loss + bc_loss + ic_loss
    return total_loss, pde_loss, bc_loss, ic_loss


# ============================================================================
# 训练和计时
# ============================================================================
def train_and_time(model, optimizer, epochs, batch, domain_config, device, warmup=5):
    """训练指定轮数并计时"""
    model.train()
    nu = domain_config.diffusion
    
    # Warmup
    print("  Warmup中...")
    for _ in range(warmup):
        loss, _, _, _ = compute_total_loss(model, batch, nu)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 正式计时
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    print(f"  开始训练 {epochs} 轮...")
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        loss, pde_loss, bc_loss, ic_loss = compute_total_loss(model, batch, nu)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss={loss.item():.6e}, PDE={pde_loss.item():.6e}, BC={bc_loss.item():.6e}, IC={ic_loss.item():.6e}")
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start_time
    
    return elapsed, loss.item(), pde_loss.item(), bc_loss.item(), ic_loss.item()


# ============================================================================
# 主程序
# ============================================================================
def main():
    print("=" * 80)
    print(" 新旧KAN实现在真实Gene配置上的100 Epoch速度对比测试")
    print("=" * 80)
    
    # 读取gene配置
    gene_file = Path(__file__).parent / "search_results.json"
    print(f"\n读取gene配置: {gene_file}")
    with open(gene_file, 'r') as f:
        gene_data = json.load(f)
    
    # 构建gene对象
    genes: Gene = []
    for layer_dict in gene_data["gene"]:
        layer_gene = LayerGene(
            layer_type=LayerType(layer_dict["layer_type"]),
            params=layer_dict["params"]
        )
        genes.append(layer_gene)
    
    print("\nGene架构:")
    for i, layer in enumerate(genes):
        print(f"  Layer {i+1}: {layer.layer_type.value} - {layer.params}")
    
    # 配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 100
    WARMUP = 5
    LR = 1e-3
    INPUT_DIM = 2
    
    print(f"\n训练配置:")
    print(f"  设备:           {DEVICE}")
    print(f"  训练轮数:       {EPOCHS}")
    print(f"  Warmup轮数:     {WARMUP}")
    print(f"  学习率:         {LR}")
    print(f"  输入维度:       {INPUT_DIM}")
    
    device = torch.device(DEVICE)
    dtype = torch.float32
    
    # 创建配置
    domain_config = DomainConfig()
    training_config = TrainingConfig(
        collocation_points=512,
        boundary_points=128,
        initial_points=128,
        device=DEVICE
    )
    
    print(f"\n采样配置:")
    print(f"  Collocation点:  {training_config.collocation_points}")
    print(f"  Boundary点:     {training_config.boundary_points}")
    print(f"  Initial点:      {training_config.initial_points}")
    
    # 生成固定的训练数据
    print("\n生成训练数据...")
    batch = generate_training_batch(domain_config, training_config, device, dtype)
    print(f"  Collocation shape: {batch.collocation.shape}")
    print(f"  Boundary shape:    {batch.boundary_inputs.shape}")
    print(f"  Initial shape:     {batch.initial_inputs.shape}")
    
    # ========== 测试老版本 KAN ==========
    print("\n" + "=" * 80)
    print(" [1/2] 测试老版本 KAN (带循环)")
    print("=" * 80)
    
    model_old = HybridPINN(genes, INPUT_DIM, KANLayerOld).to(device)
    optimizer_old = torch.optim.Adam(model_old.parameters(), lr=LR)
    
    # 计算参数量
    total_params_old = sum(p.numel() for p in model_old.parameters())
    print(f"  模型参数量:     {total_params_old:,}")
    
    time_old, loss_old, pde_old, bc_old, ic_old = train_and_time(
        model_old, optimizer_old, EPOCHS, batch, domain_config, device, WARMUP
    )
    
    avg_time_old = time_old / EPOCHS
    print(f"\n  总时间:         {time_old:.4f}s")
    print(f"  平均每轮:       {avg_time_old*1000:.2f}ms")
    print(f"  最终损失:       Total={loss_old:.6e}, PDE={pde_old:.6e}, BC={bc_old:.6e}, IC={ic_old:.6e}")
    
    # ========== 测试新版本 KAN ==========
    print("\n" + "=" * 80)
    print(" [2/2] 测试新版本 KAN (完全向量化)")
    print("=" * 80)
    
    model_new = HybridPINN(genes, INPUT_DIM, KANLayerNew).to(device)
    optimizer_new = torch.optim.Adam(model_new.parameters(), lr=LR)
    
    # 计算参数量
    total_params_new = sum(p.numel() for p in model_new.parameters())
    print(f"  模型参数量:     {total_params_new:,}")
    
    time_new, loss_new, pde_new, bc_new, ic_new = train_and_time(
        model_new, optimizer_new, EPOCHS, batch, domain_config, device, WARMUP
    )
    
    avg_time_new = time_new / EPOCHS
    print(f"\n  总时间:         {time_new:.4f}s")
    print(f"  平均每轮:       {avg_time_new*1000:.2f}ms")
    print(f"  最终损失:       Total={loss_new:.6e}, PDE={pde_new:.6e}, BC={bc_new:.6e}, IC={ic_new:.6e}")
    
    # ========== 对比总结 ==========
    speedup = time_old / time_new
    time_saved = time_old - time_new
    
    print("\n" + "=" * 80)
    print(" 对比总结")
    print("=" * 80)
    print(f"  老版本总时间:   {time_old:.4f}s  ({avg_time_old*1000:.2f}ms/epoch)")
    print(f"  新版本总时间:   {time_new:.4f}s  ({avg_time_new*1000:.2f}ms/epoch)")
    print(f"  加速比:         {speedup:.2f}x")
    print(f"  节省时间:       {time_saved:.4f}s")
    
    print(f"\n  外推到不同训练规模:")
    for scale in [500, 1000, 3000, 5000]:
        old_time = time_old * scale / EPOCHS
        new_time = time_new * scale / EPOCHS
        saved_time = old_time - new_time
        print(f"    {scale:5d} epochs: 老版本={old_time/60:6.2f}分钟, 新版本={new_time/60:6.2f}分钟, 节省={saved_time/60:6.2f}分钟")
    
    print("=" * 80)
    
    # 内存使用情况
    if DEVICE == 'cuda':
        print(f"\nGPU内存使用:")
        print(f"  已分配:         {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"  峰值:           {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    
    print("\n测试完成！✓")


if __name__ == "__main__":
    main()
