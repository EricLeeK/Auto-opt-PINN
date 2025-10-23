# Auto-PINN 实现说明

本说明文档总结 `src/` 目录下 Auto-PINN 项目的核心代码逻辑与模块关系，帮助快速理解整体实现细节。

---

## 1. 项目结构概览

```
src/
├─ main.py                    # 命令行入口：启动遗传搜索流程
└─ auto_pinn/
   ├─ __init__.py             # 对外暴露的公共 API
   ├─ config.py               # 配置项与默认参数
   ├─ data.py                 # 训练数据采样（配点、边界、初值）
   ├─ gene.py                 # 基因与层类型定义
   ├─ genetic_algorithm.py    # 遗传算法核心逻辑
   ├─ pinn.py                 # 混合 PINN 模型及定制层实现
   └─ trainer.py              # PINN 训练循环与适应度评估
```

---

## 2. 主流程（`main.py`）

1. 加载默认配置 `DEFAULT_CONFIG`。
2. 实例化 `PINNFitnessEvaluator`，用于评估每个候选架构。
3. 调用 `run_genetic_search` 执行遗传算法，返回最佳基因与适应度。
4. 打印最佳层级结构与分数，并将结果写入 `search_results.json`。

该脚本可直接运行：

```powershell
python src/main.py
```

---

## 3. 配置体系（`config.py`）

- `DomainConfig`：定义 PDE 领域范围与粘性系数（针对 Burgers 方程）。
- `SearchSpace`：限制网络层数和各类型层的超参候选值。
- `GAConfig`：遗传算法相关超参（种群规模、进化代数、交叉/变异率、锦标赛规模等）。
- `TrainingConfig`：PINN 训练采样规模与训练超参（学习率、损失权重、设备等）。
- `RuntimeConfig`：运行时控制（随机种子、数据类型、日志频率）。
- `ProjectConfig`：将上述配置封装为整体配置对象；`DEFAULT_CONFIG` 给出默认值。

通过组合式 dataclass，让模块间共享统一的配置入口，并便于后续调参。

---

## 4. 基因编码（`gene.py`）

- `LayerType`：枚举三类候选层（`DNN`、`KAN`、`Attention`）。
- `LayerGene`：记录单层的层类型与参数字典。
- `Gene`：一组 `LayerGene` 的列表，表示一条候选网络架构。

每层参数:
- `DNN`：`units`
- `KAN`：`width`, `grid_points`, `spline_order`
- `Attention`：`embed_dim`, `heads`

---

## 5. 遗传算法（`genetic_algorithm.py`）

### 5.1 种群初始化
- `create_random_gene`：按 `SearchSpace` 随机生成层数与层配置。
- `initialize_population`：批量生成初始种群。

### 5.2 选择、交叉、变异
- `tournament_selection`：锦标赛选择，复制优胜个体。
- `crossover`：单点交叉，前半段来自父代 A，后半段来自父代 B。
- `mutate`：随机执行层类型替换、参数扰动、插入、删除（尊重层数上下限）。

### 5.3 主循环 `run_genetic_search`
1. 对每代所有个体调用适应度评估器，记录最佳基因。
2. 精英保留：直接复制若干最优个体到下一代。
3. 其余个体通过选择、交叉、变异生成，直到填满种群。
4. 返回搜索到的全局最优基因及适应度。

遗传算法通过 `FitnessEvaluator` 回调解耦架构构建逻辑与训练逻辑。

---

## 6. 数据采样（`data.py`）

针对 Burgers 方程提供三类数据：
- **配点**：`collocation`，在空间-时间区域内均匀采样，用于计算 PDE 残差。
- **边界点**：`boundary_inputs` + `boundary_targets`，边界条件保持为零。
- **初值点**：`initial_inputs` + `initial_targets`，初始条件为 `-sin(πx)`。

返回 `TrainingBatch`，包含所有张量（转换为目标设备与 dtype）。

---

## 7. 模型构建（`pinn.py`）

### 7.1 自定义层
- `KANLayer`：简化的 KAN 风格层，使用可学习中心与尺度的径向基展开，再线性混合并归一化。
- `AttentionLayer`：多头注意力风格层，针对向量输入实现 head-wise gating，并加入输入投影的残差连接。

### 7.2 `HybridPINN`
按基因顺序累积构建模块：
1. 遍历 `Gene`，调用 `_build_layer` 根据层类型实例化对应模块并更新通道数。
2. 最后补充线性输出层（输出标量 `u(x, t)`）。
3. 整体使用 `nn.Sequential` 封装，确保前向传播结果一致。

---

## 8. 训练评估（`trainer.py`）

### 8.1 `PINNFitnessEvaluator`
- 构造函数：保存配置、确定设备与 dtype。
- `__call__`：以防御方式运行 `_train_gene`，若训练失败返回 0 fitness。

### 8.2 `_train_gene`
1. 设置 Python、NumPy、PyTorch 随机种子，保证可复现。
2. 实例化 `HybridPINN`，迁移到目标设备与精度。
3. 使用 Adam 优化器与 MSE 损失。
4. 迭代 `epochs` 次：
   - 调用 `_sample_batch` 重新采样训练数据。
   - `_train_step` 完成一次反向传播与参数更新。
   - 追踪历史最小损失 `best_loss` 作为适应度依据。
5. Fitness = `1 / (best_loss + 1e-8)`，返回 `TrainingResult`。

### 8.3 `_train_step`
- 对配点启用自动微分，计算 `u_x`、`u_t`、`u_xx`。
- 构造 Burgers 方程残差：`u_t + u * u_x - ν * u_xx`。
- 分别计算 PDE / 边界 / 初值损失，并按照配置权重加权求和。
- 反向传播并执行优化器更新。

---

## 9. 配置调优建议

- **搜索空间**：可在 `SearchSpace` 中增减候选层类型、宽度等参数。
- **训练力度**：调整 `TrainingConfig` 中的点数与迭代次数以平衡速度和精度。
- **遗传超参**：`GAConfig` 的代数、种群规模决定搜索探索深度。
- **设备配置**：将 `TrainingConfig.device` 设置为 `cuda` 以利用 GPU 加速。

---

## 10. 运行结果输出

- 搜索完成后，最佳架构及 fitness 指标会打印在终端。
- 同步将结果保存为 JSON 文件，便于下游加载与复现。
- 可进一步扩展：例如将最佳基因再训练更长时间、绘制残差收敛曲线、导出模型权重等。

---

通过以上模块拆分，Auto-PINN 代码实现了“遗传算法 + 混合 PINN” 的自动架构搜索流程，实现了从配置管理、架构生成、数据采样到训练评估的完整闭环。欢迎在此基础上继续扩展更多 PDE、层类型或搜索策略。