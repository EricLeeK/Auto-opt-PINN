"""
Kaggle Notebook: Compare Three Baseline Architectures (DNN, KAN, Attention)
训练三个基线模型，每个 25000 轮，参数量匹配最佳 gene

使用说明：
1. 上传整个 Auto_PINN 项目到 Kaggle Dataset
2. 在 Kaggle Notebook 中逐步运行以下代码块
3. 建议使用 GPU T4 x2 加速器
"""

# ============================================================================
# Cell 1: 环境准备和数据加载
# ============================================================================
import os
import sys
import json
from pathlib import Path

# 挂载 Kaggle 数据集（假设你上传的数据集名为 auto-pinn）
# 修改为你的实际数据集路径
DATASET_PATH = "/kaggle/input/auto-pinn"  # 修改这里！
WORKING_PATH = "/kaggle/working/Auto_PINN"

# 复制项目到 working 目录（可读写）
!cp -r {DATASET_PATH} {WORKING_PATH}
!ls -la {WORKING_PATH}

# 添加到 Python 路径
sys.path.insert(0, f"{WORKING_PATH}/src")

# 验证导入
try:
    from auto_pinn.pinn import HybridPINN
    from auto_pinn.config import ProjectConfig
    print("✅ Successfully imported auto_pinn modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Python path: {sys.path}")


# ============================================================================
# Cell 2: 准备配置文件 - DNN Baseline
# ============================================================================
dnn_config = {
    "targets": ["dnn"],
    "skip_reference": True,
    "epochs": 25000,
    "log_every": 500,
    "device": "cuda",
    "tolerance": 0.10,
    "allow_mismatch": True,
    "evaluate": True,
    "plot": True,
    "show": False,
    
    "dnn": {
        "depth": 4,
        "widths": [64, 48, 32, 24],  # 根据参数量自动调整
        "base_width": 50
    },
    
    "output_dir": "/kaggle/working/comparison_runs/dnn_baseline"
}

# 保存配置
config_path_dnn = "/kaggle/working/config_dnn.json"
with open(config_path_dnn, "w") as f:
    json.dump(dnn_config, f, indent=2)

print(f"✅ DNN config saved to {config_path_dnn}")
print(json.dumps(dnn_config, indent=2))


# ============================================================================
# Cell 3: 准备配置文件 - KAN Baseline
# ============================================================================
kan_config = {
    "targets": ["kan"],
    "skip_reference": True,
    "epochs": 25000,
    "log_every": 500,
    "device": "cuda",
    "tolerance": 0.10,
    "allow_mismatch": True,
    "evaluate": True,
    "plot": True,
    "show": False,
    
    "kan": {
        "depth": 3,
        "widths": [32, 24, 16],  # 根据参数量自动调整
        "base_width": 24,
        "grids": [5, 5, 5],
        "base_grid": 5,
        "orders": [2, 2, 2],
        "base_order": 2
    },
    
    "output_dir": "/kaggle/working/comparison_runs/kan_baseline"
}

# 保存配置
config_path_kan = "/kaggle/working/config_kan.json"
with open(config_path_kan, "w") as f:
    json.dump(kan_config, f, indent=2)

print(f"✅ KAN config saved to {config_path_kan}")
print(json.dumps(kan_config, indent=2))


# ============================================================================
# Cell 4: 准备配置文件 - Attention Baseline
# ============================================================================
attention_config = {
    "targets": ["attention"],
    "skip_reference": True,
    "epochs": 25000,
    "log_every": 500,
    "device": "cuda",
    "tolerance": 0.10,
    "allow_mismatch": True,
    "evaluate": True,
    "plot": True,
    "show": False,
    
    "attention": {
        "depth": 3,
        "embeds": [48, 32, 24],  # 根据参数量自动调整
        "base_embed": 40,
        "heads": [2, 2, 2],
        "base_heads": 2
    },
    
    "output_dir": "/kaggle/working/comparison_runs/attention_baseline"
}

# 保存配置
config_path_attention = "/kaggle/working/config_attention.json"
with open(config_path_attention, "w") as f:
    json.dump(attention_config, f, indent=2)

print(f"✅ Attention config saved to {config_path_attention}")
print(json.dumps(attention_config, indent=2))


# ============================================================================
# Cell 5: 运行 DNN Baseline (约需 2-3 小时)
# ============================================================================
print("=" * 70)
print("开始训练 DNN Baseline - 25000 epochs")
print("=" * 70)

!cd {WORKING_PATH} && python src/compare_architectures.py \
    --reference-results search_results.json \
    --mat src/Allen_Cahn.mat \
    --local-config {config_path_dnn}

print("\n✅ DNN Baseline training completed!")
print(f"Results saved to: /kaggle/working/comparison_runs/dnn_baseline")


# ============================================================================
# Cell 6: 运行 KAN Baseline (约需 3-4 小时，KAN 较慢)
# ============================================================================
print("=" * 70)
print("开始训练 KAN Baseline - 25000 epochs")
print("=" * 70)

!cd {WORKING_PATH} && python src/compare_architectures.py \
    --reference-results search_results.json \
    --mat src/Allen_Cahn.mat \
    --local-config {config_path_kan}

print("\n✅ KAN Baseline training completed!")
print(f"Results saved to: /kaggle/working/comparison_runs/kan_baseline")


# ============================================================================
# Cell 7: 运行 Attention Baseline (约需 2-3 小时)
# ============================================================================
print("=" * 70)
print("开始训练 Attention Baseline - 25000 epochs")
print("=" * 70)

!cd {WORKING_PATH} && python src/compare_architectures.py \
    --reference-results search_results.json \
    --mat src/Allen_Cahn.mat \
    --local-config {config_path_attention}

print("\n✅ Attention Baseline training completed!")
print(f"Results saved to: /kaggle/working/comparison_runs/attention_baseline")


# ============================================================================
# Cell 8: 查看训练结果
# ============================================================================
import json
import pandas as pd

def load_summary(baseline_name):
    """加载某个基线的训练结果"""
    summary_path = f"/kaggle/working/comparison_runs/{baseline_name}_baseline/{baseline_name}/summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    else:
        print(f"⚠️  Summary not found: {summary_path}")
        return None

# 加载所有结果
results = {}
for baseline in ["dnn", "kan", "attention"]:
    print(f"\n{'='*70}")
    print(f"{baseline.upper()} Baseline Results:")
    print('='*70)
    
    summary = load_summary(baseline)
    if summary:
        results[baseline] = summary
        print(json.dumps(summary, indent=2))
    
    # 检查是否有训练历史
    history_path = f"/kaggle/working/comparison_runs/{baseline}_baseline/{baseline}/training_history.csv"
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        print(f"\n📊 Training History Preview (last 10 epochs):")
        print(df.tail(10))
        print(f"\n📈 Final Loss: {df['total'].iloc[-1]:.6e}")
        print(f"📉 Best Loss: {df['total'].min():.6e}")

# 创建对比表格
if results:
    comparison_df = pd.DataFrame({
        name: {
            "Parameters": data.get("parameters", "N/A"),
            "Best Loss": data.get("best_loss", "N/A"),
            "Fitness": data.get("fitness", "N/A"),
            "Relative L2 Error": data.get("relative_l2_error", "N/A")
        }
        for name, data in results.items()
    }).T
    
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70)
    print(comparison_df.to_string())


# ============================================================================
# Cell 9: 可视化训练曲线
# ============================================================================
import matplotlib.pyplot as plt
import pandas as pd

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, baseline in enumerate(["dnn", "kan", "attention"]):
    history_path = f"/kaggle/working/comparison_runs/{baseline}_baseline/{baseline}/training_history.csv"
    
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        ax = axes[idx]
        
        # 绘制总损失
        ax.plot(df.index, df['total'], label='Total Loss', alpha=0.8)
        ax.plot(df.index, df['pde'], label='PDE Loss', alpha=0.6)
        ax.plot(df.index, df['boundary'], label='Boundary Loss', alpha=0.6)
        ax.plot(df.index, df['initial'], label='Initial Loss', alpha=0.6)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{baseline.upper()} Training History')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Training curves saved to: /kaggle/working/training_comparison.png")
plt.show()


# ============================================================================
# Cell 10: 检查输出文件
# ============================================================================
print("📁 Output Directory Structure:")
print("="*70)
!tree -L 3 /kaggle/working/comparison_runs 2>/dev/null || find /kaggle/working/comparison_runs -type f -name "*.pt" -o -name "*.json" -o -name "*.csv" -o -name "*.png"

print("\n📦 Files to download:")
print("  1. /kaggle/working/comparison_runs/dnn_baseline/")
print("  2. /kaggle/working/comparison_runs/kan_baseline/")
print("  3. /kaggle/working/comparison_runs/attention_baseline/")
print("  4. /kaggle/working/training_comparison.png")


# ============================================================================
# Cell 11: 打包结果（可选）
# ============================================================================
print("📦 Creating archive for download...")
!cd /kaggle/working && tar -czf comparison_baselines_results.tar.gz comparison_runs/
print("✅ Archive created: /kaggle/working/comparison_baselines_results.tar.gz")
!ls -lh /kaggle/working/comparison_baselines_results.tar.gz
