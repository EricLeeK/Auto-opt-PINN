"""
Kaggle 简化版：一次性运行三个基线模型
适合长时间运行（约 8-10 小时）
"""

# ============================================================================
# Step 1: 环境设置
# ============================================================================
import sys
import os

# 修改为你的 Kaggle 数据集路径
DATASET_PATH = "/kaggle/input/auto-pinn"  # ⚠️ 修改这里！
WORKING_PATH = "/kaggle/working/Auto_PINN"

# 复制项目
!cp -r {DATASET_PATH} {WORKING_PATH}
sys.path.insert(0, f"{WORKING_PATH}/src")

# 验证
from auto_pinn.pinn import HybridPINN
print("✅ Environment ready")


# ============================================================================
# Step 2: 上传配置文件并运行
# ============================================================================
# 方案 A: 使用项目中的配置文件（需要先上传 kaggle_config_all_baselines.json）
config_path = f"{WORKING_PATH}/kaggle_config_all_baselines.json"

# 方案 B: 直接在 Kaggle 创建配置
import json

config = {
    "targets": ["dnn", "kan", "attention"],
    "skip_reference": True,
    "epochs": 25000,
    "log_every": 500,
    "device": "cuda",
    "tolerance": 0.15,
    "allow_mismatch": True,
    "evaluate": True,
    "plot": True,
    "show": False,
    
    "dnn": {
        "depth": 4,
        "widths": [64, 48, 32, 24],
        "base_width": 50
    },
    
    "kan": {
        "depth": 3,
        "widths": [32, 24, 16],
        "base_width": 24,
        "grids": [5, 5, 5],
        "base_grid": 5,
        "orders": [2, 2, 2],
        "base_order": 2
    },
    
    "attention": {
        "depth": 3,
        "embeds": [48, 32, 24],
        "base_embed": 40,
        "heads": [2, 2, 2],
        "base_heads": 2
    },
    
    "output_dir": "/kaggle/working/comparison_runs"
}

config_path = "/kaggle/working/config_all.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"✅ Config saved: {config_path}")


# ============================================================================
# Step 3: 运行比较（⚠️ 这将运行很长时间！）
# ============================================================================
print("="*70)
print("开始训练三个基线模型 (DNN, KAN, Attention)")
print("预计时间: 8-10 小时")
print("="*70)

!cd {WORKING_PATH} && python src/compare_architectures.py \
    --reference-results search_results.json \
    --mat src/Allen_Cahn.mat \
    --local-config {config_path}

print("✅ All baselines training completed!")


# ============================================================================
# Step 4: 查看结果
# ============================================================================
!ls -la /kaggle/working/comparison_runs/

# 查看 summary
import json
summary_path = "/kaggle/working/comparison_runs/summary.json"
if os.path.exists(summary_path):
    with open(summary_path, "r") as f:
        summary = json.load(f)
    print("\n📊 Training Summary:")
    print(json.dumps(summary, indent=2))


# ============================================================================
# Step 5: 下载结果
# ============================================================================
!cd /kaggle/working && tar -czf baselines_results.tar.gz comparison_runs/
print("✅ Download: /kaggle/working/baselines_results.tar.gz")
