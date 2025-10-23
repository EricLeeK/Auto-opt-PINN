# Auto-PINN: Automated Physics-Informed Neural Networks

🤖 使用遗传算法自动搜索最优的混合神经网络架构来求解偏微分方程（PDE）。

## 📖 项目简介

Auto-PINN 是一个自动化的 PINN 架构搜索框架，能够：
- 🔍 自动探索混合网络架构（DNN + KAN + Attention）
- 🧬 使用遗传算法优化网络结构
- 📊 在 Burgers 方程上验证性能

## 🚀 快速开始

### 安装依赖

```bash
pip install torch numpy
```

### 运行搜索

```bash
python main.py
```

## 📁 项目结构

```
Auto-PINN/
├── auto_pinn/          # 核心包
│   ├── config.py       # 配置管理
│   ├── gene.py         # 基因表示
│   ├── genetic_algorithm.py  # 遗传算法
│   ├── pinn.py         # 混合 PINN 模型
│   ├── trainer.py      # 训练和评估
│   └── data.py         # 数据采样
├── main.py             # 入口脚本
└── README.md           # 项目说明
```

## 🎯 核心特性

- **混合架构**: 支持 DNN、KAN、Attention 层的任意组合
- **智能搜索**: 遗传算法自动优化网络结构
- **可配置**: 所有超参数都可通过配置文件调整
- **模块化设计**: 清晰的代码结构，易于扩展

## 📊 示例结果

搜索结果将保存在 `search_results.json` 中。

## 🛠️ 配置说明

编辑 `auto_pinn/config.py` 来调整：
- 搜索空间（层类型、参数范围）
- 遗传算法参数（种群大小、代数）
- 训练超参数（学习率、轮次）

## 📝 许可证

MIT License

## 👤 作者

[你的名字]

---

⭐ 如果这个项目对你有帮助，请给它一个 Star！
