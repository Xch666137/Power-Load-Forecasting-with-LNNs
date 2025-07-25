# 电力负荷时间序列预测系统 - 基于液态神经网络

## 项目简介

本项目旨在开发一个基于液态神经网络（Liquid Neural Networks）的电力负荷时间序列预测系统，专注于变压器终端的短期日内负荷预测。液态神经网络是一类能够处理连续时间信号的动态神经网络模型，在处理具有复杂时间依赖性的电力负荷数据方面具有优势。

## 技术栈

- Python 3.8+
- PyTorch / TensorFlow
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

## 项目结构

```
power-load-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_data/
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── liquid_neural_network.py
│   │   ├── training.py
│   │   └── evaluation.py
│   ├── utils/
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── main.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── tests/
│   ├── test_data_loader.py
│   ├── test_models.py
│   └── test_utils.py
└── configs/
    └── model_config.yaml
```

## 安装指南

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python src/main.py
```

## 液态神经网络简介

液态神经网络（Liquid Neural Networks）是一种受生物神经网络启发的新一代机器学习模型，具有以下特点：
- 时间连续性：能够处理连续时间信号
- 动态行为：可以适应输入信号的变化模式
- 记忆能力：具有内置的时间记忆机制
- 高效性：相比传统RNN在某些任务上更加高效

## Git工作流

- 主分支：main（稳定版本）
- 开发分支：develop（开发版本）
- 功能分支：feature/*（新功能开发）
- 发布分支：release/*（版本发布）
- 修复分支：hotfix/*（紧急修复）