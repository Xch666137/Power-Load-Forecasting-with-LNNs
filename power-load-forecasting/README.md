# 电力负荷时间序列预测系统 - 基于液态神经网络

## 项目简介

本项目旨在开发一个基于液态神经网络（Liquid Neural Networks）的电力负荷时间序列预测系统，专注于变压器终端的短期日内负荷预测。液态神经网络是一类能够处理连续时间信号的动态神经网络模型，在处理具有复杂时间依赖性的电力负荷数据方面具有优势。

## 技术栈

- Python 3.8+
- PyTorch
- torchdiffeq
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- PyYAML

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
│   │   ├── data_interface.py
│   │   ├── data_loader.py
│   │   ├── data_factory.py
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

## 项目工作流程

本项目的工作流程包括以下几个主要步骤：

### 1. 数据加载
系统支持多种数据集格式，包括ETT小时级数据集(ETTh1, ETTh2)、ETT分钟级数据集(ETTm1, ETTm2)以及自定义数据集。数据加载模块通过工厂模式创建对应的数据加载器实例。

### 2. 数据预处理
数据预处理模块负责特征工程，包括：
- 时间特征提取（年、月、日、小时、星期等）
- 滞后特征构建（1, 2, 3, 24, 168小时前的负荷值）
- 滚动窗口统计特征（均值、标准差等）
- 序列数据构建（将时间序列转换为监督学习问题）

### 3. 模型构建
项目实现了两种液态神经网络模型：
- **Liquid ODE模型**：基于常微分方程的液态神经网络，能够处理连续时间信号
- **Liquid LSTM模型**：结合LSTM和液态神经网络思想的混合模型

### 4. 模型训练
训练模块支持以下功能：
- GPU加速训练（如果可用）
- 验证集监控训练过程
- 可视化训练历史（损失曲线）

### 5. 模型评估
评估模块提供多种评估指标：
- MSE（均方误差）
- RMSE（均方根误差）
- MAE（平均绝对误差）
- MAPE（平均绝对百分比误差）
- R²（决定系数）

### 6. 结果可视化
可视化模块提供以下图表：
- 训练历史图表
- 预测结果对比图
- 预测误差分布图

## 配置文件说明

项目使用YAML格式的配置文件来管理各种参数，包括数据配置、模型配置、训练配置和评估配置。用户可以通过修改[configs/model_config.yaml](configs/model_config.yaml)文件来调整模型参数和训练设置。

主要配置项包括：
- 数据集类型和路径
- 序列长度和预测步长
- 模型类型和超参数
- 训练参数（学习率、批次大小、轮数等）

## 使用方法

```bash
python src/main.py
```

程序运行时会按照以下流程执行：
1. 加载配置文件
2. 加载并预处理数据
3. 创建模型
4. 训练模型
5. 评估模型性能
6. 生成可视化结果

## 液态神经网络简介

液态神经网络（Liquid Neural Networks）是一种受生物神经网络启发的新一代机器学习模型，具有以下特点：
- **时间连续性**：能够处理连续时间信号
- **动态行为**：可以适应输入信号的变化模式
- **记忆能力**：具有内置的时间记忆机制
- **高效性**：相比传统RNN在某些任务上更加高效

这些特性使得液态神经网络特别适合处理电力负荷这类具有复杂时间依赖性的时序数据。