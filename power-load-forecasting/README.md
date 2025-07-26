"""
PyCharm SSH解释器训练脚本
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp_STSF import ExpSTSF
from experiments.exp_model_comparison import ExpModelComparison

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='PyCharm SSH解释器训练脚本')
    parser.add_argument('--experiment', type=str, default='STSF',
                        help='实验类型: STSF 或 ModelComparison')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--remote-config', type=str, default='configs/remote_config.yaml',
                        help='远程配置文件路径')
    parser.add_argument('--itr', type=int, default=1,
                        help='实验重复次数')
    
    args = parser.parse_args()

    try:
        if args.experiment == 'STSF':
            exp_class = ExpSTSF
        elif args.experiment == 'ModelComparison':
            exp_class = ExpModelComparison
        else:
            raise ValueError(f"未知的实验类型: {args.experiment}")
        
        # 初始化实验类
        exp = exp_class(args)
        
        # 运行实验
        logger.info(f"开始运行 {args.experiment} 实验")
        exp.run()
        logger.info(f"{args.experiment} 实验运行完成")
        
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
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
├── configs/
│   ├── model_config.yaml          # 模型配置文件
│   └── remote_config.yaml         # 远程训练配置文件
├── experiments/
│   ├── code/
│   │   ├── evaluate.py
│   │   ├── train.py
│   │   ├── version_control.py
│   │   └── visualize.py
│   ├── results/
│   │   └── README.md
│   ├── exp_STSF.py
│   ├── exp_base.py
│   └── exp_model_comparison.py
├── scripts/
│   ├── cli.py                     # 命令行接口
│   ├── run_experiment.py          # Python版一键执行脚本
│   ├── run_experiment.sh          # Shell版一键执行脚本
│   ├── remote_train.py            # Python版远程训练脚本
│   ├── remote_train.sh            # Shell版远程训练脚本
│   └── pycharm_train.py           # PyCharm SSH解释器训练脚本
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── attention_layer.py
│   │   ├── base_attention.py
│   │   ├── multihead_attention.py
│   │   └── prob_attention.py
│   ├── data_provider/
│   │   ├── __init__.py
│   │   ├── data_factory.py
│   │   ├── data_loader.py
│   │   └── data_interface.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── informer.py
│   │   ├── liquid_lstm.py
│   │   ├── liquid_neural_network.py
│   │   ├── lstm.py
│   │   └── transformer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── __init__.py
├── tests/
│   ├── test_data_loader.py
│   └── test_models.py
└── data/
    └── ETDataset/
        ├── README.md
        ├── README_CN.md
        └── ETT-small/
            ├── ETTh1.csv
            ├── ETTh2.csv
            ├── ETTm1.csv
            └── ETTm2.csv
```

## 安装指南

### 环境要求
- Python 3.8或更高版本
- pip包管理器
- Git（可选，用于版本控制）

### 安装步骤
1. 首先确保已安装Python 3.8+：
   ```bash
   python --version
   ```

2. 安装项目依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 验证安装：
   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import numpy; print(numpy.__version__)"
   ```

### 依赖包说明
安装过程会安装以下类型的依赖包：
- **核心科学计算库**：numpy, pandas, scipy
- **深度学习框架**：torch, torchdiffeq
- **数据可视化工具**：matplotlib, seaborn
- **机器学习工具**：scikit-learn
- **配置文件处理**：PyYAML
- **Jupyter Notebook支持**：jupyter
- **测试框架**：pytest
- **代码质量工具**：flake8, black

## 项目工作流程

本项目采用模块化实验设计，通过统一入口运行不同实验。主要流程包括以下几个步骤：

### 1. 实验设计
项目采用实验类的设计模式，每个预测任务对应一个实验类，便于管理和扩展。

### 2. 数据加载
系统支持多种数据集格式，包括ETT小时级数据集(ETTh1, ETTh2)、ETT分钟级数据集(ETTm1, ETTm2)以及自定义数据集。数据加载模块通过工厂模式创建对应的数据加载器实例。

数据加载由[data_provider](file:///F:/PyProgram/power-load-forecasting/src/data_provider)模块负责，该模块包含以下组件：

- [data_factory.py](file:///F:/PyProgram/power-load-forecasting/src/data_provider/data_factory.py)：数据集工厂，用于创建不同类型的ETT数据集实例
- [data_loader.py](file:///F:/PyProgram/power-load-forecasting/src/data_provider/data_loader.py)：数据加载器，提供PyTorch DataLoader接口
- [data_interface.py](file:///F:/PyProgram/power-load-forecasting/src/data_provider/data_interface.py)：数据接口定义，定义了统一的数据集接口

ETT数据集包含以下特征列：
- **date**：时间戳
- **HUFL**：High UseFul Load (高有用负荷)
- **HULL**：High UseLess Load (高无用负荷)
- **MUFL**：Middle UseFul Load (中有用负荷)
- **MULL**：Middle UseLess Load (中无用负荷)
- **LUFL**：Low UseFul Load (低有用负荷)
- **LULL**：Low UseLess Load (低无用负荷)
- **OT**：Oil Temperature (变压器油温，目标变量)

### 3. 数据预处理
数据预处理模块负责特征工程，包括：
- 时间特征提取（年、月、日、小时、星期等）
- 滞后特征构建（1, 2, 3, 24, 168小时前的负荷值）
- 滚动窗口统计特征（均值、标准差等）
- 序列数据构建（将时间序列转换为监督学习问题）

### 4. 模型构建
项目实现了多种时间序列预测模型：
- **Liquid ODE模型**：基于常微分方程的液态神经网络，能够处理连续时间信号
- **Liquid LSTM模型**：结合LSTM和液态神经网络思想的混合模型
- **传统模型**：包括LSTM、Transformer、Informer等用于对比实验

### 5. 模型训练
训练模块支持以下功能：
- GPU加速训练（如果可用）
- 验证集监控训练过程
- 可视化训练历史（损失曲线）

### 6. 模型评估
评估模块提供多种评估指标：
- MSE（均方误差）
- RMSE（均方根误差）
- MAE（平均绝对误差）
- MAPE（平均绝对百分比误差）
- R²（决定系数）

### 7. 结果可视化
可视化模块提供以下图表：
- 训练历史图表
- 预测结果对比图
- 预测误差分布图

## 配置文件说明

项目使用YAML格式的配置文件来管理各种参数，包括数据配置、模型配置、训练配置和评估配置。用户可以通过修改[configs/model_config.yaml](configs/model_config.yaml)文件来调整模型参数和训练设置。

### 配置文件结构说明

```
data:
  # 数据集根路径
  root_path: ./data/ETDataset/
  
  # 数据集类型: ETTh1, ETTh2, ETTm1, ETTm2
  data: ETTh1
  
  # 特征类型: S (单变量), M (多变量), MS (多变量预测单变量)
  features: M
  
  # 目标变量
  target: OT
  
  # 输入序列长度
  seq_len: 96
  
  # 标签序列长度
  label_len: 48
  
  # 预测序列长度
  pred_len: 24
  
  # 数据频率 (h: 小时, t: 分钟)
  freq: h
  
  # 批次大小
  batch_size: 32
  
  # 数据加载进程数
  num_workers: 0

model:
  # 模型类型: liquid_ode, lstm, liquid_lstm, transformer, informer, lnn_informer
  type: liquid_ode
  
  # 隐藏层大小
  hidden_size: 64
  
  # 学习率
  learning_rate: 0.001
  
  # 训练轮数
  epochs: 100
  
  # 批次大小
  batch_size: 32
  
  # ODE求解方法 (适用于liquid_ode)
  ode_method: dopri5
  
  # LSTM层数 (适用于lstm, liquid_lstm)
  lstm_layers: 2

training:
  # 验证集比例
  validation_split: 0.2
  
  # 是否打乱数据
  shuffle: True

evaluation:
  # 评估指标
  metrics:
    - MSE
    - RMSE
    - MAE
    - MAPE
    - R2
```

### 配置优化建议
- **数据集选择**：ETTh系列为小时级数据，ETTm系列为分钟级数据，根据预测需求选择合适的数据集
- **特征类型**：
  - 'S': 单变量预测，仅使用目标变量(OT)进行预测
  - 'M': 多变量预测，使用所有变量进行预测
  - 'MS': 多变量预测单变量，使用所有变量预测目标变量
- **序列长度**：seq_len表示输入序列长度，label_len表示标签序列长度，pred_len表示预测序列长度
- **数据集参数**：root_path应指向ETT数据集的根目录，data参数指定具体使用的数据集文件
- **ODE求解方法**：对于液态ODE模型，dopri5是推荐的求解方法，具有良好的精度和效率平衡

## 使用方法

### 本地运行

```bash
# 运行默认实验
python run.py

# 运行短时预测实验
python run.py --experiment STSF

# 运行模型对比实验
python run.py --experiment ModelComparison

# 指定配置文件
python run.py --config configs/model_config.yaml

## 液态神经网络简介

液态神经网络（Liquid Neural Networks）是一种受生物神经网络启发的新一代机器学习模型，具有以下特点：
- **时间连续性**：能够处理连续时间信号
- **动态行为**：可以适应输入信号的变化模式
- **记忆能力**：具有内置的时间记忆机制
- **高效性**：相比传统RNN在某些任务上更加高效

这些特性使得液态神经网络特别适合处理电力负荷这类具有复杂时间依赖性的时序数据。