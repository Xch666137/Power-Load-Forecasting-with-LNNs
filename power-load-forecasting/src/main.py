"""
电力负荷预测主程序
"""
import numpy as np
import pandas as pd
import torch
import os
import sys
import argparse
import yaml
from datetime import datetime

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import PowerLoadDataLoader, load_power_data
from data.preprocessing import PowerLoadPreprocessor, preprocess_power_data
from models.liquid_neural_network import create_liquid_model
from models.training import ModelTrainer, train_model
from models.evaluation import ModelEvaluator
from utils.metrics import calculate_all_metrics
from utils.visualization import plot_time_series, plot_training_history


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config or {}
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认配置")
        return {}
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        return {}


def main():
    """
    主函数
    """
    print("电力负荷预测系统 - 基于液态神经网络")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    
    # 默认配置
    default_config = {
        'data': {
            'dataset_type': 'custom',  # 数据集类型: custom, ETTh1, ETTh2, ETTm1, ETTm2
            'data_path': None,         # 数据文件路径
            'sequence_length': 24,
            'forecast_horizon': 1
        },
        'model': {
            'type': 'liquid_ode',
            'hidden_size': 64,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        },
        'training': {
            'validation_split': 0.2,
            'shuffle': True
        }
    }
    
    # 合并配置
    for key in default_config:
        if key not in config:
            config[key] = default_config[key]
        else:
            for sub_key, sub_value in default_config[key].items():
                if sub_key not in config[key]:
                    config[key][sub_key] = sub_value
    
    # 1. 数据加载
    print("1. 加载数据...")
    data_loader = PowerLoadDataLoader(
        dataset_type=config['data']['dataset_type'],
        data_path=config['data']['data_path']
    )
    
    try:
        features, target = data_loader.load_data()
        print(f"特征数据形状: {features.shape}")
        print(f"目标数据形状: {target.shape}")
        
        # 显示数据集信息
        dataset_info = data_loader.get_dataset_info()
        print(f"数据集类型: {dataset_info['dataset_name']}")
        print(f"数据集大小: {dataset_info['shape']}")
        print(f"目标列: {dataset_info['target_column']}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    preprocessor = PowerLoadPreprocessor()
    
    # 合并特征和目标数据用于预处理
    raw_data = features.copy()
    target_col = target.columns[0]
    raw_data[target_col] = target[target_col]
    
    # 创建时间特征
    processed_data = preprocessor.create_time_features(raw_data)
    
    # 创建滞后特征
    processed_data = preprocessor.create_lag_features(processed_data, target_column=target_col)
    
    # 创建滚动统计特征
    processed_data = preprocessor.create_rolling_features(processed_data, target_column=target_col)
    
    # 删除包含NaN的行
    processed_data = processed_data.dropna().reset_index(drop=True)
    print(f"处理后数据形状: {processed_data.shape}")
    
    # 3. 准备训练数据
    print("\n3. 准备训练数据...")
    X, y = preprocessor.prepare_sequences(
        processed_data,
        sequence_length=config['data']['sequence_length'],
        forecast_horizon=config['data']['forecast_horizon']
    )
    
    print(f"特征数据形状: {X.shape}")
    print(f"标签数据形状: {y.shape}")
    
    # 4. 数据集划分
    print("\n4. 划分数据集...")
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 5. 创建模型
    print("\n5. 创建液态神经网络模型...")
    input_size = X.shape[2]  # 特征维度
    output_size = y.shape[1] if len(y.shape) > 1 else 1  # 输出维度
    
    model = create_liquid_model(
        input_size=input_size,
        hidden_size=config['model']['hidden_size'],
        output_size=output_size,
        model_type=config['model']['type']
    )
    
    print(f"模型结构: {model}")
    
    # 6. 训练模型
    print("\n6. 训练模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    trainer = ModelTrainer(model, device)
    trainer.compile(
        optimizer='adam',
        loss='mse',
        learning_rate=config['model']['learning_rate']
    )
    
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=config['model']['epochs'],
        batch_size=config['model']['batch_size'],
        validation_split=config['training']['validation_split'],
        shuffle=config['training']['shuffle']
    )
    
    print("训练完成!")
    
    # 7. 模型评估
    print("\n7. 评估模型...")
    # 在测试集上进行预测
    y_pred = trainer.predict(X_test)
    
    # 确保y_test是正确的形状
    y_test_flat = y_test.squeeze() if len(y_test.shape) > 1 else y_test
    y_pred_flat = y_pred.squeeze() if len(y_pred.shape) > 1 else y_pred
    
    # 计算评估指标
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test_flat, y_pred_flat)
    
    print("测试集评估结果:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 8. 可视化结果
    print("\n8. 生成可视化结果...")
    
    # 绘制训练历史
    plot_training_history(history, title="模型训练历史")
    
    # 绘制预测结果
    evaluator.plot_predictions(y_test_flat, y_pred_flat, title="电力负荷预测结果")
    
    # 绘制误差分布
    evaluator.plot_error_distribution(y_test_flat, y_pred_flat, title="预测误差分布")
    
    print("\n程序执行完成!")


if __name__ == "__main__":
    main()