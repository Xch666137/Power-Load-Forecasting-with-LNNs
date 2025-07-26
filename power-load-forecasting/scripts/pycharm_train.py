#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyCharm SSH解释器训练脚本
专门用于在PyCharm中通过SSH解释器运行训练任务
"""

import os
import sys
import argparse
import yaml

# 添加项目根目录到Python路径
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)

from experiments.exp_STSF import ExpSTSF
from experiments.exp_model_comparison import ExpModelComparison


def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    # 如果是相对路径，则转换为绝对路径
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_path, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    主函数 - 用于PyCharm SSH解释器执行训练
    """
    parser = argparse.ArgumentParser(description='PyCharm SSH解释器训练脚本')
    parser.add_argument('--experiment', type=str, default='STSF',
                        choices=['STSF', 'ModelComparison'],
                        help='实验类型: STSF (短时预测), ModelComparison (模型对比)')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='配置文件路径')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        print(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        
        # 根据实验类型运行相应实验
        if args.experiment == 'STSF':
            print("运行短时预测实验...")
            experiment = ExpSTSF()
            experiment.run(config)
        elif args.experiment == 'ModelComparison':
            print("运行模型对比实验...")
            experiment = ExpModelComparison()
            experiment.run(config)
        else:
            print(f"未知的实验类型: {args.experiment}")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"运行实验时发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 如果没有命令行参数，则使用默认设置
    if len(sys.argv) == 1:
        # 默认运行STSF实验
        print("使用默认设置运行短时预测实验...")
        config_path = os.path.join(base_path, 'configs', 'model_config.yaml')
        config = load_config(config_path)
        
        experiment = ExpSTSF()
        experiment.run(config)
    else:
        # 使用命令行参数
        main()