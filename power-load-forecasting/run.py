#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动训练的主程序
"""
import os
import sys
import yaml

# 启用CUDA设备端断言
os.environ['TORCH_USE_CUDA_DSA'] = '1'
# 启用CUDA阻塞式启动
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import argparse
from experiments.exp_STSF import ExpSTSF
from utils.visualization import plot_training_history

# 设置PyTorch多进程共享策略，解决"Too many open files"问题
torch.multiprocessing.set_sharing_strategy('file_system')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from experiments.exp_STSF import ExpSTSF
from experiments.exp_model_comparison import ExpModelComparison


def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置参数
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='电力负荷预测系统')
    parser.add_argument('--experiment', type=str, default='STSF',
                        help='实验类型: STSF, ModelComparison')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--itr', type=int, default=1,
                        help='实验重复次数')
    parser.add_argument('--disable-progress-bar', action='store_true',
                        help='禁用训练进度条显示')
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_config(args.config)
        print(f"配置文件加载成功: {args.config}")
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        return
    
    # 将进度条设置添加到配置中
    config['disable_progress_bar'] = args.disable_progress_bar
    
    # 根据实验类型运行实验
    if args.experiment == 'STSF':
        print("运行短时预测实验...")
        exp = ExpSTSF()
        exp.run(config)
    elif args.experiment == 'ModelComparison':
        print("运行模型对比实验...")
        exp = ExpModelComparison()
        exp.run(config)
    else:
        print(f"未知的实验类型: {args.experiment}")
        return


if __name__ == '__main__':
    main()