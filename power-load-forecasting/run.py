#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实验通用入口文件
"""

import os
import sys
import argparse
import yaml

# 添加项目根目录到Python路径
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path)

# 静态导入实验模块
from experiments.exp_STSF import ExpSTSF

def load_config(config_path="configs/model_config.yaml"):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(base_path, config_path)
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        print(f"警告: 配置文件 {config_file} 不存在，将使用默认配置")
        return {}


def main():
    """
    主函数
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='电力负荷预测实验入口')
    parser.add_argument('--experiment', '-e', type=str, default='STSF',
                        help='实验类型: STSF (短时预测)等')
    parser.add_argument('--config', '-c', type=str, default='configs/model_config.yaml',
                        help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 根据实验类型运行相应实验
    if args.experiment.upper() == 'STSF':
        print("运行短时预测实验...")
        # 静态导入实验类
        experiment = ExpSTSF()
        experiment.run(config)
    else:
        print(f"未知实验类型: {args.experiment}")
        print("支持的实验类型:")
        print("STSF: 短时预测实验")
        sys.exit(1)

if __name__ == "__main__":
    main()