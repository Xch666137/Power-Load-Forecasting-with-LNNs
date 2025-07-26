#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
电力负荷预测系统入口文件
"""

import os
import sys

# 添加项目根目录到Python路径
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path)

# 检查是否在远程环境中运行（通过PyCharm SSH解释器）
def is_remote_execution():
    """
    检查是否在远程环境执行（如PyCharm SSH解释器）
    """
    # 检查环境变量
    if os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_CLIENT'):
        return True
    
    # 检查主机名
    hostname = os.environ.get('HOSTNAME', '')
    if 'gpu' in hostname.lower() or 'server' in hostname.lower():
        return True
    
    return False

if __name__ == "__main__":
    # 检查是否通过PyCharm SSH解释器运行
    if is_remote_execution():
        print("检测到通过SSH解释器运行，直接执行训练...")
        # 导入实验类
        from experiments.exp_STSF import ExpSTSF
        import yaml
        
        # 加载配置
        config_path = os.path.join(base_path, 'configs', 'model_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 运行实验
        experiment = ExpSTSF()
        experiment.run(config)
    else:
        # 导入并运行CLI接口
        from scripts.cli import main
        main()