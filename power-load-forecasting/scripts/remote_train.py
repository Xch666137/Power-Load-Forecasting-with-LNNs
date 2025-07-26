#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
电力负荷预测系统 - 远程GPU训练脚本
通过SSH连接到远程服务器并在GPU上运行实验
"""

import os
import sys
import argparse
import subprocess
import yaml
import json
from pathlib import Path

# 添加项目根目录到Python路径
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)


def load_remote_config(config_path):
    """
    加载远程配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_path, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"远程配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_ssh_command(ssh_config, command, capture_output=False):
    """
    通过SSH运行远程命令
    
    Args:
        ssh_config (dict): SSH配置
        command (str): 要执行的命令
        capture_output (bool): 是否捕获输出
        
    Returns:
        subprocess.CompletedProcess: 命令执行结果
    """
    # 构建SSH命令
    ssh_cmd = ["ssh"]
    
    # 添加SSH选项
    ssh_cmd.extend(["-o", "ConnectTimeout=10"])
    ssh_cmd.extend(["-o", "StrictHostKeyChecking=no"])
    
    # 如果指定了私钥文件
    if ssh_config.get('identity_file'):
        identity_file = os.path.expanduser(ssh_config['identity_file'])
        if os.path.exists(identity_file):
            ssh_cmd.extend(["-i", identity_file])
    
    # 添加端口
    if ssh_config.get('port') and ssh_config['port'] != 22:
        ssh_cmd.extend(["-p", str(ssh_config['port'])])
    
    # 添加用户和主机
    ssh_cmd.append(f"{ssh_config['user']}@{ssh_config['host']}")
    
    # 添加要执行的命令
    ssh_cmd.append(command)
    
    # 执行命令
    try:
        if capture_output:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
        else:
            result = subprocess.run(ssh_cmd, check=True)
        return result
    except subprocess.CalledProcessError as e:
        raise Exception(f"SSH命令执行失败: {e}")


def sync_code(ssh_config, sync_config):
    """
    同步本地代码到远程服务器
    
    Args:
        ssh_config (dict): SSH配置
        sync_config (dict): 同步配置
    """
    print("正在同步代码到远程服务器...")
    
    # 构建rsync命令
    rsync_cmd = ["rsync", "-avz", "--delete"]
    
    # 添加排除项
    if sync_config.get('exclude'):
        for exclude in sync_config['exclude']:
            rsync_cmd.extend(["--exclude", exclude])
    
    # 添加SSH选项
    rsync_rsh = f"ssh -o StrictHostKeyChecking=no"
    if ssh_config.get('identity_file'):
        identity_file = os.path.expanduser(ssh_config['identity_file'])
        if os.path.exists(identity_file):
            rsync_rsh += f" -i {identity_file}"
    
    if ssh_config.get('port') and ssh_config['port'] != 22:
        rsync_rsh += f" -p {ssh_config['port']}"
    
    rsync_cmd.extend(["-e", rsync_rsh])
    
    # 添加源目录和目标目录
    src_dir = os.path.join(base_path, "")  # 添加尾部斜杠
    dest = f"{ssh_config['user']}@{ssh_config['host']}:{ssh_config['project_path']}"
    rsync_cmd.extend([src_dir, dest])
    
    # 执行rsync
    try:
        subprocess.run(rsync_cmd, check=True)
        print("代码同步完成")
    except subprocess.CalledProcessError as e:
        raise Exception(f"代码同步失败: {e}")


def setup_remote_environment(ssh_config, remote_config):
    """
    在远程服务器上设置环境
    
    Args:
        ssh_config (dict): SSH配置
        remote_config (dict): 远程环境配置
    """
    print("正在设置远程环境...")
    
    # 激活虚拟环境（如果配置了）
    if remote_config.get('python_venv'):
        activate_cmd = f"source {remote_config['python_venv']}/bin/activate"
        run_ssh_command(ssh_config, activate_cmd)
    
    # 设置CUDA环境变量
    if remote_config.get('cuda_env'):
        for key, value in remote_config['cuda_env'].items():
            env_cmd = f"export {key}={value}"
            run_ssh_command(ssh_config, env_cmd)


def run_remote_experiment(ssh_config, remote_config, experiment_type, config_file):
    """
    在远程服务器上运行实验
    
    Args:
        ssh_config (dict): SSH配置
        remote_config (dict): 远程环境配置
        experiment_type (str): 实验类型
        config_file (str): 配置文件路径
    """
    print(f"正在远程服务器上运行{experiment_type}实验...")
    
    # 构建远程命令
    remote_project_path = ssh_config['project_path']
    remote_config_path = os.path.join(remote_project_path, config_file).replace('\\', '/')
    
    # 切换到项目目录并运行实验
    cd_cmd = f"cd {remote_project_path}"
    
    # 激活虚拟环境（如果配置了）
    if remote_config.get('python_venv'):
        activate_cmd = f"source {remote_config['python_venv']}/bin/activate && "
    else:
        activate_cmd = ""
    
    # 设置CUDA环境变量（如果配置了）
    cuda_env_cmd = ""
    if remote_config.get('cuda_env'):
        for key, value in remote_config['cuda_env'].items():
            cuda_env_cmd += f"export {key}={value} && "
    
    # 运行实验命令
    experiment_cmd = f"python run.py --experiment {experiment_type} --config {remote_config_path}"
    full_cmd = f"{cd_cmd} && {cuda_env_cmd}{activate_cmd}{experiment_cmd}"
    
    # 执行命令
    run_ssh_command(ssh_config, full_cmd)


def main():
    """
    主函数 - 远程训练脚本入口
    """
    parser = argparse.ArgumentParser(description='电力负荷预测系统 - 远程GPU训练')
    parser.add_argument('--experiment', type=str, default='STSF',
                        choices=['STSF', 'ModelComparison'],
                        help='实验类型: STSF (短时预测), ModelComparison (模型对比)')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='本地配置文件路径')
    parser.add_argument('--remote-config', type=str, default='configs/remote_config.yaml',
                        help='远程配置文件路径')
    parser.add_argument('--sync-only', action='store_true',
                        help='仅同步代码，不运行实验')
    
    args = parser.parse_args()
    
    try:
        # 加载远程配置
        print("加载远程配置文件...")
        remote_config = load_remote_config(args.remote_config)
        
        # 获取配置部分
        ssh_config = remote_config.get('ssh', {})
        sync_config = remote_config.get('sync', {})
        remote_env_config = remote_config.get('remote', {})
        
        # 验证必要配置
        if not ssh_config.get('user') or not ssh_config.get('host') or not ssh_config.get('project_path'):
            raise ValueError("SSH配置不完整，请检查remote_config.yaml文件")
        
        # 同步代码到远程服务器
        if sync_config.get('sync_code', True):
            sync_code(ssh_config, sync_config)
        
        if args.sync_only:
            print("仅同步代码完成")
            return
        
        # 设置远程环境
        setup_remote_environment(ssh_config, remote_env_config)
        
        # 运行远程实验
        run_remote_experiment(ssh_config, remote_env_config, args.experiment, args.config)
        
        print("远程实验执行完成")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()