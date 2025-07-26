import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .data_factory import ETTDataset, StandardScaler


def load_dataset(root_path, data_path, flag, size, features, target, scale=True, inverse=False, timeenc=0, freq='h'):
    """
    加载ETT数据集
    
    参数:
        root_path (str): 根路径
        data_path (str): 数据文件路径
        flag (str): 数据集类型 ('train', 'val', 'test')
        size (list): [seq_len, label_len, pred_len]
        features (str): 特征类型 ('S' - 单变量, 'M' - 多变量, 'MS' - 多变量预测单变量)
        target (str): 目标变量名
        scale (bool): 是否标准化数据
        inverse (bool): 是否逆变换输出数据
        timeenc (int): 时间编码方式 (0: 离散特征, 1: 周期编码)
        freq (str): 时间频率 ('h' - 小时, 't' - 分钟等)
    
    返回:
        ETTDataset: 数据集对象
    """
    dataset = ETTDataset(
        root_path=root_path,
        flag=flag,
        size=size,
        features=features,
        data_path=data_path,
        target=target,
        scale=scale,
        inverse=inverse,
        timeenc=timeenc,
        freq=freq
    )
    return dataset


def get_data_loader(dataset, batch_size, shuffle=True, num_workers=0, drop_last=True):
    """
    创建数据加载器
    
    参数:
        dataset (Dataset): 数据集对象
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        num_workers (int): 数据加载进程数
        drop_last (bool): 是否丢弃最后一个不完整的批次
    
    返回:
        DataLoader: 数据加载器对象
    """
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )
    return data_loader


# 数据集类型映射
dataset_mapping = {
    'ETTh1': 'ETT-small/ETTh1.csv',
    'ETTh2': 'ETT-small/ETTh2.csv',
    'ETTm1': 'ETT-small/ETTm1.csv',
    'ETTm2': 'ETT-small/ETTm2.csv',
}


def data_provider(args, flag):
    """
    根据参数提供数据集和数据加载器
    
    参数:
        args: 包含数据配置的参数对象
        flag (str): 数据集类型 ('train', 'val', 'test')
        
    返回:
        tuple: (数据集对象, 数据加载器对象)
    """
    # 获取数据文件路径
    data_path = dataset_mapping.get(args.data, args.data)
    
    # 构造数据集大小参数
    size = [args.seq_len, args.label_len, args.pred_len]
    
    # 加载数据集
    dataset = load_dataset(
        root_path=args.root_path,
        data_path=data_path,
        flag=flag,
        size=size,
        features=args.features,
        target=args.target,
        scale=True,
        inverse=False,
        timeenc=0,
        freq=args.freq
    )
    
    # 打印数据集信息
    print(flag, len(dataset))
    
    # 创建数据加载器
    batch_size = args.batch_size if flag == 'train' else 1
    data_loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=True if flag == 'train' else False,
        num_workers=args.num_workers,
        drop_last=True if flag == 'train' else False
    )
    
    return dataset, data_loader