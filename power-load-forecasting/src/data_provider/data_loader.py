"""
电力负荷数据加载模块
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from .data_interface import BaseDataset
from .data_factory import DatasetFactory


class PowerLoadDataLoader:
    """
    电力负荷数据加载器
    """
    
    def __init__(self, dataset_type: str = 'custom', **dataset_kwargs):
        """
        初始化数据加载器
        
        Args:
            dataset_type: 数据集类型
            **dataset_kwargs: 数据集参数
        """
        self.dataset_type = dataset_type
        self.dataset_kwargs = dataset_kwargs
        self.dataset: Optional[BaseDataset] = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载电力负荷数据
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 特征数据和目标数据
        """
        # 使用工厂模式创建数据集实例
        self.dataset = DatasetFactory.create_dataset(self.dataset_type, **self.dataset_kwargs)
        
        # 加载数据
        features, target = self.dataset.load_data()
        
        return features, target
    
    def get_dataset_info(self) -> dict:
        """
        获取数据集信息
        
        Returns:
            dict: 数据集信息
        """
        if self.dataset is None:
            self.load_data()
            
        return self.dataset.get_data_info()
    
    def get_transformer_load_data(self, transformer_id: str) -> pd.DataFrame:
        """
        获取特定变压器的负荷数据（保持向后兼容）
        
        Args:
            transformer_id: 变压器ID
            
        Returns:
            DataFrame: 变压器负荷数据
        """
        # 加载数据
        features, target = self.load_data()
        
        # 合并特征和目标数据
        data = features.copy()
        target_col = target.columns[0]
        data[target_col] = target[target_col]
        
        # 添加变压器ID列（示例）
        data['transformer_id'] = transformer_id
        return data


def load_power_data(dataset_type: str = 'custom', **dataset_kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载电力负荷数据的便捷函数
    
    Args:
        dataset_type: 数据集类型
        **dataset_kwargs: 数据集参数
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 特征数据和目标数据
    """
    loader = PowerLoadDataLoader(dataset_type, **dataset_kwargs)
    return loader.load_data()