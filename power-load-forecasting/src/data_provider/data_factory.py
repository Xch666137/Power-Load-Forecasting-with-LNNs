"""
数据工厂模块
用于根据配置创建不同的数据集实例
"""
from typing import Dict, Any, Optional
from .data_interface import BaseDataset, ETTHDataset, ETTmDataset, CustomDataset


class DatasetFactory:
    """
    数据集工厂类
    """
    
    # 数据集类型映射
    DATASET_TYPES = {
        'ETTh1': ETTHDataset,
        'ETTh2': ETTHDataset,
        'ETTm1': ETTmDataset,
        'ETTm2': ETTmDataset,
        'custom': CustomDataset
    }
    
    @classmethod
    def create_dataset(cls, dataset_type: str, **kwargs) -> BaseDataset:
        """
        创建数据集实例
        
        Args:
            dataset_type: 数据集类型
            **kwargs: 数据集参数
            
        Returns:
            BaseDataset: 数据集实例
            
        Raises:
            ValueError: 当数据集类型不支持时
        """
        if dataset_type not in cls.DATASET_TYPES:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                           f"Supported types: {list(cls.DATASET_TYPES.keys())}")
        
        dataset_class = cls.DATASET_TYPES[dataset_type]
        return dataset_class(**kwargs)
    
    @classmethod
    def register_dataset_type(cls, name: str, dataset_class):
        """
        注册新的数据集类型
        
        Args:
            name: 数据集类型名称
            dataset_class: 数据集类
        """
        cls.DATASET_TYPES[name] = dataset_class


def create_dataset(dataset_type: str, **kwargs) -> BaseDataset:
    """
    创建数据集实例的便捷函数
    
    Args:
        dataset_type: 数据集类型
        **kwargs: 数据集参数
        
    Returns:
        BaseDataset: 数据集实例
    """
    return DatasetFactory.create_dataset(dataset_type, **kwargs)