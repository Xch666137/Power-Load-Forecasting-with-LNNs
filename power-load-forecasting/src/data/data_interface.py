"""
电力负荷数据接口模块
定义数据集接口，用于适配不同数据源
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseDataset(ABC):
    """
    数据集接口基类
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.features = None
        self.target = None
    
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载数据，返回特征和目标值
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 特征数据和目标数据
        """
        pass
    
    @abstractmethod
    def get_data_info(self) -> dict:
        """
        获取数据集信息
        
        Returns:
            dict: 数据集信息
        """
        pass


class ETTHDataset(BaseDataset):
    """
    ETT小时级数据集 (ETTh1, ETTh2)
    """
    
    def __init__(self, data_path: str, dataset_name: str = "ETTh1"):
        """
        初始化ETT小时级数据集
        
        Args:
            data_path: 数据文件路径
            dataset_name: 数据集名称 (ETTh1, ETTh2)
        """
        super().__init__(data_path)
        self.dataset_name = dataset_name
        self.feature_columns = None
        self.target_column = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载ETT小时级数据
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 特征数据和目标数据
        """
        # 读取数据
        self.data = pd.read_csv(self.data_path)
        
        # 转换日期列
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # ETT数据集默认特征列（除日期和目标列外的所有列）
        all_columns = list(self.data.columns)
        self.feature_columns = [col for col in all_columns if col not in ['date']]
        self.target_column = 'OT'  # Oil Temperature 作为默认目标列
        
        # 设置特征和目标
        self.features = self.data[self.feature_columns]
        self.target = self.data[[self.target_column]]
        
        return self.features, self.target
    
    def get_data_info(self) -> dict:
        """
        获取数据集信息
        
        Returns:
            dict: 数据集信息
        """
        if self.data is None:
            self.load_data()
            
        return {
            'dataset_name': self.dataset_name,
            'shape': self.data.shape,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'date_range': {
                'start': self.data['date'].min(),
                'end': self.data['date'].max()
            },
            'missing_values': self.data.isnull().sum().to_dict()
        }


class ETTmDataset(BaseDataset):
    """
    ETT分钟级数据集 (ETTm1, ETTm2)
    """
    
    def __init__(self, data_path: str, dataset_name: str = "ETTm1"):
        """
        初始化ETT分钟级数据集
        
        Args:
            data_path: 数据文件路径
            dataset_name: 数据集名称 (ETTm1, ETTm2)
        """
        super().__init__(data_path)
        self.dataset_name = dataset_name
        self.feature_columns = None
        self.target_column = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载ETT分钟级数据
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 特征数据和目标数据
        """
        # 读取数据
        self.data = pd.read_csv(self.data_path)
        
        # 转换日期列
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # ETT数据集默认特征列（除日期和目标列外的所有列）
        all_columns = list(self.data.columns)
        self.feature_columns = [col for col in all_columns if col not in ['date']]
        self.target_column = 'OT'  # Oil Temperature 作为默认目标列
        
        # 设置特征和目标
        self.features = self.data[self.feature_columns]
        self.target = self.data[[self.target_column]]
        
        return self.features, self.target
    
    def get_data_info(self) -> dict:
        """
        获取数据集信息
        
        Returns:
            dict: 数据集信息
        """
        if self.data is None:
            self.load_data()
            
        return {
            'dataset_name': self.dataset_name,
            'shape': self.data.shape,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'date_range': {
                'start': self.data['date'].min(),
                'end': self.data['date'].max()
            },
            'missing_values': self.data.isnull().sum().to_dict()
        }


class CustomDataset(BaseDataset):
    """
    自定义数据集
    """
    
    def __init__(self, data_path: str, feature_columns: list = None, target_column: str = 'load'):
        """
        初始化自定义数据集
        
        Args:
            data_path: 数据文件路径
            feature_columns: 特征列名列表
            target_column: 目标列名
        """
        super().__init__(data_path)
        self.feature_columns = feature_columns
        self.target_column = target_column
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载自定义数据
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 特征数据和目标数据
        """
        # 根据文件扩展名选择加载方式
        if self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
            self.data = pd.read_excel(self.data_path)
        else:
            raise ValueError("Unsupported file format")
        
        # 如果未指定特征列，则使用除目标列外的所有列
        if self.feature_columns is None:
            all_columns = list(self.data.columns)
            self.feature_columns = [col for col in all_columns if col != self.target_column]
        
        # 设置特征和目标
        self.features = self.data[self.feature_columns]
        self.target = self.data[[self.target_column]]
        
        return self.features, self.target
    
    def get_data_info(self) -> dict:
        """
        获取数据集信息
        
        Returns:
            dict: 数据集信息
        """
        if self.data is None:
            self.load_data()
            
        return {
            'dataset_name': 'Custom',
            'shape': self.data.shape,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'missing_values': self.data.isnull().sum().to_dict()
        }