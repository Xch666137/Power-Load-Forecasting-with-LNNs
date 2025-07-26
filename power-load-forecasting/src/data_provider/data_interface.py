"""
电力负荷数据接口模块
定义数据集接口，用于适配不同数据源
"""
import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import warnings


class BaseDataset(ABC):
    """
    数据集接口基类
    """

    def __init__(self, root_path: str, data_path: str):
        """
        初始化数据集

        Args:
            root_path: 根路径
            data_path: 数据文件路径
        """
        self.root_path = root_path
        self.data_path = data_path
        self.raw_data = None
        self.data = None
        self.features = None
        self.target = None

    @abstractmethod
    def __read_data__(self):
        """
        读取数据的抽象方法
        """
        pass

    def get_data_info(self) -> dict:
        """
        获取数据集信息

        Returns:
            dict: 数据集信息
        """
        if self.data is None:
            self.__read_data__()
            
        info = {
            'data_shape': self.data.shape if self.data is not None else None,
            'feature_shape': self.features.shape if self.features is not None else None,
            'target_shape': self.target.shape if self.target is not None else None,
            'root_path': self.root_path,
            'data_path': self.data_path
        }
        return info


class ETTDataset(BaseDataset):
    """
    ETT数据集类 (Electricity Transformer Temperature)
    适配ETT-small数据集，包括ETTh1, ETTh2, ETTm1, ETTm2
    """

    def __init__(self, root_path: str, data_path: str, dataset_name: str = "ETT"):
        """
        初始化ETT数据集

        Args:
            root_path: 根路径
            data_path: 数据文件路径
            dataset_name: 数据集名称
        """
        super().__init__(root_path, data_path)
        self.dataset_name = dataset_name
        self.feature_columns = None
        self.target_column = None
        self.data_stamp = None

    def __read_data__(self):
        """
        读取ETT数据
        """
        # 构建完整路径
        full_path = os.path.join(self.root_path, self.data_path)
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"数据文件不存在: {full_path}")
        
        # 读取数据
        self.raw_data = pd.read_csv(full_path)
        
        # 处理日期列
        if 'date' in self.raw_data.columns:
            self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
        else:
            # 尝试自动检测日期列
            date_columns = [col for col in self.raw_data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                self.raw_data[date_columns[0]] = pd.to_datetime(self.raw_data[date_columns[0]])
                self.raw_data.rename(columns={date_columns[0]: 'date'}, inplace=True)
            else:
                raise ValueError("未找到日期列，请确保数据中包含日期时间信息")
        
        # 保存处理后的数据
        self.data = self.raw_data.copy()
        
        # 设置特征列和目标列
        all_columns = list(self.data.columns)
        self.feature_columns = [col for col in all_columns if col != 'date']
        self.target_column = 'OT'  # Oil Temperature 作为默认目标列
        
        # 分离特征和目标
        self.features = self.data[self.feature_columns]
        self.target = self.data[[self.target_column]]

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载ETT数据

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 特征数据和目标数据
        """
        if self.data is None:
            self.__read_data__()
            
        return self.features, self.target