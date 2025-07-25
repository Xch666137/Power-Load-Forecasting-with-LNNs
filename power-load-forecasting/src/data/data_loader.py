"""
电力负荷数据加载模块
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class PowerLoadDataLoader:
    """
    电力负荷数据加载器
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """
        加载电力负荷数据
        
        Returns:
            DataFrame: 电力负荷数据
        """
        if self.data_path:
            # 根据文件扩展名选择加载方式
            if self.data_path.endswith('.csv'):
                self.data = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
                self.data = pd.read_excel(self.data_path)
            else:
                raise ValueError("Unsupported file format")
        else:
            # 生成示例数据
            self.data = self._generate_sample_data()
            
        return self.data
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """
        生成示例电力负荷数据
        
        Returns:
            DataFrame: 示例电力负荷数据
        """
        # 生成时间序列数据（以小时为单位，生成30天的数据）
        dates = pd.date_range('2023-01-01', periods=30*24, freq='H')
        
        # 模拟电力负荷数据（包含趋势、周期性和噪声）
        np.random.seed(42)
        trend = np.linspace(1000, 1200, len(dates))  # 长期趋势
        daily_pattern = 500 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)  # 日周期性
        weekly_pattern = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*7))  # 周周期性
        noise = np.random.normal(0, 50, len(dates))  # 随机噪声
        
        load = trend + daily_pattern + weekly_pattern + noise
        
        # 创建DataFrame
        data = pd.DataFrame({
            'datetime': dates,
            'load': load
        })
        
        return data
    
    def get_transformer_load_data(self, transformer_id: str) -> pd.DataFrame:
        """
        获取特定变压器的负荷数据
        
        Args:
            transformer_id: 变压器ID
            
        Returns:
            DataFrame: 变压器负荷数据
        """
        # 在实际应用中，这里会根据变压器ID筛选数据
        # 目前我们使用所有数据作为示例
        if self.data is None:
            self.load_data()
            
        # 添加变压器ID列（示例）
        self.data['transformer_id'] = transformer_id
        return self.data


def load_power_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    加载电力负荷数据的便捷函数
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        DataFrame: 电力负荷数据
    """
    loader = PowerLoadDataLoader(data_path)
    return loader.load_data()