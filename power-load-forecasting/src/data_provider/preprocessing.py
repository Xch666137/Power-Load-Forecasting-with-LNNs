"""
电力负荷数据预处理模块
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List, Optional


class PowerLoadPreprocessor:
    """
    电力负荷数据预处理器
    """
    
    def __init__(self):
        """
        初始化预处理器
        """
        self.scaler = None
        self.feature_columns = []
    
    def create_time_features(self, data: pd.DataFrame, datetime_column: str = 'datetime') -> pd.DataFrame:
        """
        创建时间特征
        
        Args:
            data: 原始数据
            datetime_column: 时间列名
            
        Returns:
            DataFrame: 添加时间特征后的数据
        """
        df = data.copy()
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        
        # 提取时间特征
        df['year'] = df[datetime_column].dt.year
        df['month'] = df[datetime_column].dt.month
        df['day'] = df[datetime_column].dt.day
        df['hour'] = df[datetime_column].dt.hour
        df['dayofweek'] = df[datetime_column].dt.dayofweek
        df['weekend'] = (df[datetime_column].dt.dayofweek >= 5).astype(int)
        df['quarter'] = df[datetime_column].dt.quarter
        
        return df
    
    def create_lag_features(self, data: pd.DataFrame, 
                           target_column: str = 'load',
                           lags: List[int] = [1, 2, 3, 24, 168]) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            data: 原始数据
            target_column: 目标列名
            lags: 滞后时间步
            
        Returns:
            DataFrame: 添加滞后特征后的数据
        """
        df = data.copy()
        
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
            
        return df
    
    def create_rolling_features(self, data: pd.DataFrame,
                               target_column: str = 'load',
                               windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """
        创建滚动统计特征
        
        Args:
            data: 原始数据
            target_column: 目标列名
            windows: 窗口大小列表
            
        Returns:
            DataFrame: 添加滚动特征后的数据
        """
        df = data.copy()
        
        for window in windows:
            df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
            df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
            df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window=window).min()
            df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window=window).max()
            
        return df
    
    def prepare_sequences(self, data: pd.DataFrame,
                         target_column: str = 'load',
                         sequence_length: int = 24,
                         forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备时间序列数据用于训练
        
        Args:
            data: 输入数据
            target_column: 目标列名
            sequence_length: 序列长度
            forecast_horizon: 预测步长
            
        Returns:
            Tuple: (特征序列, 目标值)
        """
        # 选择数值型特征列
        feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in feature_columns:
            feature_columns.remove(target_column)
        
        # 保存特征列名
        self.feature_columns = feature_columns
        
        # 提取特征和目标数据
        features = data[feature_columns].values
        targets = data[target_column].values
        
        X, y = [], []
        for i in range(sequence_length, len(features) - forecast_horizon + 1):
            X.append(features[i-sequence_length:i])
            y.append(targets[i:i+forecast_horizon])
            
        return np.array(X), np.array(y)
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        对特征进行标准化
        
        Args:
            X: 特征数据
            fit: 是否重新拟合缩放器
            
        Returns:
            np.ndarray: 缩放后的特征数据
        """
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        else:
            X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
        return X_scaled
    
    def inverse_scale_targets(self, y: np.ndarray) -> np.ndarray:
        """
        对目标值进行反向缩放
        
        Args:
            y: 缩放后的目标值
            
        Returns:
            np.ndarray: 原始尺度的目标值
        """
        # 这里需要根据实际情况调整，因为我们只对特征进行了缩放
        # 如果目标值也需要缩放，需要单独的缩放器
        return y


def preprocess_power_data(data: pd.DataFrame,
                         datetime_column: str = 'datetime',
                         target_column: str = 'load') -> pd.DataFrame:
    """
    预处理电力负荷数据的便捷函数
    
    Args:
        data: 原始数据
        datetime_column: 时间列名
        target_column: 目标列名
        
    Returns:
        DataFrame: 预处理后的数据
    """
    preprocessor = PowerLoadPreprocessor()
    
    # 创建时间特征
    data = preprocessor.create_time_features(data, datetime_column)
    
    # 创建滞后特征
    data = preprocessor.create_lag_features(data, target_column)
    
    # 创建滚动统计特征
    data = preprocessor.create_rolling_features(data, target_column)
    
    # 删除包含NaN的行
    data = data.dropna().reset_index(drop=True)
    
    return data