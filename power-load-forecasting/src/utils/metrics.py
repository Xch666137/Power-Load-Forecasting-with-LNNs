"""
评估指标工具模块
"""
import numpy as np
from typing import Union


def mean_absolute_percentage_error(y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> float:
    """
    计算平均绝对百分比误差 (MAPE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        MAPE值（百分比）
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100
    return mape


def normalized_mean_absolute_error(y_true: np.ndarray,
                                 y_pred: np.ndarray) -> float:
    """
    计算归一化平均绝对误差 (NMAE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        NMAE值（百分比）
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    nmae = mae / np.mean(np.abs(y_true)) * 100
    return nmae


def mean_absolute_error(y_true: np.ndarray,
                       y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差 (MAE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        MAE值
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true: np.ndarray,
                           y_pred: np.ndarray) -> float:
    """
    计算均方根误差 (RMSE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        RMSE值
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_squared_error(y_true: np.ndarray,
                      y_pred: np.ndarray) -> float:
    """
    计算均方误差 (MSE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        MSE值
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def r_squared(y_true: np.ndarray,
             y_pred: np.ndarray) -> float:
    """
    计算决定系数 (R²)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        R²值
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def directional_accuracy(y_true: np.ndarray,
                        y_pred: np.ndarray) -> float:
    """
    计算方向精度 (DA)
    衡量预测值变化方向与真实值变化方向一致的比例
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        DA值（百分比）
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # 计算变化方向
    true_direction = np.diff(y_true)
    pred_direction = np.diff(y_pred)
    
    # 计算方向一致的比例
    correct_direction = np.sign(true_direction) == np.sign(pred_direction)
    da = np.mean(correct_direction) * 100
    
    return da


def theil_u_statistic(y_true: np.ndarray,
                     y_pred: np.ndarray) -> float:
    """
    计算Theil不等系数 (Theil's U)
    衡量预测误差相对于基准误差的比例，值越小表示预测效果越好
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        Theil's U值
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # 计算分子（预测误差）
    numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 计算分母（基准误差）
    denominator = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    
    # 避免除以零
    if denominator == 0:
        return np.inf if numerator > 0 else 0
    
    return numerator / denominator


def calculate_all_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray) -> dict:
    """
    计算所有评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        包含所有评估指标的字典
    """
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'NMAE': normalized_mean_absolute_error(y_true, y_pred),
        'R2': r_squared(y_true, y_pred),
        'DA': directional_accuracy(y_true, y_pred),
        'Theil_U': theil_u_statistic(y_true, y_pred)
    }