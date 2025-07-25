"""
模型评估模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    """
    模型评估器
    """
    
    def __init__(self):
        """
        初始化模型评估器
        """
        pass
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            评估指标字典
        """
        # 确保输入是numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 移除可能的额外维度
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        
        # 计算各种评估指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算MAPE (Mean Absolute Percentage Error)
        # 避免除以零的情况
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # 计算NMAE (Normalized Mean Absolute Error)
        nmae = mae / np.mean(np.abs(y_true)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'NMAE': nmae,
            'R2': r2
        }
    
    def plot_predictions(self, 
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        title: str = "预测结果对比",
                        save_path: Optional[str] = None):
        """
        绘制预测结果对比图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            save_path: 保存路径
        """
        # 确保输入是numpy数组
        y_true = np.array(y_true).squeeze()
        y_pred = np.array(y_pred).squeeze()
        
        plt.figure(figsize=(12, 6))
        
        # 绘制真实值和预测值
        plt.plot(y_true, label='真实值', alpha=0.7)
        plt.plot(y_pred, label='预测值', alpha=0.7)
        
        plt.title(title)
        plt.xlabel('时间步')
        plt.ylabel('电力负荷')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_error_distribution(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               title: str = "预测误差分布",
                               save_path: Optional[str] = None):
        """
        绘制预测误差分布图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            save_path: 保存路径
        """
        # 确保输入是numpy数组
        y_true = np.array(y_true).squeeze()
        y_pred = np.array(y_pred).squeeze()
        
        # 计算误差
        errors = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        
        # 绘制误差直方图
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.title(title)
        plt.xlabel('预测误差')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        plt.axvline(mean_error, color='red', linestyle='--', 
                   label=f'平均误差: {mean_error:.2f}')
        plt.axvline(mean_error + std_error, color='orange', linestyle=':', 
                   label=f'+1标准差: {mean_error + std_error:.2f}')
        plt.axvline(mean_error - std_error, color='orange', linestyle=':', 
                   label=f'-1标准差: {mean_error - std_error:.2f}')
        
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_scatter_predictions(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                title: str = "预测值 vs 真实值散点图",
                                save_path: Optional[str] = None):
        """
        绘制预测值与真实值的散点图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            save_path: 保存路径
        """
        # 确保输入是numpy数组
        y_true = np.array(y_true).squeeze()
        y_pred = np.array(y_pred).squeeze()
        
        plt.figure(figsize=(10, 8))
        
        # 绘制散点图
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # 绘制完美预测线
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.title(title)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.grid(True, alpha=0.3)
        
        # 计算R2并添加到图中
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray) -> pd.DataFrame:
        """
        生成评估报告
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            评估报告DataFrame
        """
        # 计算评估指标
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # 转换为DataFrame
        report = pd.DataFrame.from_dict(metrics, orient='index', columns=['值'])
        report.index.name = '指标'
        
        return report


def evaluate_model(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  plot: bool = True) -> Dict[str, float]:
    """
    评估模型的便捷函数
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        plot: 是否绘制图表
        
    Returns:
        评估指标字典
    """
    evaluator = ModelEvaluator()
    
    # 计算评估指标
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    
    if plot:
        # 绘制预测结果对比图
        evaluator.plot_predictions(y_true, y_pred)
        
        # 绘制误差分布图
        evaluator.plot_error_distribution(y_true, y_pred)
        
        # 绘制散点图
        evaluator.plot_scatter_predictions(y_true, y_pred)
    
    return metrics