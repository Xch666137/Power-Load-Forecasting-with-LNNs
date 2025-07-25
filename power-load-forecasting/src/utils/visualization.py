"""
可视化工具模块
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any


def setup_visualization_style():
    """
    设置可视化样式
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def plot_time_series(data: pd.DataFrame,
                    datetime_column: str,
                    value_column: str,
                    title: str = "时间序列图",
                    xlabel: str = "时间",
                    ylabel: str = "值",
                    figsize: tuple = (15, 8),
                    save_path: Optional[str] = None):
    """
    绘制时间序列图
    
    Args:
        data: 包含时间序列数据的DataFrame
        datetime_column: 时间列名
        value_column: 值列名
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        figsize: 图表大小
        save_path: 保存路径
    """
    setup_visualization_style()
    
    plt.figure(figsize=figsize)
    plt.plot(data[datetime_column], data[value_column], linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    
    # 自动旋转日期标签
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_multiple_time_series(data: pd.DataFrame,
                             datetime_column: str,
                             value_columns: List[str],
                             title: str = "多时间序列对比图",
                             xlabel: str = "时间",
                             ylabel: str = "值",
                             figsize: tuple = (15, 8),
                             save_path: Optional[str] = None):
    """
    绘制多个时间序列对比图
    
    Args:
        data: 包含时间序列数据的DataFrame
        datetime_column: 时间列名
        value_columns: 值列名列表
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        figsize: 图表大小
        save_path: 保存路径
    """
    setup_visualization_style()
    
    plt.figure(figsize=figsize)
    
    for column in value_columns:
        plt.plot(data[datetime_column], data[column], label=column, linewidth=1)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 自动旋转日期标签
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_matrix(data: pd.DataFrame,
                           title: str = "相关性矩阵热力图",
                           figsize: tuple = (12, 10),
                           save_path: Optional[str] = None):
    """
    绘制相关性矩阵热力图
    
    Args:
        data: 数据DataFrame
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
    """
    setup_visualization_style()
    
    # 计算相关性矩阵
    corr_matrix = data.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(feature_names: List[str],
                           importance_scores: np.ndarray,
                           title: str = "特征重要性",
                           top_n: Optional[int] = None,
                           figsize: tuple = (10, 8),
                           save_path: Optional[str] = None):
    """
    绘制特征重要性图
    
    Args:
        feature_names: 特征名称列表
        importance_scores: 特征重要性分数
        title: 图表标题
        top_n: 显示前N个重要特征
        figsize: 图表大小
        save_path: 保存路径
    """
    setup_visualization_style()
    
    # 创建DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    # 选择前N个特征
    if top_n:
        importance_df = importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
    plt.title(title)
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history: Dict[str, List[float]],
                         title: str = "训练历史",
                         figsize: tuple = (12, 6),
                         save_path: Optional[str] = None):
    """
    绘制训练历史图
    
    Args:
        history: 训练历史字典，包含'train_loss'和'val_loss'
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
    """
    setup_visualization_style()
    
    plt.figure(figsize=figsize)
    
    # 绘制训练损失
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='训练损失')
    
    # 绘制验证损失
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='验证损失')
    
    plt.title(title)
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数刻度
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_prediction_intervals(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             lower_bound: Optional[np.ndarray] = None,
                             upper_bound: Optional[np.ndarray] = None,
                             title: str = "预测结果与置信区间",
                             figsize: tuple = (15, 8),
                             save_path: Optional[str] = None):
    """
    绘制预测结果与置信区间
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        lower_bound: 下界（可选）
        upper_bound: 上界（可选）
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
    """
    setup_visualization_style()
    
    plt.figure(figsize=figsize)
    
    x = np.arange(len(y_true))
    
    # 绘制真实值
    plt.plot(x, y_true, label='真实值', alpha=0.7)
    
    # 绘制预测值
    plt.plot(x, y_pred, label='预测值', alpha=0.7)
    
    # 绘制置信区间
    if lower_bound is not None and upper_bound is not None:
        plt.fill_between(x, lower_bound, upper_bound, alpha=0.3, label='置信区间')
    
    plt.title(title)
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()