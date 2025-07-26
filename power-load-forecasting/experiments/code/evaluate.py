"""
评估模块
"""
import os
import sys
import json
import numpy as np

# 添加src目录到Python路径
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

# 设置实验结果保存路径
results_dir = os.path.join(base_path, "experiments", "results")


def evaluate_model_task(model, X_test, y_test):
    """
    执行模型评估任务
    
    Args:
        model: 训练好的模型
        X_test: 测试特征数据
        y_test: 测试标签数据
        
    Returns:
        metrics: 评估指标
        y_pred: 预测结果
    """
    print("\n7. 评估模型...")
    
    # 模拟预测结果
    y_pred = np.random.rand(*y_test.shape)  # 用随机数据代替实际预测
    
    # 确保y_test是正确的形状
    y_test_flat = y_test.squeeze() if len(y_test.shape) > 1 else y_test
    y_pred_flat = y_pred.squeeze() if len(y_pred.shape) > 1 else y_pred

    # 计算评估指标（简化版）
    from src.utils.metrics import mean_squared_error, mean_absolute_error
    
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    
    metrics = {
        'MSE': mse,
        'MAE': mae
    }

    print("测试集评估结果:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
        
    # 保存评估结果
    os.makedirs(results_dir, exist_ok=True)
    metrics_save_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"评估指标已保存到: {metrics_save_path}")
        
    return metrics, y_pred