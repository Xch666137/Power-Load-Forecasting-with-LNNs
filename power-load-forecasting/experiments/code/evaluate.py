"""
评估模块
"""

import numpy as np
from src.utils.metrics import calculate_all_metrics


def evaluate_model_task(model, X_test, y_test):
    """
    执行模型评估任务
    
    Args:
        model: 训练好的模型
        X_test: 测试特征数据
        y_test: 测试标签数据
        
    Returns:
        metrics: 评估指标字典
        y_pred: 预测结果
    """
    print("\n7. 模型评估...")
    
    # 模型预测
    # 注意：不同模型的输入要求可能不同，这里假设所有模型都接受相同格式的输入
    try:
        y_pred = model(X_test)
        # 如果输出是(批次, 1, 特征)形状，则压缩中间维度
        if len(y_pred.shape) == 3 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(1)
    except Exception as e:
        print(f"预测时出错: {e}")
        # 尝试另一种方式
        try:
            y_pred = model.predict(X_test)
        except:
            raise RuntimeError("无法使用模型进行预测")
    
    # 确保y_test和y_pred形状一致
    if len(y_test.shape) != len(y_pred.shape):
        if len(y_test.shape) == 2 and len(y_pred.shape) == 3:
            y_pred = y_pred.squeeze(1)
        elif len(y_test.shape) == 3 and len(y_pred.shape) == 2:
            y_pred = y_pred.unsqueeze(1)
    
    # 计算评估指标
    metrics = calculate_all_metrics(y_test, y_pred)
    
    print("评估完成!")
    return metrics, y_pred