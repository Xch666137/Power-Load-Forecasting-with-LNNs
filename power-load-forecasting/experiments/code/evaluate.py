"""
模型评估模块
"""
import torch
import torch.nn as nn
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
        metrics: 评估指标
        y_pred: 预测结果
    """
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"评估使用设备: {device}")
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 切换到评估模式
    model.eval()
    
    # 转换数据为PyTorch张量并移动到指定设备
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # 禁用梯度计算
    with torch.no_grad():
        # 模型预测
        y_pred_tensor = model(X_test_tensor)
        
        # 处理输出维度不匹配的情况
        if len(y_pred_tensor.shape) != len(y_test_tensor.shape):
            if len(y_pred_tensor.shape) == 3 and len(y_test_tensor.shape) == 2:
                y_pred_tensor = y_pred_tensor.squeeze(1)
        
        # 进一步处理维度不匹配问题
        if y_pred_tensor.shape != y_test_tensor.shape:
            # 如果输出是 (batch_size, seq_len, features) 而目标是 (batch_size, pred_len, features)
            if len(y_pred_tensor.shape) == 3 and len(y_test_tensor.shape) == 3:
                # 取输出的最后pred_len个时间步
                pred_len = y_test_tensor.shape[1]
                if y_pred_tensor.shape[1] != pred_len:
                    y_pred_tensor = y_pred_tensor[:, -pred_len:, :]
            # 如果输出是 (batch_size, features) 而目标是 (batch_size, pred_len, features)
            elif len(y_pred_tensor.shape) == 2 and len(y_test_tensor.shape) == 3:
                # 添加时间维度并扩展
                y_pred_tensor = y_pred_tensor.unsqueeze(1).repeat(1, y_test_tensor.shape[1], 1)
        
        # 计算损失
        criterion = nn.MSELoss()
        test_loss = criterion(y_pred_tensor, y_test_tensor)
        
        # 转换为numpy数组用于计算指标
        y_pred = y_pred_tensor.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()
        
        # 计算评估指标
        metrics = calculate_all_metrics(y_test_np, y_pred)
        metrics['test_loss'] = test_loss.item()
        
        print(f"\n测试集损失: {test_loss.item():.4f}")
        print("评估指标:")
        for metric, value in metrics.items():
            if metric != 'test_loss':
                print(f"  {metric}: {value:.4f}")
    
    return metrics, y_pred


def evaluate_model_task_with_dataloader(model, test_loader):
    """
    使用数据加载器执行模型评估任务，提高GPU利用率
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        
    Returns:
        metrics: 评估指标
        y_pred: 预测结果
    """
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"评估使用设备: {device}")
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 切换到评估模式
    model.eval()
    
    # 收集所有预测结果和真实标签
    all_predictions = []
    all_targets = []
    
    # 禁用梯度计算
    with torch.no_grad():
        # 模型预测
        for batch_x, batch_y, _, _ in test_loader:
            # 将数据移动到指定设备并确保数据类型为Float
            batch_x = batch_x.to(device, non_blocking=True).float()
            batch_y = batch_y.to(device, non_blocking=True).float()
            
            y_pred_tensor = model(batch_x)
            
            # 处理输出维度不匹配的情况
            if len(y_pred_tensor.shape) != len(batch_y.shape):
                if len(y_pred_tensor.shape) == 3 and len(batch_y.shape) == 2:
                    y_pred_tensor = y_pred_tensor.squeeze(1)
            
            # 进一步处理维度不匹配问题
            if y_pred_tensor.shape != batch_y.shape:
                # 如果输出是 (batch_size, seq_len, features) 而目标是 (batch_size, pred_len, features)
                if len(y_pred_tensor.shape) == 3 and len(batch_y.shape) == 3:
                    # 取输出的最后pred_len个时间步
                    pred_len = batch_y.shape[1]
                    if y_pred_tensor.shape[1] != pred_len:
                        y_pred_tensor = y_pred_tensor[:, -pred_len:, :]
                # 如果输出是 (batch_size, features) 而目标是 (batch_size, pred_len, features)
                elif len(y_pred_tensor.shape) == 2 and len(batch_y.shape) == 3:
                    # 添加时间维度并扩展
                    y_pred_tensor = y_pred_tensor.unsqueeze(1).repeat(1, batch_y.shape[1], 1)
            
            # 收集预测结果和真实标签
            all_predictions.append(y_pred_tensor.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
        
        # 合并所有批次的结果
        y_pred = np.concatenate(all_predictions, axis=0)
        y_test_np = np.concatenate(all_targets, axis=0)
        
        # 计算评估指标
        metrics = calculate_all_metrics(y_test_np, y_pred)
        
        print("评估指标:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return metrics, y_pred