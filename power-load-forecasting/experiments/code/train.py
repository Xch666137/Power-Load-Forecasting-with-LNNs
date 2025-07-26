"""
训练模块
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# 添加src目录到Python路径
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

# 设置实验结果保存路径
results_dir = os.path.join(base_path, "experiments", "results")

def train_model_task_with_dataloader(config, train_loader, val_loader):
    """
    使用数据加载器执行模型训练任务，提高GPU利用率
    
    Args:
        config: 配置参数
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        
    Returns:
        trained_model: 训练好的模型
        history: 训练历史记录
    """
    # 定义是否使用自动混合精度(AMP)，某些模型类型不支持AMP
    use_amp = config['model'].get('use_amp', False)
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 根据配置选择模型类型
    model_type = config['model'].get('type', 'liquid_ode')
    
    # 验证序列长度参数
    seq_len = config['data'].get('seq_len', 24)
    label_len = config['data'].get('label_len', 12)
    pred_len = config['data'].get('pred_len', 12)
    
    # 添加序列长度参数验证
    assert seq_len >= label_len + pred_len, \
        f"序列长度(seq_len={seq_len})必须大于等于label_len+pred_len={label_len+pred_len}"
    assert pred_len <= label_len, \
        f"pred_len({pred_len})不能大于label_len({label_len})"
    
    # 获取输入和输出维度
    sample_x, sample_y, _, _ = next(iter(train_loader))
    input_size = sample_x.shape[2]  # 特征维度
    output_size = sample_y.shape[2] if len(sample_y.shape) > 2 else sample_y.shape[1]  # 输出维度
    
    print(f"\n5. 创建 {model_type} 模型...")
    model = None
    
    if model_type == 'liquid_ode':
        from src.models.liquid_neural_network import create_liquid_model
        model = create_liquid_model(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            output_size=output_size,
            model_type=model_type
        )
    elif model_type == 'lstm':
        from src.models.lstm import create_lstm_model
        model = create_lstm_model(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model'].get('num_layers', 1),
            output_size=output_size,
            dropout=config['model'].get('dropout', 0.2),
            sequence_length=config['data'].get('sequence_length', 24)
        )
    elif model_type == 'liquid_lstm':
        from src.models.liquid_lstm import create_liquid_lstm_model
        model = create_liquid_lstm_model(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model'].get('num_layers', 1),
            output_size=output_size,
            dropout=config['model'].get('dropout', 0.2)
        )
    elif model_type == 'transformer':
        from src.models.transformer import create_transformer_model
        model = create_transformer_model(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            output_size=output_size,
            dropout=config['model'].get('dropout', 0.2),
            sequence_length=config['data'].get('sequence_length', 24)
        )
    elif model_type == 'informer':
        from src.models.informer import create_informer_model
        model = create_informer_model(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            output_size=output_size,
            dropout=config['model'].get('dropout', 0.2),
            sequence_length=config['data']['seq_len'],
            label_length=config['data']['label_len'],
            pred_length=config['data']['pred_len'],
            factor=config['model'].get('factor', 5)
        )
    elif model_type == 'lnn_informer':
        from src.models.lnn_informer import create_lnn_informer_model
        model = create_lnn_informer_model(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            liquid_hidden_size=config['model']['liquid_hidden_size'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            output_size=output_size,
            dropout=config['model'].get('dropout', 0.2),
            sequence_length=config['data']['seq_len'],
            label_length=config['data']['label_len'],
            pred_length=config['data']['pred_len']
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 将模型移动到指定设备
    model = model.to(device)
    print(f"模型结构: {model}")

    # 6. 训练模型
    print("\n6. 训练模型...")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    learning_rate = config['model'].get('learning_rate', 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练参数
    epochs = config['model'].get('epochs', 100)

    # 添加学习率调度器
    lr_scheduler_type = config['model'].get('lr_scheduler', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_scheduler_type == 'step':
        step_size = config['model'].get('lr_step_size', 30)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    elif lr_scheduler_type == 'exponential':
        gamma = config['model'].get('lr_gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError(f"不支持的学习率调度器类型: {lr_scheduler_type}")
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # 获取进度条设置
    disable_progress_bar = config.get('disable_progress_bar', False)

    # 启用cuDNN优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 检查是否应该使用AMP（自动混合精度）
    effective_use_amp = use_amp and torch.cuda.is_available()
    
    # 初始化GradScaler用于AMP
    scaler = torch.amp.GradScaler('cuda') if effective_use_amp else None
    if use_amp and effective_use_amp:
        print("启用自动混合精度训练 (AMP)")
    elif use_amp and not effective_use_amp:
        print("检测到液态神经网络模型，暂时禁用AMP以确保训练效率")
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_loader_with_progress = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] Training", leave=False) if not disable_progress_bar else train_loader
        for batch_x, batch_y, _, _ in train_loader_with_progress:
            # 将数据移动到指定设备并确保数据类型为Float
            batch_x = batch_x.to(device, non_blocking=True).float()
            batch_y = batch_y.to(device, non_blocking=True).float()
            
            optimizer.zero_grad()
            
            # 使用自动混合精度训练
            if effective_use_amp and scaler is not None:
                with torch.cuda.amp.autocast('cuda'):
                    outputs = model(batch_x)
                    
                    # 处理输出维度不匹配的情况
                    if len(outputs.shape) != len(batch_y.shape):
                        if len(outputs.shape) == 3 and len(batch_y.shape) == 2:
                            outputs = outputs.squeeze(1)
                    
                    # 在计算损失前，确保目标张量只包含需要预测的部分
                    pred_len = outputs.shape[1]
                    # 只取目标张量中需要预测的部分
                    batch_y = batch_y[:, -pred_len:, :]
                    
                    # 进一步处理维度不匹配问题，确保预测长度与目标一致
                    if outputs.shape != batch_y.shape:
                        # 如果输出是 (batch_size, seq_len, features) 而目标是 (batch_size, pred_len, features)
                        if len(outputs.shape) == 3 and len(batch_y.shape) == 3:
                            # 取输出的最后pred_len个时间步
                            pred_len = batch_y.shape[1]
                            if outputs.shape[1] != pred_len:
                                outputs = outputs[:, -pred_len:, :]
                        # 如果输出是 (batch_size, features) 而目标是 (batch_size, pred_len, features)
                        elif len(outputs.shape) == 2 and len(batch_y.shape) == 3:
                            # 添加时间维度并扩展
                            outputs = outputs.unsqueeze(1).repeat(1, batch_y.shape[1], 1)
                    
                    loss = criterion(outputs, batch_y)
                
                scaler.scale(loss).backward()
                
                # 添加梯度裁剪
                gradient_clipping = config['model'].get('gradient_clipping', None)
                if gradient_clipping is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x)
                
                # 处理输出维度不匹配的情况
                if len(outputs.shape) != len(batch_y.shape):
                    if len(outputs.shape) == 3 and len(batch_y.shape) == 2:
                        outputs = outputs.squeeze(1)
                
                # 在计算损失前，确保目标张量只包含需要预测的部分
                pred_len = outputs.shape[1]
                # 只取目标张量中需要预测的部分
                batch_y = batch_y[:, -pred_len:, :]
                
                # 进一步处理维度不匹配问题，确保预测长度与目标一致
                if outputs.shape != batch_y.shape:
                    # 如果输出是 (batch_size, seq_len, features) 而目标是 (batch_size, pred_len, features)
                    if len(outputs.shape) == 3 and len(batch_y.shape) == 3:
                        # 取输出的最后pred_len个时间步
                        pred_len = batch_y.shape[1]
                        if outputs.shape[1] != pred_len:
                            outputs = outputs[:, -pred_len:, :]
                    # 如果输出是 (batch_size, features) 而目标是 (batch_size, pred_len, features)
                    elif len(outputs.shape) == 2 and len(batch_y.shape) == 3:
                        # 添加时间维度并扩展
                        outputs = outputs.unsqueeze(1).repeat(1, batch_y.shape[1], 1)
                
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                
                # 添加梯度裁剪
                gradient_clipping = config['model'].get('gradient_clipping', None)
                if gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                
                optimizer.step()
            
            train_loss += loss.item()

            # 更新进度条
            if not disable_progress_bar:
                train_loader_with_progress.set_postfix({'Batch Loss': loss.item()})

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch_x, batch_y, _, _ in val_loader:
                # 将数据移动到指定设备并确保数据类型为Float
                batch_x = batch_x.to(device, non_blocking=True).float()
                batch_y = batch_y.to(device, non_blocking=True).float()
                
                # 删除AMP相关代码
                val_outputs = model(batch_x)
                batch_y = batch_y[:, -pred_len:, :]
                # 处理输出维度不匹配的情况
                if len(val_outputs.shape) != len(batch_y.shape):
                    if len(val_outputs.shape) == 3 and len(batch_y.shape) == 2:
                        val_outputs = val_outputs.squeeze(1)
                
                # 进一步处理维度不匹配问题
                if val_outputs.shape != batch_y.shape:
                    # 如果输出是 (batch_size, seq_len, features) 而目标是 (batch_size, pred_len, features)
                    if len(val_outputs.shape) == 3 and len(batch_y.shape) == 3:
                        # 取输出的最后pred_len个时间步
                        pred_len = batch_y.shape[1]
                        if val_outputs.shape[1] != pred_len:
                            val_outputs = val_outputs[:, -pred_len:, :]
                    # 如果输出是 (batch_size, features) 而目标是 (batch_size, pred_len, features)
                    elif len(val_outputs.shape) == 2 and len(batch_y.shape) == 3:
                        # 添加时间维度并扩展
                        val_outputs = val_outputs.unsqueeze(1).repeat(1, batch_y.shape[1], 1)
                
                val_loss_batch = criterion(val_outputs, batch_y)
                
                val_loss += val_loss_batch.item()
                val_steps += 1
        
        # 记录损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / val_steps if val_steps > 0 else val_loss
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # 更新学习率调度器
        scheduler.step()

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        # 打印进度
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Learning Rate: {current_lr:.6f}")

    # 保存训练好的模型
    os.makedirs(results_dir, exist_ok=True)
    model_save_path = os.path.join(results_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_save_path)
    
    return model, history