"""
训练模块
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np

# 添加src目录到Python路径
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

# 设置实验结果保存路径
results_dir = os.path.join(base_path, "experiments", "results")


def train_model_task(config, X_train, y_train, X_val, y_val):
    """
    执行模型训练任务
    
    Args:
        config: 配置参数
        X_train: 训练特征数据
        y_train: 训练标签数据
        X_val: 验证特征数据
        y_val: 验证标签数据
        
    Returns:
        trained_model: 训练好的模型
        history: 训练历史记录
    """
    # 根据配置选择模型类型
    model_type = config['model'].get('type', 'liquid_ode')
    
    print(f"\n5. 创建 {model_type} 模型...")
    input_size = X_train.shape[2]  # 特征维度
    output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1  # 输出维度
    
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
            sequence_length=config['data'].get('sequence_length', 24)
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
            sequence_length=config['data'].get('sequence_length', 24)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    print(f"模型结构: {model}")

    # 6. 训练模型
    print("\n6. 训练模型...")
    
    # 转换数据为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    batch_size = config['model'].get('batch_size', 32)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    learning_rate = config['model'].get('learning_rate', 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练参数
    epochs = config['model'].get('epochs', 100)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            # 处理输出维度不匹配的情况
            if len(outputs.shape) != len(batch_y.shape):
                if len(outputs.shape) == 3 and len(batch_y.shape) == 2:
                    outputs = outputs.squeeze(1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            # 处理输出维度不匹配的情况
            if len(val_outputs.shape) != len(y_val_tensor.shape):
                if len(val_outputs.shape) == 3 and len(y_val_tensor.shape) == 2:
                    val_outputs = val_outputs.squeeze(1)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        # 记录损失
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss.item())
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
    
    # 保存训练好的模型
    os.makedirs(results_dir, exist_ok=True)
    model_save_path = os.path.join(results_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到: {model_save_path}")
    
    print("训练完成!")
    return model, history