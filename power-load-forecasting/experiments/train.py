"""
训练模块
"""
import os
import sys

# 添加src目录到Python路径
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)

# 动态导入模块
def import_from_src(module_name):
    """从src目录导入模块"""
    import importlib.util
    module_path = os.path.join(base_path, "src", *module_name.split(".")) + ".py"
    if not os.path.exists(module_path):
        # 尝试不带.py后缀的目录结构
        module_path = os.path.join(base_path, "src", *module_name.split("."))
        if os.path.exists(module_path) and os.path.isdir(module_path):
            module_path = os.path.join(module_path, "__init__.py")
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    print("\n5. 创建液态神经网络模型...")
    input_size = X_train.shape[2]  # 特征维度
    output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1  # 输出维度

    # 导入模型创建函数
    liquid_model_module = import_from_src("models.liquid_neural_network")
    create_liquid_model = liquid_model_module.create_liquid_model
    
    model = create_liquid_model(
        input_size=input_size,
        hidden_size=config['model']['hidden_size'],
        output_size=output_size,
        model_type=config['model']['type']
    )

    print(f"模型结构: {model}")

    # 6. 训练模型
    print("\n6. 训练模型...")
    # 导入训练模块
    training_module = import_from_src("models.training")
    ModelTrainer = training_module.ModelTrainer
    
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    trainer = ModelTrainer(model, device)
    trainer.compile(
        optimizer='adam',
        loss='mse',
        learning_rate=config['model']['learning_rate']
    )

    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=config['model']['epochs'],
        batch_size=config['model']['batch_size'],
        validation_split=config['training']['validation_split'],
        shuffle=config['training']['shuffle']
    )

    print("训练完成!")
    return trainer.model, history