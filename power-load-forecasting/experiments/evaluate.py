"""
评估模块
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
    # 在测试集上进行预测
    trainer = model  # 假设传入的是trainer对象
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        # 如果传入的是纯模型对象，则需要重新创建trainer
        training_module = import_from_src("models.training")
        ModelTrainer = training_module.ModelTrainer
        
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        temp_trainer = ModelTrainer(model, device)
        y_pred = temp_trainer.predict(X_test)
    
    # 确保y_test是正确的形状
    y_test_flat = y_test.squeeze() if len(y_test.shape) > 1 else y_test
    y_pred_flat = y_pred.squeeze() if len(y_pred.shape) > 1 else y_pred

    # 计算评估指标
    evaluation_module = import_from_src("models.evaluation")
    ModelEvaluator = evaluation_module.ModelEvaluator
    
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_test_flat, y_pred_flat)

    print("测试集评估结果:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
        
    return metrics, y_pred