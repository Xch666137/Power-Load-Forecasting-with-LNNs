"""
可视化模块
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


def visualize_results_task(history, y_test, y_pred):
    """
    执行结果可视化任务
    
    Args:
        history: 训练历史记录
        y_test: 真实值
        y_pred: 预测值
    """
    print("\n8. 生成可视化结果...")
    
    # 绘制训练历史
    visualization_module = import_from_src("utils.visualization")
    plot_training_history = visualization_module.plot_training_history
    plot_training_history(history, title="模型训练历史")
    
    # 绘制预测结果
    evaluation_module = import_from_src("models.evaluation")
    ModelEvaluator = evaluation_module.ModelEvaluator
    
    evaluator = ModelEvaluator()
    evaluator.plot_predictions(y_test, y_pred, title="电力负荷预测结果")
    
    # 绘制误差分布
    evaluator.plot_error_distribution(y_test, y_pred, title="预测误差分布")
    
    print("\n可视化完成!")