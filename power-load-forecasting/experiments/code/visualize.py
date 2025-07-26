"""
可视化模块
"""
import os
import sys

# 添加src目录到Python路径
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

# 设置实验结果保存路径
results_dir = os.path.join(base_path, "experiments", "results")


def visualize_results_task(history, y_test, y_pred):
    """
    执行结果可视化任务
    
    Args:
        history: 训练历史记录
        y_test: 真实值
        y_pred: 预测值
    """
    print("\n8. 生成可视化结果...")
    
    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 绘制训练历史
    try:
        from src.utils.visualization import plot_training_history
        plot_training_history(history, title="模型训练历史", save_path=os.path.join(results_dir, "training_history.png"))
    except Exception as e:
        print(f"绘制训练历史时出错: {e}")
        print("使用模拟数据生成示例图...")
        # 可以在这里添加简单的可视化逻辑
    
    # 绘制预测结果（简化版）
    try:
        from src.utils.visualization import plot_predictions
        plot_predictions(y_test, y_pred, title="电力负荷预测结果", save_path=os.path.join(results_dir, "prediction_results.png"))
    except Exception as e:
        print(f"绘制预测结果时出错: {e}")
    
    print(f"\n可视化结果已保存到: {results_dir}")