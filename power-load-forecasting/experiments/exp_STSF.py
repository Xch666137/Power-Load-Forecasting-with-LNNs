"""
短时预测实验类
"""

import os
import sys

# 添加项目根目录到Python路径
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)

from exp_base import ExpBase


class ExpSTSF(ExpBase):
    """
    短时预测实验类 (Short-Term Forecasting Experiment)
    """

    def __init__(self):
        """
        初始化短时预测实验
        """
        super().__init__("Short-Term Load Forecasting")
        
    def run(self, config=None):
        """
        运行短时预测实验
        
        Args:
            config: 配置参数
        """
        print("开始运行短时电力负荷预测实验...")
        
        # 实验设置
        self.setup()
        
        try:
            # 1. 数据准备
            self._prepare_data(config)
            
            # 2. 模型训练
            model, history = self._train_model(config)
            
            # 3. 模型评估
            metrics, y_pred = self._evaluate_model(model)
            
            # 4. 结果可视化
            self._visualize_results(history, metrics, y_pred)
            
            # 5. 版本控制
            self._version_control(config)
            
        except Exception as e:
            print(f"实验运行出错: {e}")
            raise e
            
        finally:
            # 实验结束
            self.teardown()
            
    def _prepare_data(self, config):
        """
        准备数据
        
        Args:
            config: 配置参数
        """
        print("\n1-4. 数据加载和预处理...")
        # 数据加载和预处理将通过动态导入实现
        data_loader = self.import_from_code("data_loader")
        X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_and_preprocess_data(config)
        self.data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test
        }
        
    def _train_model(self, config):
        """
        训练模型
        
        Args:
            config: 配置参数
            
        Returns:
            trained_model: 训练好的模型
            history: 训练历史
        """
        print("\n5-6. 模型创建和训练...")
        # 动态导入train模块
        from experiments.code import train
        
        # 获取训练数据
        X_train = self.data["X_train"]
        y_train = self.data["y_train"]
        X_val = self.data["X_val"]
        y_val = self.data["y_val"]
        
        # 训练模型
        model, history = train.train_model_task(config, X_train, y_train, X_val, y_val)
        return model, history
        
    def _evaluate_model(self, model):
        """
        评估模型
        
        Args:
            model: 训练好的模型
            
        Returns:
            metrics: 评估指标
            y_pred: 预测结果
        """
        print("\n7. 模型评估...")
        # 动态导入evaluate模块
        from experiments.code import evaluate
        
        # 获取测试数据
        X_test = self.data["X_test"]
        y_test = self.data["y_test"]
        
        # 评估模型
        metrics, y_pred = evaluate.evaluate_model_task(model, X_test, y_test)
        return metrics, y_pred
        
    def _visualize_results(self, history, metrics, y_pred):
        """
        可视化结果
        
        Args:
            history: 训练历史
            metrics: 评估指标
            y_pred: 预测结果
        """
        print("\n8. 结果可视化...")
        # 动态导入visualize模块
        from experiments.code import visualize
        
        # 结果可视化
        visualize.visualize_results_task(history, y_test, y_pred)
        
    def _version_control(self, config):
        """
        版本控制
        
        Args:
            config: 配置参数
        """
        print("\n9. 实验版本控制...")
        from experiments.code import version_control
        version_id = version_control.ExperimentVersionControl().create_version(config, "短时预测实验")
        print(f"实验版本已创建: {version_id}")