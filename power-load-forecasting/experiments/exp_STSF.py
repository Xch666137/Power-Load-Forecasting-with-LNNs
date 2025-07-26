"""
短时预测实验类
"""

import sys
import os
import numpy as np

# 标准库导入
from sklearn.model_selection import train_test_split

# 项目特定导入
from .exp_base import ExpBase
from .code import train
from .code import evaluate
from .code import visualize

# 使用标准导入
from src.data.data_loader import load_power_data
from src.data.preprocessing import preprocess_power_data, PowerLoadPreprocessor


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

        
        # 加载数据
        features, target = load_power_data()
        
        # 预处理数据
        # 注意：这里需要根据实际数据结构调整
        data = features.copy()
        # 假设target是单独的一列，我们需要将其添加到数据中
        if len(target.columns) > 0:
            target_col = target.columns[0]
            data[target_col] = target[target_col]
        
        processed_data = preprocess_power_data(data)
        
        # 准备序列数据
        preprocessor = PowerLoadPreprocessor()
        X, y = preprocessor.prepare_sequences(processed_data)
        
        # 分割数据集
        # 先分割出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 再从剩余数据中分割出训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
        
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

        
        # 获取测试数据
        X_test = self.data["X_test"]
        y_test = self.data["y_test"]
        
        # 结果可视化
        visualize.visualize_results_task(history, y_test, y_pred)
        
    def _version_control(self, config):
        """
        版本控制
        
        Args:
            config: 配置参数
        """
        print("\n9. 实验版本控制...")
        from .code import version_control
        version_id = version_control.ExperimentVersionControl().create_version(config, "短时预测实验")
        print(f"实验版本已创建: {version_id}")