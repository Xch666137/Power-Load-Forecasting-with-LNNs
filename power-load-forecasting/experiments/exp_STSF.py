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
from src.data_provider.data_loader import data_provider


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

        # 创建一个模拟的args对象，用于传递给data_provider
        class Args:
            def __init__(self, config):
                data_config = config.get('data', {})
                self.root_path = data_config.get('root_path', './data/ETDataset/')
                self.data = data_config.get('data', 'ETTh1')
                self.features = data_config.get('features', 'M')
                self.target = data_config.get('target', 'OT')
                self.seq_len = data_config.get('seq_len', 96)
                self.label_len = data_config.get('label_len', 48)
                self.pred_len = data_config.get('pred_len', 24)
                self.freq = data_config.get('freq', 'h')
                self.batch_size = data_config.get('batch_size', 32)
                self.num_workers = data_config.get('num_workers', 4)  # 增加num_workers以提高数据加载效率
                self.pin_memory = data_config.get('pin_memory', True)  # 添加pin_memory以加速GPU数据传输
                self.drop_last = data_config.get('drop_last', True)  # 添加drop_last以提高batch处理效率

        args = Args(config)
        
        # 加载训练数据
        train_dataset, train_loader = data_provider(args, 'train')
        
        # 加载验证数据
        val_dataset, val_loader = data_provider(args, 'val')
        
        # 加载测试数据
        test_dataset, test_loader = data_provider(args, 'test')
        
        self.data = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset
        }
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
    def _train_model(self, config):
        """
        训练模型
        
        Args:
            config: 配置参数
            
        Returns:
            tuple: (训练好的模型, 训练历史)
        """
        print("\n2. 模型训练...")
        
        # 直接使用数据加载器而不是提取numpy数组
        train_loader = self.data["train_loader"]
        val_loader = self.data["val_loader"]
        
        # 模型训练逻辑将在这里实现
        model, history = train.train_model_task_with_dataloader(config, train_loader, val_loader)
        return model, history

    def _evaluate_model(self, model):
        """
        评估模型
        
        Args:
            model: 训练好的模型
            
        Returns:
            tuple: (评估指标, 预测结果)
        """
        print("\n3. 模型评估...")
        
        # 获取测试数据加载器
        test_loader = self.data["test_loader"]
        
        # 模型评估逻辑将在这里实现
        metrics, y_pred = evaluate.evaluate_model_task_with_dataloader(model, test_loader)
        return metrics, y_pred

    def _visualize_results(self, history, metrics, y_pred):
        """
        可视化结果
        
        Args:
            history: 训练历史
            metrics: 评估指标
            y_pred: 预测结果
        """
        print("\n4. 结果可视化...")
        
        # 结果可视化逻辑将在这里实现
        visualize.visualize_results_task(history, metrics, y_pred)

    def _version_control(self, config):
        """
        版本控制
        
        Args:
            config: 配置参数
        """
        print("\n5. 版本控制...")
        
        # 版本控制逻辑将在这里实现
        # 这里可以保存配置、模型权重、结果等