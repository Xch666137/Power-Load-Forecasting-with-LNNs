"""
模型对比实验类
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 添加项目根目录到Python路径
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

# 项目特定导入
from .exp_base import ExpBase
from .code import train
from .code import evaluate
from src.data_provider.data_loader import load_power_data
from src.data_provider.preprocessing import preprocess_power_data, PowerLoadPreprocessor


class ExpModelComparison(ExpBase):
    """
    模型对比实验类，在ETT数据集上比较不同模型的性能
    """

    def __init__(self):
        """
        初始化模型对比实验
        """
        super().__init__("Model Comparison on ETT Dataset")
        self.models_config = {
            'LNN': {
                'type': 'liquid_ode',
                'hidden_size': 64
            },
            'LSTM': {
                'type': 'lstm',
                'hidden_size': 64,
                'num_layers': 2
            },
            'LiquidLSTM': {
                'type': 'liquid_lstm',
                'hidden_size': 64,
                'num_layers': 2
            },
            'Transformer': {
                'type': 'transformer',
                'hidden_size': 64,
                'num_layers': 2,
                'num_heads': 8
            },
            'Informer': {
                'type': 'informer',
                'hidden_size': 64,
                'num_layers': 2,
                'num_heads': 8
            },
            'LNN-Informer': {
                'type': 'lnn_informer',
                'hidden_size': 64,
                'liquid_hidden_size': 32,
                'num_layers': 2,
                'num_heads': 8
            }
        }
        
    def run(self, config=None):
        """
        运行模型对比实验
        
        Args:
            config: 配置参数
        """
        print("开始运行模型对比实验...")
        
        # 实验设置
        self.setup()
        
        try:
            # 1. 数据准备
            self._prepare_data(config)
            
            # 2. 为每个模型训练和评估
            results = {}
            for model_name, model_config in self.models_config.items():
                print(f"\n{'='*50}")
                print(f"训练和评估 {model_name} 模型")
                print(f"{'='*50}")
                
                # 更新配置
                temp_config = config.copy() if config else {}
                if 'model' not in temp_config:
                    temp_config['model'] = {}
                temp_config['model'].update(model_config)
                
                # 训练模型
                model, history = self._train_model(temp_config, model_name)
                
                # 评估模型
                metrics, y_pred = self._evaluate_model(model)
                
                # 保存结果
                results[model_name] = {
                    'metrics': metrics,
                    'history': history
                }
                
                print(f"{model_name} 模型评估结果:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
            
            # 3. 结果对比和可视化
            self._compare_results(results)
            
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

        # 加载数据 - 使用ETT数据集
        features, target = load_power_data(dataset_type="ETTh1")  # 默认使用ETTh1数据集
        
        # 预处理数据
        data = features.copy()
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
        
    def _train_model(self, config, model_name):
        """
        训练模型
        
        Args:
            config: 配置参数
            model_name: 模型名称
            
        Returns:
            trained_model: 训练好的模型
            history: 训练历史
        """
        print(f"\n训练 {model_name} 模型...")

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
        print("评估模型...")
        
        # 获取测试数据
        X_test = self.data["X_test"]
        y_test = self.data["y_test"]
        
        # 评估模型
        metrics, y_pred = evaluate.evaluate_model_task(model, X_test, y_test)
        return metrics, y_pred
        
    def _compare_results(self, results):
        """
        对比不同模型的结果
        
        Args:
            results: 各模型的评估结果
        """
        print("\n" + "="*60)
        print("模型对比结果")
        print("="*60)
        
        # 创建结果对比表
        comparison_data = []
        metrics_names = set()
        
        for model_name, result in results.items():
            row = {'Model': model_name}
            for metric, value in result['metrics'].items():
                row[metric] = value
                metrics_names.add(metric)
            comparison_data.append(row)
        
        # 显示结果表格
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # 保存结果到文件
        results_file = os.path.join(self.results_dir, "model_comparison_results.csv")
        df.to_csv(results_file, index=False)
        print(f"\n结果已保存到: {results_file}")
        
        # 可视化对比结果
        self._visualize_comparison(results)
        
    def _visualize_comparison(self, results):
        """
        可视化模型对比结果
        
        Args:
            results: 各模型的评估结果
        """
        print("\n可视化模型对比结果...")
        
        try:
            import matplotlib.pyplot as plt
            
            # 创建指标对比图
            metrics_names = list(set(metric for result in results.values() 
                                   for metric in result['metrics'].keys()))
            
            fig, axes = plt.subplots(len(metrics_names), 1, figsize=(10, 4*len(metrics_names)))
            if len(metrics_names) == 1:
                axes = [axes]
            
            model_names = list(results.keys())
            x_pos = np.arange(len(model_names))
            
            for i, metric in enumerate(metrics_names):
                values = [results[model]['metrics'].get(metric, 0) for model in model_names]
                axes[i].bar(x_pos, values, color=['skyblue', 'lightgreen', 'lightcoral', 'orange', 'gold', 'mediumpurple'])
                axes[i].set_xlabel('Models')
                axes[i].set_ylabel(metric)
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_xticks(x_pos)
                axes[i].set_xticklabels(model_names, rotation=45, ha='right')
                
                # 添加数值标签
                for j, v in enumerate(values):
                    axes[i].text(j, v + max(values) * 0.01, f'{v:.4f}', 
                                ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            comparison_plot_file = os.path.join(self.results_dir, "model_comparison.png")
            plt.savefig(comparison_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"对比图已保存到: {comparison_plot_file}")
            
        except ImportError:
            print("无法导入matplotlib，跳过可视化")
        except Exception as e:
            print(f"可视化过程中出错: {e}")