"""
实验基类
"""

import os
import sys
from abc import ABC, abstractmethod

# 添加项目根目录到Python路径
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)


class ExpBase(ABC):
    """
    实验基类，所有实验类都应该继承此类
    """

    def __init__(self, experiment_name=""):
        """
        初始化实验基类
        
        Args:
            experiment_name: 实验名称
        """
        self.experiment_name = experiment_name
        self.base_path = base_path
        self.experiments_dir = os.path.join(base_path, "experiments")
        self.code_dir = os.path.join(self.experiments_dir, "code")
        self.results_dir = os.path.join(self.experiments_dir, "results")
        
        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        
    @abstractmethod
    def run(self, *args, **kwargs):
        """
        运行实验的抽象方法，子类必须实现
        """
        pass

    def setup(self):
        """
        实验设置方法，可在run方法前调用
        """
        print(f"设置实验: {self.experiment_name}")
        
    def teardown(self):
        """
        实验结束后的清理方法，可在run方法后调用
        """
        print(f"实验 {self.experiment_name} 完成")