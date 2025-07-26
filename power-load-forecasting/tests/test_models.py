"""
模型测试
"""
import unittest
import torch
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.liquid_neural_network import LiquidNeuralNetwork
from src.models.liquid_lstm import LiquidLSTM


class TestLiquidNeuralNetwork(unittest.TestCase):
    
    def setUp(self):
        """测试前的准备工作"""
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 1
        self.batch_size = 32
        self.sequence_length = 24
        
        # 创建模型
        self.model = LiquidNeuralNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        )
    
    def test_model_initialization(self):
        """测试模型初始化"""
        # 检查模型类型
        self.assertIsInstance(self.model, LiquidNeuralNetwork)
        
        # 检查模型参数
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.output_size, self.output_size)
    
    def test_forward_pass(self):
        """测试前向传播"""
        # 创建随机输入数据
        x = torch.randn(self.batch_size, self.sequence_length, self.input_size)
        
        # 执行前向传播
        output = self.model(x)
        
        # 检查输出形状
        expected_shape = (self.batch_size, self.sequence_length, self.output_size)
        self.assertEqual(output.shape, expected_shape)
    
    def test_liquid_ode_func(self):
        """测试液态ODE函数"""
        from src.models.liquid_neural_network import LiquidODEFunc
        
        # 创建ODE函数
        ode_func = LiquidODEFunc(self.hidden_size)
        
        # 创建测试输入
        t = torch.tensor(0.0)
        h = torch.randn(self.batch_size, self.hidden_size)
        ode_func.input = torch.randn(self.batch_size, self.hidden_size)
        
        # 执行前向传播
        dhdt = ode_func(t, h)
        
        # 检查输出形状
        self.assertEqual(dhdt.shape, h.shape)


class TestLiquidLSTM(unittest.TestCase):
    
    def setUp(self):
        """测试前的准备工作"""
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 1
        self.num_layers = 2
        self.batch_size = 32
        self.sequence_length = 24
        
        # 创建模型
        self.model = LiquidLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size
        )
    
    def test_model_initialization(self):
        """测试模型初始化"""
        # 检查模型类型
        self.assertIsInstance(self.model, LiquidLSTM)
        
        # 检查模型参数
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.output_size, self.output_size)
    
    def test_forward_pass(self):
        """测试前向传播"""
        # 创建随机输入数据
        x = torch.randn(self.batch_size, self.sequence_length, self.input_size)
        
        # 执行前向传播
        output = self.model(x)
        
        # 检查输出形状
        expected_shape = (self.batch_size, 1, self.output_size)
        self.assertEqual(output.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()