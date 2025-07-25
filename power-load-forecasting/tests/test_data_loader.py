"""
数据加载器测试
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.data_loader import PowerLoadDataLoader


class TestPowerLoadDataLoader(unittest.TestCase):
    
    def setUp(self):
        """测试前的准备工作"""
        self.loader = PowerLoadDataLoader()
    
    def test_generate_sample_data(self):
        """测试生成示例数据"""
        data = self.loader._generate_sample_data()
        
        # 检查返回的数据类型
        self.assertIsInstance(data, pd.DataFrame)
        
        # 检查必要的列
        self.assertIn('datetime', data.columns)
        self.assertIn('load', data.columns)
        
        # 检查数据形状
        self.assertGreater(len(data), 0)
        
        # 检查时间列是否为datetime类型
        self.assertIsInstance(data['datetime'].iloc[0], pd.Timestamp)
        
        # 检查负荷数据是否为数值型
        self.assertTrue(pd.api.types.is_numeric_dtype(data['load']))
    
    def test_load_data_without_path(self):
        """测试不提供路径时加载数据"""
        data = self.loader.load_data()
        
        # 检查返回的数据类型
        self.assertIsInstance(data, pd.DataFrame)
        
        # 检查必要的列
        self.assertIn('datetime', data.columns)
        self.assertIn('load', data.columns)
        
        # 检查数据不为空
        self.assertGreater(len(data), 0)
    
    def test_get_transformer_load_data(self):
        """测试获取变压器负荷数据"""
        transformer_id = "T001"
        data = self.loader.get_transformer_load_data(transformer_id)
        
        # 检查返回的数据类型
        self.assertIsInstance(data, pd.DataFrame)
        
        # 检查变压器ID列
        self.assertIn('transformer_id', data.columns)
        
        # 检查变压器ID值
        self.assertEqual(data['transformer_id'].iloc[0], transformer_id)


if __name__ == '__main__':
    unittest.main()