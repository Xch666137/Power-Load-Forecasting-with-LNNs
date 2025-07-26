"""
数据加载器测试模块
"""

import unittest
import numpy as np
import pandas as pd
import os
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from src.data_provider.data_loader import PowerLoadDataLoader
from src.data_provider.data_interface import ETTHDataset, ETTmDataset, CustomDataset
from src.data_provider.data_factory import DatasetFactory


class TestPowerLoadDataLoader(unittest.TestCase):
    """测试电力负荷数据加载器"""
    
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
    
    @patch("pandas.read_csv")
    def test_load_custom_data_without_path(self, mock_read_csv):
        """测试不提供路径时加载自定义数据"""
        # 模拟pandas.read_csv返回示例数据
        sample_data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=10, freq='H'),
            'load': np.random.rand(10) * 1000
        })
        mock_read_csv.return_value = sample_data
        
        # 创建一个临时文件路径用于测试
        with patch('src.data_provider.data_interface.os.path.exists', return_value=True):
            loader = PowerLoadDataLoader('custom', data_path='fake_path.csv')
            features, target = loader.load_data()
            
            # 检查返回的数据类型
            self.assertIsInstance(features, pd.DataFrame)
            self.assertIsInstance(target, pd.DataFrame)
            
            # 检查数据不为空
            self.assertGreater(len(features), 0)
            self.assertGreater(len(target), 0)
    
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


    def test_dataset_factory(self):
        """测试数据集工厂"""
        # 测试创建ETTH数据集
        etth_dataset = DatasetFactory.create_dataset('ETTh1', data_path='fake_path.csv')
        self.assertIsInstance(etth_dataset, ETTHDataset)
        
        # 测试创建ETTm数据集
        ettm_dataset = DatasetFactory.create_dataset('ETTm1', data_path='fake_path.csv')
        self.assertIsInstance(ettm_dataset, ETTmDataset)
        
        # 测试创建自定义数据集
        custom_dataset = DatasetFactory.create_dataset('custom', data_path='fake_path.csv')
        self.assertIsInstance(custom_dataset, CustomDataset)
        
        # 测试不支持的数据集类型
        with self.assertRaises(ValueError):
            DatasetFactory.create_dataset('unsupported_type')
            
            
class TestDatasetInterfaces(unittest.TestCase):
    def setUp(self):
        self.mock_data = pd.DataFrame({
            'HUFL': [5.827, 5.8124],
            'HULL': [2.0096, 2.0049],
            'MUFL': [1.5997, 1.5851],
            'MULL': [0.6534, 0.6467],
            'LUFL': [2.7506, 2.7401],
            'LULL': [1.0583, 1.0521],
            'OT': [25.827, 25.8124]
        })

    def test_etth_dataset(self):
        """测试ETTH数据集"""
        with patch('src.data_provider.data_interface.pd.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.mock_data
            dataset = ETTHDataset('fake_path.csv', size=[24, 0, 1], features='S', data_path='ETTh1.csv')
            self.assertIsInstance(dataset, ETTHDataset)

    def test_ettm_dataset(self):
        """测试ETTm数据集"""
        with patch('src.data_provider.data_interface.pd.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.mock_data
            dataset = ETTmDataset('fake_path.csv', size=[24, 0, 1], features='S', data_path='ETTm1.csv')
            self.assertIsInstance(dataset, ETTmDataset)

    def test_custom_dataset(self):
        """测试自定义数据集"""
        with patch('src.data_provider.data_interface.pd.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.mock_data
            dataset = CustomDataset('fake_path.csv', size=[24, 0, 1], features='S')
            self.assertIsInstance(dataset, CustomDataset)
            
if __name__ == '__main__':
    unittest.main()