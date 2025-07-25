"""
数据加载器测试
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, mock_open

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.data_loader import PowerLoadDataLoader
from src.data.data_interface import ETTHDataset, ETTmDataset, CustomDataset
from src.data.data_factory import DatasetFactory


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
        with patch('src.data.data_interface.os.path.exists', return_value=True):
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
    
    def test_etth_dataset(self):
        """测试ETTH数据集接口"""
        # 模拟ETT数据
        mock_data = '''date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.0096,1.5997,0.6534,2.7506,1.0583,25.827
2016-07-01 01:00:00,5.8124,2.0049,1.5851,0.6467,2.7401,1.0521,25.8124'''
        
        with patch("builtins.open", mock_open(read_data=mock_data)):
            with patch('src.data.data_interface.os.path.exists', return_value=True):
                dataset = ETTHDataset(data_path='fake_etth.csv', dataset_name='ETTh1')
                features, target = dataset.load_data()
                
                # 检查返回的数据类型
                self.assertIsInstance(features, pd.DataFrame)
                self.assertIsInstance(target, pd.DataFrame)
                
                # 检查必要的列
                self.assertIn('OT', target.columns)  # OT是目标列
                
                # 检查数据集信息
                info = dataset.get_data_info()
                self.assertEqual(info['dataset_name'], 'ETTh1')
                self.assertIn('feature_columns', info)
                self.assertIn('target_column', info)
    
    def test_ettm_dataset(self):
        """测试ETTm数据集接口"""
        # 模拟ETT数据
        mock_data = '''date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.0096,1.5997,0.6534,2.7506,1.0583,25.827
2016-07-01 00:15:00,5.8124,2.0049,1.5851,0.6467,2.7401,1.0521,25.8124'''
        
        with patch("builtins.open", mock_open(read_data=mock_data)):
            with patch('src.data.data_interface.os.path.exists', return_value=True):
                dataset = ETTmDataset(data_path='fake_ettm.csv', dataset_name='ETTm1')
                features, target = dataset.load_data()
                
                # 检查返回的数据类型
                self.assertIsInstance(features, pd.DataFrame)
                self.assertIsInstance(target, pd.DataFrame)
                
                # 检查必要的列
                self.assertIn('OT', target.columns)  # OT是目标列
                
                # 检查数据集信息
                info = dataset.get_data_info()
                self.assertEqual(info['dataset_name'], 'ETTm1')
                self.assertIn('feature_columns', info)
                self.assertIn('target_column', info)
                
if __name__ == '__main__':
    unittest.main()