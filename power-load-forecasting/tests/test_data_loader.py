import sys
import os

# 添加项目根目录到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_provider import data_provider, ETTDatasetInterface
import argparse


class Args:
    """模拟参数类"""
    def __init__(self):
        self.root_path = '../data/ETDataset'
        self.data = 'ETTh1'
        self.features = 'M'
        self.target = 'OT'
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        self.batch_size = 32
        self.num_workers = 0
        self.freq = 'h'


def test_data_interface():
    """测试数据接口功能"""
    print("测试ETT数据接口...")
    
    try:
        # 创建数据接口实例
        dataset_interface = ETTDatasetInterface(
            root_path='../data/ETDataset',
            data_path='ETT-small/ETTh1.csv',
            dataset_name='ETTh1'
        )
        
        # 加载数据
        features, target = dataset_interface.load_data()
        print(f"   特征数据形状: {features.shape}")
        print(f"   目标数据形状: {target.shape}")
        
        # 获取数据信息
        info = dataset_interface.get_data_info()
        print(f"   数据集信息: {info}")
        
        print("   数据接口测试成功!")
        
    except Exception as e:
        print(f"   数据接口测试失败: {e}")


def test_data_loading():
    """测试数据加载功能"""
    print("\n测试ETT数据加载器...")
    
    # 创建参数对象
    args = Args()
    
    # 测试训练数据集
    print("\n1. 测试训练数据集:")
    try:
        train_dataset, train_loader = data_provider(args, 'train')
        print(f"   训练集大小: {len(train_dataset)}")
        print(f"   训练批次数量: {len(train_loader)}")
        
        # 获取一个批次的数据
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            print(f"   输入数据形状: {batch_x.shape}")
            print(f"   输出数据形状: {batch_y.shape}")
            print(f"   输入时间标记形状: {batch_x_mark.shape}")
            print(f"   输出时间标记形状: {batch_y_mark.shape}")
            break
    except Exception as e:
        print(f"   训练集加载失败: {e}")
    
    # 测试验证数据集
    print("\n2. 测试验证数据集:")
    try:
        val_dataset, val_loader = data_provider(args, 'val')
        print(f"   验证集大小: {len(val_dataset)}")
        print(f"   验证批次数量: {len(val_loader)}")
        
        # 获取一个批次的数据
        for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
            print(f"   输入数据形状: {batch_x.shape}")
            print(f"   输出数据形状: {batch_y.shape}")
            break
    except Exception as e:
        print(f"   验证集加载失败: {e}")
    
    # 测试测试数据集
    print("\n3. 测试测试数据集:")
    try:
        test_dataset, test_loader = data_provider(args, 'test')
        print(f"   测试集大小: {len(test_dataset)}")
        print(f"   测试批次数量: {len(test_loader)}")
        
        # 获取一个批次的数据
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            print(f"   输入数据形状: {batch_x.shape}")
            print(f"   输出数据形状: {batch_y.shape}")
            break
    except Exception as e:
        print(f"   测试集加载失败: {e}")
    
    print("\n数据加载测试完成!")


if __name__ == "__main__":
    test_data_interface()
    test_data_loading()