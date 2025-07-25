"""
版本控制模块
"""
import os
import json
import uuid
from datetime import datetime

class ExperimentVersionControl:
    """
    实验版本控制类
    """
    def __init__(self, experiment_dir):
        """
        初始化版本控制
        
        Args:
            experiment_dir: 实验目录路径
        """
        self.experiment_dir = experiment_dir
        self.versions_dir = os.path.join(experiment_dir, "versions")
        
        # 创建版本目录（如果不存在）
        if not os.path.exists(self.versions_dir):
            os.makedirs(self.versions_dir)
    
    def create_version(self, config, description=""):
        """
        创建新版本
        
        Args:
            config: 当前配置
            description: 版本描述
            
        Returns:
            版本ID
        """
        # 生成唯一版本ID
        version_id = str(uuid.uuid4())
        
        # 创建版本信息
        version_info = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "description": description
        }
        
        # 保存版本信息
        version_file = os.path.join(self.versions_dir, f"{version_id}.json")
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, indent=2, ensure_ascii=False)
            
        return version_id
"""
实验运行主文件
"""
import os
import sys
import yaml
from datetime import datetime

# 添加src目录到Python路径
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)

# 动态导入模块
def import_from_src(module_name):
    """从src目录导入模块"""
    import importlib.util
    
    # 构建模块路径
    module_path = os.path.join(base_path, "src", *module_name.split(".")) + ".py"
    
    # 如果文件不存在，尝试目录结构（带__init__.py）
    if not os.path.exists(module_path):
        dir_path = os.path.join(base_path, "src", *module_name.split("."))
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            module_path = os.path.join(dir_path, "__init__.py")
        else:
            raise ImportError(f"模块 {module_name} 未找到")
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"无法加载模块 {module_name}")
        
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config or {}
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认配置")
        return {}
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        return {}


def main():
    """
    主函数
    """
    print("电力负荷预测系统 - 基于液态神经网络")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    
    # 创建实验版本
    try:
        # 导入版本控制模块
        version_control_module = import_from_src("experiments.version_control")
        ExperimentVersionControl = version_control_module.ExperimentVersionControl
        
        version_control = ExperimentVersionControl("experiments")
        version_id = version_control.create_version(config, "自动创建的实验版本")
        print(f"实验版本已创建: {version_id}")
    except Exception as e:
        print(f"创建实验版本时出错: {e}")
    
    # 默认配置
    default_config = {
        'data': {
            'dataset_type': 'custom',  # 数据集类型: custom, ETTh1, ETTh2, ETTm1, ETTm2
            'data_path': None,         # 数据文件路径
            'sequence_length': 24,
            'forecast_horizon': 1
        },
        'model': {
            'type': 'liquid_ode',
            'hidden_size': 64,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        },
        'training': {
            'validation_split': 0.2,
            'shuffle': True
        }
    }
    
    # 合并配置
    for key in default_config:
        if key not in config:
            config[key] = default_config[key]
        else:
            for sub_key, sub_value in default_config[key].items():
                if sub_key not in config[key]:
                    config[key][sub_key] = sub_value
    
    # 1. 数据加载
    print("1. 加载数据...")
    # 导入数据加载模块
    data_loader_module = import_from_src("data.data_loader")
    PowerLoadDataLoader = data_loader_module.PowerLoadDataLoader
    
    data_loader = PowerLoadDataLoader(
        dataset_type=config['data']['dataset_type'],
        data_path=config['data']['data_path']
    )
    
    try:
        features, target = data_loader.load_data()
        print(f"特征数据形状: {features.shape}")
        print(f"目标数据形状: {target.shape}")
        
        # 显示数据集信息
        dataset_info = data_loader.get_dataset_info()
        print(f"数据集类型: {dataset_info['dataset_name']}")
        print(f"数据集大小: {dataset_info['shape']}")
        print(f"目标列: {dataset_info['target_column']}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    # 导入数据预处理模块
    preprocessing_module = import_from_src("data.preprocessing")
    PowerLoadPreprocessor = preprocessing_module.PowerLoadPreprocessor
    
    preprocessor = PowerLoadPreprocessor()
    
    # 合并特征和目标数据用于预处理
    target_col = target.columns[0]
    raw_data = features.copy()
    raw_data[target_col] = target[target_col]
    
    # 创建时间特征
    processed_data = preprocessor.create_time_features(raw_data)
    
    # 创建滞后特征
    processed_data = preprocessor.create_lag_features(processed_data, target_column=target_col)
    
    # 创建滚动统计特征
    processed_data = preprocessor.create_rolling_features(processed_data, target_column=target_col)
    
    # 删除包含NaN的行
    processed_data = processed_data.dropna().reset_index(drop=True)
    print(f"处理后数据形状: {processed_data.shape}")
    
    # 3. 准备训练数据
    print("\n3. 准备训练数据...")
    X, y = preprocessor.prepare_sequences(
        processed_data,
        sequence_length=config['data']['sequence_length'],
        forecast_horizon=config['data']['forecast_horizon']
    )
    
    print(f"特征数据形状: {X.shape}")
    print(f"标签数据形状: {y.shape}")
    
    # 4. 数据集划分
    print("\n4. 划分数据集...")
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 5. 执行训练任务
    # 动态导入实验模块
    train_module = import_from_src("experiments.train")
    model, history = train_module.train_model_task(config, X_train, y_train, X_val, y_val)
    
    # 6. 执行评估任务
    evaluate_module = import_from_src("experiments.evaluate")
    metrics, y_pred = evaluate_module.evaluate_model_task(model, X_test, y_test)
    
    # 7. 执行可视化任务
    visualize_module = import_from_src("experiments.visualize")
    y_test_flat = y_test.squeeze() if len(y_test.shape) > 1 else y_test
    visualize_module.visualize_results_task(history, y_test_flat, y_pred)
    
    print("\n程序执行完成!")


if __name__ == "__main__":
    main()