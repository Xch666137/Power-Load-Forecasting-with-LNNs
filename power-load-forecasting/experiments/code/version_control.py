"""
实验版本控制模块
"""
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path


class ExperimentVersionControl:
    """
    实验版本控制类
    """
    
    def __init__(self, experiment_dir="experiments"):
        """
        初始化版本控制
        
        Args:
            experiment_dir: 实验目录路径
        """
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.experiment_dir = os.path.join(self.base_path, experiment_dir)
        self.code_dir = os.path.join(self.experiment_dir, "code")
        self.version_file = os.path.join(self.experiment_dir, "experiment_versions.json")
        self._ensure_version_file()
    
    def _ensure_version_file(self):
        """
        确保版本文件存在
        """
        if not os.path.exists(self.version_file):
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    
    def _get_file_hash(self, file_path):
        """
        计算文件的哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件的哈希值
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return None
    
    def _get_config_hash(self, config):
        """
        计算配置的哈希值
        
        Args:
            config: 配置字典
            
        Returns:
            配置的哈希值
        """
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()
    
    def create_version(self, config, description=""):
        """
        创建新版本
        
        Args:
            config: 实验配置
            description: 版本描述
            
        Returns:
            version_id: 版本ID
        """
        # 获取当前时间戳作为版本ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v_{timestamp}"
        
        # 计算关键文件的哈希值
        file_hashes = {}
        experiment_files = [
            "train.py",
            "evaluate.py", 
            "visualize.py",
            "version_control.py"
        ]
        
        for file_name in experiment_files:
            file_path = os.path.join(self.code_dir, file_name)
            if os.path.exists(file_path):
                file_hashes[file_name] = self._get_file_hash(file_path)
        
        # 计算配置哈希
        config_hash = self._get_config_hash(config)
        
        # 构建版本信息
        version_info = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "config_hash": config_hash,
            "file_hashes": file_hashes,
            "config": config
        }
        
        # 读取现有版本信息
        with open(self.version_file, 'r', encoding='utf-8') as f:
            versions = json.load(f)
        
        # 添加新版本
        versions[version_id] = version_info
        
        # 保存更新后的版本信息
        with open(self.version_file, 'w', encoding='utf-8') as f:
            json.dump(versions, f, ensure_ascii=False, indent=2)
        
        print(f"创建实验版本: {version_id}")
        return version_id
    
    def list_versions(self):
        """
        列出所有版本
        
        Returns:
            versions: 版本列表
        """
        with open(self.version_file, 'r', encoding='utf-8') as f:
            versions = json.load(f)
        return versions
    
    def get_version(self, version_id):
        """
        获取特定版本信息
        
        Args:
            version_id: 版本ID
            
        Returns:
            version_info: 版本信息
        """
        with open(self.version_file, 'r', encoding='utf-8') as f:
            versions = json.load(f)
        
        return versions.get(version_id, None)
    
    def compare_versions(self, version_id1, version_id2):
        """
        比较两个版本
        
        Args:
            version_id1: 第一个版本ID
            version_id2: 第二个版本ID
            
        Returns:
            comparison: 比较结果
        """
        version1 = self.get_version(version_id1)
        version2 = self.get_version(version_id2)
        
        if not version1 or not version2:
            return None
        
        comparison = {
            "version1": version_id1,
            "version2": version_id2,
            "config_diff": version1["config"] != version2["config"],
            "file_diffs": {}
        }
        
        # 比较文件哈希
        all_files = set(version1["file_hashes"].keys()) | set(version2["file_hashes"].keys())
        for file_name in all_files:
            hash1 = version1["file_hashes"].get(file_name)
            hash2 = version2["file_hashes"].get(file_name)
            comparison["file_diffs"][file_name] = hash1 != hash2
        
        return comparison