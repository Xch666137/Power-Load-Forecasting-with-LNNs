"""
模型训练模块
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import logging


class ModelTrainer:
    """
    模型训练器
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        初始化模型训练器
        
        Args:
            model: 待训练的模型
            device: 训练设备（CPU或GPU）
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 训练历史记录
        self.train_history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 损失函数和优化器
        self.criterion = None
        self.optimizer = None
        
        # 日志配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def compile(self, 
                optimizer: str = 'adam',
                loss: str = 'mse',
                learning_rate: float = 0.001,
                **kwargs):
        """
        编译模型（配置损失函数和优化器）
        
        Args:
            optimizer: 优化器类型
            loss: 损失函数类型
            learning_rate: 学习率
            **kwargs: 其他参数
        """
        # 设置损失函数
        if loss.lower() == 'mse':
            self.criterion = nn.MSELoss()
        elif loss.lower() == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
        
        # 设置优化器
        if optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, **kwargs)
        elif optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, **kwargs)
        elif optimizer.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2,
              shuffle: bool = True,
              verbose: int = 1) -> Dict[str, list]:
        """
        训练模型
        
        Args:
            X_train: 训练特征数据
            y_train: 训练标签数据
            X_val: 验证特征数据
            y_val: 验证标签数据
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例
            shuffle: 是否打乱数据
            verbose: 日志详细程度
            
        Returns:
            训练历史记录
        """
        # 数据转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # 处理验证数据
        if X_val is None or y_val is None:
            # 拆分训练集和验证集
            X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
                X_train_tensor, y_train_tensor, 
                test_size=validation_split, 
                shuffle=shuffle,
                random_state=42
            )
        else:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        
        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(batch_X)
                outputs = outputs.squeeze()  # 移除多余的维度
                
                # 计算损失
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                
                # 参数更新
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            avg_train_loss = train_loss / len(train_loader)
            self.train_history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_outputs = val_outputs.squeeze()
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
            
            self.train_history['val_loss'].append(val_loss)
            
            # 打印训练进度
            if verbose > 0 and (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch [{epoch+1}/{epochs}], "
                               f"Train Loss: {avg_train_loss:.4f}, "
                               f"Val Loss: {val_loss:.4f}")
        
        return self.train_history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X_test: 测试特征数据
            y_test: 测试标签数据
            
        Returns:
            评估结果
        """
        self.model.eval()
        
        # 转换为PyTorch张量
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        with torch.no_grad():
            # 预测
            predictions = self.model(X_test_tensor)
            predictions = predictions.squeeze()
            
            # 计算损失
            mse = nn.MSELoss()(predictions, y_test_tensor).item()
            mae = nn.L1Loss()(predictions, y_test_tensor).item()
            
            # 计算RMSE
            rmse = np.sqrt(mse)
            
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X: 输入特征数据
            
        Returns:
            预测结果
        """
        self.model.eval()
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            # 预测
            predictions = self.model(X_tensor)
            predictions = predictions.squeeze()
            
            # 转换为NumPy数组并移至CPU
            return predictions.cpu().numpy()


def train_model(model: nn.Module,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None,
                epochs: int = 100,
                batch_size: int = 32,
                learning_rate: float = 0.001,
                device: torch.device = None) -> Tuple[nn.Module, Dict[str, list]]:
    """
    训练模型的便捷函数
    
    Args:
        model: 待训练的模型
        X_train: 训练特征数据
        y_train: 训练标签数据
        X_val: 验证特征数据
        y_val: 验证标签数据
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        device: 训练设备
        
    Returns:
        训练好的模型和训练历史记录
    """
    # 创建训练器
    trainer = ModelTrainer(model, device)
    
    # 编译模型
    trainer.compile(optimizer='adam', loss='mse', learning_rate=learning_rate)
    
    # 训练模型
    history = trainer.train(X_train, y_train, X_val, y_val, epochs, batch_size)
    
    return trainer.model, history