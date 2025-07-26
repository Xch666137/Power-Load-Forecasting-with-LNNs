"""
原始Transformer模型实现
"""
import torch
import torch.nn as nn
import math
import sys
import os

# 添加src目录到Python路径以支持相对导入
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)

# 使用相对导入
from src.attention.multihead_attention import MultiheadAttention


class PositionalEncoding(nn.Module):
    """
    位置编码层
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            添加位置编码的张量
        """
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """
    原始Transformer模型
    
    使用Transformer编码器处理时间序列数据，适用于电力负荷预测等任务
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 sequence_length: int = 24):
        """
        初始化Transformer模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度（必须能被num_heads整除）
            num_layers: Transformer编码器层数
            num_heads: 注意力头数
            output_size: 输出维度
            dropout: Dropout比率
            sequence_length: 输入序列长度
        """
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_size, sequence_length)
        
        # 使用自定义的MultiheadAttention替换nn.TransformerEncoderLayer中的注意力机制
        encoder_layers = []
        for _ in range(num_layers):
            # 创建编码器层，这里为了简化直接使用PyTorch的实现
            # 在实际应用中可以替换为使用我们自定义的注意力机制
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True
            )
            encoder_layers.append(encoder_layer)
        
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        
        # 输出层
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        
        # Dropout和激活函数
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, sequence_length, input_size)
            
        Returns:
            输出张量
        """
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        # x = self.pos_encoding(x)
        
        # Transformer编码器
        for layer in self.transformer_encoder:
            x = layer(x)
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        
        # 全连接层
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.unsqueeze(1)  # 增加一个维度以匹配输出格式


def create_transformer_model(input_size: int,
                             hidden_size: int = 64,
                             num_layers: int = 2,
                             num_heads: int = 8,
                             output_size: int = 1,
                             dropout: float = 0.2,
                             sequence_length: int = 24) -> nn.Module:
    """
    创建Transformer模型的便捷函数
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏层维度
        num_layers: Transformer编码器层数
        num_heads: 注意力头数
        output_size: 输出维度
        dropout: Dropout比率
        sequence_length: 输入序列长度
        
    Returns:
        Transformer模型
    """
    # 确保hidden_size能被num_heads整除
    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        
    return TransformerModel(input_size, hidden_size, num_layers, num_heads, output_size, dropout, sequence_length)