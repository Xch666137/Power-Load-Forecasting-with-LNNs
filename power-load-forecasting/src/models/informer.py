"""
Informer模型实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# 添加src目录到Python路径以支持相对导入
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)

# 使用相对导入
from ..attention.prob_attention import ProbAttention
from ..attention.attention_layer import AttentionLayer


class EncoderLayer(nn.Module):
    """
    编码器层
    """
    
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        初始化编码器层
        
        Args:
            attention: 注意力机制
            d_model: 模型维度
            d_ff: 前馈网络维度
            dropout: Dropout比率
            activation: 激活函数
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        """
        前向传播
        """
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class InformerEncoder(nn.Module):
    """
    Informer编码器
    """
    
    def __init__(self, enc_layers):
        """
        初始化Informer编码器
        
        Args:
            enc_layers: 编码器层列表
        """
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList(enc_layers)

    def forward(self, x, attn_mask=None):
        """
        前向传播
        """
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)

        return x, attns


class InformerModel(nn.Module):
    """
    Informer模型
    
    基于Informer的高效长序列时间序列预测模型
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
        初始化Informer模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度（必须能被num_heads整除）
            num_layers: 编码器层数
            num_heads: 注意力头数
            output_size: 输出维度
            dropout: Dropout比率
            sequence_length: 输入序列长度
        """
        super(InformerModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # 构建编码器层
        enc_layers = []
        for i in range(num_layers):
            attention = ProbAttention(mask_flag=False, factor=5, attention_dropout=dropout)
            enc_layers.append(EncoderLayer(
                AttentionLayer(attention, hidden_size, num_heads),
                hidden_size,
                dropout=dropout
            ))
        
        # 编码器
        self.encoder = InformerEncoder(enc_layers)
        
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
        
        # 编码器
        x, _ = self.encoder(x)
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        
        # 全连接层
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.unsqueeze(1)  # 增加一个维度以匹配输出格式


def create_informer_model(input_size: int,
                          hidden_size: int = 64,
                          num_layers: int = 2,
                          num_heads: int = 8,
                          output_size: int = 1,
                          dropout: float = 0.2,
                          sequence_length: int = 24) -> nn.Module:
    """
    创建Informer模型的便捷函数
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏层维度
        num_layers: 编码器层数
        num_heads: 注意力头数
        output_size: 输出维度
        dropout: Dropout比率
        sequence_length: 输入序列长度
        
    Returns:
        Informer模型
    """
    # 确保hidden_size能被num_heads整除
    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        
    return InformerModel(input_size, hidden_size, num_layers, num_heads, output_size, dropout, sequence_length)