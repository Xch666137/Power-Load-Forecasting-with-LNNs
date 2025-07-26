"""
基础注意力机制模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class BaseAttention(nn.Module):
    """
    基础注意力机制类
    """
    
    def __init__(self):
        """
        初始化基础注意力机制
        """
        super(BaseAttention, self).__init__()
        
    def forward(self, queries, keys, values, attn_mask=None):
        """
        前向传播
        
        Args:
            queries: 查询张量
            keys: 键张量
            values: 值张量
            attn_mask: 注意力掩码
            
        Returns:
            注意力输出和注意力权重
        """
        raise NotImplementedError("This method should be implemented by subclasses")


class ScaledDotProductAttention(BaseAttention):
    """
    缩放点积注意力机制
    """
    
    def __init__(self, scale=None, attention_dropout=0.1):
        """
        初始化缩放点积注意力机制
        
        Args:
            scale: 缩放因子
            attention_dropout: 注意力dropout比率
        """
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask=None):
        """
        前向传播
        
        Args:
            queries: 查询张量 [B, L_Q, H, D]
            keys: 键张量 [B, L_K, H, D]
            values: 值张量 [B, L_V, H, D]
            attn_mask: 注意力掩码
            
        Returns:
            注意力输出和注意力权重
        """
        # 计算QK^T
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        # 缩放
        if self.scale is not None:
            scores = scores * self.scale
        
        # 应用掩码
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
            
        # 计算注意力权重
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 计算输出
        context = torch.matmul(attn, values)
        
        return context, attn