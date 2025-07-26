"""
Multi-head Attention实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """
    Multi-head Attention实现
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        初始化Multi-head Attention
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout比率
        """
        super(MultiheadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, keys, values, attn_mask=None):
        """
        前向传播
        
        Args:
            queries: 查询张量 [B, L_Q, D]
            keys: 键张量 [B, L_K, D]
            values: 值张量 [B, L_V, D]
            attn_mask: 注意力掩码
            
        Returns:
            注意力输出和注意力权重
        """
        B, L_Q, _ = queries.shape
        _, L_K, _ = keys.shape
        
        # 线性变换并分头
        Q = self.w_q(queries).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L_Q, D_K]
        K = self.w_k(keys).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)      # [B, H, L_K, D_K]
        V = self.w_v(values).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)     # [B, H, L_V, D_K]
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, H, L_Q, L_K]
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)
            
        attn = F.softmax(scores, dim=-1)  # [B, H, L_Q, L_K]
        attn = self.dropout(attn)
        
        # 应用注意力权重
        context = torch.matmul(attn, V)  # [B, H, L_Q, D_K]
        
        # 合并头
        context = context.transpose(1, 2).contiguous().view(B, L_Q, self.d_model)  # [B, L_Q, D]
        
        # 输出线性变换
        output = self.w_o(context)
        
        return output, attn