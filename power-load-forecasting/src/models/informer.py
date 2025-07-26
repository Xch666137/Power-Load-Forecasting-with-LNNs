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


class PositionalEmbedding(nn.Module):
    """
    位置编码
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播
        修改位置编码的设备处理，确保与输入数据在同一个设备上
        """
        # 获取输入张量的设备信息
        device = x.device
        
        # 如果位置编码不在正确的设备上，则移动它
        if self.pe.device != device:
            self.pe = self.pe.to(device)
            
        # 获取输入张量的序列长度
        seq_len = x.size(1)
        
        # 返回相应长度的位置编码
        return self.pe[:, :seq_len]


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


class DecoderLayer(nn.Module):
    """
    解码器层
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class InformerDecoder(nn.Module):
    """
    Informer解码器
    """
    def __init__(self, layers, norm_layer=None):
        super(InformerDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, enc_out):
        # 添加输入维度验证
        assert x.dim() == 3, f"Expected 3D input (B, L, H), got {x.dim()}D"
        B, L_Q, _ = x.shape
        B, L_K, _ = enc_out.shape
        
        # 添加更详细的错误提示
        if L_Q > L_K:
            raise ValueError(f"Query length {L_Q} exceeds key length {L_K} in cross attention. "
                           "This usually indicates a configuration issue. "
                           f"Consider adjusting the pred_length parameter in your config (currently {L_Q})")
        
        for layer in self.layers:
            x = layer(x, enc_out)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x


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
                 sequence_length: int = 24,
                 label_length: int = 12,
                 pred_length: int = 24,
                 factor: int = 5):
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
            label_length: 标签序列长度
            pred_length: 预测序列长度
            factor: ProbAttention中的采样因子
        """
        super(InformerModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.label_length = label_length
        self.pred_length = pred_length
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position_embedding = PositionalEmbedding(d_model=hidden_size)
        
        # 构建编码器层
        enc_layers = []
        for i in range(num_layers):
            attention = ProbAttention(mask_flag=False, factor=factor, attention_dropout=dropout)
            enc_layers.append(EncoderLayer(
                AttentionLayer(attention, hidden_size, num_heads),
                hidden_size,
                dropout=dropout
            ))
        
        # 编码器
        self.encoder = InformerEncoder(enc_layers)
        
        # 构建解码器层
        dec_layers = []
        for i in range(num_layers):
            self_attention = ProbAttention(mask_flag=True, factor=factor, attention_dropout=dropout)
            cross_attention = ProbAttention(mask_flag=False, factor=factor, attention_dropout=dropout)
            dec_layers.append(DecoderLayer(
                AttentionLayer(self_attention, hidden_size, num_heads),
                AttentionLayer(cross_attention, hidden_size, num_heads),
                hidden_size,
                dropout=dropout
            ))
        
        # 解码器
        self.decoder = InformerDecoder(dec_layers)
        
        # 输出层
        self.output_projection = nn.Linear(hidden_size, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, sequence_length, input_size)
            
        Returns:
            输出张量
        """
        # 输入投影和位置编码
        x_enc = self.input_projection(x) + self.position_embedding(x)
        x_enc = self.dropout(x_enc)
        
        # 编码器
        enc_out, _ = self.encoder(x_enc)
        
        # 添加编码器输出长度验证
        B, L_K, _ = enc_out.shape
        assert L_K >= self.label_length, \
            f"Encoder output length {L_K} must be at least label_length {self.label_length}"
        
        # 构造解码器输入 (简化版本 - 实际应用中可能需要更复杂的处理)
        # 这里我们使用编码器输出的一部分和零填充来构造解码器输入
        # 确保解码器输入的长度等于label_length + pred_length
        dec_input_zeros = torch.zeros([B, self.pred_length, enc_out.shape[-1]], 
                                      dtype=enc_out.dtype, device=enc_out.device)
        
        # 确保编码器输出和解码器输入的连接部分维度匹配
        enc_out_last = enc_out[:, -self.label_length:, :]
        
        dec_input = torch.cat([enc_out_last, dec_input_zeros], dim=1)
        dec_input = dec_input + self.position_embedding(dec_input)
        dec_input = self.dropout(dec_input)
        
        # 解码器
        dec_out = self.decoder(dec_input, enc_out)
        
        # 输出投影
        output = self.output_projection(dec_out)
        
        # 确保输出只包含预测部分（取最后pred_length个时间步）
        output = output[:, -self.pred_length:, :]
        
        return output


def create_informer_model(input_size: int,
                          hidden_size: int = 64,
                          num_layers: int = 2,
                          num_heads: int = 8,
                          output_size: int = 1,
                          dropout: float = 0.2,
                          sequence_length: int = 24,
                          label_length: int = 12,
                          pred_length: int = 24,
                          factor: int = 5) -> nn.Module:
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
        label_length: 标签序列长度
        pred_length: 预测序列长度
        factor: ProbAttention中的采样因子
        
    Returns:
        Informer模型
    """
    # 确保hidden_size能被num_heads整除
    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        
    return InformerModel(input_size, hidden_size, num_layers, num_heads, output_size, 
                         dropout, sequence_length, label_length, pred_length, factor)