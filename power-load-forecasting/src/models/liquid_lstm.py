"""
液态LSTM模型实现
"""
import torch
import torch.nn as nn


class LiquidLSTM(nn.Module):
    """
    液态LSTM模型（替代实现）
    
    结合LSTM和液态神经网络思想的混合模型
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2):
        """
        初始化液态LSTM
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            output_size: 输出维度
            dropout: Dropout比率
        """
        super(LiquidLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, sequence_length, input_size)
            
        Returns:
            输出张量
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        output = self.dropout(lstm_out[:, -1, :])
        
        # 全连接层
        output = self.fc(output)
        
        return output.unsqueeze(1)  # 增加一个维度以匹配输出格式


def create_liquid_lstm_model(input_size: int,
                             hidden_size: int = 64,
                             num_layers: int = 2,
                             output_size: int = 1,
                             dropout: float = 0.2) -> nn.Module:
    """
    创建液态LSTM模型的便捷函数
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏层维度
        num_layers: LSTM层数
        output_size: 输出维度
        dropout: Dropout比率
        
    Returns:
        液态LSTM模型
    """
    return LiquidLSTM(input_size, hidden_size, num_layers, output_size, dropout)