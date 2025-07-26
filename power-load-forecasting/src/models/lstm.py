"""
标准LSTM模型实现
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    标准LSTM模型
    
    使用LSTM层处理时间序列数据，适用于电力负荷预测等任务
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 sequence_length: int = 24):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            output_size: 输出维度
            dropout: Dropout比率
            sequence_length: 输入序列长度
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        
        # Dropout层
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
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        output = lstm_out[:, -1, :]
        
        # 全连接层
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output.unsqueeze(1)  # 增加一个维度以匹配输出格式


def create_lstm_model(input_size: int,
                      hidden_size: int = 64,
                      num_layers: int = 2,
                      output_size: int = 1,
                      dropout: float = 0.2,
                      sequence_length: int = 24) -> nn.Module:
    """
    创建LSTM模型的便捷函数
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏层维度
        num_layers: LSTM层数
        output_size: 输出维度
        dropout: Dropout比率
        sequence_length: 输入序列长度
        
    Returns:
        LSTM模型
    """
    return LSTMModel(input_size, hidden_size, num_layers, output_size, dropout, sequence_length)