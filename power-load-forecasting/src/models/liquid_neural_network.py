"""
液态神经网络模型实现
"""
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
from typing import Tuple, Optional


class LiquidNeuralNetwork(nn.Module):
    """
    液态神经网络模型实现
    
    液态神经网络是一种受生物神经网络启发的动态系统模型，
    能够处理连续时间信号并具有记忆能力。
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 output_size: int = 1,
                 ode_method: str = 'euler'):
        """
        初始化液态神经网络
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            ode_method: ODE求解方法
        """
        super(LiquidNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.ode_method = ode_method
        
        # 输入到隐藏层的线性变换
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # 液态层（ODE定义的动态系统）
        self.liquid_layer = LiquidODEFunc(hidden_size)
        
        # 隐藏层到输出层的线性变换
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # 激活函数
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor, time_steps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, sequence_length, input_size)
            time_steps: 时间步长
            
        Returns:
            输出张量
        """
        batch_size, seq_length, _ = x.shape
        
        # 初始化隐藏状态
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # 存储每个时间步的输出
        outputs = []
        
        # 如果没有提供时间步，创建默认的时间步
        if time_steps is None:
            time_steps = torch.linspace(0, seq_length, seq_length + 1).to(x.device)
        
        # 对每个时间步进行处理
        for t in range(seq_length):
            # 输入到隐藏层
            input_t = self.input_layer(x[:, t, :])
            
            # 液态层处理（通过ODE求解器）
            # 修复：确保时间点是严格递增的
            dt = time_steps[t+1] - time_steps[t]
            # 确保dt为正数且不为0
            if dt <= 0:
                dt = torch.abs(dt) + 1e-6  # 添加小的正值确保严格递增
                
            h = self._solve_ode(h, input_t, dt)
            
            # 隐藏层到输出层
            output = self.output_layer(self.tanh(h))
            outputs.append(output)
        
        # 堆叠所有时间步的输出
        outputs = torch.stack(outputs, dim=1)
        return outputs
    
    def _solve_ode(self, h: torch.Tensor, input_t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        使用ODE求解器解决液态层动态
        
        Args:
            h: 当前隐藏状态
            input_t: 当前输入
            dt: 时间间隔
            
        Returns:
            更新后的隐藏状态
        """
        # 定义时间点，确保是严格递增的
        t_points = torch.tensor([0, torch.abs(dt).item()]).to(h.device)
        
        # 将输入添加到液态层
        self.liquid_layer.input = input_t
        
        # 使用ODE求解器，使用odeint_adjoint以节省内存
        h_sol = odeint(self.liquid_layer, h, t_points, method=self.ode_method, adjoint_method=self.ode_method)
        
        # 返回最后一个时间点的解
        return h_sol[-1]


class LiquidODEFunc(nn.Module):
    """
    液态ODE函数
    
    定义液态神经网络中的常微分方程
    """
    
    def __init__(self, hidden_size: int):
        """
        初始化液态ODE函数
        
        Args:
            hidden_size: 隐藏层维度
        """
        super(LiquidODEFunc, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 液态系统的参数
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_ih = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        
        # 偏置项
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        # 时间常数
        self.tau = nn.Parameter(torch.ones(hidden_size))
        
        # 输入占位符
        self.input = None
        
        # 激活函数
        self.tanh = nn.Tanh()
        
    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        计算dh/dt
        
        Args:
            t: 时间
            h: 隐藏状态
            
        Returns:
            隐藏状态的变化率
        """
        # 液态系统的动态方程
        # tau * dh/dt = -h + tanh(W_hh * h + W_ih * input + bias)
        
        # 计算输入对隐藏状态的影响
        if self.input is not None:
            # 使用批量矩阵乘法提高效率
            input_effect = torch.mm(self.input, self.W_ih.t())
        else:
            input_effect = 0
            
        # 计算隐藏状态的自反馈
        hidden_effect = torch.mm(self.tanh(h), self.W_hh.t())
        
        # 计算dh/dt
        dhdt = (-h + hidden_effect + input_effect + self.bias) / self.tau
        
        return dhdt


def create_liquid_model(input_size: int, 
                       hidden_size: int = 64,
                       output_size: int = 1,
                       model_type: str = 'liquid_ode') -> nn.Module:
    """
    创建液态神经网络模型的便捷函数
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏层维度
        output_size: 输出维度
        model_type: 模型类型 ('liquid_ode')
        
    Returns:
        液态神经网络模型
    """
    if model_type == 'liquid_ode':
        return LiquidNeuralNetwork(input_size, hidden_size, output_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")