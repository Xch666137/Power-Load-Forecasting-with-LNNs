"""
概率稀疏注意力机制（来自Informer）
"""
import torch
import torch.nn as nn
import numpy as np


class ProbAttention(nn.Module):
    """
    概率稀疏注意力机制
    """
    
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        """
        初始化概率稀疏注意力
        
        Args:
            mask_flag: 是否使用掩码
            factor: 采样因子
            scale: 缩放因子
            attention_dropout: 注意力dropout比率
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        计算概率QK
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # 计算采样点
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        
        # 添加维度检查和调试信息
        assert K_expand.shape[-2] == L_K, \
            f"K_expand last dimension size {K_expand.shape[-2]} does not match L_K {L_K}"
        
        # 添加更详细的错误信息
        if torch.any(index_sample >= L_K) or torch.any(index_sample < 0):
            raise ValueError(f"Invalid index_sample range: min={index_sample.min().item()}, "
                             f"max={index_sample.max().item()}, "
                             f"valid range=[0, {L_K-1}], "
                             f"L_K={L_K}, "
                             f"shape of K_expand={K_expand.shape}")
        
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        
        # 计算QK
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # 找到topk查询项
        M = Q_K_sample.max(-1)[0] - torch.div(torch.sum(Q_K_sample, -1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # 添加均匀先验
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        获取初始上下文
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        """
        更新上下文
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            # 创建二维attn_mask [L_Q, L_Q]
            attn_mask = torch.ones(L_Q, L_Q, dtype=torch.bool, device=scores.device)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            
            # 应用掩码到scores，注意scores的维度是[B, H, n_top, L_V]
            # 其中n_top是index的长度，index的形状是[B, H, n_top]
            # 我们需要创建一个适用于scores的掩码
            # scores.shape = [B, H, n_top, L_V]，其中n_top=20, L_V=36
            # 需要创建一个形状为[B, H, n_top, L_V]的掩码
            
            # 从scores的形状获取n_top
            n_top = scores.shape[-2]
            
            # 创建适用于scores的掩码 [n_top, L_Q]
            # 这里我们只关注序列的前n_top个位置的掩码
            causal_mask = attn_mask[:n_top, :L_Q]  # 形状 [n_top, L_Q]
            
            # 扩展到scores的维度 [B, H, n_top, L_Q]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
            
            # 应用掩码到scores
            scores.masked_fill_(causal_mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        
        if self.training:
            attns = (torch.ones([B, H, L_V, L_V], device=attn.device) / L_V).type_as(attn)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask=None):
        """
        前向传播
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # 添加缩放因子
        scale = self.scale or 1. / np.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        return context.transpose(2, 1).contiguous(), attn