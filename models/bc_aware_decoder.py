"""
bc_aware_decoder.py — 边界条件感知模态参数解码器。

替代 global_mean_pool 的物理动机:
  均值池化将关键边界条件节点 (典型 <50 个, 在 ~4000 节点中) 的信号
  稀释至 1/4000。但模态刚度 K_k = 结构刚度 + Σ_j K_spring_j·φ_k(x_j)²,
  角点弹簧节点对 ω_k 的影响远超数百个内部节点之和。

解决方案: 可学习模态查询 token 对所有节点做交叉注意力。
  - mode_queries: (K, D) 可学习参数, 每模态一个查询向量
  - BC gate: 逐节点学习重要性权重, 显式建模 "哪些节点影响全局模态参数"
  - Cross-attention: mode_queries (Q) attend to node_tokens (K, V)
  - 输出: 每模态的特征 → head_modal → Δω, ζ

类比 FEM: 这相当于网络学习 "看向" 边界条件节点和几何薄弱区来预测 ω_k,
  而非盲目平均所有节点的特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCAwareModalDecoder(nn.Module):
    """
    边界条件感知全局池化 → 模态参数。

    结构:
      1. BC Gate: node_tokens → sigmoid → per-node importance (N, 1)
         物理直觉: 弹簧节点和薄筋节点的模态参与度远高于内部节点

      2. Cross-Attention: learnable mode_queries (K, D) attend to node_tokens (N, D)
         物理直觉: 每阶模态 "扫描" 所有节点, 寻找对其 ω_k 影响最大的区域

      3. head_modal: 拼接的模态特征 → Δω, ζ (B, 2K)
    """

    def __init__(self, token_dim: int = 256, n_modes: int = 2,
                 num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            token_dim:  节点 token 维度
            n_modes:    模态阶数 K
            num_heads:  交叉注意力头数
            dropout:    dropout 率
        """
        super().__init__()
        self.token_dim = token_dim
        self.n_modes = n_modes
        self.num_heads = num_heads

        # 可学习模态查询 token — 每阶模态一个
        # 初始化为小值, 使初始交叉注意力接近均匀
        self.mode_queries = nn.Parameter(
            torch.randn(n_modes, token_dim) * 0.02
        )

        # 交叉注意力: mode_queries attend to node_tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # BC 重要性门控: 逐节点学习权重
        # 输入 token 维度, 输出标量 (通过 sigmoid)
        self.bc_gate = nn.Sequential(
            nn.Linear(token_dim, token_dim // 4),
            nn.ReLU(),
            nn.Linear(token_dim // 4, 1),
            nn.Sigmoid(),
        )

        # 模态特征 → Δω, ζ
        # 输入: 拼接 K 个模态 token → (K * token_dim)
        # 输出: 2K (K 个 Δω + K 个 ζ)
        modal_in_dim = n_modes * token_dim
        self.head_modal = nn.Sequential(
            nn.Linear(modal_in_dim, token_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(token_dim, n_modes * 2),
        )

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, node_tokens, batch):
        """
        Args:
            node_tokens: (total_N, token_dim) 所有样本的节点 token (Transformer 输出)
            batch:       (total_N,)           批次索引
        Returns:
            modal_out: (B, 2*K) 原始 Δω 和 ζ 预测
        """
        B = int(batch.max().item()) + 1
        device = node_tokens.device

        # BC 重要性门控权重
        bc_weight = self.bc_gate(node_tokens)  # (N, 1)

        modal_features_list = []

        for b in range(B):
            mask = batch == b
            tokens_b = node_tokens[mask]   # (N_b, D)
            weight_b = bc_weight[mask]      # (N_b, 1)

            # 加权节点 token
            weighted_tokens = tokens_b * weight_b  # (N_b, D)

            # 交叉注意力: mode_queries (K, D) attend to weighted_tokens (N_b, D)
            # 将 mode_queries 扩展 batch 维度: (1, K, D)
            queries = self.mode_queries.unsqueeze(0)  # (1, K, D)
            kv = weighted_tokens.unsqueeze(0)         # (1, N_b, D)

            attn_out, attn_weights = self.cross_attn(
                query=queries,
                key=kv,
                value=kv,
            )  # attn_out: (1, K, D)

            # 展平 K 个模态 token → (K*D,)
            modal_feat = attn_out.squeeze(0).flatten()  # (K*D,)
            modal_features_list.append(modal_feat)

        modal_features = torch.stack(modal_features_list)  # (B, K*D)

        # 输出 Δω, ζ
        modal_out = self.head_modal(modal_features)  # (B, 2K)

        return modal_out

    def get_attention_weights(self, node_tokens, batch):
        """
        获取交叉注意力权重 (用于可视化和可解释性分析)。

        Returns:
            attn_per_mode: list of (N_b, K) tensors, 每个样本的每节点注意力权重
        """
        B = int(batch.max().item()) + 1
        bc_weight = self.bc_gate(node_tokens)

        attn_list = []
        for b in range(B):
            mask = batch == b
            tokens_b = node_tokens[mask]
            weight_b = bc_weight[mask]
            weighted_tokens = tokens_b * weight_b

            queries = self.mode_queries.unsqueeze(0)  # (1, K, D)
            kv = weighted_tokens.unsqueeze(0)         # (1, N_b, D)

            _, attn_weights = self.cross_attn(query=queries, key=kv, value=kv,
                                              average_attn_weights=False)
            # attn_weights: (1, num_heads, K, N_b)
            # 平均头 → (K, N_b) → 转置 → (N_b, K)
            attn_avg = attn_weights.mean(dim=1).squeeze(0).transpose(0, 1)  # (N_b, K)
            attn_list.append(attn_avg)

        return attn_list
