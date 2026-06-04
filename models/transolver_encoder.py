"""
transolver_encoder.py — Transolver 几何编码器。

基于 Transolver (Wu et al., ICML 2024) 的 Physics-Attention (Slice Attention):
  1. SLICE:   每个节点通过 softmax 软分配到 M 个可学习切片
  2. ATTEND:  自注意力仅在 M 个切片间计算 (O(M²), 非 O(N²))
  3. DESLICE: 切片特征广播回全部节点

处理全部 N 个节点, 不需要 FPS 降采样, 不丢失凹槽边缘和薄筋的局部几何信息。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SliceAttention(nn.Module):
    """
    Transolver Slice Attention.

    每条节点软分配到 M 个切片, 注意力仅在切片间计算。
    复杂度 O(N·M + M²), 与 N 线性。
    """

    def __init__(self, dim=256, num_heads=8, slice_num=64, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.slice_num = slice_num

        # 节点 → K, V 投影
        self.to_kv = nn.Linear(dim, dim * 2)

        # K → 切片权重 (每个head独立分配)
        self.to_slice_weights = nn.Linear(self.head_dim, slice_num)

        # 可学习温度参数
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1) * 0.5)

        # 切片 → Q, K, V (切片间自注意力)
        self.to_q_s = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_k_s = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_v_s = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 初始化切片权重投影为正交
        nn.init.orthogonal_(self.to_slice_weights.weight)

    def forward(self, x):
        """
        Args:
            x: (N, dim) 节点特征
        Returns:
            out: (N, dim)
        """
        N, D = x.shape
        H = self.num_heads
        M = self.slice_num
        d = self.head_dim

        # ---- SLICE: 节点 → 切片 ----
        kv = self.to_kv(x)  # (N, 2*D)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(N, H, d)      # (N, H, d)
        v = v.view(N, H, d)

        # 切片权重: softmax 归一化
        slice_logits = self.to_slice_weights(k) / self.temperature  # (N, H, M)
        w = F.softmax(slice_logits, dim=-1)  # (N, H, M)

        # 加权聚合节点→切片: s = w^T @ v / sum(w)
        s = torch.einsum('nhm,nhd->hmd', w, v)  # w^T @ v: (H, M, d_head)
        s_norm = w.sum(dim=0).unsqueeze(-1) + 1e-5  # (H, M, 1)
        s = s / s_norm  # (H, M, d)

        # ---- ATTEND: 切片间自注意力 O(M²) ----
        q_s = self.to_q_s(s)  # (H, M, d)
        k_s = self.to_k_s(s)
        v_s = self.to_v_s(s)

        attn = torch.matmul(q_s, k_s.transpose(-2, -1)) / math.sqrt(d)  # (H, M, M)
        attn = F.softmax(attn, dim=-1)
        s_out = torch.matmul(attn, v_s)  # (H, M, d)

        # ---- DESLICE: 切片 → 节点 ----
        x_out = torch.einsum('nhm,hmd->nhd', w, s_out)  # w @ s_out: (N, H, d)
        x_out = x_out.reshape(N, D)
        x_out = self.proj(x_out)
        x_out = self.dropout(x_out)

        return x_out + x  # 残差连接


class TransolverLayer(nn.Module):
    """单个 Transolver 层: Pre-LN SliceAttention + Pre-LN FFN."""

    def __init__(self, dim=256, num_heads=8, slice_num=64,
                 ff_expand=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SliceAttention(dim, num_heads, slice_num, dropout)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_expand),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim * ff_expand, dim),
        )
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class TransolverEncoder(nn.Module):
    """
    Transolver 几何编码器。

    输入: points(N,3) + point_features(N,7) → Linear(10, 256)
    处理: n_layers 个 TransolverLayer (SliceAttention + FFN)
    输出: node_tokens(N, 256)

    全程不降采样, 全部 N 个节点参与。
    """

    def __init__(self, coord_dim=3, feat_dim=7, hidden_dim=256,
                 n_layers=4, num_heads=8, slice_num=64, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 输入嵌入: 坐标+特征 → hidden_dim
        self.embed = nn.Linear(coord_dim + feat_dim, hidden_dim)

        # Transolver 层
        self.layers = nn.ModuleList([
            TransolverLayer(hidden_dim, num_heads, slice_num, ff_expand=2, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, points, point_features):
        """
        Args:
            points:         (N, 3) 原始坐标
            point_features: (N, F) 逐节点特征 (F=7)
        Returns:
            node_tokens: (N, hidden_dim)
        """
        x = torch.cat([points, point_features], dim=-1)
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x
