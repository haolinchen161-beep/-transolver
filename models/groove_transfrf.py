"""
groove_transfrf.py — GrooveTransFRF: Transolver 物理感知网络。

架构: Transolver Encoder + GlobalAttentionPool + 双流残差 ω

  points(N,3) + pt_feat(N,7)
          │
  ┌───────┴──────────┐
  │ TransolverEncoder │  全分辨率 SliceAttention (无FPS)
  │ → node_tokens(N,256) │
  └───────┬──────────┘
          │
    ┌─────┴─────┐
    │           │
┌───┴──┐  ┌─────┴──────────┐
│head_phi│  │GlobalAttentionPool│ 可学习查询交叉注意力
│→φ(N,K)│  │→ modal_out(B,2K) │
└───┬──┘  └─────┬──────────┘
    │           │
    └─────┬─────┘
          │
  ┌───────┴───────┐
  │ macro+micro ω │ softplus(macro(4标量))×15000 + tanh(micro(modal_out))×5000
  │ ζ = softplus×0.004+1e-4 │
  └───────┬───────┘
          │
  ┌───────┴───────┐
  │ PhysicsDecoder │ H=Σφ_k(x)φ_k(x_f)/(ω_k²-ω²+j2ζ_kω_kω)
  │ → FRF (asinh)  │
  └───────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry_data import GeometryData
from .transolver_encoder import TransolverEncoder
from .bc_aware_decoder import BCAwareModalDecoder
from .physics_decoder import PhysicsDecoder


class GrooveTransFRF(nn.Module):
    """
    几何感知物理 Transformer — FRF 预测。

    输入/输出接口与 ModalFRFModel 完全兼容。
    """

    def __init__(self, coord_dim=3, point_feat_dim=7,
                 hidden_dim=256, n_modes=3,
                 n_transolver_layers=3, num_heads=8,
                 slice_num=64, dropout=0.1,
                 amp_scale=500000.0, freq_min=1.0, freq_max=5000.0):
        """
        Args:
            coord_dim:            坐标维度 (3)
            point_feat_dim:       逐节点特征维度 (7)
            hidden_dim:           隐藏维度
            n_modes:              模态阶数 K
            n_transolver_layers:  Transolver 层数
            num_heads:            注意力头数
            slice_num:            切片数 M
            dropout:              Dropout 率
            amp_scale:            FRF 幅值缩放
            freq_min/max:         频率范围 (Hz)
        """
        super().__init__()
        self.n_modes = n_modes
        self.hidden_dim = hidden_dim

        # ============================================
        # Transolver 几何编码器 (全分辨率, 无FPS)
        # ============================================
        self.encoder = TransolverEncoder(
            coord_dim=coord_dim,
            feat_dim=point_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=n_transolver_layers,
            num_heads=num_heads,
            slice_num=slice_num,
            dropout=dropout,
        )

        # ============================================
        # 全局注意力池化 (可学习查询对全部N节点交叉注意力)
        # ============================================
        self.global_pool = BCAwareModalDecoder(
            token_dim=hidden_dim,
            n_modes=n_modes,
            num_heads=max(4, num_heads // 2),
            dropout=dropout,
        )

        # ============================================
        # head_phi: 逐节点振型预测
        # ============================================
        self.head_phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, n_modes),
        )

        # ============================================
        # 双流残差 ω 解码
        # ============================================
        self.macro_omega = nn.Sequential(
            nn.Linear(4, 64), nn.GELU(),           # 4 = ws, avg_logK, constraint_ratio, mean_Z/H
            nn.Linear(64, n_modes)
        )
        self.micro_omega = nn.Sequential(          # 微观流: modal_out前K + mean+max双池化->Delta_omega
            nn.Linear(n_modes + hidden_dim * 2, 64), nn.GELU(),
            nn.Linear(64, n_modes)
        )
        nn.init.constant_(self.macro_omega[-1].weight, 0.0)
        nn.init.constant_(self.macro_omega[-1].bias, 0.0)
        nn.init.constant_(self.micro_omega[-1].weight, 0.0)
        nn.init.constant_(self.micro_omega[-1].bias, 0.0)

        # ============================================
        # PhysicsDecoder (无参数)
        # ============================================
        self.physics = PhysicsDecoder(
            amp_scale=amp_scale,
            freq_min=freq_min,
            freq_max=freq_max,
        )

    def _compute_physics_prior(self, point_features, batch):
        """
        计算物理先验 (B, 4): [√(E/ρ), avg_logK, constraint_ratio, mean_Z/H]

        四个真物理量 (与 ω 正相关):
          1. √(E/ρ):            材料波速, ω ∝ √(E/ρ) — 振动理论严格结论
          2. avg_logK(bc):       边界弹簧平均刚度 — ω 随约束增强而增大
          3. constraint_ratio:   被约束节点占比 — 总约束刚度 ∝ K × 节点数
          4. mean_Z/H:           全局等效厚度 — 反映凹槽深度造成的刚度损失

        Args:
            point_features: (total_N, 7) [E,PRXY,DENS,is_fixed,logK,logC,Z/H]
            batch:          (total_N,)
        Returns:
            physics: (B, 4)
        """
        E_r = point_features[:, 0]
        rho_r = point_features[:, 2]
        logK = point_features[:, 4]
        z_norm = point_features[:, 6]  # Z/H 归一化厚度

        wave_speed = torch.sqrt(torch.abs(E_r / (rho_r + 1e-6)))

        B = int(batch.max().item()) + 1
        device = point_features.device

        physics_list = []
        for b in range(B):
            mask = batch == b
            total_nodes = mask.sum().float()

            ws_b = wave_speed[mask][0]

            bc_mask = logK[mask] > 0
            if bc_mask.any():
                avg_logK = logK[mask][bc_mask].mean()
                constraint_ratio = bc_mask.sum().float() / (total_nodes + 1e-6)
            else:
                avg_logK = torch.tensor(0.0, device=device)
                constraint_ratio = torch.tensor(0.0, device=device)

            mean_thickness = z_norm[mask].mean()

            physics_list.append(torch.stack([ws_b, avg_logK, constraint_ratio, mean_thickness]))

        return torch.stack(physics_list)  # (B, 4)

    def _prepare_inputs(self, geometry_data):
        """
        将 GeometryData 统一为 (total_N, 3/7) + (total_N,) batch 格式。

        Returns:
            points:         (total_N, 3)
            point_features: (total_N, F)
            batch:          (total_N,)
        """
        points = geometry_data.points
        point_feat = geometry_data.point_features
        batch = geometry_data.batch

        if points.ndim == 3:
            # 固定 N: (B, N, 3) → (B*N, 3)
            B, N_max, _ = points.shape
            points = points.reshape(-1, 3)
            if point_feat is not None:
                point_feat = point_feat.reshape(-1, point_feat.shape[-1])
            batch = torch.arange(B, device=points.device).repeat_interleave(N_max)

        return points, point_feat, batch

    def forward(self, geometry_data, frequencies=None, phi_exc=None):
        """
        前向传播 — 全分辨率, 不降采样。

        流程:
          1. Transolver 几何编码 (全部 N 节点)
          2. head_phi → φ (逐节点)
          3. GlobalAttentionPool → modal_out → micro_omega, zeta
          4. macro_omega(physics_prior) → ω_coarse
          5. PhysicsDecoder → FRF

        Args:
            geometry_data: GeometryData
            frequencies:   (B, F) 归一化频率 或 None
            phi_exc:       (B, K) 激励点振型值
        Returns:
            frf, omega, zeta, phi
        """
        points, point_feat, batch = self._prepare_inputs(geometry_data)

        if point_feat is None:
            point_feat = torch.zeros(points.shape[0], 7, device=points.device)
            point_feat[:, 3] = 1.0

        # ============================================
        # 步骤1: Transolver 几何编码 (全部 N 节点, 无降采样)
        # 坐标逐样本归一化到 [-1,1], 帮助 SliceAttention 均匀切片
        # ============================================
        pts_norm = points.clone()
        B_norm = int(batch.max().item()) + 1
        for b in range(B_norm):
            mask = batch == b
            p_b = points[mask]
            lo, hi = p_b.min(dim=0, keepdim=True)[0], p_b.max(dim=0, keepdim=True)[0]
            pts_norm[mask] = (p_b - lo) / (hi - lo + 1e-8) * 2.0 - 1.0

        node_tokens = self.encoder(pts_norm, point_feat)
        # (total_N, hidden_dim)

        # ============================================
        # 步骤2: 双路径解码
        # ============================================
        # 局部路径: 模态振型 (逐节点)
        phi = self.head_phi(node_tokens)  # (total_N, K)

        # 全局路径: 注意力池化 → 模态参数
        modal_out = self.global_pool(node_tokens, batch)  # (B, 2K)

        # ============================================
        # 步骤3: 双流残差 ω + ζ
        # ============================================
        physics_prior = self._compute_physics_prior(point_feat, batch)  # (B, 4)

        # 微观流: modal_out前K + mean + TopK(128)池化→几何特征
        # TopK替代max: 取前128个最强节点平均, 每个得1/128梯度 (vs max仅1个节点)
        B_val = int(batch.max().item()) + 1
        dev = node_tokens.device
        K_topk = 128
        geo_mean = torch.zeros(B_val, self.hidden_dim, device=dev)
        geo_topk = torch.zeros(B_val, self.hidden_dim, device=dev)
        for b in range(B_val):
            mask = batch == b
            toks = node_tokens[mask]  # (N_b, D)
            geo_mean[b] = toks.mean(dim=0)
            kt = min(K_topk, toks.size(0))
            geo_topk[b] = torch.topk(toks, kt, dim=0)[0].mean(dim=0)
        micro_in = torch.cat([modal_out[:, :self.n_modes], geo_mean, geo_topk], dim=-1)

        omega_coarse = F.softplus(self.macro_omega(physics_prior)) * 15000.0
        omega_fine = torch.tanh(self.micro_omega(micro_in)) * 5000.0
        omega = omega_coarse + omega_fine

        zeta = F.softplus(modal_out[:, self.n_modes:]) * 0.004 + 1e-4

        # ============================================
        # 步骤4: 物理重建 FRF
        # ============================================
        if frequencies is not None:
            frf_raw = self.physics(phi, omega, zeta, frequencies, phi_exc,
                                   batch_idx=batch)
            frf = torch.asinh(frf_raw.clamp(-1e4, 1e4))
        else:
            frf = None

        return frf, omega, zeta, phi
