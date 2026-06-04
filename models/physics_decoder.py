"""
physics_decoder.py — 无参数物理解码器。

使用模态叠加公式从模态参数重建复数频响函数 (FRF):
  H(x, x_f, ω) = Σ_k φ_k(x)·φ_k(x_f) / (ω_k² - ω² + j·2ζ_k·ω_k·ω)

该模块不含任何可学习参数, 所有运算严格遵循振动理论,
确保 FRF 预测在物理上一致。
"""
import torch
import torch.nn as nn


class PhysicsDecoder(nn.Module):
    """无参数物理解码器: φ + ω + ζ + φ_exc + freqs → FRF(Re,Im)

    H(x, x_f, ω) = Σ_k φ_k(x)·φ_k(x_f) / (ω_k²-ω²+j·2ζ_k·ω_k·ω)
    """

    def __init__(self, amp_scale: float = 500000.0,
                 freq_min: float = 1.0, freq_max: float = 5000.0):
        super().__init__()
        self.amp_scale = amp_scale
        self.freq_min = freq_min
        self.freq_max = freq_max

    def forward(self, phi, omega, zeta, frequencies, phi_exc=None, batch_idx=None):
        """
        Args:
            phi:         (total_N, K) 可变N 或 (B, N, K) 同节点数 — 模态振型
            omega:       (B, K)     rad/s 固有圆频率
            zeta:        (B, K)     阻尼比
            frequencies: (B, F)     归一化查询频率 [-1,1]
            phi_exc:     (B, K)     激励点振型值 φ_k(x_f)
            batch_idx:   (total_N,) 可变N时的批次索引
        Returns:
            frf: (total_N, F, 2) 或 (B, N, F, 2) — 复数 FRF [Re, Im]
        """
        K = omega.shape[1]
        B, F = frequencies.shape
        var_n = (batch_idx is not None)

        f_phys = (frequencies + 1) / 2 * (self.freq_max - self.freq_min) + self.freq_min
        omega_q = 2.0 * torch.pi * f_phys  # (B, F)

        if var_n:
            total_N = phi.shape[0]
            frf_re = torch.zeros(total_N, F, device=phi.device)
            frf_im = torch.zeros(total_N, F, device=phi.device)
            for k in range(K):
                wk = omega[:, k]       # (B,)
                zk = zeta[:, k]        # (B,)
                pk = phi[:, k]         # (total_N,)
                if phi_exc is not None:
                    pk = pk * phi_exc[:, k][batch_idx]

                dw = wk.unsqueeze(1)**2 - omega_q**2  # (B, F)
                gamma = 2.0 * zk.unsqueeze(1) * wk.unsqueeze(1) * omega_q
                D = torch.clamp(dw**2 + gamma**2 + 1e-6, min=1.0)
                H_re = self.amp_scale * dw / D   # (B, F)
                H_im = -self.amp_scale * gamma / D

                frf_re += pk.unsqueeze(-1) * H_re[batch_idx]
                frf_im += pk.unsqueeze(-1) * H_im[batch_idx]
            return torch.stack([frf_re, frf_im], dim=-1)  # (total_N, F, 2)

        else:
            N = phi.shape[1]
            frf_re = torch.zeros(B, N, F, device=phi.device)
            frf_im = torch.zeros(B, N, F, device=phi.device)
            for k in range(K):
                wk = omega[:, k]       # (B,)
                zk = zeta[:, k]        # (B,)
                pk = phi[:, :, k]      # (B, N)
                if phi_exc is not None:
                    pk = pk * phi_exc[:, k].unsqueeze(1)

                dw = wk.unsqueeze(1)**2 - omega_q**2  # (B, F)
                gamma = 2.0 * zk.unsqueeze(1) * wk.unsqueeze(1) * omega_q
                D = torch.clamp(dw**2 + gamma**2 + 1e-6, min=1.0)
                H_re = self.amp_scale * dw / D   # (B, F)
                H_im = -self.amp_scale * gamma / D

                frf_re += pk.unsqueeze(-1) * H_re.unsqueeze(1)
                frf_im += pk.unsqueeze(-1) * H_im.unsqueeze(1)
            return torch.stack([frf_re, frf_im], dim=-1)  # (B, N, F, 2)
