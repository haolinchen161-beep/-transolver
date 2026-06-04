"""
losses.py — 模态参数损失 + 符号对齐 + 透视打印。
"""
import torch
import torch.nn.functional as F


def modal_loss(omega_pred, omega_target,
               zeta_pred, zeta_target,
               phi_pred, phi_target, batch_idx=None,
               zeta_weight=1000.0):
    loss_omega = torch.mean((omega_pred - omega_target)**2
                            / (omega_target**2 + 1e-8)) * 20000.0
    loss_zeta  = torch.mean((zeta_pred - zeta_target)**2
                            / (zeta_target**2 + 1e-8)) * zeta_weight

    if batch_idx is not None:
        if phi_pred.dim() == 3:
            phi_pred = phi_pred.view(-1, phi_pred.shape[-1])
            phi_target = phi_target.view(-1, phi_target.shape[-1])

        loss_phi = 0.0
        num_graphs = int(batch_idx.max().item()) + 1
        for i in range(num_graphs):
            mask = (batch_idx == i)
            p_p = phi_pred[mask]
            p_t = phi_target[mask]
            dot = torch.sum(p_p * p_t, dim=0, keepdim=True)
            sign = torch.sign(dot + 1e-8)
            aligned_t = p_t * sign
            loss_phi += F.mse_loss(p_p, aligned_t)
        loss_phi = (loss_phi / num_graphs) * 100.0
    else:
        phi_p = phi_pred.reshape(-1, phi_pred.shape[-1])
        phi_t = phi_target.reshape(-1, phi_target.shape[-1])
        dot = torch.sum(phi_p * phi_t, dim=0, keepdim=True)
        loss_phi = F.mse_loss(phi_p, phi_t * torch.sign(dot + 1e-8)) * 100.0

    return loss_omega + loss_zeta + loss_phi, loss_omega, loss_phi


def frf_loss(frf_pred, frf_target):
    # frf_pred 已是 asinh 空间 (模型 forward 中做了 asinh)
    # frf_target 也是 asinh 空间 (dataset 归一化)
    # 直接比较, 无需额外变换
    return F.mse_loss(frf_pred, frf_target)
