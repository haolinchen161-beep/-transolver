"""
losses.py — 模态参数损失 + MAC 物理对齐。
"""
import torch
import torch.nn.functional as F


def mac_loss(pred_phi, true_phi):
    """MAC (Modal Assurance Criterion) 损失。
    MAC = (φ_pred·φ_true)² / (‖φ_pred‖² · ‖φ_true‖²)
    值域 [0,1], 1=完全相同, 0=正交。天然处理符号和尺度歧义。
    """
    num = (pred_phi * true_phi).sum(dim=0) ** 2
    den = ((pred_phi ** 2).sum(dim=0) * (true_phi ** 2).sum(dim=0)) + 1e-8
    mac = num / den
    return 1.0 - mac.mean()


def modal_loss(omega_pred, omega_target,
               zeta_pred, zeta_target,
               phi_pred, phi_target, batch_idx=None,
               omega_weight=50.0, zeta_weight=2.0, phi_weight=1.0):
    """模态参数损失。

    权重设计: ω相对误差天然小(~1e-4), 需omega_weight=50补偿;
             ζ相对误差(~1e-2)适中; φ MAC值~0.01-0.1。
             训练初期 ω 主导, 后期自动让位给 φ/ζ。
    """
    loss_omega = torch.mean(((omega_pred - omega_target) / (omega_target + 1e-8))**2) * omega_weight
    loss_zeta  = torch.mean(((zeta_pred - zeta_target) / (zeta_target + 1e-8))**2) * zeta_weight

    if batch_idx is not None:
        if phi_pred.dim() == 3:
            phi_pred = phi_pred.view(-1, phi_pred.shape[-1])
            phi_target = phi_target.view(-1, phi_target.shape[-1])

        loss_phi = 0.0
        num_graphs = int(batch_idx.max().item()) + 1
        for i in range(num_graphs):
            mask = (batch_idx == i)
            loss_phi += mac_loss(phi_pred[mask], phi_target[mask])
        loss_phi = (loss_phi / num_graphs) * phi_weight
    else:
        phi_p = phi_pred.reshape(-1, phi_pred.shape[-1])
        phi_t = phi_target.reshape(-1, phi_target.shape[-1])
        loss_phi = mac_loss(phi_p, phi_t) * phi_weight

    return loss_omega + loss_zeta + loss_phi, loss_omega, loss_zeta, loss_phi


def frf_loss(frf_pred, frf_target):
    # frf_pred 已是 asinh 空间, frf_target 也是 asinh 空间, 直接比较
    return F.mse_loss(frf_pred, frf_target)
