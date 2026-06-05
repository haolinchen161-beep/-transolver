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
               omega_weight=200.0, zeta_weight=0.2, phi_weight=0.3):
    """模态参数损失。
    ω×50, ζ×2, φ×0.3: φ 降权让 ω 在 Phase1 占主导 (>70%)
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
    # CDF Wasserstein: 共振峰位置偏差→横向引力, MSE: 峰值高度精修
    loss_mse = F.mse_loss(frf_pred, frf_target)

    amp_pred = torch.norm(frf_pred, dim=-1) + 1e-8
    amp_target = torch.norm(frf_target, dim=-1) + 1e-8

    amp_pred_norm = amp_pred / amp_pred.sum(dim=-1, keepdim=True)
    amp_target_norm = amp_target / amp_target.sum(dim=-1, keepdim=True)

    cdf_pred = torch.cumsum(amp_pred_norm, dim=-1)
    cdf_target = torch.cumsum(amp_target_norm, dim=-1)

    loss_cdf = F.l1_loss(cdf_pred, cdf_target)
    return loss_mse + 10.0 * loss_cdf
