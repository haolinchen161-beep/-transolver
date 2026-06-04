"""
trainer.py — GrooveTransFRF 两阶段训练循环 + 评估。

训练策略: Phase1 (纯模态) → 动态解锁 → Phase2 (模态+FRF)

数据流:
    geometry + frequencies → net → per_point_frf (B, N, n_freqs[, out_dim])
    损失: modal_loss (ω, ζ, φ) + frf_loss (物理约束)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from .losses import modal_loss, frf_loss
from .augmentations import create_augmenter


def train(args, config, model_cfg, net, dataloader, optimizer,
          valloader, scheduler, logger=None, start_epoch=0):
    """
    GrooveTransFRF 两阶段训练循环。

    阶段1 (0 ~ 动态解锁):      全解冻纯模态, 仅modal_loss
    阶段2 (动态解锁 ~ total):  模态+FRF×50
    
    
    
    """
    lowest = np.inf
    net.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    total_epochs = config.get('epochs', 2000)

    frf_weight = config.get('frf_loss_weight', 50.0)

    # 数据增强器
    augmenter = create_augmenter(config)

    # 损失日志
    import csv
    os.makedirs(args.dir, exist_ok=True)
    log_path = os.path.join(args.dir, "loss_log.csv")
    log_exists = os.path.exists(log_path) and start_epoch > 0
    log_file = open(log_path, 'a', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['轮次', '训练损失', 'ω相对误差%', 'ζ相对误差%', 'φ误差', '验证asinhMSE', '幅值MAE', '幅值MAPE%', '学习率'])

    phase2_unlocked = False

    try:
      for epoch in range(start_epoch, total_epochs):
        losses, omega_losses, zeta_losses = [], [], []
        weighted_w_losses, weighted_z_losses, weighted_p_losses = [], [], []

        # ---- 两阶段判定 (动态解锁: ω<1% 或 epoch>600) ----
        in_phase1 = not phase2_unlocked
        in_phase2 = phase2_unlocked

        # ---- 阶段切换 ----
        if in_phase1 and epoch == 0:
            _log("=== 阶段1: 全解冻纯模态, ω<1% 或 epoch>600 解锁 FRF ===", logger)
        elif in_phase2 and epoch > 0 and not getattr(net, '_phase2_logged', False):
            _log(f"=== 阶段2: FRF 联合训练 (第 {epoch} 轮解锁) ===", logger)
            net._phase2_logged = True
            lowest = np.inf

        for batch in dataloader:
            optimizer.zero_grad()

            # 数据增强
            if augmenter is not None:
                augmenter.train()
                batch = augmenter(batch)

            geometry = batch['geometry'].to(args.device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                if in_phase2:
                    frequencies = batch['frequencies'].to(args.device)
                    phi_exc = batch.get('modal_phi_exc')
                    phi_exc = phi_exc.to(args.device) if phi_exc is not None else None
                    # φ_exc 符号对齐
                    if phi_exc is not None:
                        with torch.no_grad():
                            _, _, _, phi_scan = net(geometry, frequencies, None)
                        modal_phi = batch['modal_phi'].to(args.device)
                        phi_exc_corrected = phi_exc.clone()
                        b_idx = geometry.batch
                        B = phi_exc.shape[0]
                        if b_idx is not None:
                            for i in range(int(b_idx.max().item()) + 1):
                                mask = (b_idx == i)
                                dot = torch.sum(phi_scan[mask] * modal_phi[mask], dim=0)
                                phi_exc_corrected[i] = phi_exc[i] * torch.sign(dot + 1e-8)
                        else:
                            N_per = phi_scan.shape[0] // B
                            phi_s = phi_scan.view(B, N_per, -1)
                            mp_s = modal_phi.view(B, N_per, -1)
                            for i in range(B):
                                dot = torch.sum(phi_s[i] * mp_s[i], dim=0)
                                phi_exc_corrected[i] = phi_exc[i] * torch.sign(dot + 1e-8)
                        phi_exc = phi_exc_corrected
                    frf_pred, omega_pred, zeta_pred, phi_pred = net(geometry, frequencies, phi_exc)
                    loss_m, l_w, l_z, l_p = modal_loss(
                        omega_pred, batch['modal_omega'].to(args.device),
                        zeta_pred, batch['modal_zeta'].to(args.device),
                        phi_pred, batch['modal_phi'].to(args.device),
                        batch_idx=geometry.batch)
                    loss = loss_m + frf_weight * frf_loss(frf_pred, batch['point_frf'].to(args.device))
                else:
                    _, omega_pred, zeta_pred, phi_pred = net(geometry)
                    loss_m, l_w, l_z, l_p = modal_loss(
                        omega_pred, batch['modal_omega'].to(args.device),
                        zeta_pred, batch['modal_zeta'].to(args.device),
                        phi_pred, batch['modal_phi'].to(args.device),
                        batch_idx=geometry.batch)
                    loss = loss_m

            losses.append(loss.detach().cpu().item())
            omega_target = batch['modal_omega'].to(args.device)
            omega_rel_err = torch.abs(omega_pred - omega_target) / (omega_target + 1e-8)
            omega_losses.append(omega_rel_err.mean().detach().cpu().item())
            zeta_target = batch['modal_zeta'].to(args.device)
            zeta_rel_err = torch.abs(zeta_pred - zeta_target) / (zeta_target + 1e-8)
            zeta_losses.append(zeta_rel_err.mean().detach().cpu().item())
            weighted_w_losses.append(l_w.detach().cpu().item())
            weighted_z_losses.append(l_z.detach().cpu().item())
            weighted_p_losses.append(l_p.detach().cpu().item())

            scaler.scale(loss).backward()

            # 分组件梯度裁剪
            _apply_gradient_clip(net, config)

            scaler.step(optimizer)
            scaler.update()

        mean_loss = np.mean(losses)

        # 学习率调度: ReduceLROnPlateau 需要传入 loss
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)
            else:
                scheduler.step()

        raw_w = np.mean(omega_losses) if omega_losses else 0
        raw_z = np.mean(zeta_losses) if zeta_losses else 0
        wgt_w = np.mean(weighted_w_losses) if weighted_w_losses else 0
        wgt_z = np.mean(weighted_z_losses) if weighted_z_losses else 0
        wgt_p = np.mean(weighted_p_losses) if weighted_p_losses else 0
        omega_pct = raw_w * 100
        zeta_pct  = raw_z * 100
        phi_pct = wgt_p * 100  # 1-MAC, 直接乘100
        omega_share = wgt_w / mean_loss * 100 if mean_loss > 0 else 0
        zeta_share  = wgt_z / mean_loss * 100 if mean_loss > 0 else 0
        phi_share   = wgt_p / mean_loss * 100 if mean_loss > 0 else 0
        _log(f"Epoch {epoch:4d} | w={omega_pct:.1f}% z={zeta_pct:.1f}% phi={phi_pct:.1f}% | 占比 w{omega_share:.0f}% z{zeta_share:.0f}% phi{phi_share:.0f}% | total={mean_loss:.2e}", logger)

        # 动态解锁: ω误差 < 5.0% 即可, FRF 介入帮 ω 对齐共振峰
        # ω需<1% (半功率带宽~6Hz, 10Hz误差就跑偏)
        if not phase2_unlocked and (omega_pct < 1.0 or epoch > 600):
            trigger = 'ω<1%' if omega_pct < 1.0 else f'epoch>{600}'
            phase2_unlocked = True
            _log(f">>> {trigger} 动态解锁 Phase2! <<<", logger)

        lr = optimizer.param_groups[0]['lr']
        val_freq = config.get('validation_frequency', 5)
        if epoch % val_freq == 0 or epoch % int(total_epochs / 10) == 0:
            save_model(args.dir, epoch, net, optimizer, loss, "checkpoint_last")
            if in_phase1:
                # Phase 1 不训 FRF, 跳过 FRF 评估, 仅记录 ω_MAE
                val_results = evaluate(args, config, net, valloader, logger, epoch)
                omega_mae = val_results.get("ω_MAE (rad/s)", -1)
                _log(f"Epoch {epoch:4d} | ω_MAE={omega_mae:.1f} rad/s (Phase1: FRF metrics skipped)", logger)
                log_writer.writerow([epoch, f'{mean_loss:.2e}', f'{omega_pct:.3f}', f'{zeta_pct:.3f}', f'{phi_pct:.2f}', '', '', '', f'{lr:.2e}'])
            else:
                val_results = evaluate(args, config, net, valloader, logger, epoch)
                val_loss = val_results["loss (asinh-MSE)"]
                log_writer.writerow([epoch, f'{mean_loss:.2e}', f'{omega_pct:.3f}', f'{zeta_pct:.3f}', f'{phi_pct:.2f}', f'{val_loss:.4f}',
                                     f'{val_results.get("Amplitude MAE", 0):.4f}',
                                     f'{val_results.get("Amplitude MAPE (%)", 0):.2f}',
                                     f'{lr:.2e}'])
            log_file.flush()
            use_val_metric = not in_phase1
            if use_val_metric:
                val_loss = val_results["loss (asinh-MSE)"]
                best_metric = val_loss
                metric_name = "val_loss"
                fmt = ".6f"
            else:
                # Phase1: 用验证 ω_MAE 选最优 (比 train_loss 更直接)
                best_metric = val_results.get("ω_MAE (rad/s)", mean_loss)
                metric_name = "ω_MAE"
                fmt = ".1f"
            if best_metric < lowest:
                _log(f"best model ({metric_name}={best_metric:{fmt}})", logger)
                save_model(args.dir, epoch, net, optimizer, best_metric)
                lowest = best_metric
        else:
            log_writer.writerow([epoch, f'{mean_loss:.2e}', f'{omega_pct:.3f}', f'{zeta_pct:.3f}', f'{phi_pct:.2f}', '', '', '', f'{lr:.2e}'])

        if epoch == (total_epochs - 1):
            path = os.path.join(args.dir, "checkpoint_best")
            if os.path.exists(path):
                net.load_state_dict(torch.load(path, map_location='cpu')["model_state_dict"])
            evaluate(args, config, net, valloader, logger, epoch, verbose=True)

    finally:
        log_file.close()

    return net


def _apply_gradient_clip(net, config):
    """分组件梯度裁剪 (GrooveTransFRF 专用)。"""
    grad_clip = config.get('optimizer', {}).get('gradient_clip')
    if grad_clip is None:
        return

    # Transformer: max_norm=1.0 (注意力易大梯度)
    _clip_module(net, 'encoder', config.get('optimizer', {}).get('gradient_clip_transolver', 3.0))
    # head_phi (MLP): max_norm=5.0
    _clip_module(net, 'head_phi', config.get('optimizer', {}).get('gradient_clip_head_phi', 5.0))
    # Modal decoder + skip: max_norm=2.0
    _clip_module(net, 'global_pool', config.get('optimizer', {}).get('gradient_clip_modal', 2.0))
    _clip_module(net, 'macro_omega', config.get('optimizer', {}).get('gradient_clip_modal', 2.0))
    _clip_module(net, 'micro_omega', config.get('optimizer', {}).get('gradient_clip_modal', 2.0))


def _clip_module(net, prefix, max_norm):
    """按参数名前缀裁剪梯度。"""
    params = [p for name, p in net.named_parameters()
              if name.startswith(prefix + '.') and p.grad is not None]
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def evaluate(args, config, net, dataloader, logger=None, epoch=None, verbose=True):
    """验证/测试评估"""
    prediction, output, omega_errs = _generate_preds(args, config, net, dataloader)
    results = _evaluate(prediction, output, omega_errs, logger, epoch, verbose)
    return results


def _generate_preds(args, config, net, dataloader):
    net.eval()
    with torch.no_grad():
        predictions, outputs = [], []
        omega_errs = []
        for batch in dataloader:
            geometry = batch['geometry'].to(args.device)
            target = batch['point_frf']
            frequencies = batch['frequencies']
            phi_exc = batch.get('modal_phi_exc')
            omega_true = batch.get('modal_omega')

            # 可变F: 逐样本处理
            if isinstance(frequencies, list):
                for i, freqs_i in enumerate(frequencies):
                    gd_i = _extract_single_geometry(geometry, i)
                    pe_i = phi_exc[i:i+1].to(args.device) if phi_exc is not None else None
                    # φ_exc符号对齐
                    if pe_i is not None:
                        with torch.no_grad():
                            _, _, _, phi_scan = net(gd_i, freqs_i.unsqueeze(0).to(args.device), None)
                        b_idx = geometry.batch
                        mask_i = (b_idx == i) if b_idx is not None else slice(None)
                        dot = torch.sum(phi_scan.squeeze(0) * batch['modal_phi'].to(args.device)[mask_i], dim=0)
                        pe_i = pe_i * torch.sign(dot + 1e-8).unsqueeze(0)
                    result_i = net(gd_i, freqs_i.unsqueeze(0).to(args.device), pe_i)
                    if isinstance(result_i, tuple):
                        pred_i = result_i[0]  # 模型已输出 asinh 空间, 无需再变换
                        omega_errs.append((result_i[1].cpu() - omega_true[i]).abs())
                    else:
                        pred_i = result_i
                    predictions.append(pred_i.squeeze(0).cpu())
                    outputs.append(target[i].cpu())
            else:
                target = target.to(args.device)
                frequencies = frequencies.to(args.device)
                phi_exc = phi_exc.to(args.device) if phi_exc is not None else None
                # φ_exc符号对齐
                if phi_exc is not None:
                    with torch.no_grad():
                        _, _, _, phi_scan = net(geometry, frequencies, None)
                    modal_phi = batch['modal_phi'].to(args.device)
                    phi_exc_c = phi_exc.clone()
                    if geometry.batch is not None:
                        for i in range(int(geometry.batch.max().item()) + 1):
                            m = (geometry.batch == i)
                            dot = torch.sum(phi_scan[m] * modal_phi[m], dim=0)
                            phi_exc_c[i] = phi_exc[i] * torch.sign(dot + 1e-8)
                    else:
                        # stacked: (B,N,...), phi_scan=(B*N,K), modal_phi=(B*N,K)
                        B = phi_exc.shape[0]
                        N_per = phi_scan.shape[0] // B
                        phi_scan_view = phi_scan.view(B, N_per, -1)
                        modal_phi_view = modal_phi.view(B, N_per, -1)
                        for i in range(B):
                            dot = torch.sum(phi_scan_view[i] * modal_phi_view[i], dim=0)
                            phi_exc_c[i] = phi_exc[i] * torch.sign(dot + 1e-8)
                    phi_exc = phi_exc_c
                result = net(geometry, frequencies, phi_exc)
                if isinstance(result, tuple):
                    prediction = result[0]  # 模型已输出 asinh 空间
                    if omega_true is not None:
                        omega_errs.append((result[1].detach().cpu() - omega_true).abs())
                else:
                    prediction = result
                # 对齐形状: 预测永远是展平的 (total_N,...), target 可能是 (B,N,...)
                pred_out = prediction.detach().cpu()
                tgt_out = target.detach().cpu()
                if pred_out.ndim == 3 and tgt_out.ndim == 4:
                    tgt_out = tgt_out.reshape(-1, *tgt_out.shape[2:])
                predictions.append(pred_out)
                outputs.append(tgt_out)

    try:
        return torch.cat(predictions, dim=0), torch.cat(outputs, dim=0), omega_errs
    except RuntimeError:
        return predictions, outputs, omega_errs


def _extract_single_geometry(geometry, idx):
    """从批处理的geometry中提取第idx个样本."""
    from models.geometry_data import GeometryData
    batch_idx = geometry.batch
    if batch_idx is not None:
        mask = batch_idx == idx
        pts = geometry.points[mask].unsqueeze(0)
        pf = geometry.point_features[mask].unsqueeze(0) if geometry.point_features is not None else None
        return GeometryData(points=pts, point_features=pf)
    else:
        # stacked: (B, N, ...) → 取第idx个
        pts = geometry.points[idx:idx+1]
        pf = geometry.point_features[idx:idx+1] if geometry.point_features is not None else None
        return GeometryData(points=pts, point_features=pf)


def _evaluate(prediction, output, omega_errs, logger, epoch, verbose=True):
    """评估: asinh→物理空间, 计算幅值 MAE 和百分比 MAPE."""
    if isinstance(prediction, list):
        asinh_mse_vals = [F.mse_loss(p, o).item() for p, o in zip(prediction, output)]
        results = {"loss (asinh-MSE)": np.mean(asinh_mse_vals)}
        mae_list, mape_list = [], []
        for p_asinh, o_asinh in zip(prediction, output):
            p_phys = torch.sinh(p_asinh)
            o_phys = torch.sinh(o_asinh)
            p_amp = torch.sqrt(p_phys[..., 0]**2 + p_phys[..., 1]**2 + 1e-8)
            o_amp = torch.sqrt(o_phys[..., 0]**2 + o_phys[..., 1]**2 + 1e-8)
            mae_list.append(F.l1_loss(p_amp, o_amp).item())
            mape_list.append((torch.abs(p_amp - o_amp) / (o_amp + 1e-6)).mean().item() * 100.0)
        results["Amplitude MAE"] = np.mean(mae_list)
        results["Amplitude MAPE (%)"] = np.mean(mape_list)
    else:
        results = {}
        # 兜底: 确保 prediction 和 output 形状一致
        if prediction.shape != output.shape:
            output = output.reshape(prediction.shape)
        results["loss (asinh-MSE)"] = F.mse_loss(prediction, output).item()
        if prediction.ndim >= 3 and prediction.shape[-1] == 2:
            p_phys = torch.sinh(prediction)
            o_phys = torch.sinh(output)
            p_amp = torch.sqrt(p_phys[..., 0]**2 + p_phys[..., 1]**2 + 1e-8)
            o_amp = torch.sqrt(o_phys[..., 0]**2 + o_phys[..., 1]**2 + 1e-8)
            results["Amplitude MAE"] = F.l1_loss(p_amp, o_amp).item()
            results["Amplitude MAPE (%)"] = (torch.abs(p_amp - o_amp) / (o_amp + 1e-6)).mean().item() * 100.0

    # ω误差
    if omega_errs:
        results["ω_MAE (rad/s)"] = torch.cat([e.flatten() for e in omega_errs]).mean().item()

    if verbose:
        for key, val in results.items():
            _log(f"{key} = {val:4.4f}" if isinstance(val, float) else f"{key} = {val:4.4}", logger)

    return results


def save_model(savepath, epoch, model, optimizer, loss, name="checkpoint_best"):
    os.makedirs(savepath, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(savepath, name))



def _log(msg, logger):
    """简易日志: 若 logger 可用则用 logger，否则 print"""
    if logger and hasattr(logger, 'info'):
        logger.info(msg)
    else:
        print(msg)
