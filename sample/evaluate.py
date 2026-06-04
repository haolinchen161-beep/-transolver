"""
evaluate.py — 训练后评估+可视化。
加载检查点 → 预测模态参数 → 物理重建FRF → 对比+保存。
支持可变节点数和可变频率点数 (ANSYS per-sample-group 格式)。

用法: F:\pytorch_cuda12\python.exe geometric_frf/sample/evaluate.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from models import build_geometric_model, GeometryData
from data.dataset import GeometricHDF5Dataset

CONFIG = {'freq_min': 1.0, 'freq_max': 5000.0}
MODEL_CFG = {
    'encoder_kwargs': {
        'coord_dim': 3, 'point_feat_dim': 7,
        'hidden_dim': 256, 'n_modes': 3,
        'n_transolver_layers': 3, 'num_heads': 8, 'slice_num': 64, 'dropout': 0.05,
        'amp_scale': 500000.0, 'freq_min': 1.0, 'freq_max': 5000.0,
    },
    'decoder_kwargs': {},
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join(os.path.dirname(__file__), "..", "ansys", "data")
out_dir  = os.path.join(os.path.dirname(__file__), "output")
ckpt_path = os.path.join(out_dir, "checkpoint_best")


def main():
    print("=" * 60)
    print("模型评估 + 可视化 (模态参数预测)")
    print("=" * 60)

    testset = GeometricHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir,
                                   normalization=True, test=True)
    testset_raw = GeometricHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir,
                                       normalization=False, test=True)
    print(f"测试集: {len(testset)} 样本")

    model = build_geometric_model(MODEL_CFG['encoder_kwargs'],
                                  MODEL_CFG['decoder_kwargs']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Checkpoint: epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    all_preds, all_targets, all_freqs = [], [], []
    all_preds_re, all_preds_im = [], []
    all_targets_re, all_targets_im = [], []
    all_points_list = []
    omega_errs, zeta_errs = [], []

    for idx in range(len(testset)):
        s_norm = testset[idx]; s_raw = testset_raw[idx]
        gd = GeometryData(
            points=s_norm['geometry'].points.unsqueeze(0),
            point_features=s_norm['geometry'].point_features.unsqueeze(0)
                if s_norm['geometry'].point_features is not None else None,
        ).to(device)
        phi_exc = s_norm.get('modal_phi_exc')
        phi_exc_t = phi_exc.unsqueeze(0).to(device) if phi_exc is not None else None
        with torch.no_grad():
            if phi_exc_t is not None:
                _, _, _, phi_scan = model(gd, s_norm['frequencies'].unsqueeze(0).to(device), None)
                modal_phi_i = s_norm['modal_phi'].unsqueeze(0).to(device)
                dot = torch.sum(phi_scan * modal_phi_i, dim=1)
                phi_exc_t = phi_exc_t * torch.sign(dot + 1e-8)
            frf_p, op, zp, pp = model(gd, s_norm['frequencies'].unsqueeze(0).to(device), phi_exc_t)
        frf_p = frf_p.squeeze(0).cpu()
        p = torch.clamp(frf_p, -5000, 5000)
        t = s_raw['point_frf']  # 已是物理空间, 无需 undo_normalize

        omega_errs.append((op.cpu() - s_norm['modal_omega']).abs())
        zeta_errs.append((zp.cpu() - s_norm['modal_zeta']).abs())

        all_preds.append(torch.sqrt(p[...,0]**2+p[...,1]**2+1e-8).numpy())
        all_targets.append(torch.sqrt(t[...,0]**2+t[...,1]**2+1e-8).numpy())
        all_preds_re.append(p[...,0].numpy()); all_preds_im.append(p[...,1].numpy())
        all_targets_re.append(t[...,0].numpy()); all_targets_im.append(t[...,1].numpy())
        all_freqs.append(s_raw['frequencies'].numpy())
        all_points_list.append(s_raw['geometry'].points.numpy())

    # 可变大小 → object 数组
    def to_obj(arr_list):
        out = np.empty(len(arr_list), dtype=object)
        for i, a in enumerate(arr_list):
            out[i] = a
        return out

    omega_mae = torch.cat(omega_errs).mean().item()
    zeta_mae = torch.cat(zeta_errs).mean().item()
    # 逐样本MSE (标量均值)
    mse_vals = [np.mean((all_preds[i] - all_targets[i])**2) for i in range(len(all_preds))]
    l1_vals = [np.mean(np.abs(all_preds[i] - all_targets[i])) for i in range(len(all_preds))]
    print(f"幅值MSE={np.mean(mse_vals):.1f} L1={np.mean(l1_vals):.1f} | ω_MAE={omega_mae:.1f}rad/s ζ_MAE={zeta_mae:.5f}")

    np.savez(os.path.join(out_dir, "final_results.npz"),
             points=to_obj(all_points_list), frequencies=to_obj(all_freqs),
             predicted_frf=to_obj(all_preds), target_frf=to_obj(all_targets),
             predicted_re=to_obj(all_preds_re), target_re=to_obj(all_targets_re),
             predicted_im=to_obj(all_preds_im), target_im=to_obj(all_targets_im))
    print(f"数据保存: {out_dir}/final_results.npz")
    print(f"评估完成!")


if __name__ == '__main__':
    main()
