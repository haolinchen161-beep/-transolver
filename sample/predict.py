"""
predict.py — 加载 checkpoint 对测试样本预测 FRF。
支持可变节点数和频率点数。

用法: F:\pytorch_cuda12\python.exe geometric_frf/sample/predict.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
from models import build_geometric_model, GeometryData
from data.dataset import GeometricHDF5Dataset

CONFIG = {'freq_min': 1.0, 'freq_max': 5000.0}
MODEL_CFG = {
    'encoder_kwargs': {
        'coord_dim': 3, 'point_feat_dim': 7,
        'hidden_dim': 256, 'n_modes': 3,
        'n_transolver_layers': 3, 'num_heads': 8, 'slice_num': 64, 'dropout': 0.1,
        'amp_scale': 500000.0, 'freq_min': 1.0, 'freq_max': 5000.0,
    },
    'decoder_kwargs': {},
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join(os.path.dirname(__file__), "..", "ansys", "data")
out_dir  = os.path.join(os.path.dirname(__file__), "output")


def main():
    testset = GeometricHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir, normalization=True, test=True)
    testset_raw = GeometricHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir, normalization=False, test=True)

    net = build_geometric_model(MODEL_CFG['encoder_kwargs'], MODEL_CFG['decoder_kwargs']).to(device)
    ckpt = torch.load(os.path.join(out_dir, "checkpoint_best"), map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    print(f"Checkpoint epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f}, params={sum(p.numel() for p in net.parameters()):,}")

    all_preds, all_targets = [], []
    for idx in range(len(testset)):
        sn, sr = testset[idx], testset_raw[idx]
        gd = GeometryData(points=sn['geometry'].points.unsqueeze(0),
                          point_features=sn['geometry'].point_features.unsqueeze(0)
                          if sn['geometry'].point_features is not None else None).to(device)
        phi_exc = sn.get('modal_phi_exc')
        phi_exc_t = phi_exc.unsqueeze(0).to(device) if phi_exc is not None else None
        with torch.no_grad():
            if phi_exc_t is not None:
                _, _, _, phi_scan = net(gd, sn['frequencies'].unsqueeze(0).to(device), None)
                modal_phi_i = sn['modal_phi'].unsqueeze(0).to(device)
                dot = torch.sum(phi_scan * modal_phi_i, dim=1)
                phi_exc_t = phi_exc_t * torch.sign(dot + 1e-8)
            frf_p, _, _, _ = net(gd, sn['frequencies'].unsqueeze(0).to(device), phi_exc_t)
        p = torch.clamp(frf_p.squeeze(0).cpu(), -5000, 5000)
        t = testset_raw.undo_normalize(sn['point_frf'])
        all_preds.append(torch.sqrt(p[...,0]**2+p[...,1]**2+1e-8).numpy())
        all_targets.append(torch.sqrt(t[...,0]**2+t[...,1]**2+1e-8).numpy())

    mse_vals = [np.mean((all_preds[i] - all_targets[i])**2) for i in range(len(all_preds))]
    print(f"测试集幅值MSE (均值): {np.mean(mse_vals):.6f}")

    def to_obj(arr_list):
        out = np.empty(len(arr_list), dtype=object)
        for i, a in enumerate(arr_list):
            out[i] = a
        return out
    np.savez(os.path.join(out_dir, "predictions.npz"),
             predicted=to_obj(all_preds), target=to_obj(all_targets))
    print(f"保存: {out_dir}/predictions.npz")


if __name__ == '__main__':
    main()
