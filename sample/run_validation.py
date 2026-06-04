"""
GrooveTransFRF 模态参数预测训练 — ANSYS 凹槽工件。
用法: F:\pytorch_cuda12\python.exe sample/run_validation.py
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore', message='Detected call of')
warnings.filterwarnings('ignore', message='To get the last learning rate')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, torch
from models import build_geometric_model
from training import train, evaluate, modal_loss

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'epochs': 2000,
    'validation_frequency': 5,

    # 两阶段边界
    'phase1_epochs': 150,       # Phase1: 全解冻纯模态, 动态切换由omega<5%或epoch>200控制

    # 损失权重
    'frf_loss_weight': 50.0,
    'zeta_loss_weight': 200.0,

    # 频率范围
    'freq_min': 1.0,
    'freq_max': 5000.0,

    # 数据路径
    'data_path_train': ['train.h5'],
    'data_path_val': ['val.h5'],
    'data_path_test': ['test.h5'],

    # 数据增强
    'augmentation': {
        'enabled': True,
        'coord_noise': 1e-4,
        'feat_noise_scale': 0.005,
        'node_dropout': 0.15,
        'freq_subsample': 32,
    },

    # 优化器
    'optimizer': {
        'name': 'AdamW',
        'kwargs': {'lr': 0.0003, 'weight_decay': 0.00008, 'betas': (0.9, 0.999)},
        'gradient_clip': 2.0,
        'gradient_clip_transolver': 1.0,
        'gradient_clip_head_phi': 5.0,
        'gradient_clip_modal': 2.0,
    },
}

MODEL_CFG = {
    'encoder_kwargs': {
        'coord_dim': 3,
        'point_feat_dim': 7,
        'hidden_dim': 256,
        'n_modes': 3,
        'n_transolver_layers': 3,
        'num_heads': 8,
        'slice_num': 64,
        'dropout': 0.1,
        'amp_scale': 500000.0,
        'freq_min': 1.0,
        'freq_max': 5000.0,
    },
    'decoder_kwargs': {},
}


class SimpleArgs:
    def __init__(self):
        self.batch_size = 4
        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fp16 = False
        self.dir = os.path.join(os.path.dirname(__file__), "output")
        self.debug = False


def main():
    print("=" * 60)
    print("GrooveTransFRF 模态参数预测 — ANSYS 3D 凹槽工件")
    print("=" * 60)
    args = SimpleArgs()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "ansys", "data")
    print(f"设备: {args.device}, FP16: {args.fp16}, Batch: {args.batch_size}")

    # 步骤1: 数据
    print("\n--- 步骤1: 构建 DataLoader ---")
    from data.dataset import GeometricHDF5Dataset, collate_geometry_batch

    trainset = GeometricHDF5Dataset(['train.h5'], CONFIG, data_dir=data_dir, normalization=True, test=False)
    valset = GeometricHDF5Dataset(['val.h5'], CONFIG, data_dir=data_dir, normalization=True, test=True)
    testset = GeometricHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir, normalization=True, test=True)

    gen = torch.Generator(device='cpu').manual_seed(args.seed)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_geometry_batch, generator=gen)
    valloader = torch.utils.data.DataLoader(valset, batch_size=2, drop_last=False, shuffle=False,
        num_workers=0, collate_fn=collate_geometry_batch)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, drop_last=False, shuffle=False,
        num_workers=0, collate_fn=collate_geometry_batch)

    batch = next(iter(trainloader))
    gd = batch['geometry']
    print(f"  训练: {len(trainset)}样本, {len(trainloader)}批次")
    print(f"  points: {gd.points.shape}, freq: {batch['frequencies'].shape}")
    print(f"  omega: {batch['modal_omega'].shape}, phi: {batch['modal_phi'].shape}")

    # 步骤2: 模型
    print("\n--- 步骤2: 构建模型 ---")
    net = build_geometric_model(MODEL_CFG['encoder_kwargs'], MODEL_CFG['decoder_kwargs']).to(args.device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"  参数量: {total_params:,}")

    # 步骤3: 前向测试
    print("\n--- 步骤3: 前向传播测试 ---")
    net.eval()
    with torch.no_grad():
        phi_exc = batch.get('modal_phi_exc')
        phi_exc = phi_exc.to(args.device) if phi_exc is not None else None
        frf_p, op, zp, pp = net(gd.to(args.device), batch['frequencies'].to(args.device), phi_exc)
    print(f"  frf={list(frf_p.shape)}, omega={list(op.shape)}, phi={list(pp.shape)}")
    print("  前向传播 PASS")

    # 步骤4: 初始Loss
    print("\n--- 步骤4: 初始 Loss ---")
    with torch.no_grad():
        init_loss, _, _ = modal_loss(
            op, batch['modal_omega'].to(args.device),
            zp, batch['modal_zeta'].to(args.device),
            pp, batch['modal_phi'].to(args.device))
        tgt = batch['point_frf']
        if isinstance(tgt, list):
            init_mse = torch.nn.functional.mse_loss(
                torch.asinh(frf_p[0:1].clamp(-1e4, 1e4)),
                tgt[0].unsqueeze(0).to(args.device)).item()
        else:
            init_mse = torch.nn.functional.mse_loss(
                torch.asinh(frf_p.clamp(-1e4, 1e4)),
                tgt.to(args.device)).item()
    print(f"  初始 Loss: {init_loss.item():.0f}, MSE: {init_mse:.4f}")

    # 步骤5: 训练
    print("\n--- 步骤5: 训练 ---")
    optimizer = torch.optim.AdamW(net.parameters(), **CONFIG['optimizer']['kwargs'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6)
    start_epoch = 0

    ckpt_path = os.path.join(args.dir, "checkpoint_last")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=args.device)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  从 epoch {start_epoch} 续训")

    print(f"  训练 {CONFIG['epochs']} epochs...")
    t0 = time.time()
    net = train(args, CONFIG, MODEL_CFG, net, trainloader, optimizer, valloader, scheduler, logger=None, start_epoch=start_epoch)
    elapsed = time.time() - t0
    print(f"  完成, 耗时 {elapsed:.1f}s")

    # 步骤6: 验证
    print("\n--- 步骤6: 验证 ---")
    best_path = os.path.join(args.dir, "checkpoint_best")
    if os.path.exists(best_path):
        net.load_state_dict(torch.load(best_path, map_location=args.device)["model_state_dict"])
    results = evaluate(args, CONFIG, net, testloader, verbose=True)

    print(f"\n{'='*60}")
    print(f"验证完成 | 设备:{args.device} | 参数:{total_params:,} | 耗时:{elapsed:.0f}s")
    print(f"初始MSE:{init_mse:.4f} | Test MSE:{results.get('loss (asinh-MSE)', -1):.4f}")
    return 0


if __name__ == '__main__':
    exit(main())
