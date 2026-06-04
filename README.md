# geometric_frf_groove — 基于几何的频响函数预测

输入 3D 几何 + 边界条件 → GrooveTransFRF (几何感知物理 Transformer) → 模态参数 (ω, ζ, φ) → 物理公式重建 FRF。

## 1. 目录结构

```
├── models/
│   ├── groove_transfrf.py      主模型: GrooveTransFRF
│   ├── transolver_encoder.py   Transolver 几何编码器 (SliceAttention)
│   ├── frf_model.py             模型工厂 build_geometric_model()
│   ├── bc_aware_decoder.py     全局注意力池化
│   ├── physics_decoder.py       无参数物理解码器 (模态叠加→FRF)
│   ├── geometry_data.py         GeometryData 数据容器
├── data/
│   └── dataset.py               HDF5 数据集 (per-sample-group) + collate
├── training/
│   ├── losses.py                modal_loss + frf_loss
│   ├── trainer.py               两阶段训练循环 + 评估
│   └── augmentations.py         数据增强 (坐标/特征噪声, 节点dropout, 频率子采样)
├── ansys/
│   ├── generate_3d_test.py      ANSYS MAPDL 数据生成 (凹槽工件)
│   ├── data/                    train/val/test.h5
│   └── mesh_viz/                网格截图
└── sample/
    ├── run_validation.py        训练入口
    ├── evaluate.py              评估 + 保存 final_results.npz
    ├── 测试.py                  查看原始 FRF
    ├── 对比图.py                预测 vs 真实对比图
    ├── predict.py               推理
    └── output/                  checkpoint + 图表 + npz
```

## 2. 架构

```
输入: points(N,3) + point_features(N,7) + frequencies(B,F) + phi_exc(B,K)
                    │
    ┌───────────────┴───────────────┐
    │  TransolverEncoder            │  SliceAttention (全分辨率, 无降采样)
    │  → node_tokens (N, 256)       │
    └───────────────┬───────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
    ┌────┴────┐          ┌─────┴──────┐
    │head_phi │          │BC-Aware    │  可学习模态token交叉注意力
    │→φ_k(x)  │          │ModalDecoder│
    │(N,K)    │          │→Δω,ζ (B,2K)│
    └────┬────┘          └─────┬──────┘
         │                     │
    ┌────┴─────────────────────┴──────┐
    │  ω = softplus(macro(4标量))×15000 + tanh(micro(modal_out[:K]))×5000  │
    │  ζ = softplus(Δζ)×0.004 + 1e-4            │
    └────────────────┬────────────────┘
                     │
    ┌────────────────┴────────────────┐
    │  PhysicsDecoder (无参数)         │  H=Σφ_k(x)φ_k(x_f)/(ω_k²-ω²+j2ζ_kω_kω)
    │  → FRF (N, F, 2) [Re, Im]      │
    └─────────────────────────────────┘
```

## 3. 数据

### ANSYS 凹槽工件

| 参数 | 值 |
|------|-----|
| 工件尺寸 | 160×60×10mm (铝7075, E=71.7GPa, ρ=2810) |
| 凹槽 | 5/6/7 方案, 深度随机 30-60% 厚度 |
| 装夹 | 4角 XYZ 弹簧 + 3侧面 Y 向弹簧 (COMBIN14) |
| 弹簧刚度 | K∈[1e6, 1e8] N/m, 随机 |
| 弹簧阻尼 | C∈[10, 500] N·s/m, 随机 |
| 阻尼 | ζ=材料0.002 + 边界耗散 (物理公式) |
| 模态 | 前3阶, Z向振型 |
| 网格 | 6mm 自由四面体 (SOLID187), ~4k 节点/样本 |
| 频率 | 80点自适应网格 |

### HDF5 格式 (per-sample-group)

```
/sample_0/
├── points         (N₀, 3)       节点坐标 [x, y, z] (m)
├── point_frf      (N₀, F₀, 2)  复数频响函数 [Re, Im]
├── frequencies    (F₀,)         频率采样点 (Hz)
├── point_features (N₀, 7)       逐节点特征:
│                                [E/E_base, PRXY, ρ/ρ_base, is_fixed, log10(K), log10(C), Z/H]
├── modal_omega    (K,)          固有圆频率 (rad/s)
├── modal_zeta     (K,)          阻尼比
├── modal_phi      (N₀, K)       模态振型 φ_k(x)
└── modal_phi_exc  (K,)          激励点振型值
```

## 4. 训练

### 两阶段策略

| 阶段 | Epoch | 动作 | 损失 | 目的 |
|------|-------|------|------|------|
| 1: 模态预热 | 0-300 | 全解冻, φ权重×1 | modal_loss | ω/ζ 独享梯度, 快速收敛 |
| 2: FRF联合 | 300-2000 | 全解冻, φ权重×100 | modal + FRF×50 | 端到端全耦合 |

### 超参数

| 参数 | 值 |
|------|-----|
| 模型 | GrooveTransFRF (~1.95M params) |
| hidden_dim / num_heads | 256 / 8 |
| n_transolver_layers / slice_num | 3 / 64 |
| dropout | 0.1 |
| 损失权重 | rel_ω×20K + rel_ζ×200 + φ×100 + FRF×50 |
| 优化器 | AdamW, lr=3e-4, wd=8e-5, ReduceLROnPlateau |
| 梯度裁剪 | Transolver=3.0, head_phi=5.0, Modal=2.0 |
| batch_size | 4 |

## 5. 快速开始

```bash
# 生成数据 (需 ANSYS MAPDL license)
F:/pytorch_cuda12/python.exe ansys/generate_3d_test.py

# 查看原始 FRF
F:/pytorch_cuda12/python.exe sample/测试.py

# 训练
F:/pytorch_cuda12/python.exe sample/run_validation.py

# 评估
F:/pytorch_cuda12/python.exe sample/evaluate.py

# 对比图
F:/pytorch_cuda12/python.exe sample/对比图.py
```
