"""
augmentations.py — 几何数据增强。

针对凹槽工件数据集 (可变拓扑+可变边界条件) 设计的增强策略。
所有增强保持物理一致性: 坐标扰动模拟不同网格种子,
节点 dropout 强制鲁棒表示, 频率子采样提升频谱泛化。

用法:
    from training.augmentations import GeometryAugmenter
    augmenter = GeometryAugmenter(...)
    batch = augmenter(batch)  # 原地修改 geometry + point_frf + frequencies
"""

import torch
import numpy as np


class GeometryAugmenter:
    """
    几何数据增强器。

    增强策略:
      1. 坐标扰动: N(0, coord_noise) — 模拟不同网格离散化
      2. 特征噪声: N(0, feat_noise) — 模拟材料/BC 不确定性
      3. 节点 Dropout: 随机丢弃 node_dropout 比例的节点
      4. 频率子采样: 从 F 个频率点中随机选取 freq_subsample 个

    所有增强仅用于训练, 验证/测试时自动跳过。
    """

    def __init__(self,
                 coord_noise: float = 1e-4,      # 坐标噪声标准差 (m), 1e-4m=0.1mm
                 feat_noise_scale: float = 0.005,  # 特征噪声相对幅度
                 node_dropout: float = 0.15,       # 节点丢弃比例
                 freq_subsample: int = 32,         # 频率子采样点数 (从40中选32)
                 enabled: bool = True):
        self.coord_noise = coord_noise
        self.feat_noise_scale = feat_noise_scale
        self.node_dropout = node_dropout
        self.freq_subsample = freq_subsample
        self.enabled = enabled
        self.training = True  # 默认训练模式

    def __call__(self, batch):
        """对批次数据原地应用增强。"""
        if not self.enabled:
            return batch
        if not self.training:
            return batch

        batch = self._augment_coords(batch)
        batch = self._augment_features(batch)
        batch = self._augment_node_dropout(batch)
        batch = self._augment_frequencies(batch)
        return batch

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    # ---- 内部方法 ----

    def _augment_coords(self, batch):
        """坐标高斯噪声: 模拟不同网格种子产生的节点位置差异。"""
        geometry = batch['geometry']
        points = geometry.points
        noise = torch.randn_like(points) * self.coord_noise
        # 不扰动装夹节点 (保持 BC 几何精确)
        if geometry.point_features is not None:
            is_fixed = geometry.point_features[:, 3]  # is_fixed
            noise[is_fixed > 0.5] *= 0.1  # 装夹节点噪声降低 90%
        geometry.points = points + noise
        return batch

    def _augment_features(self, batch):
        """特征噪声: 模拟材料属性测量不确定性和 BC 参数波动。"""
        geometry = batch['geometry']
        feat = geometry.point_features
        if feat is not None:
            # 逐特征维度的噪声尺度
            noise_scales = torch.tensor([
                0.005,  # 0: E/E_base
                0.0,    # 1: PRXY (固定)
                0.003,  # 2: rho/rho_base
                0.0,    # 3: is_fixed (固定)
                0.05,   # 4: log10(K)
                0.05,   # 5: log10(C)
                0.005,  # 6: Z/H (局部厚度, ±0.5%)
            ], device=feat.device)
            noise = torch.randn_like(feat) * noise_scales * self.feat_noise_scale
            geometry.point_features = feat + noise
        return batch

    def _augment_node_dropout(self, batch):
        """
        节点 Dropout: 随机丢弃部分节点。
        强制模型学习不依赖特定网格节点的鲁棒表示。
        保留所有 BC 节点 (is_fixed > 0) 以确保边界条件完整。
        """
        if self.node_dropout <= 0:
            return batch

        geometry = batch['geometry']
        point_frf = batch['point_frf']
        points = geometry.points
        batch_idx = geometry.batch
        point_feat = geometry.point_features

        B = int(batch_idx.max().item()) + 1
        keep_masks = []

        for b in range(B):
            mask_b = batch_idx == b
            N_b = int(mask_b.sum().item())

            # BC 节点必须保留
            if point_feat is not None:
                is_bc = point_feat[mask_b][:, 3] > 0  # is_fixed > 0
            else:
                is_bc = torch.zeros(N_b, dtype=torch.bool, device=points.device)

            bc_indices = torch.where(mask_b)[0][is_bc]
            interior_mask = mask_b.clone()
            interior_mask[bc_indices] = False
            interior_indices = torch.where(interior_mask)[0]

            # 从内部节点中随机丢弃
            n_interior = len(interior_indices)
            n_drop = int(n_interior * self.node_dropout)
            if n_drop > 0 and n_interior > 0:
                drop_local = torch.randperm(n_interior, device=points.device)[:n_drop]
                # 映射: 局部内部索引 → 局部样本索引 (跳过BC节点)
                local_interior_pos = torch.where(~is_bc)[0]
                keep_mask = torch.ones(N_b, dtype=torch.bool, device=points.device)
                keep_mask[local_interior_pos[drop_local]] = False
            else:
                keep_mask = torch.ones(N_b, dtype=torch.bool, device=points.device)

            keep_masks.append(keep_mask)

        keep_all = torch.cat(keep_masks, dim=0)

        # 应用 mask
        geometry.points = points[keep_all]
        if geometry.point_features is not None:
            geometry.point_features = point_feat[keep_all]
        geometry.batch = batch_idx[keep_all]
        if point_frf is not None:
            if isinstance(point_frf, list):
                batch['point_frf'] = [
                    frf_i[keep_masks[i]] for i, frf_i in enumerate(point_frf)
                ]
            else:
                batch['point_frf'] = point_frf[keep_all]

        # 同样处理 modal_phi (如果存在)
        if 'modal_phi' in batch and batch['modal_phi'] is not None:
            batch['modal_phi'] = batch['modal_phi'][keep_all]

        return batch

    def _augment_frequencies(self, batch):
        """
        频率子采样: 从 F 个频率点中随机选取 freq_subsample 个。
        强制模型学习频率轴上的插值能力。
        """
        if self.freq_subsample <= 0:
            return batch

        frequencies = batch['frequencies']
        frf = batch['point_frf']

        if isinstance(frequencies, list):
            new_freqs, new_frfs = [], []
            for i, freqs_i in enumerate(frequencies):
                F_i = len(freqs_i)
                n_sample = min(self.freq_subsample, F_i)
                indices = torch.randperm(F_i, device=freqs_i.device)[:n_sample]
                indices = torch.sort(indices)[0]  # 保持频率单调递增
                new_freqs.append(freqs_i[indices])
                if isinstance(frf, list):
                    new_frfs.append(frf[i][:, indices, :])
            batch['frequencies'] = new_freqs
            if isinstance(frf, list):
                batch['point_frf'] = new_frfs
        else:
            # stacked 格式: (B, F) / (B, N, F, 2)
            B, F_all = frequencies.shape
            n_sample = min(self.freq_subsample, F_all)
            indices = torch.randperm(F_all, device=frequencies.device)[:n_sample]
            indices = torch.sort(indices)[0]
            batch['frequencies'] = frequencies[:, indices]
            if frf.ndim == 4:
                batch['point_frf'] = frf[:, :, indices, :]
            elif frf.ndim == 3:
                batch['point_frf'] = frf[:, indices, :]

        return batch


def create_augmenter(config):
    """从配置创建 GeometryAugmenter。"""
    aug_cfg = config.get('augmentation', {})
    return GeometryAugmenter(
        coord_noise=aug_cfg.get('coord_noise', 1e-4),
        feat_noise_scale=aug_cfg.get('feat_noise_scale', 0.005),
        node_dropout=aug_cfg.get('node_dropout', 0.15),
        freq_subsample=aug_cfg.get('freq_subsample', 32),
        enabled=aug_cfg.get('enabled', True),
    )
