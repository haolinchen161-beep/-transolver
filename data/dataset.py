"""
dataset.py — 几何数据 Dataset + DataLoader。

支持 per-sample-group HDF5 格式 (ANSYS 凹槽工件数据):
  /sample_0/points, /sample_0/point_frf, /sample_0/frequencies, ...
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os

from models.geometry_data import GeometryData


class GeometricHDF5Dataset(Dataset):
    """几何数据 HDF5 数据集 (per-sample-group 格式)。"""

    def __init__(self, data_paths, config, data_dir=".",
                 test=False, normalization=True):
        self.config = config
        self.normalization = normalization
        self.test = test
        self.freq_min = config.get('freq_min', 1.0)
        self.freq_max = config.get('freq_max', 5000.0)
        self._samples = []  # [(file_path, group_name), ...]

        full_paths = [os.path.join(data_dir, p) for p in data_paths]
        self._load_index(full_paths)

    def _load_index(self, full_paths):
        """扫描所有 HDF5 文件, 建立样本索引。"""
        for fp in full_paths:
            with h5py.File(fp, 'r') as f:
                for key in sorted(f.keys(), key=lambda k: int(k.split('_')[-1])):
                    if key.startswith('sample_'):
                        self._samples.append((fp, key))
        if len(self._samples) == 0:
            raise RuntimeError(f"未找到 per-sample-group 格式数据: {full_paths}")

    def undo_normalize(self, frf):
        """asinh → 物理空间"""
        return torch.sinh(frf)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        fp, grp_name = self._samples[idx]
        with h5py.File(fp, 'r') as f:
            grp = f[grp_name]
            points = torch.from_numpy(grp['points'][:]).float()
            freqs = torch.from_numpy(grp['frequencies'][:]).float()
            frf = torch.from_numpy(grp['point_frf'][:]).float()

            # 逐节点特征
            point_feat = None
            if 'point_features' in grp:
                gf = torch.from_numpy(grp['point_features'][:]).float()
                if gf.ndim == 1:
                    # 兼容旧格式: 全局特征广播到每个节点
                    point_feat = gf.unsqueeze(0).expand(points.shape[0], -1)
                else:
                    point_feat = gf

            # 模态参数
            out = {}
            for key in ['modal_omega', 'modal_zeta', 'modal_phi', 'modal_phi_exc']:
                if key in grp:
                    out[key] = torch.from_numpy(grp[key][:]).float()

        # 归一化
        if self.normalization:
            freqs = (freqs - self.freq_min) / (self.freq_max - self.freq_min) * 2 - 1
            frf = torch.asinh(frf)

        geometry = GeometryData(points=points, point_features=point_feat)
        result = {'geometry': geometry, 'point_frf': frf, 'frequencies': freqs}
        for key, val in out.items():
            result[key] = val
        return result


def collate_geometry_batch(batch):
    """批次整理: 同节点数→stack, 不同→拼接。可变F→list。"""
    n_points_list = [item['geometry'].points.shape[0] for item in batch]
    all_same_n = all(n == n_points_list[0] for n in n_points_list)
    f_lens = [item['frequencies'].shape[0] for item in batch]
    all_same_f = all(f == f_lens[0] for f in f_lens)

    if all_same_f:
        frequencies = torch.stack([item['frequencies'] for item in batch])
        if all_same_n:
            point_frf = torch.stack([item['point_frf'] for item in batch])
            points = torch.stack([item['geometry'].points for item in batch])
            point_feat = torch.stack([item['geometry'].point_features for item in batch]) \
                         if batch[0]['geometry'].point_features is not None else None
            geometry = GeometryData(points=points, point_features=point_feat,
                                    edge_index=None, batch=None)
        else:
            all_points, all_features, all_frfs, all_batch = [], [], [], []
            for i, item in enumerate(batch):
                n_pts = item['geometry'].points.shape[0]
                all_points.append(item['geometry'].points)
                all_frfs.append(item['point_frf'])
                all_batch.append(torch.full((n_pts,), i, dtype=torch.long))
                if item['geometry'].point_features is not None:
                    all_features.append(item['geometry'].point_features)
            points = torch.cat(all_points, dim=0)
            point_frf = torch.cat(all_frfs, dim=0)
            point_feat = torch.cat(all_features, dim=0) if all_features else None
            batch_tensor = torch.cat(all_batch, dim=0)
            geometry = GeometryData(points=points, point_features=point_feat,
                                    edge_index=None, batch=batch_tensor)
    else:
        # 可变F: FRF和频率不可stack
        frequencies = [item['frequencies'] for item in batch]
        point_frf = [item['point_frf'] for item in batch]
        all_points, all_features, all_batch = [], [], []
        for i, item in enumerate(batch):
            n_pts = item['geometry'].points.shape[0]
            all_points.append(item['geometry'].points)
            all_batch.append(torch.full((n_pts,), i, dtype=torch.long))
            if item['geometry'].point_features is not None:
                all_features.append(item['geometry'].point_features)
        points = torch.cat(all_points, dim=0)
        point_feat = torch.cat(all_features, dim=0) if all_features else None
        batch_tensor = torch.cat(all_batch, dim=0)
        geometry = GeometryData(points=points, point_features=point_feat,
                                edge_index=None, batch=batch_tensor)

    out = {'geometry': geometry, 'point_frf': point_frf, 'frequencies': frequencies}
    modal = _stack_modal(batch)
    if modal:
        out.update(modal)
    return out


def _stack_modal(batch):
    """堆叠模态参数。"""
    for key in ['modal_omega', 'modal_zeta', 'modal_phi']:
        if key not in batch[0] or batch[0][key] is None:
            return {}
    result = {}
    for key in ['modal_omega', 'modal_zeta', 'modal_phi_exc']:
        if key in batch[0] and batch[0][key] is not None:
            result[key] = torch.stack([item[key] for item in batch])
    result['modal_phi'] = torch.cat([item['modal_phi'] for item in batch], dim=0)
    return result


def get_geometric_dataloader(args, config, data_dir=".", num_workers=0,
                             shuffle=True, normalization=True):
    """构建 DataLoader。"""
    batch_size = args.batch_size
    torch.manual_seed(args.seed)

    trainset = GeometricHDF5Dataset(
        config['data_path_train'], config, data_dir=data_dir,
        normalization=normalization, test=False,
    )
    valset = GeometricHDF5Dataset(
        config['data_path_val'], config, data_dir=data_dir,
        normalization=normalization, test=True,
    )
    testset = None
    if config.get('data_path_test') is not None:
        testset = GeometricHDF5Dataset(
            config['data_path_test'], config, data_dir=data_dir,
            normalization=normalization, test=True,
        )
    else:
        testset = valset

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, drop_last=shuffle, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_geometry_batch,
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=2, drop_last=False, shuffle=False,
        num_workers=num_workers, collate_fn=collate_geometry_batch,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=2, drop_last=False, shuffle=False,
        num_workers=num_workers, collate_fn=collate_geometry_batch,
    )
    return trainloader, valloader, testloader, trainset, valset, testset
