"""
geometry_data.py — 统一几何数据容器。

定义 GeometryData dataclass，在不同模块间传递几何信息。
支持点坐标、可选点特征、网格拓扑边索引等几何表示。
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class GeometryData:
    """
    统一几何数据容器。

    属性:
        points:         点坐标 (N, 3) 或 batch模式 (B*N, 3) 或 (B, N, 3)
        point_features: 可选点特征 (N, F)，如法向量、厚度、材料参数等
        edge_index:     可选边索引 (2, E)，网格单元拓扑连接，用于GNN
        batch:          batch索引 (N,)，标识每个点属于哪个样本（变点数时使用）
    """
    points: torch.Tensor
    point_features: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    batch: Optional[torch.Tensor] = None

    def to(self, device):
        """将所有张量移动到指定设备"""
        self.points = self.points.to(device)
        if self.point_features is not None:
            self.point_features = self.point_features.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self
