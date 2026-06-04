"""
frf_model.py — 模型构建入口。

构建 GrooveTransFRF: 几何感知物理 Transformer 用于 FRF 预测。
"""
from .groove_transfrf import GrooveTransFRF


def build_geometric_model(encoder_kwargs=None, decoder_kwargs=None):
    """构建 GrooveTransFRF 模型。"""
    kwargs = {}
    kwargs.update(encoder_kwargs or {})
    kwargs.update(decoder_kwargs or {})
    return GrooveTransFRF(**kwargs)
