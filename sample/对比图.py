"""
对比图：指定样本和坐标，画幅值+实部+虚部预测vs真实。
支持 variable N/F 的 object-array npz 格式。

用法: F:\pytorch_cuda12\python.exe geometric_frf/sample/对比图.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

# ============ 配置 ============
SAMPLE_IDX  = 0           # 样本索引
POINT_X     = 0.095       # 悬臂自由端附近 (L≈0.10)
POINT_Y     = 0.055       # 自由端角 (W≈0.06)
POINT_Z     = 0.01        # 目标 z (m), 具体值或 None(自动取顶面)
NEAR_R      = 0.02        # 附近点搜索半径 (m)
FREQ_MIN    = 1
FREQ_MAX    = 5000
# ============================

data = np.load(os.path.join(os.path.dirname(__file__), 'output', 'final_results.npz'),
               allow_pickle=True)
# object arrays: data['frequencies'][i] 等
f_np   = data['frequencies'][SAMPLE_IDX]
points = data['points'][SAMPLE_IDX]
t_amp  = data['target_frf'][SAMPLE_IDX]
p_amp  = data['predicted_frf'][SAMPLE_IDX]
t_re   = data['target_re'][SAMPLE_IDX]
p_re   = data['predicted_re'][SAMPLE_IDX]
t_im   = data['target_im'][SAMPLE_IDX]
p_im   = data['predicted_im'][SAMPLE_IDX]

print(f'X: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]  '
      f'Y: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]  '
      f'Freq: [{f_np[0]:.1f}, {f_np[-1]:.1f}] Hz  '
      f'Shape: {t_amp.shape}')

# 峰值检测
f_abs = t_amp.max(axis=0)
peaks = []
for i in range(2, len(f_abs)-2):
    if f_abs[i] > f_abs[i-1] and f_abs[i] > f_abs[i+1] and f_abs[i] > f_abs.max()*0.015:
        peaks.append(f_np[i])
pk1, pk2 = sorted(peaks)[:2]
print(f'峰值: {pk1:.1f} Hz, {pk2:.1f} Hz')

# 拉伸x轴 (带宽按峰值频率自适应)
bw1, bw2 = 0.012 * pk1, 0.012 * pk2
w1 = 15.0 * np.exp(-0.5 * ((f_np - pk1) / (bw1 * 0.5))**2)
w2 =  4.0 * np.exp(-0.5 * ((f_np - pk2) / (bw2 * 0.5))**2)
xs = np.zeros_like(f_np)
for i in range(1, len(f_np)):
    xs[i] = xs[i-1] + (w1[i]+w2[i]+w1[i-1]+w2[i-1]+2) / 2 * (f_np[i] - f_np[i-1])
tfs = np.unique(np.sort([f_np[0], pk1-bw1, pk1, pk1+bw1, pk2-bw2, pk2, pk2+bw2, f_np[-1]]))
tls = np.interp(tfs, f_np, xs)
tlbs = [f'{f:.1f}' if abs(f-pk1)<0.1 or abs(f-pk2)<0.1 else f'{f:.0f}' for f in tfs]

# 选中点+2附近点 (含Z轴)
pz = POINT_Z if POINT_Z is not None else points[:, 2].max()
if POINT_Z is None:
    print(f'POINT_Z=None → 自动取顶面 z={pz:.4f}')
dist_xyz = np.sqrt((points[:,0] - POINT_X)**2 + (points[:,1] - POINT_Y)**2 + (points[:,2] - pz)**2)
idx_main = np.argmin(dist_xyz)
sel = points[idx_main]
dist_min = dist_xyz[idx_main]
print(f'目标: ({POINT_X}, {POINT_Y}, {pz})')
print(f'选中节点#{idx_main}: ({sel[0]:.5f}, {sel[1]:.5f}, {sel[2]:.5f}), 距离={dist_min:.5f}m')
if dist_min > 0.01:
    print(f'  ⚠️ 距离 > 1cm，该节点可能不在目标位置附近！')
    print(f'  提示: 可调整 POINT_X/Y/Z 使坐标落在实际网格节点上')

rng = np.random.RandomState(42)
near = []
x0, y0, z0 = points[idx_main, 0], points[idx_main, 1], points[idx_main, 2]
candidates = np.where(np.abs(points[:, 2] - z0) < 0.005)[0]  # 同层节点
if len(candidates) < 3:
    candidates = np.arange(len(points))  # 退化: 全搜
for _ in range(2):
    cx = x0 + rng.uniform(-NEAR_R, NEAR_R)
    cy = y0 + rng.uniform(-NEAR_R, NEAR_R)
    sub_dist = np.sqrt((points[candidates, 0] - cx)**2 + (points[candidates, 1] - cy)**2)
    ni = candidates[np.argmin(sub_dist)]
    near.append(ni)

all_i = [idx_main] + near
tags = ['SELECTED', 'Near #1', 'Near #2']

out_dir = os.path.join(os.path.dirname(__file__), 'output')

def plot_1x3(t_arr, p_arr, ylabel, out_name, do_ylim=False):
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5), sharex=True)
    for ax, i, tag in zip(axes, all_i, tags):
        xp, yp = points[i,0], points[i,1]
        ax.plot(xs, t_arr[i], 'b-', linewidth=1.0, alpha=0.8, label='Target')
        ax.plot(xs, p_arr[i], 'r--', linewidth=1.0, alpha=0.8, label='Predicted')
        for lo, hi in [(pk1-bw1, pk1+bw1), (pk2-bw2, pk2+bw2)]:
            ax.axvspan(np.interp(lo, f_np, xs), np.interp(hi, f_np, xs), color='gray', alpha=0.06)
        if do_ylim:
            t_max = max(1e-6, t_arr[i].max())
            ax.set_ylim(max(-50, t_arr[i].min()*1.15), t_max*1.15)
        ax.set_title(f'{tag} (x={xp:.4f},y={yp:.4f})', fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.15)
        ax.set_ylabel(ylabel, fontsize=9)
    for a in axes:
        a.set_xticks(tls); a.set_xticklabels(tlbs, fontsize=8); a.set_xlabel('Hz', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, out_name), dpi=150, bbox_inches='tight'); plt.close()
    print(f'{ylabel}: {out_dir}/{out_name}')

plot_1x3(t_amp, p_amp, 'Amplitude', '对比图.png', True)
plot_1x3(t_re, p_re, 'Real', '对比图_re.png')
plot_1x3(t_im, p_im, 'Imag', '对比图_im.png')
