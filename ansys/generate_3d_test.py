"""
ANSYS 凹槽工件数据集生成 — 模态参数 + FRF。
拓扑时变 (凹槽铣削) + 边界阻抗扰动 (COMBIN14 弹簧阻尼器)。

策略: 固定工件外形，保留凹槽分组逻辑，随机化网格划分 + 装夹参数

特征维度: 7维 [E, PRXY, DENS, is_fixed, log10(K), log10(C), Z/H]

物理正确的位移频响函数 (Z向面外, 薄板颤振最敏感方向):
    H(x, x_f, ω) = Σ_k φ_k(x)·φ_k(x_f) / (ω_k² - ω² + j·2ζ_k·ω_k·ω)
"""
import random
import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'  # 避免VTK显示问题
from ansys.mapdl.core import launch_mapdl
from scipy.stats import qmc
import numpy as np
import h5py
import time
import csv

# ============ 全局随机种子 (修改此值可生成不同数据集) ============
SEED = 2
np.random.seed(SEED)
random.seed(SEED)

# ============ 配置 ============
N_SAMPLES = 100
N_TRAIN = int(N_SAMPLES * 0.7)   # 70% 训练集
N_VAL = int(N_SAMPLES * 0.15)    # 15% 验证集
N_TEST = N_SAMPLES - N_TRAIN - N_VAL  # 剩余测试集
N_MODES = 3
N_FREQS = 60  # 每峰至少15点, 总60点保证共振峰幅值精度
FREQ_MIN, FREQ_MAX = 1.0, 5000.0
MESH_SIZE = 0.006  # 6mm网格
ZETA_MATERIAL = 0.002
AMPLITUDE_SCALE = 500000.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
VIZ_DIR = os.path.join(os.path.dirname(__file__), "mesh_viz")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# ============ 固定工件尺寸 ============
E_BASE, RHO_BASE, PRXY_BASE = 71.7e9, 2810.0, 0.33
L_BASE, W_BASE, H_BASE = 0.160, 0.060, 0.010  # 160×60×10mm (固定)

# ============ 材料参数随机范围 ============
E_RANGE = (0.95, 1.05)   # E ±5%
RHO_RANGE = (0.97, 1.03)  # ρ ±3%

# ============ 凹槽参数 ============
GRID_JITTER = 0.15                 # 行列尺寸随机偏移: ±15%
GAP_ABS = 0.006                    # 结构筋间距: 6mm (绝对尺寸, ≥ 1个网格单元)
BORDER_ABS = 0.006                 # 外围边界: 6mm (绝对尺寸, ≥ 1个网格单元)
POCKET_DEPTH_RANGE = (0.30, 0.60)  # 凹槽深度: 30%~60% × H (10mm时: 3~6mm深, 剩4~7mm底)

# ============ 凹槽区域划分 ============
# 区域划分: 把工件表面分成 行×列 的区域，用于定义凹槽位置和大小
# 外围有边界（用于装夹），行列之间有间距（结构筋）
# 与ANSYS有限元网格完全无关!

def generate_region_division(n_cols, n_rows, L, W, jitter=GRID_JITTER,
                              gap=GAP_ABS, border=BORDER_ABS):
    """
    生成区域划分 (带外围边界和结构筋间距)
    L, W: 工件的X/Y方向总长度 (米)
    gap: 结构筋绝对宽度 (米)
    border: 外围边界绝对宽度 (米)
    返回: x_pockets, y_pockets (每个区域的归一化坐标列表, 0~1)
    """
    n_gaps_x = n_cols - 1
    n_gaps_y = n_rows - 1
    available_x = L - 2 * border - n_gaps_x * gap
    available_y = W - 2 * border - n_gaps_y * gap

    # 归一化随机: 生成随机权重后归一化, 保证总和 = 可用空间
    weights_x = np.array([1.0 + np.random.uniform(-jitter, jitter) for _ in range(n_cols)])
    weights_x = weights_x / weights_x.sum() * available_x

    weights_y = np.array([1.0 + np.random.uniform(-jitter, jitter) for _ in range(n_rows)])
    weights_y = weights_y / weights_y.sum() * available_y

    x_pockets = []
    y_pockets = []

    current_x = border
    for i in range(n_cols):
        col_w = weights_x[i]
        x_pockets.append((current_x / L, (current_x + col_w) / L))
        current_x += col_w + gap

    current_y = border
    for i in range(n_rows):
        row_h = weights_y[i]
        y_pockets.append((current_y / W, (current_y + row_h) / W))
        current_y += row_h + gap

    return x_pockets, y_pockets


def get_pocket_from_cells(x_pockets, y_pockets, cell_indices, n_cols):
    """
    根据区域单元索引计算凹槽边界 (合并多个单元)
    cell_indices: 单元索引列表 (从1开始, 从左到右, 从上到下)
    返回: (xmin_frac, xmax_frac, ymin_frac, ymax_frac)
    """
    rows = [(idx - 1) // n_cols for idx in cell_indices]
    cols = [(idx - 1) % n_cols for idx in cell_indices]

    xmin = min(x_pockets[c][0] for c in cols)
    xmax = max(x_pockets[c][1] for c in cols)
    ymin = min(y_pockets[r][0] for r in rows)
    ymax = max(y_pockets[r][1] for r in rows)

    return (xmin, xmax, ymin, ymax)


# 5个凹槽方案 (4列×3行网格)
# 区域: P1={1,2,5,6,9,10}, P2={3,4}, P3={8,12}, P4={7}, P5={11}
POCKET_CELLS_5 = [
    [1, 2, 5, 6, 9, 10],   # P1: 列1+2全宽
    [3, 4],                 # P2: 列3+4顶部
    [8, 12],                # P3: 列4中下部
    [7],                    # P4: 列3中部
    [11],                   # P5: 列3下部
]

# 6个凹槽方案 (4列×3行网格)
# 区域: P1={1,5,9}, P2={2}, P3={3,7}, P4={4,8,12}, P5={6}, P6={10,11}
POCKET_CELLS_6 = [
    [1, 5, 9],              # P1: 列1全宽
    [2],                    # P2: 列2顶部
    [3, 7],                 # P3: 列3上部
    [4, 8, 12],             # P4: 列4全宽
    [6],                    # P5: 列2中部
    [10, 11],               # P6: 列2+3下部
]

# 7个凹槽方案 (5列×3行网格)
# 区域: P1={1,6,11}, P2={4,9,14}, P3={5,10}, P4={15}, P5={2,3}, P6={7,8}, P7={12,13}
POCKET_CELLS_7 = [
    [1, 6, 11],             # P1: 列1全宽
    [4, 9, 14],             # P2: 列4全宽
    [5, 10],                # P3: 列5上部
    [15],                   # P4: 列5下部
    [2, 3],                 # P5: 列2+3顶部
    [7, 8],                 # P6: 列2+3中部
    [12, 13],               # P7: 列2+3下部
]

# 弹簧阻尼器参数范围 (角点与侧面分开, 反映螺栓 vs 顶杆的物理差异)
# 角点 XYZ: 螺栓/压板紧固, 刚度高 (总刚度除以节点数后施加三向弹簧)
K_CORNER_RANGE = (5e6, 1e8)    # 角点刚度 (N/m), 螺栓连接典型值
# 侧面 Y: 侧向顶杆/可调支撑, 刚度低一个量级
K_SIDE_RANGE = (1e6, 3e7)      # 侧面刚度 (N/m), 顶杆连接典型值
# C 不独立随机, 通过 C = 2·ζ_joint·√(K·M_ref) 与各自 K 耦合
ZETA_JOINT_RANGE = (0.005, 0.05)  # 机械连接阻尼比
M_REF = 0.01  # kg, 夹持区局部有效质量


print(">>> 正在生成 Sobol 低偏差序列 (材料参数)...")
SOBOL_BUFFER = 50  # 预留额外样本应对布尔失败重试
sampler = qmc.Sobol(d=2, scramble=True, seed=SEED)
sobol_samples = sampler.random(n=N_SAMPLES + SOBOL_BUFFER)
l_bounds = [E_RANGE[0], RHO_RANGE[0]]
u_bounds = [E_RANGE[1], RHO_RANGE[1]]
scaled_sobol = qmc.scale(sobol_samples, l_bounds, u_bounds)

print(f"配置: {N_SAMPLES}样本, {N_MODES}阶模态, {N_FREQS}频率点")
print(f"工件: {L_BASE*1000:.0f}×{W_BASE*1000:.0f}×{H_BASE*1000:.0f}mm (固定)")
print(f"凹槽方案: 5/6/7个凹槽, 深度随机{POCKET_DEPTH_RANGE[0]*100:.0f}~{POCKET_DEPTH_RANGE[1]*100:.0f}%, 结构筋{GAP_ABS*1000:.0f}mm, 外围边界{BORDER_ABS*1000:.0f}mm")
print(f"装夹: 四角螺栓(K_c∈{K_CORNER_RANGE}) + 3侧面顶杆(K_s∈{K_SIDE_RANGE}), C=2ζ√(K·M_ref)")
print(f"阻尼: ζ_material={ZETA_MATERIAL} + C/(2√(K·M_ref)), M_ref={M_REF}kg, ζ_joint∈{ZETA_JOINT_RANGE}")

print("\n>>> 正在连接 ANSYS 求解器...")
mapdl = launch_mapdl(override=True)
print(f">>> 连接成功! 版本: {mapdl.version}\n")

# 预分配
all_points, all_frf, all_freqs = [], [], []
all_omega, all_zeta, all_phi, all_phi_exc = [], [], [], []
all_features = []
t0 = time.time()

# CSV日志文件
csv_path = os.path.join(OUT_DIR, "sample_log.csv")
csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    '样本编号', '节点总数', '加工凹槽数', '凹槽方案总数',
    '凹槽深度范围(%)', '切削区节点数',
    '激励点X坐标(mm)', '激励点Y坐标(mm)', '激励点Z坐标(mm)',
    '弹簧区域数', '弹簧节点数',
    'K_corner_0~3(N/m)', 'C_corner_0~3(N·s/m)', 'K_side_0~2(N/m)', 'C_side_0~2(N·s/m)',
    '阻尼比ζ₁', '阻尼比ζ₂', '阻尼比ζ₃',
    '固有频率f₁(Hz)', '固有频率f₂(Hz)', '固有频率f₃(Hz)',
    '弹性模量比', '密度比', '列数', '行数',
    '凹槽区域定义'
])

valid_samples = 0
attempt_count = 0

while valid_samples < N_SAMPLES:
    attempt_count += 1
    sobol_idx = attempt_count - 1  # Sobol索引基于总尝试次数
    if sobol_idx >= len(scaled_sobol):
        sobol_idx = sobol_idx % len(scaled_sobol)  # 超出时循环使用
    print(f"[有效样本 {valid_samples+1}/{N_SAMPLES}] (尝试第{attempt_count}次)", end=" ", flush=True)
    try:
        mapdl.clear()
        mapdl.prep7()
    except Exception:
        print("(reconnect)", end=" ", flush=True)
        mapdl.exit()
        time.sleep(2)
        mapdl = launch_mapdl(override=True)
        mapdl.clear()
        mapdl.prep7()

    # 1. 采样参数 (仅材料属性随机，工件尺寸固定)
    E = E_BASE * scaled_sobol[sobol_idx, 0]
    rho = RHO_BASE * scaled_sobol[sobol_idx, 1]
    L, W, H = L_BASE, W_BASE, H_BASE

    # 随机化弹簧阻尼参数: 每装夹区独立采样 (4角 + 3侧面)
    K_corners = []; C_corners = []; K_sides = []; C_sides = []
    for _ in range(4):
        kc = 10 ** random.uniform(np.log10(K_CORNER_RANGE[0]), np.log10(K_CORNER_RANGE[1]))
        zc = random.uniform(*ZETA_JOINT_RANGE)
        K_corners.append(kc); C_corners.append(2.0 * zc * np.sqrt(kc * M_REF))
    for _ in range(3):
        ks = 10 ** random.uniform(np.log10(K_SIDE_RANGE[0]), np.log10(K_SIDE_RANGE[1]))
        zs = random.uniform(*ZETA_JOINT_RANGE)
        K_sides.append(ks); C_sides.append(2.0 * zs * np.sqrt(ks * M_REF))

    # 2. 材料定义
    mapdl.mp("EX", 1, E)
    mapdl.mp("PRXY", 1, PRXY_BASE)
    mapdl.mp("DENS", 1, rho)

    # 3. 凹槽方案选择 (在建模之前)
    num_machined = random.choice([5, 6, 7])
    if num_machined == 5:
        pocket_cells = POCKET_CELLS_5
        n_cols, n_rows = 4, 3
    elif num_machined == 6:
        pocket_cells = POCKET_CELLS_6
        n_cols, n_rows = 4, 3
    else:
        pocket_cells = POCKET_CELLS_7
        n_cols, n_rows = 5, 3

    # 生成随机区域划分 (带外围边界和结构筋间距)
    x_pockets, y_pockets = generate_region_division(n_cols, n_rows, L, W)

    # 随机选择加工几个凹槽 (1~num_machined)
    n_pockets_to_machine = random.randint(1, num_machined)
    pockets_to_machine = random.sample(range(num_machined), n_pockets_to_machine)

    # 4. 建模: 生成带凹槽的真实几何 (布尔运算)
    # 4.1 创建基础长方体
    mapdl.btol(0.0001)  # 布尔运算容差 0.1mm
    mapdl.block(0, L, 0, W, 0, H)
    wk_vol = int(mapdl.geometry.vnum[0])  # 显式记录工件体积编号

    # 4.2 对每个要加工的凹槽，用布尔运算切除
    pocket_depth_fracs = []  # 记录每个凹槽的实际深度比例
    bool_ok = True
    for pocket_idx in pockets_to_machine:
        # 【优化】每个凹槽独立采样深度
        pocket_depth_frac_k = random.uniform(*POCKET_DEPTH_RANGE)
        pocket_depth_fracs.append(pocket_depth_frac_k)
        pocket_zmin = H - pocket_depth_frac_k * H

        cells = pocket_cells[pocket_idx]
        xmin_frac, xmax_frac, ymin_frac, ymax_frac = get_pocket_from_cells(
            x_pockets, y_pockets, cells, n_cols)
        xmin_p = max(xmin_frac * L, 0.0)
        xmax_p = min(xmax_frac * L, L)
        ymin_p = max(ymin_frac * W, 0.0)
        ymax_p = min(ymax_frac * W, W)

        if xmax_p <= xmin_p or ymax_p <= ymin_p or pocket_zmin >= H:
            continue

        mapdl.allsel()
        old_vols = set(mapdl.geometry.vnum)

        mapdl.block(xmin_p, xmax_p, ymin_p, ymax_p, pocket_zmin, H + 0.001)

        new_vols = set(mapdl.geometry.vnum) - old_vols
        if len(new_vols) == 0:
            continue
        pk_vol = int(list(new_vols)[0])

        try:
            mapdl.vsbv(wk_vol, pk_vol)
        except Exception as e:
            print(f"  VSBV失败(pocket {pocket_idx}): {e}")
            bool_ok = False
            break

        mapdl.allsel()
        remaining_vols = mapdl.geometry.vnum
        if len(remaining_vols) > 0:
            wk_vol = int(remaining_vols[0])
        else:
            bool_ok = False
            break
    
    if not bool_ok:
        print(f"  跳过本次尝试 (布尔运算失败)")
        mapdl.clear()
        continue
    
    # 4.3 划分网格
    mapdl.et(1, "SOLID187")
    mapdl.mshape(1, "3D")
    mapdl.mshkey(0)
    mapdl.esize(MESH_SIZE)
    try:
        mapdl.vmesh("ALL")
    except Exception:
        mapdl.smrtsize(4)
        mapdl.vmesh("ALL")

    # 5. 初始化节点特征矩阵 (N, 7)
    all_node_ids = mapdl.mesh.nnum
    all_node_coords = np.array(mapdl.mesh.nodes, dtype=np.float32)
    n_nodes_total = len(all_node_ids)
    node_id_to_idx = {int(nid): idx for idx, nid in enumerate(all_node_ids)}

    node_features = np.zeros((n_nodes_total, 7), dtype=np.float32)
    node_features[:, 0] = E / E_BASE
    node_features[:, 1] = PRXY_BASE
    node_features[:, 2] = rho / RHO_BASE
    node_features[:, 3] = 0.0   # is_fixed: 0=自由, 0.5=侧面, 1.0=角点 (后面设)
    node_features[:, 4] = -1.0  # log10(K): -1 = 无弹簧
    node_features[:, 5] = -1.0  # log10(C): -1 = 无弹簧
    node_features[:, 6] = all_node_coords[:, 2] / H  # Z/H: 局部厚度比
    
    # 记录凹槽底面可用节点 (用于激励点选择, 模拟铣削刀具切削位置)
    # 【修复】每个凹槽深度独立，需分别按各自Z坐标选取底面节点
    pocket_cut_indices = []
    pocket_bottom_any_indices = []
    tool_r = MESH_SIZE / 2
    cut_band = MESH_SIZE * 0.6

    mapdl.allsel()
    for pocket_i, pocket_idx in enumerate(pockets_to_machine):
        pocket_depth_k = pocket_depth_fracs[pocket_i] * H
        pocket_bottom_z = H - pocket_depth_k
        cells = pocket_cells[pocket_idx]
        xmin_frac, xmax_frac, ymin_frac, ymax_frac = get_pocket_from_cells(
            x_pockets, y_pockets, cells, n_cols)
        xmin_p, xmax_p = xmin_frac * L, xmax_frac * L
        ymin_p, ymax_p = ymin_frac * W, ymax_frac * W

        # 用ANSYS选取该凹槽底面Z高度的节点
        mapdl.nsel("S", "LOC", "Z", pocket_bottom_z, pocket_bottom_z + 1e-6)
        # XY范围向内收缩0.1mm，防止选到侧壁与底面交线上的节点
        margin = 1e-4
        mapdl.nsel("R", "LOC", "X", xmin_p + margin, xmax_p - margin)
        mapdl.nsel("R", "LOC", "Y", ymin_p + margin, ymax_p - margin)
        bottom_nids = set(int(nid) for nid in mapdl.mesh.nnum)

        for nid in bottom_nids:
            if nid not in node_id_to_idx:
                continue
            idx = node_id_to_idx[nid]
            if idx in pocket_bottom_any_indices:
                continue  # 避免重复添加
            pocket_bottom_any_indices.append(idx)
            x, y = all_node_coords[idx, 0], all_node_coords[idx, 1]
            dist_to_wall = min(x - xmin_p, xmax_p - x, y - ymin_p, ymax_p - y)
            if abs(dist_to_wall - tool_r) < cut_band:
                pocket_cut_indices.append(idx)
    mapdl.allsel()

    if len(pocket_bottom_any_indices) == 0:
        print(f"  警告: 未找到凹槽底面节点!")

    # ==========================================
    # 6 & 7. 混合柔性装夹 (4角 + 3个侧边浮动点)，全采用 XYZ 三向解耦弹簧
    # ==========================================
    mapdl.et(2, "COMBIN14"); mapdl.keyopt(2, 2, 1)  # UX
    mapdl.et(3, "COMBIN14"); mapdl.keyopt(3, 2, 2)  # UY
    mapdl.et(4, "COMBIN14"); mapdl.keyopt(4, 2, 3)  # UZ

    clamp_len = 0.010  # 四角的夹持长度 10mm
    # 定义 4 个角的侧面区域 (xmin, xmax, ymin, ymax)
    all_clamp_areas = [
        (0, clamp_len, 0, 1e-4),
        (L - clamp_len, L, 0, 1e-4),
        (0, clamp_len, W - 1e-4, W),
        (L - clamp_len, L, W - 1e-4, W)
    ]

    # 生成 3 个随机侧边浮动点
    CORNER_EXCL = clamp_len + H / 2
    x_min, x_max = CORNER_EXCL, L - CORNER_EXCL
    side_choices = [0, 0, 1] if random.random() < 0.5 else [1, 1, 0]
    sides_y = [0, W]

    for side_idx in (0, 1):
        n_on_side = sum(1 for s in side_choices if s == side_idx)
        if n_on_side == 0: continue
        xs = []
        min_gap = 2 * H
        for _ in range(n_on_side):
            for _attempt in range(100):
                x_try = random.uniform(x_min, x_max)
                if all(abs(x_try - x_exist) >= min_gap for x_exist in xs):
                    xs.append(x_try)
                    break
            else:
                xs.append(x_min + len(xs) * H)

        cy = sides_y[side_idx]
        for x_c in xs:
            all_clamp_areas.append((x_c - H/2, x_c + H/2, cy - 1e-4, cy + 1e-4))

    max_node_id = int(all_node_ids.max())
    spring_info = []  # [(ansys_nid, Cx, Cy, Cz)]
    spring_node_set = set()  # 已施加弹簧的节点集合，防止重叠
    real_const_num = 2

    for idx_area, (xmin, xmax, ymin, ymax) in enumerate(all_clamp_areas):
        mapdl.nsel("S", "LOC", "X", xmin, xmax)
        mapdl.nsel("R", "LOC", "Y", ymin, ymax)
        mapdl.nsel("R", "LOC", "Z", 0, H)

        n_selected = mapdl.mesh.n_node
        if n_selected > 0:
            clamp_nodes = mapdl.mesh.nnum

            # 【防呆】前4个区域(0,1,2,3)是角落，之后的是浮动点
            is_corner = (idx_area < 4)
            K_this = K_corners[idx_area] if is_corner else K_sides[idx_area - 4]
            C_this = C_corners[idx_area] if is_corner else C_sides[idx_area - 4]
            K_each = K_this / n_selected
            C_each = C_this / n_selected

            mapdl.r(real_const_num, K_each, 0.0)
            for n1 in clamp_nodes:
                n1_int = int(n1)
                if n1_int in node_id_to_idx and n1_int not in spring_node_set:
                    spring_node_set.add(n1_int)
                    idx_n1 = node_id_to_idx[n1_int]
                    x1, y1, z1 = all_node_coords[idx_n1]

                    max_node_id += 1
                    n2 = max_node_id
                    mapdl.n(n2, x1, y1, z1)
                    mapdl.d(n2, "ALL")

                    if is_corner:
                        mapdl.type(2); mapdl.real(real_const_num); mapdl.e(n1_int, n2)
                        mapdl.type(3); mapdl.real(real_const_num); mapdl.e(n1_int, n2)
                        mapdl.type(4); mapdl.real(real_const_num); mapdl.e(n1_int, n2)
                        node_features[idx_n1, 3] = 1.0
                        spring_info.append((n1_int, C_each, C_each, C_each))
                    else:
                        mapdl.type(3); mapdl.real(real_const_num); mapdl.e(n1_int, n2)
                        node_features[idx_n1, 3] = 0.5
                        spring_info.append((n1_int, 0.0, C_each, 0.0))

                    node_features[idx_n1, 4] = np.log10(K_each)
                    node_features[idx_n1, 5] = np.log10(C_each)
            real_const_num += 1

    mapdl.allsel()

    # 8. 模态分析
    mapdl.slashsolu()
    mapdl.antype("MODAL")
    mapdl.modopt("LANB", N_MODES, nrmkey="ON")  # nrmkey=ON: 质量归一化振型
    mapdl.solve()

    # ==========================================
    # 9. 绝对安全的模态结果提取 (引入X向以求空间耗散)
    # ==========================================
    mapdl.post1()
    current_nnum = mapdl.mesh.nnum
    current_id_to_idx = {int(nid): idx for idx, nid in enumerate(current_nnum)}

    omega_k = np.zeros(N_MODES, dtype=np.float32)
    phi_x_safe = np.zeros((n_nodes_total, N_MODES), dtype=np.float32)
    phi_y_safe = np.zeros((n_nodes_total, N_MODES), dtype=np.float32)
    phi_z_safe = np.zeros((n_nodes_total, N_MODES), dtype=np.float32)

    for k in range(1, N_MODES + 1):
        mapdl.set(1, k)
        f_hz = mapdl.post_processing.freq
        omega_k[k - 1] = 2.0 * np.pi * f_hz
        disp = np.array(mapdl.post_processing.nodal_displacement("ALL"), dtype=np.float32)

        for idx_curr, nid in enumerate(current_nnum):
            nid_int = int(nid)
            if nid_int in node_id_to_idx:
                idx_orig = node_id_to_idx[nid_int]
                phi_x_safe[idx_orig, k - 1] = disp[idx_curr, 0]
                phi_y_safe[idx_orig, k - 1] = disp[idx_curr, 1]
                phi_z_safe[idx_orig, k - 1] = disp[idx_curr, 2]

    # ==========================================
    # 10. 激励点提取 (切削区几何中心最近的节点)
    # ==========================================
    if len(pocket_cut_indices) > 0:
        cut_coords = all_node_coords[pocket_cut_indices]
        center = cut_coords.mean(axis=0)
        dists = np.linalg.norm(cut_coords - center, axis=1)
        exc_idx = pocket_cut_indices[int(np.argmin(dists))]
    elif len(pocket_bottom_any_indices) > 0:
        any_coords = all_node_coords[pocket_bottom_any_indices]
        center = any_coords.mean(axis=0)
        dists = np.linalg.norm(any_coords - center, axis=1)
        exc_idx = pocket_bottom_any_indices[int(np.argmin(dists))]
    else:
        exc_idx = np.random.randint(0, n_nodes_total)

    # 【核心修改】：使用 Z 向振型 (面外方向) 计算模态参与因子
    phi_exc_k_z = phi_z_safe[exc_idx, :].copy()
    exc_actual = all_node_coords[exc_idx]  # 真实物理敲击坐标

    # ==========================================
    # 11. 阻尼比计算 (三维空间耗散求和)
    # ==========================================
    zeta_k = np.zeros(N_MODES, dtype=np.float32)
    for k in range(N_MODES):
        wk = omega_k[k]
        zeta_boundary_k = 0.0

        for ansys_nid, cx, cy, cz in spring_info:
            if ansys_nid in node_id_to_idx:
                idx_orig = node_id_to_idx[ansys_nid]
                phi_x = phi_x_safe[idx_orig, k]
                phi_y = phi_y_safe[idx_orig, k]
                phi_z = phi_z_safe[idx_orig, k]

                dissipation = cx * (phi_x ** 2) + cy * (phi_y ** 2) + cz * (phi_z ** 2)
                zeta_boundary_k += dissipation / (2.0 * wk)

        zeta_k[k] = ZETA_MATERIAL + zeta_boundary_k

    # ==========================================
    # 12. 自适应频率网格 (保持不变)
    # ==========================================
    freqs_parts = []
    prev = FREQ_MIN
    for idx_k, f_k in enumerate(omega_k / (2 * np.pi)):
        bw = 2.0 * zeta_k[idx_k] * f_k
        lo = max(FREQ_MIN, f_k - 3.0 * bw)
        hi = min(FREQ_MAX, f_k + 3.0 * bw)
        if prev < lo:
            freqs_parts.append(np.logspace(np.log10(max(prev, 0.1)), np.log10(lo),
                                max(2, int(5 * (lo - prev) / FREQ_MAX)), endpoint=False))
        freqs_parts.append(np.linspace(lo, hi, max(15, int(20 * (hi - lo) / FREQ_MAX)), endpoint=True))
        prev = hi
    if prev < FREQ_MAX:
        freqs_parts.append(np.logspace(np.log10(max(prev, 0.1)), np.log10(FREQ_MAX),
                            max(2, int(5 * (FREQ_MAX - prev) / FREQ_MAX)), endpoint=True))
    freqs = np.unique(np.sort(np.concatenate(freqs_parts)))
    if len(freqs) > N_FREQS:
        # 标记峰值保护区: ±1 半功率带宽内不下采样
        protected = np.zeros(len(freqs), dtype=bool)
        for fk, zk in zip(omega_k / (2*np.pi), zeta_k):
            bw_half = 2.0 * zk * fk  # 半功率带宽 = 2*zeta*f
            protected |= (freqs >= fk - bw_half) & (freqs <= fk + bw_half)
        # 削点: 优先删对数区最密集处 (间隔最小, 信息冗余最多)
        n_remove = len(freqs) - N_FREQS
        keep = np.ones(len(freqs), dtype=bool)
        gap_indices = np.where(~protected)[0]
        if len(gap_indices) >= n_remove:
            gap_edges = np.diff(freqs[gap_indices])
            remove_order = gap_indices[:-1][np.argsort(gap_edges)][:n_remove]
            keep[remove_order] = False
        else:
            # 对数区不够: 先删全部对数区, 再均匀削峰值区
            keep[~protected] = False
            peak_indices = np.where(protected)[0]
            n_remove_peak = n_remove - len(gap_indices)
            step = max(2, len(peak_indices) // n_remove_peak)
            keep[peak_indices[::step][:n_remove_peak]] = False
        freqs = freqs[keep]
    elif len(freqs) < N_FREQS:
        # 把剩余预算按间隙宽度比例分配到对数区
        shortage = N_FREQS - len(freqs)
        gaps = np.diff(freqs)
        weights = gaps / gaps.sum()
        extra_per_gap = np.round(weights * shortage).astype(int)
        diff = shortage - extra_per_gap.sum()
        order = np.argsort(-gaps)
        for j in range(int(diff)):
            extra_per_gap[order[j % len(order)]] += 1
        # 在每个间隙中插入对数点
        filled = [freqs[0]]
        for lo, hi, n_extra in zip(freqs[:-1], freqs[1:], extra_per_gap):
            if n_extra > 0:
                filled.extend(np.logspace(np.log10(lo), np.log10(hi), n_extra + 2)[1:-1])
            filled.append(hi)
        freqs = np.unique(np.sort(filled))[:N_FREQS]
    freqs = freqs.astype(np.float32)

    # ==========================================
    # 13. FRF 计算 (Z向面外频响, 薄板颤振最敏感方向)
    # ==========================================
    omega_q = 2.0 * np.pi * freqs
    # 直接生成正确尺寸的 FRF 矩阵
    frf_safe = np.zeros((n_nodes_total, len(freqs), 2), dtype=np.float32)
    for k in range(N_MODES):
        wk = omega_k[k]
        zk = zeta_k[k]
        
        # 【核心修正】：使用 Z 向振型 (面外方向) 计算模态参与因子
        pk_z = phi_z_safe[:, k] * phi_exc_k_z[k]
        
        dw = wk**2 - omega_q**2
        gm = 2.0 * zk * wk * omega_q
        D = np.maximum(dw**2 + gm**2 + 1e-6, 1.0)  # 与 PhysicsDecoder 一致
        
        frf_safe[:, :, 0] += np.outer(pk_z, AMPLITUDE_SCALE * dw / D)
        frf_safe[:, :, 1] += np.outer(pk_z, -AMPLITUDE_SCALE * gm / D)

    # ==========================================
    # 14. 纯净的数据保存 (再也不需要用 [:valid_len] 冒险截断了)
    # ==========================================
    all_points.append(all_node_coords)
    all_frf.append(frf_safe)
    all_freqs.append(freqs)
    all_omega.append(omega_k)
    all_zeta.append(zeta_k)
    all_phi.append(phi_z_safe)  # 与FRF计算方向一致：Z向振型 (面外)
    all_phi_exc.append(phi_exc_k_z)  # Z向激励振型
    all_features.append(node_features)

    exc_actual = all_node_coords[exc_idx]
    n_spring_areas = len(all_clamp_areas)
    n_spring_nodes = len(spring_info)
    n_cut_nodes = len(pocket_cut_indices)
    n_any_nodes = len(pocket_bottom_any_indices)
    depth_min = min(pocket_depth_fracs) * 100 if pocket_depth_fracs else 0
    depth_max = max(pocket_depth_fracs) * 100 if pocket_depth_fracs else 0
    print(f"[有效样本 {valid_samples+1}/{N_SAMPLES}] N={n_nodes_total}, 加工{n_pockets_to_machine}/{num_machined}个凹槽, "
          f"深度{depth_min:.0f}~{depth_max:.0f}%, 切削区={n_cut_nodes}/底面={n_any_nodes}, "
          f"激励点=({exc_actual[0]*1000:.1f},{exc_actual[1]*1000:.1f},{exc_actual[2]*1000:.1f})mm, "
          f"弹簧区域={n_spring_areas}个, 弹簧节点={n_spring_nodes}个, "
          f"Kc∈[{min(K_corners):.1e}~{max(K_corners):.1e}], "
          f"Ks∈[{min(K_sides):.1e}~{max(K_sides):.1e}], "
          f"ζ₁={zeta_k[0]:.4f}, ζ₂={zeta_k[1]:.4f}" + (f", ζ₃={zeta_k[2]:.4f}" if N_MODES >= 3 else "") + ", "
          f"f₁={omega_k[0]/(2*np.pi):.1f}Hz, f₂={omega_k[1]/(2*np.pi):.1f}Hz"
          + (f", f₃={omega_k[2]/(2*np.pi):.1f}Hz" if N_MODES >= 3 else ""))
    
    # 写入CSV日志
    zeta_cols = [f"{zk:.6f}" for zk in zeta_k]  # ζ₁, ζ₂, [ζ₃]
    freq_cols = [f"{wk/(2*np.pi):.2f}" for wk in omega_k]  # f₁, f₂, [f₃]
    csv_row = [
        valid_samples + 1, n_nodes_total, n_pockets_to_machine, num_machined,
        f"{depth_min:.1f}~{depth_max:.1f}", f"{n_cut_nodes}/{n_any_nodes}",
        f"{exc_actual[0]*1000:.2f}", f"{exc_actual[1]*1000:.2f}", f"{exc_actual[2]*1000:.2f}",
        n_spring_areas, n_spring_nodes,
        ";".join(f"{k:.2e}" for k in K_corners),
        ";".join(f"{c:.2f}" for c in C_corners),
        ";".join(f"{k:.2e}" for k in K_sides),
        ";".join(f"{c:.2f}" for c in C_sides),
        *zeta_cols, *freq_cols,
        f"{E/E_BASE:.4f}", f"{rho/RHO_BASE:.4f}", n_cols, n_rows,
        str(pocket_cells)
    ]
    csv_writer.writerow(csv_row)
    csv_file.flush()
    
    time.sleep(0.5)

    # 15. 网格可视化 (所有样本)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import pyvista as pv

        mapdl.allsel()
        grid = mapdl.mesh._grid

        plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
        plotter.add_mesh(grid, color='lightblue', show_edges=True,
                       edge_color='gray', line_width=0.3, opacity=0.8)

        if len(pocket_bottom_any_indices) > 0:
            bottom_points = all_node_coords[pocket_bottom_any_indices]
            plotter.add_points(bottom_points, color='red', point_size=5,
                             render_points_as_spheres=True)

        plotter.add_points(exc_actual.reshape(1, -1), color='green', point_size=15,
                         render_points_as_spheres=True)

        clamp_points_plot = []
        for (xmin, xmax, ymin, ymax) in all_clamp_areas:
            mapdl.allsel()
            mapdl.nsel("S", "LOC", "X", xmin, xmax)
            mapdl.nsel("R", "LOC", "Y", ymin, ymax)
            mapdl.nsel("R", "LOC", "Z", 0, H)
            for nid in mapdl.mesh.nnum:
                nid_int = int(nid)
                if nid_int in node_id_to_idx:
                    clamp_points_plot.append(all_node_coords[node_id_to_idx[nid_int]])
        mapdl.allsel()
        if clamp_points_plot:
            clamp_points_plot = np.array(clamp_points_plot)
            plotter.add_points(clamp_points_plot, color='yellow', point_size=8,
                             render_points_as_spheres=True)

        plotter.add_text(f'Sample {valid_samples+1}: {n_pockets_to_machine}/{num_machined} pockets, '
                       f'depth={depth_min:.0f}~{depth_max:.0f}%\n'
                       f'LightBlue=Mesh, Red=Bottom, Green=Excitation, Yellow=Clamp',
                       font_size=10)
        plotter.camera_position = 'iso'
        plotter.screenshot(os.path.join(VIZ_DIR, f'sample_{valid_samples:03d}_mesh.png'))
        plotter.close()
    except Exception as e:
        print(f"  可视化失败: {e}")

    # 成功走完所有流程，有效样本数 + 1
    valid_samples += 1

# 关闭CSV日志文件
csv_file.close()
print(f"CSV日志已保存: {csv_path}")

mapdl.exit()
elapsed = time.time() - t0
print(f"\n生成完成, 耗时 {elapsed:.0f}s")
print(f"总样本数: {N_SAMPLES}个 (总尝试{attempt_count}次)")


# ============ 保存 HDF5 ============
def save_h5(name, idx_slice):
    idxs = list(idx_slice)
    with h5py.File(os.path.join(OUT_DIR, name), 'w') as f:
        for i, idx in enumerate(idxs):
            grp = f.create_group(f'sample_{i}')
            grp.create_dataset('points', data=all_points[idx])
            grp.create_dataset('point_frf', data=all_frf[idx])
            grp.create_dataset('frequencies', data=all_freqs[idx])
            grp.create_dataset('modal_omega', data=all_omega[idx])
            grp.create_dataset('modal_zeta', data=all_zeta[idx])
            grp.create_dataset('modal_phi', data=all_phi[idx])
            grp.create_dataset('modal_phi_exc', data=all_phi_exc[idx])
            grp.create_dataset('point_features', data=all_features[idx])
    print(f"  保存: {name} ({len(idxs)}样本)")


save_h5('train.h5', range(N_TRAIN))
save_h5('val.h5', range(N_TRAIN, N_TRAIN + N_VAL))
save_h5('test.h5', range(N_TRAIN + N_VAL, N_SAMPLES))

# ============ FRF 可视化 ============
print("\n生成FRF可视化...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

coords0 = all_points[0]
frf0 = all_frf[0]
freqs0 = all_freqs[0]
amp0 = np.sqrt(frf0[..., 0]**2 + frf0[..., 1]**2)
n_nodes0 = len(coords0)

# 随机选5个存活节点
alive_idx_list = np.where(all_features[0][:, 3] == 1.0)[0]
np.random.seed(42)
selected_idx = np.random.choice(alive_idx_list, size=5, replace=False)

# 双面板: 左=dB幅值 (全动态范围), 右=线性幅值 (峰高度直观)
fig = plt.figure(figsize=(20, 14))

for i, (idx, label) in enumerate(zip(selected_idx, range(5))):
    coord_str = f'({coords0[idx, 0]*1000:.0f},{coords0[idx, 1]*1000:.0f},{coords0[idx, 2]*1000:.0f})mm'
    amp = amp0[idx]

    # --- 左侧: dB 幅值 (标准 FRF 格式, 显示全部共振) ---
    ax_db = fig.add_subplot(5, 2, 2*i + 1)
    amp_db = 20 * np.log10(amp + 1e-12)
    ax_db.semilogx(freqs0, amp_db, 'b-', linewidth=1.0)
    # 标记固有频率
    for k in range(N_MODES):
        fk = all_omega[0][k] / (2*np.pi)
        zk = all_zeta[0][k]
        ax_db.axvline(fk, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                      label=f'f{k+1}={fk:.0f}Hz (ζ={zk:.4f})' if i == 0 else '')
    ax_db.set_ylabel(f'Point{label+1}\n{coord_str}\nMagnitude (dB)', fontsize=8)
    db_max = amp_db.max()
    ax_db.set_ylim(db_max - 60, db_max + 5)
    ax_db.grid(alpha=0.3)
    if i == 0:
        ax_db.legend(fontsize=7, loc='upper right')

    # --- 右侧: 线性幅值 (共振峰高度直观) ---
    ax_lin = fig.add_subplot(5, 2, 2*i + 2)
    ax_lin.semilogx(freqs0, amp, 'b-', linewidth=1.0)
    for k in range(N_MODES):
        fk = all_omega[0][k] / (2*np.pi)
        ax_lin.axvline(fk, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_lin.set_ylabel(f'Point{label+1}\nMagnitude (lin)', fontsize=8)
    ax_lin.grid(alpha=0.3)
    # 线性轴聚焦共振区, 丢弃极值以显示二阶峰
    p95 = np.percentile(amp, 95)
    p99 = np.percentile(amp, 99.9)
    ax_lin.set_ylim(0, min(p95 * 3, p99 * 0.8))  # 裁掉极端峰值以显示次峰

axes = fig.get_axes()
axes[-2].set_xlabel('Frequency (Hz)')
axes[-1].set_xlabel('Frequency (Hz)')

fig.suptitle(f'Grooved Workpiece FRF — {n_nodes0} nodes, '
             f'f₁={all_omega[0][0]/(2*np.pi):.0f}Hz, '
             f'f₂={all_omega[0][1]/(2*np.pi):.0f}Hz'
             + (f', f₃={all_omega[0][2]/(2*np.pi):.0f}Hz' if N_MODES >= 3 else ''),
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'sample_000_frf.png'), dpi=150)
plt.close()
print(f"可视化保存: {VIZ_DIR}/")
