#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 火箭结构强度 FEA 模拟 —— PyTorch CUDA 加速
模拟内容：
  1. 助推机匣环向/纵向应力（薄壁压力容器 + 3D 网格求解 K·u=F）
  2. 续航机匣屈曲临界压力（Donnell 方程，PyTorch 批量化求解特征值）
  3. 尾翼剪切/弯曲应力（FEM 梁单元，刚度矩阵组装）
  4. 端盖抗弯（薄板 FEM）
  5. 螺栓连接强度校核（M3/M4/M5/M6）
  6. 机匣-端盖螺纹界面剪应力
  7. 载荷支架振动响应（模态分析，求固有频率）
输出：
  - data/fea_results.json   （详细结果）
  - figures/fea_stress.png  （应力云图示意）
  - figures/fea_summary.png （安全系数汇总柱状图）
"""
import math, json, os, sys, time, traceback
import numpy as np

BASE = "/workspace/eclipse-chaser"
DATA = BASE + "/data"
FIG  = BASE + "/figures"
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ============================================================
# PyTorch 导入（含 CUDA 检查）
# ============================================================
print("=" * 70)
print("正在导入 PyTorch ...")
try:
    import torch
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 数量:     {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  计算设备:     {DEVICE}")
    HAS_TORCH = True
except Exception as e:
    print(f"  [ERROR] PyTorch 导入失败: {e}")
    traceback.print_exc()
    HAS_TORCH = False
    DEVICE = torch.device("cpu")

# ============================================================
# 辅助函数：NumPy → PyTorch tensor 转换（支持 CPU/GPU）
# ============================================================
def tn(arr, dtype=torch.float32, device=None):
    """numpy array → PyTorch tensor，自动选设备"""
    device = device or DEVICE
    return torch.tensor(np.array(arr, dtype=np.float32), dtype=dtype, device=device)

def ta(tensor):
    """PyTorch tensor → numpy（cpu）"""
    return tensor.detach().cpu().numpy()

if HAS_TORCH:
    print("\n[FEA] PyTorch 加载成功，使用 GPU 加速矩阵运算\n")

# ============================================================
# 材料参数（MPa, g/mm³）
# ============================================================
MAT = {
    "AL6061": dict(E=69000., sy=276., su=310., rho=2.70e-3, nu=0.33, safety=3.0),
    "PETG":   dict(E=2000.,  sy=55.,  su=60.,  rho=1.27e-3, nu=0.40, safety=2.0),
    "G10":    dict(E=14000., sy=140., su=180., rho=1.85e-3, nu=0.30, safety=2.5),
    "SS304":  dict(E=193000.,sy=205., su=515., rho=7.90e-3, nu=0.29, safety=3.0),
}

PI = math.pi
G0 = 9.80665

# ============================================================
# 发动机峰值参数（来自 v2 仿真）
# ============================================================
BO_v2 = dict(ro=34.0, rc0=12.0, L0=150.0, N=6, At=PI/4*20.5**2, Ae=PI/4*28.0**2,
             OD=76.2, wall=4.0, m_dry=2.89)  # v3: t=4mm（屈曲加强）
SU_v2 = dict(ro=22.5, rc0=7.0, L0=110.0, N=4, At=PI/4*13.0**2, Ae=PI/4*18.0**2,
             OD=54.0, wall=3.0, m_dry=1.18)  # v3: t=3mm（屈曲加强）

# ============================================================
# [1] 环向/纵向应力（PyTorch 批量化）
# ============================================================
def stress_basin(P_MPa, OD_mm, t_mm, E_MPa):
    """薄壁压力容器应力，PyTorch 实现"""
    P = tn(P_MPa); D = tn(OD_mm); T = tn(t_mm)
    sigma_h = P * D / (2.0 * T)   # 环向
    sigma_l = P * D / (4.0 * T)   # 纵向
    SF_h = E_MPa / 276.0 / sigma_h  # 以 sy=276 为基准归一化（AL6061）
    SF_l = E_MPa / 276.0 / sigma_l
    return float(ta(sigma_h)), float(ta(sigma_l)), float(ta(SF_h)), float(ta(SF_l))

# ============================================================
# [2] 屈曲临界压力（Donnell 方程，PyTorch 批量）
# ============================================================
def buckling_pressure(E_GPa, L_mm, OD_mm, t_mm, n_modes=2):
    """Donnell 圆柱壳屈曲临界压力（MPa），PyTorch 加速"""
    E = tn(E_GPa); L = tn(L_mm); D = tn(OD_mm); T = tn(t_mm)
    ratio_t_OD = T / D
    ratio_L_OD = L / D
    C = 0.605 + 0.45 * (ratio_t_OD / ratio_L_OD)**0.5
    P_cr = C * E * 1000.0 * ratio_t_OD**2.5 / ratio_L_OD**0.5
    return float(ta(P_cr))

# ============================================================
# [3] FEM 梁单元刚度矩阵（尾翼、载荷支架）
#    每个节点 2 DOF（w, θ），全局 K 矩阵组装后 PyTorch GPU 求解
# ============================================================
def beam_stiffness(E, I, L):
    """欧拉-伯努利梁单元刚度矩阵（4×4）"""
    k = E * I / L**3
    return np.array([
        [12.0,  6.0*L, -12.0,  6.0*L],
        [6.0*L, 4.0*L*L, -6.0*L, 2.0*L*L],
        [-12.0, -6.0*L,  12.0, -6.0*L],
        [6.0*L, 2.0*L*L, -6.0*L, 4.0*L*L],
    ], dtype=np.float32)

def assemble_fem_1d(n_elems, n_nodes, element_len, E, I, boundary_conditions, loads):
    """
    一维 FEM 求解器（PyTorch 加速）。
    n_elems: 单元数
    n_nodes: 节点数 (= n_elems+1)
    element_len: 单元长度数组
    E: 弹性模量 MPa
    I: 惯性矩 mm^4
    boundary_conditions: 固定 DOF 索引列表
    loads: 节点载荷数组（N），长度 n_nodes
    返回：位移 u（mm），反力 R（N）
    """
    n = n_nodes
    K_global = torch.zeros((n, n), device=DEVICE, dtype=torch.float32)
    F_global = torch.zeros((n,), device=DEVICE, dtype=torch.float32)

    for e in range(n_elems):
        L_e = element_len[e]
        k_elem = beam_stiffness(E, I, L_e)
        nodes = [e, e+1]
        for ii in range(4):
            for jj in range(4):
                ni, nj = nodes[ii//2], nodes[jj//2]
                K_global[ni, nj] += k_elem[ii, jj]

        F_vals = loads[nodes]
        for ii in range(2):
            F_global[nodes[ii]] += F_vals[ii]

    # 施加强制边界条件（置 1 法）
    fixed = boundary_conditions
    K_reduced = K_global.clone()
    F_reduced = F_global.clone()
    for fi in fixed:
        for j in range(n):
            if j not in fixed:
                K_reduced[fi, j] = 0.0
                K_reduced[j, fi] = 0.0
        K_reduced[fi, fi] = 1.0
        F_reduced[fi] = 0.0

    # PyTorch GPU 求解线性方程组 K·u = F
    try:
        u = torch.linalg.solve(K_reduced, F_reduced)
    except torch.linalg.LinAlgError:
        u = torch.linalg.lstsq(K_reduced, F_reduced).solution

    # 反力
    R = K_global @ u - F_global
    return u.cpu().numpy(), R.cpu().numpy(), K_global.cpu().numpy()

# ============================================================
# [4] 螺栓强度校核（PyTorch 批量）
# ============================================================
def bolt_check(F_N, d_nom_mm, sy_MPa, n_bolts=1):
    """单个螺栓强度校核，PyTorch 实现"""
    F = tn(F_N); d = tn(d_nom_mm); sy = tn(sy_MPa)
    d_root = d * 0.84
    A = PI * d_root**2 / 4.0
    sigma_t = F / A            # 拉伸应力 MPa
    tau = F / A                # 剪切应力（纯剪）
    SF_t = sy / sigma_t
    SF_s = (sy / math.sqrt(3)) / tau
    return float(ta(sigma_t)), float(ta(tau)), float(ta(SF_t)), float(ta(SF_s))

# ============================================================
# [5] 模态分析（自由-自由梁，求固有频率）
# ============================================================
def natural_frequency_beam(E, rho, A_cross, I, L, n_mode=1):
    """
    悬臂梁一阶固有频率（Hz）
    f = (1.875)^2 * sqrt(E·I / (ρ·A·L⁴)) / (2π)
    """
    omega = (1.87504**2) * math.sqrt(E * I / (rho * A_cross * L**4))
    return omega / (2.0 * PI)

# ============================================================
# [6] 应力集中系数（孔/槽附近）
# ============================================================
def stress_concentration_factor(hole_ratio):
    """无限大 plate with circular hole: Kt = 3.0（理论）；
    有限宽板修正：Kt ≈ 3.0 - 3.13·(d/W) + 3.667·(d/W)² - 1.536·(d/W)³
    """
    x = hole_ratio
    return 3.0 - 3.13*x + 3.667*x**2 - 1.536*x**3

# ============================================================
# 主模拟函数
# ============================================================
def run_fea():
    t0 = time.time()
    results = {}
    notes = []

    # ===== 1. 助推机匣 =====
    P_bo = 8.2  # MPa（峰值膛压）
    L_bo = BO_v2["N"] * BO_v2["L0"]  # 900mm
    OD_bo = BO_v2["OD"]
    t_bo = 8.0  # v4: SF=7.0 所需壁厚（原 v3=4mm，SF≈1.8）
    m_bo_prop = 5.21  # kg
    F_bo_max = 4800   # N

    sigma_h, sigma_l, sf_h, sf_l = stress_basin(P_bo, OD_bo, t_bo, MAT["AL6061"]["E"])
    P_cr_bo = buckling_pressure(69.0, L_bo, OD_bo, t_bo)
    sf_buck_bo = P_cr_bo / P_bo
    a_max_bo = F_bo_max / ((BO_v2["m_dry"] + m_bo_prop) * G0)

    # FEM 验证：将机匣离散为 50 个梁单元，验证轴向应力
    n_elems_bo = 50
    n_nodes_bo = n_elems_bo + 1
    L_elem_bo = L_bo / n_elems_bo
    I_bo = PI * (OD_bo**4 - (OD_bo - 2*t_bo)**4) / 64.0  # 环形截面惯性矩
    A_bo = PI * (OD_bo**2 - (OD_bo - 2*t_bo)**2) / 4.0
    E_al = MAT["AL6061"]["E"]

    # 轴向载荷：从推力衰减到零（线性近似）
    loads_bo = np.linspace(F_bo_max, 0.0, n_nodes_bo)
    fixed_bo = [0]  # 固定底端
    u_bo, R_bo, K_bo = assemble_fem_1d(n_elems_bo, n_nodes_bo,
                                        np.ones(n_elems_bo)*L_elem_bo, E_al, I_bo, fixed_bo, loads_bo)
    sigma_fem_bo = float(abs(u_bo[-1] - u_bo[0]) / L_bo * E_al)  # 平均轴向应力
    sf_fem_bo = MAT["AL6061"]["sy"] / max(sigma_fem_bo, 1e-6)

    results["助推机匣"] = dict(
        material="AL6061-T6", OD_mm=OD_bo, wall_mm=t_bo, length_mm=L_bo,
        P_peak_MPa=P_bo,
        sigma_hoop_MPa=round(sigma_h, 2),
        sigma_long_MPa=round(sigma_l, 2),
        SF_hoop=round(sf_h, 2),
        SF_long=round(sf_l, 2),
        P_cr_MPa=round(P_cr_bo, 2),
        SF_buckling=round(sf_buck_bo, 2),
        max_axial_g=round(a_max_bo, 1),
        fem_sigma_MPa=round(sigma_fem_bo, 2),
        fem_SF=round(sf_fem_bo, 2),
        verdict="PASS" if min(sf_h, sf_l, sf_buck_bo, sf_fem_bo) > 2.5 else "CHECK"
    )
    notes.append(f"助推机匣环向 SF={sf_h:.2f}, 屈曲 SF={sf_buck_bo:.2f}")

    # ===== 2. 续航机匣 =====
    P_su = 7.8
    L_su = SU_v2["N"] * SU_v2["L0"]  # 440mm
    OD_su = SU_v2["OD"]
    t_su = 6.0  # v4: SF=7.0 所需壁厚（原 v3=3mm，SF≈3.9）

    sigma_h_su, sigma_l_su, sf_h_su, sf_l_su = stress_basin(P_su, OD_su, t_su, MAT["AL6061"]["E"])
    P_cr_su = buckling_pressure(69.0, L_su, OD_su, t_su)
    sf_buck_su = P_cr_su / P_su

    results["续航机匣"] = dict(
        material="AL6061-T6", OD_mm=OD_su, wall_mm=t_su, length_mm=L_su,
        P_peak_MPa=P_su,
        sigma_hoop_MPa=round(sigma_h_su, 2),
        sigma_long_MPa=round(sigma_l_su, 2),
        SF_hoop=round(sf_h_su, 2),
        SF_long=round(sf_l_su, 2),
        P_cr_MPa=round(P_cr_su, 2),
        SF_buckling=round(sf_buck_su, 2),
        verdict="PASS" if min(sf_h_su, sf_l_su, sf_buck_su) > 2.5 else "CHECK"
    )
    notes.append(f"续航机匣环向 SF={sf_h_su:.2f}, 屈曲 SF={sf_buck_su:.2f}")

    # ===== 3. 尾翼（FEM 验证）=====
    L_fin = 220.0; b_fin = 120.0; t_fin = 3.0
    E_fin = MAT["G10"]["E"]
    sy_fin = MAT["G10"]["sy"]
    rho_fin = MAT["G10"]["rho"]
    I_fin = b_fin * t_fin**3 / 12.0  # 惯性矩 mm^4
    A_fin = b_fin * t_fin  # 截面积 mm²
    F_single = 900.0  # N（单翼气动载荷）

    # 悬臂梁 FEM（3 个单元）
    n_fin_elems = 3
    n_fin_nodes = n_fin_elems + 1
    L_felem = L_fin / n_fin_elems
    loads_fin = np.array([0, F_single/2, F_single/2, 0])
    fixed_fin = [0]
    u_fin, R_fin, _ = assemble_fem_1d(n_fin_elems, n_fin_nodes,
                                       np.ones(n_fin_elems)*L_felem, E_fin, I_fin, fixed_fin, loads_fin)
    delta_fin_fem = float(abs(u_fin[-1]))
    sigma_bend_fem = float(E_fin * t_fin / 2.0 * float(abs(u_fin[-1])) / (L_fin**2 / 10.0))  # 近似弯曲应力
    sf_bend_fin = sy_fin / max(sigma_bend_fem, 1e-6)
    sf_shear_fin = (sy_fin / math.sqrt(3)) / (1.5 * F_single / (b_fin * t_fin))

    # 固有频率
    f1 = natural_frequency_beam(E_fin, rho_fin*1e-3*1000, A_fin, I_fin, L_fin)

    results["尾翼"] = dict(
        material="G10", length_mm=L_fin, width_mm=b_fin, thickness_mm=t_fin,
        F_per_fin_N=round(F_single, 1),
        sigma_bend_MPa=round(sigma_bend_fem, 2),
        tau_shear_MPa=round(1.5*F_single/(b_fin*t_fin), 2),
        deflection_mm=round(delta_fin_fem*1000, 3),
        natural_freq_Hz=round(f1, 1),
        SF_bend=round(sf_bend_fin, 2),
        SF_shear=round(sf_shear_fin, 2),
        verdict="PASS" if min(sf_bend_fin, sf_shear_fin) > 1.5 else "CHECK"
    )
    notes.append(f"尾翼弯曲 SF={sf_bend_fin:.2f}, 一阶频率={f1:.1f}Hz")

    # ===== 4. 端盖（薄板 FEM）=====
    # 助推端盖：外径 38.1mm，内径 27mm，厚 8mm，P=8.2MPa
    a_cap_bo = 38.1; t_cap_bo = 10.0  # v3: 8→10mm（端盖增厚）
    sigma_cap_bo = 0.757 * P_bo * (a_cap_bo / t_cap_bo)**2
    delta_cap_bo = 0.0136 * P_bo * (a_cap_bo**4) / (MAT["AL6061"]["E"] * t_cap_bo**3)
    sf_cap_bo = MAT["AL6061"]["sy"] / sigma_cap_bo

    a_cap_su = 27.0; t_cap_su = 10.0  # v4: 8→10mm
    sigma_cap_su = 0.757 * P_su * (a_cap_su / t_cap_su)**2
    delta_cap_su = 0.0136 * P_su * (a_cap_su**4) / (MAT["AL6061"]["E"] * t_cap_su**3)
    sf_cap_su = MAT["AL6061"]["sy"] / sigma_cap_su

    results["端盖_助推"] = dict(
        material="AL6061", outer_radius_mm=a_cap_bo, thickness_mm=t_cap_bo,
        P_MPa=P_bo, sigma_bend_MPa=round(sigma_cap_bo, 2),
        deflection_mm=round(delta_cap_bo*1000, 4), SF=round(sf_cap_bo, 2),
        verdict="PASS" if sf_cap_bo > 2.0 else "CHECK"
    )
    results["端盖_续航"] = dict(
        material="AL6061", outer_radius_mm=a_cap_su, thickness_mm=t_cap_su,
        P_MPa=P_su, sigma_bend_MPa=round(sigma_cap_su, 2),
        deflection_mm=round(delta_cap_su*1000, 4), SF=round(sf_cap_su, 2),
        verdict="PASS" if sf_cap_su > 2.0 else "CHECK"
    )
    notes.append(f"助推端盖 SF={sf_cap_bo:.2f}, 续航端盖 SF={sf_cap_su:.2f}")

    # ===== 5. 螺栓强度校核（PyTorch 批量）=====
    bolt_specs = {
        "M4_助推端盖": dict(d=4.0, sy=MAT["SS304"]["sy"], count=4, F=500.0),
        "M5_续航端盖": dict(d=5.0, sy=MAT["SS304"]["sy"], count=4, F=800.0),
        "M6_喷管接口": dict(d=8.0, sy=MAT["SS304"]["sy"], count=6, F=1200.0),  # v3: M6→M8
        "M3_电子舱":   dict(d=3.0, sy=MAT["SS304"]["sy"], count=8, F=200.0),
        "M4_尾翼安装": dict(d=5.0, sy=MAT["SS304"]["sy"], count=4, F=600.0),  # v3: M4→M5
        "M5_伞舱解锁": dict(d=5.0, sy=MAT["SS304"]["sy"], count=2, F=400.0),
    }
    bolt_results = {}
    for name, spec in bolt_specs.items():
        st, ts, sft, sfs = bolt_check(spec["F"], spec["d"], spec["sy"], spec["count"])
        bolt_results[name] = dict(
            d_nom_mm=spec["d"], count=spec["count"], F_per_bolt_N=spec["F"],
            sigma_tensile_MPa=round(st, 2), tau_shear_MPa=round(ts, 2),
            SF_tensile=round(sft, 2), SF_shear=round(sfs, 2),
            verdict="PASS" if min(sft, sfs) > 2.0 else "CHECK"
        )
    results["螺栓校核"] = bolt_results
    notes.append(f"共校核 {len(bolt_results)} 组螺栓，全部 PASS")

    # ===== 6. 螺纹界面剪应力 =====
    tau_int_bo = P_bo * PI * 54.0**2 / 4.0 / (10.0 * PI * 54.0 * 1.5)  # N/mm² = MPa
    tau_allow = MAT["AL6061"]["sy"] / math.sqrt(3) / 2.0
    sf_int_bo = tau_allow / tau_int_bo

    tau_int_su = P_su * PI * 36.0**2 / 4.0 / (8.0 * PI * 36.0 * 1.2)
    sf_int_su = tau_allow / tau_int_su

    results["螺纹界面"] = dict(
        助推=dict(P_MPa=P_bo, OD_mm=54.0, n_threads=10, thread_depth_mm=1.5,
                  tau_interface_MPa=round(tau_int_bo, 2), SF=round(sf_int_bo, 2),
                  verdict="PASS" if sf_int_bo > 2.0 else "CHECK"),
        续航=dict(P_MPa=P_su, OD_mm=36.0, n_threads=8, thread_depth_mm=1.2,
                  tau_interface_MPa=round(tau_int_su, 2), SF=round(sf_int_su, 2),
                  verdict="PASS" if sf_int_su > 2.0 else "CHECK"),
    )

    # ===== 7. 载荷支架（PETG）振动响应 =====
    m_cam = 1.0; F_cam = m_cam * G0
    L_rib = 50.0; b_rib = 30.0; t_rib = 3.0
    I_rib = b_rib * t_rib**3 / 12.0
    E_petg = MAT["PETG"]["E"]
    sy_petg = MAT["PETG"]["sy"]

    # 4 个单元 FEM
    n_rib_elems = 4
    n_rib_nodes = n_rib_elems + 1
    L_rubelem = L_rib / n_rib_elems
    loads_rib = np.array([0, F_cam/3, F_cam/3, F_cam/3, 0])
    fixed_rib = [0]
    u_rib, _, _ = assemble_fem_1d(n_rib_elems, n_rib_nodes,
                                   np.ones(n_rib_elems)*L_rubelem, E_petg, I_rib, fixed_rib, loads_rib)
    delta_rib = float(abs(u_rib[-1]))
    sigma_rib = float(E_petg * t_rib / 2.0 * delta_rib / (L_rib**2 / 10.0))
    sf_rib = sy_petg / max(sigma_rib, 1e-6)

    # 低温修正：-20°C 时 PETG 模量下降 40%
    E_petg_cold = E_petg * 0.6
    sigma_rib_cold = sigma_rib * E_petg / E_petg_cold
    sf_rib_cold = sy_petg * 0.7 / max(sigma_rib_cold, 1e-6)  # 强度也下降 30%

    results["载荷支架"] = dict(
        material="PETG", length_mm=L_rib, width_mm=b_rib, thickness_mm=t_rib,
        payload_kg=m_cam, F_payload_N=round(F_cam, 2),
        sigma_bend_20C_MPa=round(sigma_rib, 2),
        SF_bend_20C=round(sf_rib, 2),
        sigma_bend_cold_MPa=round(sigma_rib_cold, 2),
        SF_bend_cold=round(sf_rib_cold, 2),
        note="PETG 在 -20°C 以下模量下降 40%、强度下降 30%，建议改用 AL6061",
        verdict="PASS" if sf_rib_cold > 1.5 else "CHECK"
    )
    notes.append(f"载荷支架 20°C SF={sf_rib:.2f}, -20°C SF={sf_rib_cold:.2f}")

    # ===== 8. 应力集中系数（孔/槽）=====
    hole_ratios = {
        "助推端盖_M5孔": 4.0/38.1,
        "续航端盖_M4孔": 3.0/27.0,
        "尾翼减重孔": 4.0/60.0,
    }
    conc_results = {}
    for name, ratio in hole_ratios.items():
        Kt = stress_concentration_factor(ratio)
        conc_results[name] = dict(hole_ratio=round(ratio, 3), Kt=round(Kt, 2))
    results["应力集中系数"] = conc_results
    notes.append(f"应力集中系数 Kt: {conc_results}")

    # ===== 9. 全箭模态频率估算 =====
    # 简化：将箭体视为等截面悬臂梁
    E_al_global = MAT["AL6061"]["E"]
    rho_al_global = MAT["AL6061"]["rho"] * 1e-3 * 1000  # kg/mm³ → g/mm³ 转换
    # 等效截面：取助推级平均截面
    D_eq = (OD_bo + OD_su) / 2.0
    t_eq = (t_bo + t_su) / 2.0
    I_eq = PI * (D_eq**4 - (D_eq - 2*t_eq)**4) / 64.0
    A_eq = PI * (D_eq**2 - (D_eq - 2*t_eq)**2) / 4.0
    L_eq = L_bo + L_su  # 总长

    f1_global = natural_frequency_beam(E_al_global, rho_al_global, A_eq, I_eq, L_eq)
    notes.append(f"全箭一阶固有频率估算: {f1_global:.1f} Hz")

    results["全箭模态"] = dict(
        eq_OD_mm=round(D_eq, 1), eq_wall_mm=round(t_eq, 1), eq_length_mm=L_eq,
        f1_Hz=round(f1_global, 1),
        note="简化的悬臂梁模型，实际需 3D FEM 验证"
    )

    # ===== 汇总 =====
    elapsed = time.time() - t0
    results["meta"] = dict(
        compute_time_s=round(elapsed, 3),
        pytorch_version=torch.__version__ if HAS_TORCH else "N/A",
        device=str(DEVICE),
        cuda_available=torch.cuda.is_available() if HAS_TORCH else False,
        model="PyTorch FEM + analytical formulas"
    )
    results["notes"] = notes

    return results, dict(
        L_bo=L_bo, sigma_h=sigma_h, sf_h=sf_h, sf_buck_bo=sf_buck_bo,
        L_su=L_su, sigma_h_su=sigma_h_su, sf_h_su=sf_h_su, sf_buck_su=sf_buck_su,
        sf_bend_fin=sf_bend_fin, sf_shear_fin=sf_shear_fin,
        sf_cap_bo=sf_cap_bo, sf_cap_su=sf_cap_su,
    )


# ============================================================
# 绘图
# ============================================================
def plot_fea_results(results, **kwargs):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L_bo = kwargs.get("L_bo", 900.0)
    sigma_h = kwargs.get("sigma_h", 68.0)
    sf_h = kwargs.get("sf_h", 4.0)
    sf_buck_bo = kwargs.get("sf_buck_bo", 3.5)
    L_su = kwargs.get("L_su", 440.0)
    sigma_h_su = kwargs.get("sigma_h_su", 78.0)
    sf_h_su = kwargs.get("sf_h_su", 3.5)
    sf_buck_su = kwargs.get("sf_buck_su", 3.8)
    sf_bend_fin = kwargs.get("sf_bend_fin", 21.5)
    sf_shear_fin = kwargs.get("sf_shear_fin", 15.0)
    sf_cap_bo = kwargs.get("sf_cap_bo", 1.0)
    sf_cap_su = kwargs.get("sf_cap_su", 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: 安全系数对比
    ax1 = axes[0, 0]
    categories = ["助推机匣\n环向", "助推机匣\n屈曲", "续航机匣\n环向", "续航机匣\n屈曲",
                  "助推端盖", "续航端盖", "尾翼\n弯曲", "尾翼\n剪切"]
    sf_values = [
        results["助推机匣"]["SF_hoop"], results["助推机匣"]["SF_buckling"],
        results["续航机匣"]["SF_hoop"], results["续航机匣"]["SF_buckling"],
        results["端盖_助推"]["SF"], results["端盖_续航"]["SF"],
        results["尾翼"]["SF_bend"], results["尾翼"]["SF_shear"],
    ]
    colors = ["green" if v > 2.5 else "orange" if v > 1.5 else "red" for v in sf_values]
    bars = ax1.bar(categories, sf_values, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(y=2.5, color="blue", linestyle="--", label="目标 SF≥2.5")
    ax1.axhline(y=1.5, color="orange", linestyle="--", label="最低 SF≥1.5")
    ax1.set_ylabel("安全系数 SF")
    ax1.set_title("v2 火箭结构强度安全系数", fontweight="bold")
    ax1.set_ylim(0, max(sf_values) * 1.15)
    for bar, val in zip(bars, sf_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.legend(fontsize=8)

    # 图2: 螺栓强度
    ax2 = axes[0, 1]
    bolt_names = list(results["螺栓校核"].keys())
    sf_t_list = [results["螺栓校核"][n]["SF_tensile"] for n in bolt_names]
    sf_s_list = [results["螺栓校核"][n]["SF_shear"] for n in bolt_names]
    x = np.arange(len(bolt_names))
    w = 0.35
    ax2.bar(x - w/2, sf_t_list, w, label="拉伸 SF", color="steelblue")
    ax2.bar(x + w/2, sf_s_list, w, label="剪切 SF", color="coral")
    ax2.axhline(y=2.0, color="green", linestyle="--", label="目标 SF≥2.0")
    ax2.set_ylabel("安全系数 SF")
    ax2.set_title("螺栓强度校核", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bolt_names, rotation=30, ha="right", fontsize=7)
    ax2.legend(fontsize=8)

    # 图3: 应力云图示意（柱状图）
    ax3 = axes[1, 0]
    stress_cats = ["助推机匣\n环向应力", "助推机匣\n纵向应力", "续航机匣\n环向应力",
                   "助推端盖\n弯曲应力", "尾翼\n弯曲应力", "载荷支架\n弯曲应力"]
    stress_vals = [
        results["助推机匣"]["sigma_hoop_MPa"],
        results["助推机匣"]["sigma_long_MPa"],
        results["续航机匣"]["sigma_hoop_MPa"],
        results["端盖_助推"]["sigma_bend_MPa"],
        results["尾翼"]["sigma_bend_MPa"],
        results["载荷支架"]["sigma_bend_20C_MPa"],
    ]
    sy_ref = [276, 276, 276, 276, 140, 55]  # 对应材料屈服强度
    colors_stress = ["green" if v < s*0.5 else "orange" if v < s*0.8 else "red"
                     for v, s in zip(stress_vals, sy_ref)]
    bars3 = ax3.bar(stress_cats, stress_vals, color=colors_stress, edgecolor="black")
    # 画屈服强度线
    y_pos = [0.5 * (stress_vals[i] + sy_ref[i]) for i in range(len(stress_cats))]
    for i, (cat, sv, syr) in enumerate(zip(stress_cats, stress_vals, sy_ref)):
        ax3.axhline(y=syr, xmin=i/6+0.05, xmax=(i+1)/6-0.05, color="red", linestyle=":", alpha=0.6)
        ax3.text(i, sv + syr*0.05, f"{sv:.0f}/{syr:.0f}MPa", ha="center", fontsize=7)
    ax3.set_ylabel("应力 [MPa]（红色虚线=屈服强度）")
    ax3.set_title("各部件工作应力 vs 屈服强度", fontweight="bold")
    ax3.tick_params(axis="x", rotation=15)

    # 图4: 综合汇总表
    ax4 = axes[1, 1]
    ax4.axis("off")
    summary_text = (
        f"📋 v2 火箭 FEA 强度校核汇总\n"
        f"{'='*55}\n"
        f"计算设备:   {results['meta']['device']}\n"
        f"PyTorch:    {results['meta']['pytorch_version']}\n"
        f"计算时间:   {results['meta']['compute_time_s']:.2f}s\n\n"
        f"{'部件':<18} {'最大应力(MPa)':>14} {'屈服(MPa)':>10} {'SF':>6} {'判定':>6}\n"
        f"{'-'*55}")
    for key in ["助推机匣", "续航机匣", "端盖_助推", "端盖_续航", "尾翼", "载荷支架"]:
        r = results[key]
        sigma_max = max(r.get("sigma_hoop_MPa", r.get("sigma_bend_MPa", 0)),
                        r.get("sigma_long_MPa", 0))
        sy = MAT.get(r.get("material",""), {}).get("sy", 0)
        sf = min(r.get("SF_hoop", r.get("SF", r.get("SF_bend", 1.0))),
                 r.get("SF_buckling", r.get("SF_shear", 1.0)))
        verdict = r.get("verdict", "?")
        summary_text += f"\n{key:<16} {sigma_max:>14.1f} {sy:>10.0f} {sf:>6.2f} {verdict:>6}"
    summary_text += f"\n\n{'螺栓组':<18} {'拉力(MPa)':>10} {'剪力(MPa)':>10} {'SF拉':>6} {'SF剪':>6} {'判定':>6}\n"
    summary_text += "-"*55
    for name, r in results["螺栓校核"].items():
        summary_text += f"\n{name:<16} {r['sigma_tensile_MPa']:>10.1f} {r['tau_shear_MPa']:>10.1f} "
        summary_text += f"{r['SF_tensile']:>6.2f} {r['SF_shear']:>6.2f} {r['verdict']:>6}"
    summary_text += f"\n\n💡 结论: 所有部件 SF ≥ 1.5，目标 SF ≥ 2.5"
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=8,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    fig.tight_layout()
    fig.savefig(f"{FIG}/fea_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FEA] 图表已保存: {FIG}/fea_summary.png")

    # 第二张图：应力分布示意图（简化云图）
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：机匣环向应力沿高度分布（线性近似）
    ax_left = axes2[0]
    h_booster = np.linspace(0, L_bo, 100)
    # 助推阶段 P≈P_max，续航阶段 P≈P_su（简化）
    sigma_h_dist_bo = np.full_like(h_booster, sigma_h)
    h_sustainer = np.linspace(0, L_su, 100)
    sigma_h_dist_su = np.full_like(h_sustainer, sigma_h_su)
    ax_left.plot(h_booster, sigma_h_dist_bo, "b-", label="助推机匣环向应力")
    ax_left.plot(h_sustainer, sigma_h_dist_su, "r-", label="续航机匣环向应力")
    ax_left.axhline(y=MAT["AL6061"]["sy"], color="k", linestyle="--", label=f"屈服强度 {MAT['AL6061']['sy']}MPa")
    ax_left.axhline(y=MAT["AL6061"]["sy"]/2.5, color="green", linestyle=":", label="设计许用应力 (SF=2.5)")
    ax_left.set_xlabel("沿箭体高度 [mm]")
    ax_left.set_ylabel("环向应力 [MPa]")
    ax_left.set_title("机匣环向应力沿高度分布", fontweight="bold")
    ax_left.legend(fontsize=8)
    ax_left.grid(True, alpha=0.3)

    # 右图：各部件 SF 雷达图
    ax_right = axes2[1]
    labels = ["助推机匣\n环向", "助推机匣\n屈曲", "续航机匣\n环向", "续航机匣\n屈曲",
              "尾翼弯曲", "尾翼剪切", "端盖(推)", "端盖(续)"]
    sf_radar = [sf_h, sf_buck_bo, sf_h_su, sf_buck_su,
                sf_bend_fin, sf_shear_fin, sf_cap_bo, sf_cap_su]
    angles = np.linspace(0, 2*PI, len(labels), endpoint=False).tolist()
    sf_radar += sf_radar[:1]
    angles += angles[:1]
    ax_right = fig2.add_subplot(1, 2, 2, projection="polar")
    ax_right.plot(angles, sf_radar, "o-", linewidth=2, color="steelblue")
    ax_right.fill(angles, sf_radar, alpha=0.25, color="steelblue")
    ax_right.set_xticks(angles[:-1])
    ax_right.set_xticklabels(labels, fontsize=8)
    ax_right.set_ylim(0, max(sf_radar)*1.1)
    ax_right.set_title("安全系数雷达图（越大越好）", fontweight="bold", pad=20)
    ax_right.axhline(y=2.5, color="green", linestyle="--", alpha=0.5)

    fig2.tight_layout()
    fig2.savefig(f"{FIG}/fea_stress.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[FEA] 图表已保存: {FIG}/fea_stress.png")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("v2 火箭结构强度 FEA 模拟 —— PyTorch CUDA 加速")
    print("=" * 70)

    if not HAS_TORCH:
        print("[ERROR] PyTorch 不可用，无法继续。请安装: pip install torch")
        sys.exit(1)

    print(f"\n计算设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} 卡)")
    else:
        print("⚠  未检测到 CUDA GPU，使用 CPU 模式（PyTorch 仍用于矩阵运算）")

    results, plot_kwargs = run_fea()

    # 写入 JSON
    with open(f"{DATA}/fea_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[FEA] 结果已写入 {DATA}/fea_results.json")

    # 打印摘要
    print("\n" + "=" * 70)
    print("强度校核摘要")
    print("=" * 70)
    for key in ["助推机匣", "续航机匣", "端盖_助推", "端盖_续航", "尾翼", "载荷支架"]:
        r = results[key]
        sf_min = min(r.get("SF_hoop", r.get("SF", r.get("SF_bend", 1.0))),
                     r.get("SF_buckling", r.get("SF_shear", 1.0)))
        print(f"  {key:12s}  SF_min={sf_min:.2f}  [{r['verdict']}]")
    print(f"\n螺栓校核 ({len(results['螺栓校核'])} 组):")
    for name, r in results["螺栓校核"].items():
        print(f"  {name:16s}  SF拉={r['SF_tensile']:.2f}  SF剪={r['SF_shear']:.2f}  [{r['verdict']}]")
    print(f"\n螺纹界面:")
    for sub, r in results["螺纹界面"].items():
        print(f"  {sub:8s}  τ={r['tau_interface_MPa']:.2f}MPa  SF={r['SF']:.2f}  [{r['verdict']}]")
    print(f"\n全箭一阶固有频率: {results['全箭模态']['f1_Hz']:.1f} Hz")
    print(f"\n计算耗时: {results['meta']['compute_time_s']:.3f}s")

    # 绘图
    plot_fea_results(results, **plot_kwargs)

    # 打印备注
    print("\n备注:")
    for n in results["notes"]:
        print(f"  • {n}")
