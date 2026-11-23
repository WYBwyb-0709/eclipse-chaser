#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""协和逐日 v4 详细 STL 生成（真实布尔版）
所有孔/槽/倒角都用 manifold3d 布尔运算生成，单件单壳体、水密。
输出 13 种零件，单位 mm。
"""
import math, os, json
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.geometry.point import Point

BASE = "/workspace/eclipse-chaser"
OUT = BASE + "/stl_v2_detail"
os.makedirs(OUT, exist_ok=True)
PI = math.pi
ENGINE = "manifold"

DENS = {"PETG": 1.27e-3, "AL6061": 2.70e-3, "G10": 1.85e-3, "SS304": 7.90e-3}


# ---------------- 基础几何 ----------------
def cyl_x(r, x0, x1, sec=96):
    """沿 X 轴的圆柱。"""
    m = trimesh.creation.cylinder(radius=r, height=abs(x1 - x0), sections=sec)
    m.apply_transform(trimesh.transformations.rotation_matrix(PI / 2, [0, 1, 0]))
    m.apply_translation([(x0 + x1) / 2.0, 0, 0])
    return m


def tube_x(r_out, r_in, x0, x1, sec=96):
    """沿 X 轴的圆管（外圆柱减内圆柱）。"""
    a = cyl_x(r_out, x0, x1, sec)
    b = cyl_x(r_in, x0 - 1.0, x1 + 1.0, sec)
    return a.difference(b, engine=ENGINE)


def axial_holes(mesh, holes, x0, x1, sec=32):
    """holes: [(y,z,r), ...]，沿 X 轴向打穿。"""
    cut = []
    for y, z, r in holes:
        c = cyl_x(r, x0 - 1.0, x1 + 1.0, sec)
        c.apply_translation([0, y, z])
        cut.append(c)
    return mesh.difference(cut, engine=ENGINE) if cut else mesh


def radial_holes(mesh, holes, r_out, wall, sec=32):
    """holes: [(x, theta_deg, r[, r_local]), ...]，沿径向打穿单侧壁。
    r_local 是该 x 处的实际外半径（锥体必须给），默认取 r_out。"""
    cut = []
    for h in holes:
        x, th, r = h[0], h[1], h[2]
        rl = h[3] if len(h) > 3 else r_out
        u = np.array([0.0, math.cos(math.radians(th)), math.sin(math.radians(th))])
        c = trimesh.creation.cylinder(radius=r, height=wall + 4.0, sections=sec)
        c.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], u))
        rc = rl - wall / 2.0
        c.apply_translation([x, u[1] * rc, u[2] * rc])
        cut.append(c)
    return mesh.difference(cut, engine=ENGINE) if cut else mesh


def revolve_loop(loop, nseg=96):
    """闭合 (x, r) 剖面绕 X 轴旋转，得到回转壳体。"""
    V, rings = [], []
    for x, r in loop:
        ring = []
        for k in range(nseg):
            a = 2 * PI * k / nseg
            ring.append(len(V))
            V.append((x, r * math.cos(a), r * math.sin(a)))
        rings.append(ring)
    F = []
    for i in range(len(loop)):
        i2 = (i + 1) % len(loop)
        for k in range(nseg):
            k2 = (k + 1) % nseg
            F.append((rings[i][k], rings[i][k2], rings[i2][k2]))
            F.append((rings[i][k], rings[i2][k2], rings[i2][k]))
    m = trimesh.Trimesh(vertices=np.array(V), faces=np.array(F), process=True)
    m.fix_normals()
    return m


def cone_groove(x0, x1, r0, r1, depth, nseg=96):
    """沿锥形内壁的环形槽刀具体：内壁半径从 r0 线性变到 r1，槽深 depth。"""
    return revolve_loop([(x0, r0 - 1.0), (x0, r0 + depth),
                         (x1, r1 + depth), (x1, r1 - 1.0)], nseg)


def solid_rev(xs, rs, nseg=96):
    """实心回转体（底/尖端封口），用于鼻锥等。"""
    V, rings = [], []
    for x, r in zip(xs, rs):
        if r <= 1e-6:
            continue
        ring = []
        for k in range(nseg):
            a = 2 * PI * k / nseg
            ring.append(len(V))
            V.append((float(x), float(r * math.cos(a)), float(r * math.sin(a))))
        rings.append(ring)
    c0 = len(V); V.append((float(xs[0]), 0, 0))
    tip = len(V); V.append((float(xs[-1]), 0, 0))
    F = []
    for k in range(nseg):                       # 底封口
        F.append((c0, rings[0][(k + 1) % nseg], rings[0][k]))
    for i in range(len(rings) - 1):             # 侧面
        for k in range(nseg):
            k2 = (k + 1) % nseg
            F.append((rings[i][k], rings[i][k2], rings[i + 1][k2]))
            F.append((rings[i][k], rings[i + 1][k2], rings[i + 1][k]))
    for k in range(nseg):                       # 尖端
        F.append((rings[-1][k], rings[-1][(k + 1) % nseg], tip))
    m = trimesh.Trimesh(vertices=np.array(V), faces=np.array(F), process=True)
    m.fix_normals()
    return m


def plate(poly_pts, holes, thickness, z0=0.0):
    """2D 多边形（带圆孔）沿 Z 拉伸。holes: [(x,y,r), ...]"""
    hl = [np.asarray(Point(x, y).buffer(r, resolution=24).exterior.coords) for x, y, r in holes]
    poly = Polygon(poly_pts, holes=hl)
    if not poly.is_valid:
        raise ValueError("plate: 多边形无效（孔越界或轮廓自交），shapely 会静默丢孔")
    m = trimesh.creation.extrude_polygon(poly, height=thickness)
    m.apply_translation([0, 0, z0])
    return m


# ---------------- 输出 ----------------
parts = []


def emit(name_cn, name_en, mesh, mat, n_copies=1):
    if isinstance(mesh, list):
        mesh = trimesh.util.concatenate(mesh)
    mesh.merge_vertices()
    mesh.process(validate=True)
    bodies = len(mesh.split(only_watertight=False))
    vol = abs(float(mesh.volume)) / 1000.0          # cm³
    mass = vol * 1000.0 * DENS[mat]                 # DENS 单位 g/mm³
    if not mesh.is_watertight:
        raise RuntimeError(f"{name_en}: 非水密")
    if bodies != 1:
        raise RuntimeError(f"{name_en}: 壳体数 {bodies} != 1（存在分离/重叠壳）")
    mesh.export(f"{OUT}/{name_en}.stl")
    parts.append(dict(
        name_cn=name_cn, name_en=name_en, file=f"{name_en}.stl",
        material=mat, copies=n_copies,
        volume_cm3=round(vol, 1), mass_g=round(mass, 1),
        watertight=True, open_edges=0, triangles=len(mesh.faces),
    ))
    print(f"  [{name_cn}] {name_en:26s} {mat:7s} x{n_copies}  "
          f"V={vol:7.1f}cm3  m={mass:7.1f}g  tri={len(mesh.faces)}")


print("=" * 72)
print("协和逐日 v4 详细 STL（manifold3d 真布尔）")
print("=" * 72)

# 1. 助推机匣端盖：Ø76.2，中心 Ø54 级间通道，厚 16，4×M5 底孔 Ø4.2 @ BCD65
cap_bo = tube_x(38.1, 27.0, 0.0, 16.0)
cap_bo = axial_holes(cap_bo, [(32.5, 0.0, 2.10), (0.0, 32.5, 2.10),
                              (-32.5, 0.0, 2.10), (0.0, -32.5, 2.10)], 0.0, 16.0)
emit("助推机匣端盖", "zhu_zhu_ji_xia_duan_gai", cap_bo, "AL6061")

# 2. 续航机匣端盖：Ø54，中心 Ø36 走线通道，厚 11，4×M4 底孔 Ø3.3 @ BCD45
cap_su = tube_x(27.0, 18.0, 0.0, 11.0)
cap_su = axial_holes(cap_su, [(22.5, 0.0, 1.65), (0.0, 22.5, 1.65),
                              (-22.5, 0.0, 1.65), (0.0, -22.5, 1.65)], 0.0, 11.0)
emit("续航机匣端盖", "xu_hang_ji_xia_duan_gai", cap_su, "AL6061")

# 3. 级间尾锥 76.2→54，L=80；内壁 O 形圈槽 + 2×M3 定位销孔
#    内壁 r_in(x)=36.1-0.13875x，壁厚恒为 2mm（外 r_in+2）
def _rin(x):
    return 36.1 - (36.1 - 25.0) * x / 80.0

tail = revolve_loop([(0, 38.1), (80, 27.0), (80, 25.0), (0, 36.1)], 96)
# 槽深 0.8（留 1.2mm 壁），随内锥面走，不会切穿
tail = tail.difference(cone_groove(5.0, 10.0, _rin(5.0), _rin(10.0), 0.8), engine=ENGINE)
tail = radial_holes(tail, [(40.0, 0.0, 1.5, _rin(40.0) + 2.0),
                           (40.0, 180.0, 1.5, _rin(40.0) + 2.0)], 0.0, 3.0)
emit("级间尾锥", "ji_jian_wei_zhui", tail, "AL6061")

# 4. 喷管：进气 r12 → 喉部 r10.25 → 出口 r14，外壳 r16，L=60
nozzle = revolve_loop([(0, 16.0), (60, 16.0), (60, 14.0), (30, 10.25), (0, 12.0)], 96)
emit("喷管组件", "pen_guan_zu_jian", nozzle, "G10")

# 5. 尾翼 220×120×3，根部 2×M5 安装孔 + 3×Ø10 减重孔（必须落在多边形内）
fin = plate([(0, 0), (220, 0), (180, 120), (90, 120)],
            [(20, 8, 2.75), (60, 8, 2.75),
             (110, 50, 5.0), (145, 50, 5.0), (110, 85, 5.0)],
            3.0, z0=-1.5)
emit("尾翼（含安装槽+减重孔）", "wei_yi_jian_zhong_kong", fin, "G10", n_copies=4)

# 6. 鼻锥：切线卵形，底 R27、L216、壁厚 2，顶部 2×Ø4 排气孔
R, L = 27.0, 216.0
rho = (R * R + L * L) / (2.0 * R)
xs = np.linspace(0.0, L, 90)
ro = np.sqrt(np.maximum(rho ** 2 - xs ** 2, 0.0)) + R - rho
ro[-1] = 0.0
nose_out = solid_rev(xs, ro, 96)
# 内腔向 x<0 延伸，保证底部开口（否则会留下一整块 2mm 实心底盖）；
# 尖端留 8mm 实心，避免内外卵形在 x=L 收成同一点（零壁厚、非流形接触）
xs2 = np.linspace(-10.0, L - 8.0, 110)
ri = np.where(xs2 >= 0.0,
              np.sqrt(np.maximum(rho ** 2 - xs2 ** 2, 0.0)) + R - rho - 2.0,
              R - 2.0)
ri = np.maximum(ri, 0.0)
nose_in = solid_rev(xs2, ri, 96)
nose = nose_out.difference(nose_in, engine=ENGINE)
nose = radial_holes(nose, [(180.0, 0.0, 2.0), (180.0, 180.0, 2.0)], 8.3, 2.0)
emit("鼻锥（含倒角+排气孔）", "bi_zhui_dao_jiao", nose, "PETG")

# 7. 电子舱滑板 180×45×4，8×Ø2.5
sled = plate([(0, 0), (180, 0), (180, 45), (0, 45)],
             [(x, y, 1.25) for x in (10, 50, 90, 130) for y in (10, 30)], 4.0)
emit("电子舱滑板（含M3螺纹孔×8）", "dian_zi_cang_hua_ban", sled, "PETG")

# 8. 载荷支架肋条 30×10×3（8mm 宽只剩 1.8mm 边距，加宽到 10），2×Ø4.4
rib = plate([(0, 0), (30, 0), (30, 10), (0, 10)],
            [(5, 5, 2.2), (25, 5, 2.2)], 3.0)
emit("载荷支架肋条（M4通孔×2）", "zai_he_zhi_jia_lei_tiao", rib, "PETG", n_copies=4)

# 9. 脐带舱圆管 Ø45×2.5，L100，12×M6 安装孔（径向）
umb = tube_x(22.5, 20.0, 0.0, 100.0)
umb = radial_holes(umb, [(x, th, 2.6) for x in (20, 50, 80) for th in (0, 90, 180, 270)],
                   22.5, 2.5)
emit("脐带舱圆管段（含M6安装孔×12）", "qi_dai_cang_yuan_guan_duan", umb, "PETG")

# 10. 级间传感板：Ø54，中心 Ø20，厚 4，正面 O 形圈槽
sensor = revolve_loop([(0, 10.0), (0, 27.0), (4, 27.0), (4, 10.0)], 96)
sensor = sensor.difference(tube_x(22.0, 20.0, 2.5, 5.0), engine=ENGINE)
emit("级间传感板（含通孔+O形圈槽）", "ji_jian_chuan_gan_ban", sensor, "AL6061")

# 11. 减速伞收纳筒 Ø35×3，L80，2×M5 径向解锁孔（相隔 180°）
can = tube_x(17.5, 14.5, 0.0, 80.0)
can = radial_holes(can, [(20.0, 90.0, 2.75), (60.0, 270.0, 2.75)], 17.5, 3.0)
emit("减速伞收纳筒（含M5解锁孔×2）", "jian_su_san_shou_na_tong", can, "PETG")

# 12. 喷管保持环 Ø66/Ø58，厚 6，两端倒角
ret = revolve_loop([(0, 31.5), (1.5, 33.0), (4.5, 33.0), (6, 31.5), (6, 29.0), (0, 29.0)], 96)
emit("喷管保持环（含倒角）", "pen_guan_bao_chi_huan", ret, "AL6061")

# 13. 相机安装楔块 60×40×40，2×Ø2.5
wedge = plate([(0, 0), (60, 0), (0, 40)],
              [(15, 6, 1.25), (40, 6, 1.25)], 40.0, z0=-20.0)
emit("相机安装楔块（含M3安装孔×2）", "xiang_ji_an_zhuang_xi_kuai", wedge, "PETG")

# ---------------- 汇总 ----------------
with open(f"{OUT}/stl_manifest.json", "w") as f:
    json.dump(parts, f, indent=2, ensure_ascii=False)

tot_vol = sum(p["volume_cm3"] for p in parts)
tot_mass = sum(p["mass_g"] * p["copies"] for p in parts)
tot_tris = sum(p["triangles"] * p["copies"] for p in parts)
tot_copies = sum(p["copies"] for p in parts)

print("=" * 72)
print(f"总零件数: {len(parts)} 种 / {tot_copies} 件")
print(f"总体积:   {tot_vol:.1f} cm³")
print(f"总质量:   {tot_mass:.1f} g")
print(f"总三角面: {tot_tris:,}")
print(f"全部水密: {all(p['watertight'] for p in parts)}")
print(f"输出目录: {OUT}/")
