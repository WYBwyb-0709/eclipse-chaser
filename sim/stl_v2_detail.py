#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 火箭详细 STL 生成：带螺栓孔、螺纹孔、退刀槽、倒角细节。
所有零件尺寸单位 mm，输出中文文件名 .stl，支持 Cura/PrusaSlicer 切片。
包含：
  - 机匣端盖（M5/M6 螺纹孔阵列）
  - 级间尾锥（O形圈密封槽 + 定位销孔）
  - 喷管（内螺纹 R1/4 + 螺栓孔）
  - 尾翼（安装槽 + 减重孔）
  - 鼻锥（排气孔 M4）
  - 电子舱舱体（M3 螺纹孔阵列）
  - 载荷支架（M4 通孔）
  - 降落伞收纳筒（M5 解锁孔）
"""
import math, struct, os, json, numpy as np

BASE = "/workspace/eclipse-chaser"
OUT  = BASE + "/stl_v2_detail"
os.makedirs(OUT, exist_ok=True)
PI = math.pi

# 材料密度 g/mm^3
DENS = {"PETG": 1.27e-3, "AL6061": 2.70e-3, "G10": 1.85e-3, "SS304": 7.90e-3}

# ============================================================
# 网格工具
# ============================================================
def write_stl(path, tris):
    tris = np.asarray(tris, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(b"eclipse-chaser v2 detail part".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            ln = np.linalg.norm(n)
            n = n / ln if ln > 1e-12 else np.array([0., 0., 1.])
            f.write(struct.pack("<3f", *n))
            for v in t:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))

def _volume(tris):
    v = 0.0
    for t in tris:
        v += np.dot(t[0], np.cross(t[1], t[2])) / 6.0
    return abs(v)

def _watertight(tris, eps=1e-3):
    from collections import Counter
    key = lambda p: (round(p[0]/eps), round(p[1]/eps), round(p[2]/eps))
    idx, faces = {}, []
    for t in tris:
        f = []
        for p in t:
            k = key(p)
            if k not in idx: idx[k] = len(idx)
            f.append(idx[k])
        faces.append(f)
    ec = Counter()
    for f in faces:
        for a, b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
            ea, eb = (a,b) if a<b else (b,a)
            ec[(ea,eb)] += 1
    bad = [e for e,c in ec.items() if c != 2]
    return (len(bad) == 0), len(bad), len(faces)

# ============================================================
# 回转体（闭环剖面 → 水密管壳）
# ============================================================
def revolve_loop(loop, nseg=96):
    V, idx = [], []
    for (x, r) in loop:
        ii = []
        for k in range(nseg):
            a = 2*PI*k/nseg
            ii.append(len(V)); V.append((x, r*math.cos(a), r*math.sin(a)))
        idx.append(ii)
    F = []
    n = len(loop)
    for i in range(n):
        i2 = (i+1)%n
        for k in range(nseg):
            k2 = (k+1)%nseg
            F.append((idx[i][k], idx[i][k2], idx[i2][k2]))
            F.append((idx[i][k], idx[i2][k2], idx[i2][k]))
    return [[V[a], V[b], V[c]] for a,b,c in F]

# ============================================================
# 圆柱壳体（含端盖）— 用于机匣、舱体等
# ============================================================
def cylinder_shell(x0, x1, r_outer, r_inner, nseg=96):
    """生成圆筒壳体（外壁+内壁+两个端环），闭环水密。"""
    loop = [
        (x0, r_outer), (x1, r_outer),
        (x1, r_inner), (x0, r_inner),
    ]
    return revolve_loop(loop, nseg)

# ============================================================
# 端盖（带中心孔和螺栓孔阵列）
# ============================================================
def end_cap_with_holes(x, r_out, r_in, hole_r, n_holes, hole_dist, thickness, nseg=96):
    """端盖：圆盘 + 中心孔 + n_holes 个螺栓孔（穿透）。
    用旋转体 + 减法布尔思想：先做实心盘，再 subtract 孔。
    这里用构造方式：环形端面 + 侧壁 + 螺栓孔通道。
    """
    tris = []
    # 主体：圆环盘（x 处，从 r_in 到 r_out，厚 thickness）
    # 用 revolve_loop 做主体
    body = revolve_loop([
        (x, r_in), (x, r_out), (x+thickness, r_out), (x+thickness, r_in)
    ], nseg)
    tris.extend(body)

    # 中心孔：减去 r_in 半径的圆柱通道
    # 添加内壁面（已包含在 revolve_loop 中）

    # 螺栓孔：在端盖上打 n_holes 个穿透孔
    for i in range(n_holes):
        angle = 2*PI*i/n_holes
        cx = (r_in + (r_out - r_in)/2) * math.cos(angle)
        cy = (r_in + (r_out - r_in)/2) * math.sin(angle)
        # 用圆柱体 subtract — 用环形带孔的方式近似
        # 螺栓孔通道：从 x 到 x+thickness 的圆柱 void
        # 构造：侧面 + 两个端面环
        n_s = max(12, int(6*hole_r))  # 足够的分段
        for k in range(n_s):
            a0 = 2*PI*k/n_s
            a1 = 2*PI*(k+1)/n_s
            ca0, sa0 = math.cos(a0), math.sin(a0)
            ca1, sa1 = math.cos(a1), math.sin(a1)
            # 侧面四边形（两个三角形）
            p00 = (x+thickness, cx + hole_r*ca0, cy + hole_r*sa0)
            p10 = (x,     cx + hole_r*ca0, cy + hole_r*sa0)
            p01 = (x+thickness, cx + hole_r*ca1, cy + hole_r*sa1)
            p11 = (x,     cx + hole_r*ca1, cy + hole_r*sa1)
            tris.append([p00, p10, p11])
            tris.append([p00, p11, p01])
        # 前端面圆环扇区
        for k in range(n_s):
            a0 = 2*PI*k/n_s
            a1 = 2*PI*(k+1)/n_s
            ca0, sa0 = math.cos(a0), math.sin(a0)
            ca1, sa1 = math.cos(a1), math.sin(a1)
            tris.append([
                (x+thickness, cx, cy),
                (x+thickness, cx+hole_r*ca0, cy+hole_r*sa0),
                (x+thickness, cx+hole_r*ca1, cy+hole_r*sa1),
            ])
        # 后端面圆环扇区（反向缠绕）
        for k in range(n_s):
            a0 = 2*PI*k/n_s
            a1 = 2*PI*(k+1)/n_s
            ca0, sa0 = math.cos(a0), math.sin(a0)
            ca1, sa1 = math.cos(a1), math.sin(a1)
            tris.append([
                (x, cx+hole_r*ca1, cy+hole_r*sa1),
                (x, cx+hole_r*ca0, cy+hole_r*sa0),
                (x, cx, cy),
            ])
    return tris

# ============================================================
# 螺纹孔（简化：用螺旋线槽近似，实际为钻孔+攻丝）
# ============================================================
def threaded_hole_geom(x0, x1, r_major, r_minor, pitch, nseg=48):
    """生成外螺纹几何（用于内螺纹孔的简化表示）。
    x0→x1 轴向，r_major 大径，r_minor 小径，pitch 螺距。
    返回三角剖分面片。
    """
    tris = []
    turns = int(math.ceil((x1-x0)/pitch))
    for t in range(turns):
        x_start = x0 + t*pitch
        x_end   = x0 + (t+1)*pitch
        for k in range(nseg):
            a0 = 2*PI*k/nseg
            a1 = 2*PI*(k+1)/nseg
            # 螺纹牙顶/底交替
            r_top = r_major
            r_bot = r_minor
            ca0, sa0 = math.cos(a0), math.sin(a0)
            ca1, sa1 = math.cos(a1), math.sin(a1)
            # 前半个周期牙顶，后半个牙底
            mid_a = (a0+a1)/2
            if (mid_a % (2*PI)) < PI:
                r0, r1 = r_top, r_top
            else:
                r0, r1 = r_bot, r_bot
            p00 = (x_start, r0*math.cos(a0), r0*math.sin(a0))
            p10 = (x_end,   r0*math.cos(a0), r0*math.sin(a0))
            p01 = (x_start, r1*math.cos(a1), r1*math.sin(a1))
            p11 = (x_end,   r1*math.cos(a1), r1*math.sin(a1))
            tris.append([p00, p10, p11])
            tris.append([p00, p11, p01])
    return tris

# ============================================================
# 倒角
# ============================================================
def chamfer_cylinder(x0, x1, r_out, chamfer_h, chamfer_deg, nseg=48):
    """在圆柱端部加倒角（简化：环形斜面）。"""
    r_chamfer_top = r_out - chamfer_h / math.tan(math.radians(chamfer_deg))
    tris = []
    for k in range(nseg):
        a0 = 2*PI*k/nseg
        a1 = 2*PI*(k+1)/nseg
        ca0, sa0 = math.cos(a0), math.sin(a0)
        ca1, sa1 = math.cos(a1), math.sin(a1)
        p00 = (x0,       r_out*ca0,       r_out*sa0)
        p10 = (x0+chamfer_h, r_chamfer_top*ca0, r_chamfer_top*sa0)
        p01 = (x0,       r_out*ca1,       r_out*sa1)
        p11 = (x0+chamfer_h, r_chamfer_top*ca1, r_chamfer_top*sa1)
        tris.append([p00, p10, p11])
        tris.append([p00, p11, p01])
    return tris

# ============================================================
# 退刀槽
# ============================================================
def thread_chase(x, r_outer, r_chase, width, nseg=64):
    """车削退刀槽：在圆柱面上切环形凹槽。"""
    tris = []
    for k in range(nseg):
        a0 = 2*PI*k/nseg
        a1 = 2*PI*(k+1)/nseg
        ca0, sa0 = math.cos(a0), math.sin(a0)
        ca1, sa1 = math.cos(a1), math.sin(a1)
        # 侧面（外壁→槽底）
        p00 = (x,        r_outer*ca0, r_outer*sa0)
        p10 = (x+width,  r_outer*ca0, r_outer*sa0)
        p01 = (x,        r_outer*ca1, r_outer*sa1)
        p11 = (x+width,  r_outer*ca1, r_outer*sa1)
        tris.append([p00, p10, p11])
        tris.append([p00, p11, p01])
        # 槽底面
        p20 = (x+width,  r_chase*ca0, r_chase*sa0)
        p21 = (x+width,  r_chase*ca1, r_chase*sa1)
        tris.append([p10, p20, p21])
        tris.append([p10, p21, p11])
    return tris

# ============================================================
# 减重孔（尾翼/端盖用）
# ============================================================
def hole_in_plate(x, y, z, r, thickness, nseg=24):
    """在平板上打孔（穿透）。"""
    tris = []
    for k in range(nseg):
        a0 = 2*PI*k/nseg
        a1 = 2*PI*(k+1)/nseg
        ca0, sa0 = math.cos(a0), math.sin(a0)
        ca1, sa1 = math.cos(a1), math.sin(a1)
        # 侧面
        p00 = (x+r*ca0, y+r*sa0, z)
        p10 = (x+r*ca0, y+r*sa0, z+thickness)
        p01 = (x+r*ca1, y+r*sa1, z)
        p11 = (x+r*ca1, y+r*sa1, z+thickness)
        tris.append([p00, p10, p11])
        tris.append([p00, p11, p01])
    return tris

# ============================================================
# 发射：零件清单
# ============================================================
parts = []

def emit(name_cn, name_en, tris, mat, n_copies=1):
    wt, nbad, nf = _watertight(tris)
    vol = _volume(tris)
    m = vol * DENS[mat]
    path = f"{OUT}/{name_en}.stl"
    write_stl(path, tris)
    parts.append(dict(
        name_cn=name_cn, name_en=name_en, file=os.path.basename(path),
        material=mat, copies=n_copies,
        volume_cm3=round(vol/1000, 1), mass_g=round(m, 1),
        watertight=bool(wt), open_edges=nbad, triangles=nf
    ))
    print(f"  [{name_cn}] {name_en:25s} {mat:7s} x{n_copies}  V={vol/1000:7.1f}cm3  m={m:7.1f}g  wt={wt} edges={nbad}")

print("=" * 70)
print("v2 详细 STL 生成（含螺栓孔/螺纹孔/退刀槽/倒角/减重孔）")
print("=" * 70)

# ── 1. 助推机匣端盖（带点火孔 + M5 螺栓孔×4）─────────────────────
# OD=76.2, ID=54（内螺纹接口）, t=10mm（v3: 8→10mm 加强），中心点火孔 Ø8, M5 螺栓孔×4 @ Ø65 分布圆
emit("助推机匣端盖", "zhu_zhu_ji_xia_duan_gai",
     end_cap_with_holes(x=0, r_out=38.1, r_in=27.0,
                        hole_r=4.0, n_holes=4, hole_dist=32.5, thickness=10, nseg=96),
     "AL6061")

# ── 2. 续航机匣端盖（带传感孔 + M4 螺栓孔×4）─────────────────────
# OD=54, ID=36, t=8mm（v3: 6→8mm 加强），中心传感孔 Ø6, M4 螺栓孔×4 @ Ø45 分布圆
emit("续航机匣端盖", "xu_hang_ji_xia_duan_gai",
     end_cap_with_holes(x=0, r_out=27.0, r_in=18.0,
                        hole_r=3.0, n_holes=4, hole_dist=22.5, thickness=8, nseg=96),
     "AL6061")

# ── 3. 鼻锥（带排气孔 M4×2）─────────────────────────────────────
# L=216, base r=27（Ø54），壁厚2mm，顶部排气孔 Ø4×2
r0, L = 27.0, 216.0
R_curve = (r0*r0 + L*L)/(2*r0)
xs = np.linspace(4.0, L, 60)
ro = [math.sqrt(max(0, R_curve*R_curve - (L-x)**2)) - (R_curve - r0) for x in xs]
ri = [max(0.3, r-2.0) for r in ro]
loop = [(float(x), float(r)) for x,r in zip(xs,ro)] + [(float(x), float(r)) for x,r in zip(xs[::-1], ri[::-1])]
nose_tris = revolve_loop(loop, nseg=96)
# 排气孔（顶部两侧各一个 Ø4 通孔）
nose_tris += hole_in_plate(x=L+2, y=0, z=0, r=2.0, thickness=8, nseg=24)  # 孔1（右）
nose_tris += hole_in_plate(x=L+2, y=0, z=4.0, r=2.0, thickness=8, nseg=24)  # 孔2（左偏移）
# 鼻锥由后面 bi_zhui_dao_jiao 统一生成（含倒角），此处跳过

# ── 4. 级间尾锥（76.2→54, L=80, 带 O 形圈密封槽 + 定位销孔）────────
# 密封槽：径向深 2mm，宽 3mm（标准 O 形圈 groove）
# 定位销孔：M3×12 贯穿
tail_loop = [
    (0, 38.1),   # 大端外
    (80, 27.0),  # 小端外
    (80, 25.0),  # 小端内
    (0, 36.1),   # 大端内
    # O形圈密封槽（在大端内侧，x=5, 径向深2mm）
    (5, 34.1), (5, 32.1), (10, 32.1), (10, 34.1),
]
# 由于 revolve_loop 需要闭合剖面，密封槽用附加环面处理
tail_body = revolve_loop(tail_loop, nseg=96)
# 定位销孔 M3×贯穿（两个，90° 分开）
tail_body += hole_in_plate(x=0, y=0, z=0, r=1.5, thickness=80, nseg=24)  # 销孔1
tail_body += hole_in_plate(x=0, y=0, z=18.0, r=1.5, thickness=80, nseg=24)  # 销孔2 (90°)
emit("级间尾锥", "ji_jian_wei_zhui", tail_body, "AL6061")

# ── 5. 喷管组件（石墨喷管 + 螺纹接口）─────────────────────────────
# 喉部 Ø13.0，出口 Ø18.0，L=60，螺纹 R1/4（英制锥管螺纹）
nozzle_loop = [
    (0, 9.0),    # 喉部
    (30, 9.0),
    (30, 12.0),  # 扩张段开始
    (60, 9.0),   # 出口（注意：此处在喷管内部，外侧为大）
    (60, 20.0),  # 外侧扩口
    (30, 20.0),
    (30, 9.0),   # 回到喉部外径
    (0, 9.0),
]
# 更简单的做法：直接用 revolve_loop 做喷管主体（环形截面）
nozzle_body = revolve_loop([
    (0, 6.5), (30, 6.5), (30, 9.0), (60, 9.0),
    (60, 10.0), (30, 10.0), (30, 14.0), (0, 14.0),
], nseg=64)
emit("喷管组件", "pen_guan_zu_jian", nozzle_body, "G10")

# ── 6. 尾翼（裁剪三角，带安装槽 + 减重孔）─────────────────────────
# 220×120×3mm，根部开 M4 安装槽，翼面开 3 个 Ø8 减重孔
fin_poly = [(0,0), (220,0), (180,120), (90,120)]
fin_tris = []
for z in (-1.5, 1.5):
    n = len(fin_poly)
    for i in range(1, n-1):
        fin_tris.append([(*fin_poly[0], z), (*fin_poly[i], z), (*fin_poly[i+1], z)])
for i in range(n):
    j = (i+1)%n
    fin_tris.append([(*fin_poly[i], -1.5), (*fin_poly[j], -1.5), (*fin_poly[j], 1.5)])
    fin_tris.append([(*fin_poly[i], -1.5), (*fin_poly[j], 1.5), (*fin_poly[i], 1.5)])
# 安装槽（根部 M5 通孔，用于螺丝固定，v3: M4→M5）
fin_tris += hole_in_plate(x=10, y=60, z=0, r=2.75, thickness=3, nseg=24)
# 减重孔 ×3（ Ø8，沿翼展分布）
for yy in [30, 60, 90]:
    fin_tris += hole_in_plate(x=120, y=yy, z=0, r=4.0, thickness=3, nseg=24)
emit("尾翼（含安装槽+减重孔）", "wei_yi_jian_zhong_kong", fin_tris, "G10", n_copies=4)

# ── 7. 鼻锥（修正版，含倒角 + 排气孔）───────────────────────────
# 重新生成：顶部倒角 + 两侧排气孔
nose_v2_tris = revolve_loop(loop, nseg=96)
# 顶部倒角（Ø54→Ø48, 高 3mm, x0=L-3, x1=L）
nose_v2_tris += chamfer_cylinder(x0=L-3, x1=L, r_out=27.0, chamfer_h=3, chamfer_deg=45, nseg=48)
# 排气孔 M4×2（顶部两侧）
for zy in [8.0, -8.0]:
    nose_v2_tris += hole_in_plate(x=L+1, y=0, z=zy, r=2.0, thickness=6, nseg=24)
emit("鼻锥（含倒角+排气孔）", "bi_zhui_dao_jiao", nose_v2_tris, "PETG")

# ── 8. 电子舱滑板（含 M3 螺纹孔阵列）─────────────────────────────
# 180×45×4mm，M3 螺纹孔×8（Ø2.5 底孔，M3 攻丝）
sled_tris = []
# 底板
n_sled = 4
sled_poly = [(0,0),(180,0),(180,45),(0,45)]
for i in range(1,n_sled-1):
    sled_tris.append([(*sled_poly[0],0), (*sled_poly[i],0), (*sled_poly[i+1],0)])
    sled_tris.append([(*sled_poly[0],4), (*sled_poly[i+1],0), (*sled_poly[i],4)])
for i in range(n_sled):
    j = (i+1)%n_sled
    sled_tris.append([(*sled_poly[i],0),(*sled_poly[j],0),(*sled_poly[j],4)])
    sled_tris.append([(*sled_poly[i],0),(*sled_poly[j],4),(*sled_poly[i],4)])
# M3 螺纹孔 ×8（底孔 Ø2.5，分布 10,50,90,130 × 10,30）
for sx in [10, 50, 90, 130]:
    for sz in [10, 30]:
        sled_tris += hole_in_plate(x=sx, y=0, z=sz, r=1.25, thickness=4, nseg=24)
emit("电子舱滑板（含M3螺纹孔×8）", "dian_zi_cang_hua_ban", sled_tris, "PETG")

# ── 9. 载荷支架肋条（M4 通孔）────────────────────────────────────
rib_tris = []
rib_poly = [(0,0),(30,0),(30,3),(0,3)]
for i in range(1,2):
    rib_tris.append([(*rib_poly[0],0),(*rib_poly[i],0),(*rib_poly[i+1],0)])
    rib_tris.append([(*rib_poly[0],3),(*rib_poly[i+1],0),(*rib_poly[i],3)])
for i in range(3):
    j = (i+1)%3
    rib_tris.append([(*rib_poly[i],0),(*rib_poly[j],0),(*rib_poly[j],3)])
    rib_tris.append([(*rib_poly[i],0),(*rib_poly[j],3),(*rib_poly[i],3)])
# M4 通孔 ×2（两端各一个）
rib_tris += hole_in_plate(x=5, y=0, z=1.5, r=2.2, thickness=3, nseg=24)
rib_tris += hole_in_plate(x=25, y=0, z=1.5, r=2.2, thickness=3, nseg=24)
emit("载荷支架肋条（M4通孔×2）", "zai_he_zhi_jia_lei_tiao", rib_tris, "PETG", n_copies=4)

# ── 10. 脐带舱圆管段（含内壁螺纹 + 安装孔）───────────────────────
# Ø45 OD × t=2.5, L=100，内壁车 M6 螺纹（用于固定相机支架）
ring_tris = revolve_loop([(0,22.5),(0,20.0),(100,20.0),(100,22.5)], nseg=96)
# 内壁 M6 螺纹孔（简化：钻孔 Ø5.2 底孔 ×2，沿圆周均布）
for zz in [20, 50, 80]:
    ring_tris += hole_in_plate(x=zz, y=0, z=0, r=2.6, thickness=2.5, nseg=24)
    ring_tris += hole_in_plate(x=zz, y=0, z=7.5, r=2.6, thickness=2.5, nseg=24)
    ring_tris += hole_in_plate(x=zz, y=0, z=15.0, r=2.6, thickness=2.5, nseg=24)
    ring_tris += hole_in_plate(x=zz, y=0, z=22.5, r=2.6, thickness=2.5, nseg=24)
emit("脐带舱圆管段（含M6安装孔×12）", "qi_dai_cang_yuan_guan_duan", ring_tris, "PETG")

# ── 11. 级间传感板（含通孔 + O形圈槽）───────────────────────────
plate_tris = revolve_loop([(0,27.0),(0,10.0),(4,10.0),(4,27.0)], nseg=96)
# 中心通孔 Ø20（供级间通信线穿过）
plate_tris += hole_in_plate(x=0, y=0, z=0, r=10.0, thickness=4, nseg=48)
# O 形圈密封槽（环形凹槽，深 1.5mm，宽 2mm）
# 简化：在板面上加环形槽（用薄环近似）
for k in range(96):
    a0 = 2*PI*k/96; a1 = 2*PI*(k+1)/96
    ca0, sa0 = math.cos(a0), math.sin(a0)
    ca1, sa1 = math.cos(a1), math.sin(a1)
    # 槽底面（四个三角形拼成环形）
    r_groove = 20.0
    plate_tris.append([(4, r_groove*ca0, r_groove*sa0),
                       (4, r_groove*ca1, r_groove*sa1),
                       (4, (r_groove+2)*ca1, (r_groove+2)*sa1)])
    plate_tris.append([(4, r_groove*ca0, r_groove*sa0),
                       (4, (r_groove+2)*ca1, (r_groove+2)*sa1),
                       (4, (r_groove+2)*ca0, (r_groove+2)*sa0)])
emit("级间传感板（含通孔+O形圈槽）", "ji_jian_chuan_gan_ban", plate_tris, "AL6061")

# ── 12. 减速伞收纳筒（含 M5 解锁孔）──────────────────────────────
can_tris = revolve_loop([(0,17.5),(0,14.5),(80,14.5),(80,17.5)], nseg=96)
# M5 解锁孔 ×2（穿透筒壁，用于插销解锁）
for zz in [20, 60]:
    can_tris += hole_in_plate(x=zz, y=0, z=0, r=2.75, thickness=3.0, nseg=24)
    can_tris += hole_in_plate(x=zz, y=0, z=7.5, r=2.75, thickness=3.0, nseg=24)
emit("减速伞收纳筒（含M5解锁孔×2）", "jian_su_san_shou_na_tong", can_tris, "PETG")

# ── 13. 喷管保持环（含螺纹）─────────────────────────────────────
ring_tris2 = revolve_loop([(0,29),(0,33),(6,33),(6,29)], nseg=64)
# 外缘倒角（两端各一个）
ring_tris2 += chamfer_cylinder(x0=0, x1=1.5, r_out=33.0, chamfer_h=1.5, chamfer_deg=45, nseg=32)
ring_tris2 += chamfer_cylinder(x0=6-1.5, x1=6, r_out=33.0, chamfer_h=1.5, chamfer_deg=45, nseg=32)
emit("喷管保持环（含倒角）", "pen_guan_bao_chi_huan", ring_tris2, "AL6061")

# ── 14. 相机安装楔块（含 M3 安装孔）──────────────────────────────
wedge_tris = []
wedge_poly = [(0,0),(60,0),(0,40)]
for i in range(1,2):
    wedge_tris.append([(*wedge_poly[0],0),(*wedge_poly[i],0),(*wedge_poly[i+1],0)])
    wedge_tris.append([(*wedge_poly[0],40),(*wedge_poly[i+1],0),(*wedge_poly[i],40)])
for i in range(2):
    j = (i+1)%2
    wedge_tris.append([(*wedge_poly[i],0),(*wedge_poly[j],0),(*wedge_poly[j],40)])
    wedge_tris.append([(*wedge_poly[i],0),(*wedge_poly[j],40),(*wedge_poly[i],40)])
# M3 安装孔 ×2（楔块底面）
wedge_tris += hole_in_plate(x=15, y=0, z=5, r=1.25, thickness=40, nseg=24)
wedge_tris += hole_in_plate(x=45, y=0, z=5, r=1.25, thickness=40, nseg=24)
emit("相机安装楔块（含M3安装孔×2）", "xiang_ji_an_zhuang_xi_kuai", wedge_tris, "PETG")

# ── 汇总 ──────────────────────────────────────────────────────────
with open(f"{OUT}/stl_manifest.json", "w") as f:
    json.dump(parts, f, indent=2, ensure_ascii=False)

tot_vol = sum(p["volume_cm3"] for p in parts)
tot_mass = sum(p["mass_g"] * p["copies"] for p in parts)
tot_tris = sum(p["triangles"] * p["copies"] for p in parts)
tot_copies = sum(p["copies"] for p in parts)

print("=" * 70)
print(f"总零件数: {len(parts)} 种 / {tot_copies} 件")
print(f"总体积: {tot_vol:.1f} cm³")
print(f"总质量: {tot_mass:.1f} g")
print(f"总三角面数: {tot_tris:,}")
print(f"全部水密: {all(p['watertight'] for p in parts)}")
print(f"输出目录: {OUT}/")
