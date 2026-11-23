#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v6 三维可视化：用 trimesh 读取 STL，matplotlib 离屏渲染。
输出：
  figures/rocket_3d.png   —— 全零件轴向爆炸图 + 编号清单
  figures/parts_3d.png    —— 13 个零件等轴测画廊
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

BASE = "/workspace/eclipse-chaser"
SRC = BASE + "/stl_v2_detail"
FIG = BASE + "/figures"
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MAT_COLOR = {
    "AL6061": (0.72, 0.75, 0.79),
    "SS304":  (0.52, 0.58, 0.66),
    "PETG":   (0.95, 0.55, 0.16),
    "G10":    (0.16, 0.46, 0.30),
}
LIGHT = np.array([0.45, 0.55, 0.70]); LIGHT /= np.linalg.norm(LIGHT)

# 爆炸图从尾到头的展示顺序（与总装轴向一致）
ORDER = [
    "pen_guan_zu_jian", "pen_guan_bao_chi_huan", "zhu_zhu_ji_xia_duan_gai",
    "ji_jian_wei_zhui", "wei_yi_jian_zhong_kong", "ji_jian_chuan_gan_ban",
    "qi_dai_cang_yuan_guan_duan", "jian_su_san_shou_na_tong",
    "xu_hang_ji_xia_duan_gai", "zai_he_zhi_jia_lei_tiao",
    "xiang_ji_an_zhuang_xi_kuai", "dian_zi_cang_hua_ban",
    "bi_zhui_dao_jiao",
]


def load_parts():
    items = json.load(open(os.path.join(SRC, "stl_manifest.json"), encoding="utf-8"))
    by_en = {it["name_en"]: it for it in items}
    parts = []
    for en in ORDER:
        it = by_en.get(en)
        if it is None:
            continue
        mesh = trimesh.load(os.path.join(SRC, it["file"]), force="mesh")
        parts.append((it, mesh))
    return parts


def tri_data(mesh, max_faces=200000, seed=0):
    f = mesh.faces
    if len(f) > max_faces:
        rng = np.random.default_rng(seed)
        f = f[rng.choice(len(f), max_faces, replace=False)]
    v = mesh.vertices[f]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    ln = np.linalg.norm(n, axis=1); ln[ln == 0] = 1.0
    return v, n / ln[:, None]


def facecolors(n, base):
    inten = 0.34 + 0.66 * np.clip(n @ LIGHT, 0.0, 1.0)
    rgb = np.clip(np.asarray(base)[None, :] * inten[:, None], 0.0, 1.0)
    return np.column_stack([rgb, np.ones(len(rgb))])


def add_mesh(ax, mesh, base, xoff=0.0, max_faces=200000, seed=0):
    v, n = tri_data(mesh, max_faces, seed)
    v = v.copy()
    v[:, 0] += xoff
    pc = Poly3DCollection(v, facecolors=facecolors(n, base), edgecolor="none", linewidths=0)
    ax.add_collection3d(pc)
    return v


def fit_view(ax, pts, pad=0.08, max_ratio=3.0):
    """按数据范围设置各轴 lim，box_aspect 跟随真实尺寸；
    最长轴相对于次长轴的比值超过 max_ratio 时压缩，避免细长物体在方框里变成一条线。"""
    p = np.asarray(pts).reshape(-1, 3)
    lo, hi = p.min(axis=0), p.max(axis=0)
    ext = np.maximum(hi - lo, 1.0)
    c = (lo + hi) / 2.0
    half = ext / 2.0 * (1.0 + pad)
    ax.set_xlim(c[0] - half[0], c[0] + half[0])
    ax.set_ylim(c[1] - half[1], c[1] + half[1])
    ax.set_zlim(c[2] - half[2], c[2] + half[2])

    asp = ext / ext.max()
    order = np.argsort(asp)
    small = asp[order[-2]] if asp[order[-2]] > 1e-3 else 1.0
    if asp[order[-1]] / small > max_ratio:
        asp[order[-1]] = small * max_ratio
    ax.set_box_aspect(tuple(float(a) for a in asp))
    ax.set_axis_off()


def render_exploded(parts):
    fig = plt.figure(figsize=(20, 8), facecolor="white")
    ax = fig.add_axes([0.02, 0.03, 0.76, 0.92], projection="3d")
    ax.set_facecolor("white")

    cursor = 0.0
    gap = 55.0
    allpts = []
    for i, (it, mesh) in enumerate(parts):
        v = mesh.vertices
        # 长轴对齐 +X，径向居中
        lo, hi = v.min(axis=0), v.max(axis=0)
        offset = np.array([cursor - lo[0], -(lo[1] + hi[1]) / 2.0, -(lo[2] + hi[2]) / 2.0])
        m = mesh.copy(); m.apply_translation(offset)
        pts = add_mesh(ax, m, MAT_COLOR.get(it["material"], (0.6, 0.6, 0.6)), seed=i)
        allpts.append(pts)
        xc = (lo[0] + hi[0]) / 2.0 + offset[0]
        rad = max((hi[1] - lo[1]) / 2.0, (hi[2] - lo[2]) / 2.0)
        ax.text(xc, 0.0, -rad - 8.0, str(i + 1), fontsize=8.5, color="#222",
                ha="center", va="top")
        # 间隔随该件直径增大，避免大端盖与尾锥在长轴压缩后视觉粘连
        cursor += (hi[0] - lo[0]) + gap + 0.55 * 2.0 * rad

    fit_view(ax, np.vstack(allpts), pad=0.05)
    # 近侧视图：X 沿横向，避免大直径零件在斜视角下互相遮挡
    ax.view_init(elev=8, azim=-80)
    ax.set_title("协和逐日 v6 · 全零件轴向排布图（STL，按图纸尺寸）",
                 fontsize=15, pad=8, weight="bold")

    axl = fig.add_axes([0.78, 0.05, 0.21, 0.90]); axl.axis("off")
    axl.text(0.0, 1.0, "零件清单（尾 → 头）", fontsize=12, weight="bold", va="top")
    y = 0.94
    for i, (it, _) in enumerate(parts):
        axl.text(0.0, y, f"{i+1:2d}. {it['name_cn']}", fontsize=9.5, va="top")
        axl.text(0.97, y, f"{it['mass_g']:.0f} g", fontsize=9.5, va="top", ha="right", color="#555")
        y -= 0.061

    out = os.path.join(FIG, "rocket_3d.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def render_gallery(parts):
    ncol, nrow = 4, 4
    fig = plt.figure(figsize=(15, 15), facecolor="white")
    for i, (it, mesh) in enumerate(parts):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        v = mesh.vertices
        c = (v.min(axis=0) + v.max(axis=0)) / 2.0
        m = mesh.copy(); m.apply_translation(-c)
        pts = add_mesh(ax, m, MAT_COLOR.get(it["material"], (0.6, 0.6, 0.6)), seed=100 + i)
        fit_view(ax, pts, pad=0.18)
        ax.view_init(elev=24, azim=-56)
        ax.set_title(f"{it['name_cn']}\n{it['material']} · {it['mass_g']:.1f} g",
                     fontsize=10.5)

    for k in range(len(parts) + 1, ncol * nrow + 1):
        ax = fig.add_subplot(nrow, ncol, k); ax.axis("off")

    fig.suptitle("协和逐日 v6 · 零件三维模型（trimesh 读取 STL）", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG, "parts_3d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parts = load_parts()
    tot_m = sum(it["mass_g"] * it.get("copies", 1) for it, _ in parts)
    tot_v = sum(it["volume_cm3"] * it.get("copies", 1) for it, _ in parts)
    n_pcs = sum(it.get("copies", 1) for it, _ in parts)
    tri = sum(len(m.faces) for _, m in parts)
    print(f"载入 {len(parts)} 种 / {n_pcs} 件零件，总体积 {tot_v:.1f} cm³，"
          f"总质量 {tot_m:.1f} g，三角面 {tri}")
    p1 = render_exploded(parts); print("写出", p1)
    p2 = render_gallery(parts); print("写出", p2)
