#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 KNSB 两级探空火箭的 3D 打印/加工件 STL（单位 mm），并做水密性与质量核对。
所有几何按解析剖面/棱柱构建，网格闭合，可用切片软件直接打印或 CAM 加工。
"""
import math, struct, os, json
import numpy as np

BASE="/workspace/eclipse-chaser"; OUT=BASE+"/stl"; os.makedirs(OUT,exist_ok=True)
PI=math.pi

# ---------- 网格工具 ----------
def write_stl(path,tris):
    tris=np.asarray(tris,float)
    with open(path,"wb") as f:
        f.write(b"eclipse-chaser KNSB part".ljust(80,b"\0"))
        f.write(struct.pack("<I",len(tris)))
        for t in tris:
            n=np.cross(t[1]-t[0],t[2]-t[0]); ln=np.linalg.norm(n)
            n=n/ln if ln>1e-12 else np.array([0,0,1.0])
            f.write(struct.pack("<3f",*n))
            for v in t: f.write(struct.pack("<3f",*v))
            f.write(struct.pack("<H",0))

def volume(tris):
    v=0.0
    for t in tris: v+=np.dot(t[0],np.cross(t[1],t[2]))/6.0
    return abs(v)

def watertight(tris,eps=1e-6):
    from collections import Counter
    # 顶点量化合并
    key=lambda p: (round(p[0]/eps),round(p[1]/eps),round(p[2]/eps))
    idx={}; faces=[]
    for t in tris:
        f=[]
        for p in t:
            k=key(p)
            if k not in idx: idx[k]=len(idx)
            f.append(idx[k])
        faces.append(f)
    ec=Counter()
    for f in faces:
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
            ec[(a,b) if a<b else (b,a)]+=1
    bad=[e for e,c in ec.items() if c!=2]
    return (len(bad)==0), len(bad), len(faces)

def finalize(tris):
    tris=[np.array(t,float) for t in tris]
    if volume(tris)==0: return tris
    # 统一朝外：仅用于体积符号，不做朝向修复
    return tris

# 回转体（开放剖面，r=0 端收成尖点，r>0 端加端盖）
def revolve(profile,nseg=72,caps=True):
    V=[]; F=[]; rings=[]
    for (x,r) in profile:
        if r<=1e-9:
            rings.append(("apex",[len(V)],x,r)); V.append((x,0,0))
        else:
            idx=[]
            for k in range(nseg):
                a=2*PI*k/nseg
                idx.append(len(V)); V.append((x,r*math.cos(a),r*math.sin(a)))
            rings.append(("ring",idx,x,r))
    for i in range(len(rings)-1):
        t0,d0,_,_=rings[i]; t1,d1,_,_=rings[i+1]
        if t0=="ring" and t1=="ring":
            for k in range(nseg):
                a,b,c,d=d0[k],d0[(k+1)%nseg],d1[(k+1)%nseg],d1[k]
                F.append((a,b,c));F.append((a,c,d))
        elif t0=="apex" and t1=="ring":
            for k in range(nseg): F.append((d0[0],d1[k],d1[(k+1)%nseg]))
        elif t0=="ring" and t1=="apex":
            for k in range(nseg): F.append((d0[k],d0[(k+1)%nseg],d1[0]))
    if caps:
        for ri in (0,len(rings)-1):
            tag,d,x,r=rings[ri]
            if tag!="ring": continue
            c=len(V); V.append((x,0,0))
            for k in range(nseg):
                if ri==0: F.append((c,d[(k+1)%nseg],d[k]))
                else:     F.append((c,d[k],d[(k+1)%nseg]))
    return [ [V[a],V[b],V[c]] for a,b,c in F ]

# 闭合剖面回转（管/环），剖面首尾相连 → 环面拓扑，天然水密
def revolve_loop(loop,nseg=72):
    V=[];idx=[]
    for (x,r) in loop:
        ii=[]
        for k in range(nseg):
            a=2*PI*k/nseg
            ii.append(len(V));V.append((x,r*math.cos(a),r*math.sin(a)))
        idx.append(ii)
    F=[]
    n=len(loop)
    for i in range(n):
        i2=(i+1)%n
        for k in range(nseg):
            k2=(k+1)%nseg
            F.append((idx[i][k],idx[i][k2],idx[i2][k2]))
            F.append((idx[i][k],idx[i2][k2],idx[i2][k]))
    return [[V[a],V[b],V[c]] for a,b,c in F]

# 凸多边形棱柱
def prism(poly,z0,z1):
    n=len(poly); V=[]
    for z in (z0,z1):
        for (x,y) in poly: V.append((x,y,z))
    F=[]
    for i in range(1,n-1):
        F.append((0,i,i+1))            # back cap
        F.append((n,n+i+1,n+i))        # front cap
    for i in range(n):
        j=(i+1)%n
        F.append((i,j,n+j)); F.append((i,n+j,n+i))
    return [[V[a],V[b],V[c]] for a,b,c in F]

# ---------- 零件几何 ----------
DENS=dict(PETG=1.27e-3, AL6061=2.70e-3, G10=1.85e-3)  # g/mm^3
parts=[]

def emit(name,tris,mat,n_copies=1):
    wt,nbad,nf=watertight(tris)
    vol=volume(tris)  # mm^3
    m=vol*DENS[mat]*n_copies
    path=f"{OUT}/{name}.stl"; write_stl(path,tris)
    parts.append(dict(name=name,file=os.path.basename(path),material=mat,copies=n_copies,
                      volume_cm3=round(vol/1000,1),mass_g=round(m,1),
                      watertight=bool(wt),open_edges=nbad,triangles=nf))
    print(f"  {name:16s} {mat:7s} x{n_copies}  V={vol/1000:7.1f}cm3  m={m:7.1f}g  watertight={wt} (open edges={nbad})")

print("[STL] building parts")
# 1 鼻锥 切线拱形 L=216, base r=27 (54mm)，2mm 薄壁（闭环回转 → 水密）
r0,L=27.0,216.0; R=(r0*r0+L*L)/(2*r0)
xs=np.linspace(4.0,L,60)
ro=[math.sqrt(max(0,R*R-(L-x)**2))-(R-r0) for x in xs]
ri=[max(0.3,r-2.0) for r in ro]
loop=[(float(x),float(r)) for x,r in zip(xs,ro)]+[(float(x),float(r)) for x,r in zip(xs[::-1],ri[::-1])]
emit("nose_cone_54",revolve_loop(loop), "PETG")
# 2 尾锥/级间过渡 76.2->54, L=80, 壁厚2
emit("boat_tail_76to54",revolve_loop([(0,38.1),(80,27.0),(80,25.0),(0,36.1)]),"AL6061")
# 3 尾翼 裁剪三角 (4片), 厚3mm, G10
poly=[(0,0),(220,0),(180,120),(90,120)]
emit("fin_clipped_delta",prism(poly,-1.5,1.5),"G10",n_copies=4)
# 4 相机楔块 三角形棱柱 60x40x40
emit("camera_wedge",prism([(0,0),(60,0),(0,40)],0,40),"PETG")
# 5 电子舱滑板 180x45x4
emit("avionics_sled",prism([(0,0),(180,0),(180,45),(0,45)],0,4),"PETG")
# 6 喷管保持环 OD66/ID58 t6
emit("retainer_ring_58",revolve_loop([(0,29),(0,33),(6,33),(6,29)]),"AL6061")

# ========== v2 新增零件（适配助推 6 段 / 续航 4 段 / 载荷 1 kg）==========
# 7 助推机匣端盖（法兰式，带点火孔）Ø76.2 OD × t6，中心 Ø8 点火孔
emit("booster_end_cap",revolve_loop([(0,38.1),(0,28.0),(3,28.0),(3,27.0),(6,27.0),(6,28.0),(9,28.0),(9,38.1)]),"AL6061")
# 8 续航机匣端盖 Ø54 OD × t6，中心 Ø6 孔
emit("sustainer_end_cap",revolve_loop([(0,27.0),(0,19.5),(2,19.5),(2,18.0),(5,18.0),(5,19.5),(8,19.5),(8,27.0)]),"AL6061")
# 9 载荷支架（4 条肋条）180×30×3 ×4，用于把 1kg 相机悬于脐带舱中央
emit("payload_spacer_rib",prism([(0,0),(30,0),(30,3),(0,3)],0,3),"PETG",n_copies=4)
# 10 脐带舱（承载相机/飞控/电源）Ø45 圆管段 L=100 t=2.5（环形截面闭环回转）
emit("payload_ring_section",revolve_loop([(0,22.5),(0,20.0),(100,20.0),(100,22.5)]),"PETG")
# 11 级间点火/传感板（Ø54 适配，中心通孔 Ø20 供级间通信线）
emit("interstage_sensor_plate",revolve_loop([(0,27.0),(0,10.0),(4,10.0),(4,27.0)]),"AL6061")
# 12 降落伞收纳筒（Ø35 OD × t3，L=80，用于装减速伞折叠伞包）
emit("drogue_canister",revolve_loop([(0,17.5),(0,14.5),(80,14.5),(80,17.5)]),"PETG")

with open(f"{OUT}/stl_manifest.json","w") as f: json.dump(parts,f,indent=2)
tot=sum(p["mass_g"] for p in parts)
print(f"[STL] total printed/machined mass = {tot:.0f} g across {sum(p['copies'] for p in parts)} pieces")
print("[STL] all watertight:",all(p["watertight"] for p in parts))
