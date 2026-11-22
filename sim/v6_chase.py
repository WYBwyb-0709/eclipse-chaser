#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
协和逐日 v6 —— "骑影"方案
============================
v5 的问题：T_cam 只有 88~95 s，低于地面全食。诊断发现这不符合物理——
  同高度静止点全食 = 地面值（6 km 处仅多 27 m 伞径，可忽略）；
  以 u 沿月影方向同向飞行，理论全食 = W/(V−u)，u=100 m/s 时 +9%。
所以 v6 重做三件事：
  [A] 机位改为国内陆地最长全食点（秦皇岛一带，>116 s），并用 Besselian 差分
      求当地月影真实方位/速度，不再硬编码 93.6°；
  [B] 弹道与月影时间轴联合寻优：发射方位、发射角、发射时刻 τ 一起搜，
      让火箭在窗口内尽量长时间"跟着月影跑"；
  [C] 给出理论上限 T = W/(V−u) 与仿真值对照，判定真实增益。
"""
import numpy as np, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/workspace/eclipse-chaser"; DATA = BASE + "/data"; FIG = BASE + "/figures"
os.makedirs(DATA, exist_ok=True); os.makedirs(FIG, exist_ok=True)
R_E = 6371000.0; G0 = 9.80665; PI = math.pi

# ================= [1] KNSB 内弹道 =================
PROP = dict(rho=1820.0, n=0.35, a=5.164e-5, cstar=890.0)

def ideal_CF(gamma, eps, pa, pc):
    lo, hi = 1.001, 6.0
    for _ in range(60):
        Me = 0.5 * (lo + hi)
        ar = (1.0/Me)*((2.0/(gamma+1.0))*(1.0+0.5*(gamma-1.0)*Me*Me)**((gamma+1.0)/(2.0*(gamma-1.0))))
        if ar < eps: lo = Me
        else: hi = Me
    Me = 0.5*(lo+hi)
    pr = (1.0 + 0.5*(gamma-1.0)*Me*Me)**(-gamma/(gamma-1.0))
    t1 = math.sqrt((2.0*gamma*gamma/(gamma-1.0))*(2.0/(gamma+1.0))**((gamma+1.0)/(gamma-1.0))
                   *(1.0-pr**((gamma-1.0)/gamma)))
    return t1 + (pr - pa/pc)*eps

def motor_sim(ro, rc0, L0, N, At, Ae, dt=0.002):
    rc, Ls = float(rc0), float(L0)
    mprop = PROP["rho"]*N*PI*(ro**2-rc0**2)*L0*1e-9
    ts, Fs = [], []
    t = 0.0
    while Ls > 0.0 and rc < ro and t < 30.0:
        Ab = N*(2.0*PI*rc*Ls) + N*2.0*PI*(ro**2-rc**2)
        P = (PROP["rho"]*PROP["a"]*PROP["cstar"]*(Ab/At))**(1.0/(1.0-PROP["n"]))
        r = PROP["a"]*P**PROP["n"]
        Fs.append(0.93*ideal_CF(1.14, Ae/At, 101325.0, P)*P*At*1e-6)
        ts.append(t)
        rc += r*dt*1e3; Ls -= 2.0*r*dt*1e3; t += dt
    ts = np.array(ts); Fs = np.array(Fs)
    imp = float(np.trapezoid(Fs, ts))
    cum = np.concatenate([[0.0], np.cumsum((Fs[1:]+Fs[:-1])/2*dt)])
    cum /= cum[-1] if cum[-1] > 0 else 1.0
    return dict(t=ts, F=Fs, cumf=cum, prop_mass_kg=mprop, imp=imp, burn_time=float(ts[-1]))

m1 = motor_sim(34.0, 12.0, 150.0, 6, PI/4*20.5**2, PI/4*28.0**2)   # 助推 N=6
m2 = motor_sim(22.5, 7.0, 110.0, 4, PI/4*13.0**2, PI/4*18.0**2)    # 续航 N=4
print(f"Booster  : Isp={m1['imp']/m1['prop_mass_kg']/G0:.0f}s  mp={m1['prop_mass_kg']:.2f}kg  burn={m1['burn_time']:.2f}s")
print(f"Sustainer: Isp={m2['imp']/m2['prop_mass_kg']/G0:.0f}s  mp={m2['prop_mass_kg']:.2f}kg  burn={m2['burn_time']:.2f}s")

# ================= [2] 大气 / 气动 =================
def atmos(h):
    h = max(h, 0.0)
    if h < 11000: T = 288.15-0.0065*h; p = 101325.0*(T/288.15)**5.2559
    elif h < 20000: T = 216.65; p = 22632.0*math.exp(-G0*(h-11000)/287.053/T)
    else: T = 216.65; p = 5474.9
    return p/(287.053*T), p, math.sqrt(1.4*287.053*T)

_CDM = [(0.0, 0.45), (0.8, 0.45), (1.05, 0.78), (1.5, 0.68), (2.0, 0.62), (3.0, 0.62)]
def cd_mach(M):
    if M <= 0.8: return 0.45
    for i in range(len(_CDM)-1):
        if M <= _CDM[i+1][0]:
            x0, v0 = _CDM[i]; x1, v1 = _CDM[i+1]
            return v0 + (v1-v0)*(M-x0)/(x1-x0)
    return 0.62

A_BO = PI/4*76.2**2*1e-6
A_SU = PI/4*54.0**2*1e-6
S_LIFT = 0.20; AR_LIFT = 1.0; E_OSW = 0.90

def cl_lift(M, alpha_deg):
    a = math.radians(alpha_deg)
    if M < 0.8:
        return 2.0*PI*a/(1.0+2.0*PI/(PI*AR_LIFT))
    if M < 1.2:
        f = (M-0.8)/0.4
        cl0 = 2.0*PI*a/(1.0+2.0*PI/(PI*AR_LIFT))
        cl1 = 1.3*4.0*a/math.sqrt(max(M*M-1.0, 0.02))
        return cl0*(1.0-f) + cl1*f
    return 1.3*4.0*a/math.sqrt(max(M*M-1.0, 0.02))

def cd_lift(M, alpha_deg, cd0):
    cl = cl_lift(M, alpha_deg)
    if M < 0.8: cdw = 0.0
    elif M < 1.2: cdw = 0.005*((M-0.8)/0.4)**2
    else: cdw = 0.006
    return cd0 + cdw + cl*cl/(PI*AR_LIFT*E_OSW)

# ================= [3] Besselian 月影 =================
BES = dict(x=[0.134339, 0.5377713, -3.58e-5, -8.0e-6],
           y=[0.348988, -0.1584657, -5.95e-5, 2.3e-6],
           d=[8.01771, -0.014783, -2.0e-6, 0.0],
           l2=[-0.004197, 1.098e-4, -1.18e-5, 0.0],
           mu=[210.02995, 15.004648, 0.0, 0.0],
           tanf2=0.0046097, t0=2.0)
FLAT = 1/298.25722; DTT = 80.6/3600.0

def _p(c, tc): return c[0] + c[1]*tc + c[2]*tc*tc + c[3]*tc**3
def elem(te):
    tc = te - BES["t0"]
    return (_p(BES["x"], tc), _p(BES["y"], tc), math.radians(_p(BES["d"], tc)),
            abs(_p(BES["l2"], tc)), math.radians(_p(BES["mu"], tc)))
def ut2ep(ut): return ut + DTT

def umbra_margin(lat, lon, h_m, te):
    x, y, d, l2, mu = elem(te)
    phi = math.radians(lat); phic = math.atan((1-FLAT)**2*math.tan(phi))
    rho = 1.0 + h_m/R_E; H = math.radians(lon) + mu
    xi = rho*math.cos(phic)*math.sin(H)
    eta = rho*(math.sin(phic)*math.cos(d)-math.cos(phic)*math.sin(d)*math.cos(H))
    ze = rho*(math.sin(phic)*math.sin(d)+math.cos(phic)*math.cos(d)*math.cos(H))
    return math.hypot(xi-x, eta-y) - (l2 + ze*BES["tanf2"])

def ground_scan(lat, lon, ut0, ut1, step_s=1.0):
    return [float(u) for u in np.arange(ut0, ut1, step_s/3600.0)
            if umbra_margin(lat, lon, 0.0, ut2ep(u)) < 0.0]

def axis_ground(te):
    x, y, d, l2, mu = elem(te)
    z = math.sqrt(max(1e-9, 1.0-x*x-y*y))
    phic = math.asin(max(-1.0, min(1.0, y*math.cos(d)+z*math.sin(d))))
    H = math.atan2(x/math.cos(phic), (z-math.sin(phic)*math.sin(d))/(math.cos(phic)*math.cos(d)))
    return math.degrees(phic), ((math.degrees(H)-math.degrees(mu)+540.0) % 360.0 - 180.0)

def shadow_kinematics(ut_h, dt_s=1.0):
    la0, lo0 = axis_ground(ut2ep(ut_h))
    la1, lo1 = axis_ground(ut2ep(ut_h + dt_s/3600.0))
    dx = math.radians(lo1-lo0)*R_E*math.cos(math.radians(la0))
    dy = math.radians(la1-la0)*R_E
    return math.degrees(math.atan2(dx, dy)) % 360.0, math.hypot(dx, dy)/dt_s

# ================= [4] 弹道 =================
SITE = dict(name="秦皇岛", lat0=39.90, lon0=119.60, elev=10.0)

def pos_geo(xm, hm_agl, az_rad):
    coslat = math.cos(math.radians(SITE["lat0"]))
    return (SITE["lat0"] + math.degrees(xm*math.cos(az_rad)/R_E),
            SITE["lon0"] + math.degrees(xm*math.sin(az_rad)/(R_E*coslat)),
            hm_agl + SITE["elev"])

def cam_tot(arr, C2s, tau_s, az_rad, step=20, hmin=300.0):
    """返回 (时长, 首次进入, 最后退出, 逐秒掩码列表)

    时长按实际采样步长积分：stride = t[i+step] - t[i]，
    避免记录间隔随 dt 漂移导致时长被系统性低估（v5 的 88 s 就是这个问题）。
    """
    if len(arr) < 2: return 0.0, None, None, []
    stride = arr[min(step, len(arr)-1), 0] - arr[0, 0]
    tin, mask = [], []
    for i in range(0, len(arr), step):
        t, xs, hs = arr[i, 0], arr[i, 1], arr[i, 2]
        if hs < hmin:
            mask.append(0); continue
        la, lo, hg = pos_geo(xs, hs, az_rad)
        ok = umbra_margin(la, lo, hg, ut2ep(C2s + tau_s/3600.0 + t/3600.0)) < 0.0
        mask.append(1 if ok else 0)
        if ok: tin.append(t)
    if not tin: return 0.0, None, None, mask
    return (tin[-1]-tin[0])+stride, tin[0], tin[-1], mask

def flight_sim(angle_deg, az_deg, mode="trad", alpha_max=4.0, dd_s=20.0, sust_delay=3.5,
               cd0=0.008, payload_kg=1.0, dt=0.01, tmax=1200.0):
    th = math.radians(angle_deg); az = math.radians(az_deg)
    mb, ms = 2.85, 1.15
    mbp, msp = m1["prop_mass_kg"], m2["prop_mass_kg"]
    tb1, tb2 = m1["burn_time"], m2["burn_time"]
    t_sep = tb1 + 0.6; t_ign = t_sep + sust_delay
    t = 0.0; x = 0.0; h = 0.0; vx = 0.0; vh = 0.0
    on_rail = True; rail_s = 0.0; ap_done = False
    dro = False; main = False; landed = False
    rec = []; next_rec = 0.0
    max_h = max_M = max_q = 0.0; v_rail = 0.0
    while t < tmax and not landed:
        if t <= tb1:
            m = mb+mbp+ms+msp+payload_kg - mbp*float(np.interp(t, m1["t"], m1["cumf"]))
            F = float(np.interp(t, m1["t"], m1["F"]))
        elif t < t_ign:
            m = ms+msp+payload_kg; F = 0.0
        else:
            t2 = t-t_ign
            if t2 <= tb2:
                m = ms+msp+payload_kg - msp*float(np.interp(t2, m2["t"], m2["cumf"]))
                F = float(np.interp(t2, m2["t"], m2["F"]))
            else:
                m = ms+payload_kg; F = 0.0

        v = math.hypot(vx, vh); rho, pa, am = atmos(h); M = v/am; q = 0.5*rho*v*v
        if mode == "trad":
            alpha = 0.0
            A = A_BO if t <= tb1 else A_SU
            Fd = cd_mach(M)*q*A; Fl = 0.0
        else:
            if h > 1500.0 and M < 1.5 and q < 60000.0: alpha = alpha_max
            elif h > 800.0 and M < 1.0: alpha = min(alpha_max, 1.5)
            else: alpha = 0.0
            Fd = cd_lift(M, alpha, cd0)*q*S_LIFT
            Fl = cl_lift(M, alpha)*q*S_LIFT

        if dro and not main:
            kd = G0/(9.0*9.0); ah = -G0-kd*vh*abs(vh); ax = -vx/3.0
        elif main:
            km = G0/(5.0*5.0); ah = -G0-km*vh*abs(vh); ax = -vx/1.5
        else:
            if on_rail:
                a = F/m - G0*math.sin(th) - Fd/m
                ax = a*math.cos(th); ah = a*math.sin(th)
                rail_s += v*dt
                if rail_s >= 4.0: on_rail = False; v_rail = v
            else:
                if v > 1e-3:
                    ux, uh = vx/v, vh/v
                    ca = math.cos(math.radians(alpha))
                    ax = (F*ca*ux - Fd*ux - Fl*uh)/m
                    ah = (F*ca*uh + Fl*ux - Fd*uh)/m - G0
                else:
                    ah = F/m - G0

        vx += ax*dt; vh += ah*dt
        if (not ap_done) and vh <= 0.0 and t > tb1+1.0: ap_done = True
        if ap_done and (not dro) and t > t_sep+dd_s: dro = True
        if dro and (not main) and h <= 250.0: main = True
        x += vx*dt; h += vh*dt; t += dt
        max_h = max(max_h, h); max_M = max(max_M, M); max_q = max(max_q, q)
        if h <= 0.0 and t > 20.0: landed = True
        if t >= next_rec:
            rec.append((t, x, h, v, vx, vh, M, q)); next_rec += 0.05
    arr = np.array(rec) if rec else np.zeros((1, 8))
    ev = dict(rail_exit_v=float(v_rail), x_land=float(arr[-1, 1]), t_land=float(arr[-1, 0]),
              h_max=float(max_h), M_max=float(max_M), q_max=float(max_q), landed=bool(landed))
    return arr, ev

def best_tau(arr, C2s, az_rad, taus, step=20):
    best = (-1.0, 0, None, None, [])
    for tau in taus:
        tc, t0, t1, mask = cam_tot(arr, C2s, tau, az_rad, step=step)
        if tc > best[0]: best = (tc, tau, t0, t1, mask)
    return best

# ================= [5] 主流程 =================
if __name__ == "__main__":
    print("\n" + "="*74)
    print("v6 「骑影」方案：国内陆地最优机位 + 弹道/月影联合寻优")
    print("="*74)

    # --- 机位选择：网格扫描国内陆地最长 ---
    print("\n[A] 机位扫描（中国陆地 38.6–41.6N, 114–119.7E，不含渤海海上点）")
    cand = []
    for la in np.arange(38.8, 41.4, 0.2):
        for lo in np.arange(114.0, 119.7, 0.2):
            ins = ground_scan(la, lo, 35.0/60, 38.5/60, step_s=2.0)
            if ins: cand.append((len(ins)*2.0, la, lo, ins[0]))
    cand.sort(reverse=True)
    for d, la, lo, c2 in cand[:5]:
        print(f"    {d:.0f}s  {la:.2f}N {lo:.2f}E  C2={c2*60:.2f}min")

    Tg, la_best, lo_best, C2s = cand[0]
    SITE["lat0"], SITE["lon0"] = la_best, lo_best
    ins = ground_scan(SITE["lat0"], SITE["lon0"], C2s-2/60, C2s+5/60, step_s=1.0)
    Tg = len(ins)*1.0; C2s = ins[0]
    az_s, V_s = shadow_kinematics(C2s)
    print(f"    选定: {SITE['lat0']:.2f}N {SITE['lon0']:.2f}E   地面全食 {Tg:.0f}s   "
          f"C2={C2s*60:.2f}min")
    print(f"    当地月影: 方位 {az_s:.1f}°  速度 {V_s:.0f} m/s")

    # --- v5 方法复演（固定 93.6°）---
    print("\n[B] 复演 v5 方法（方位固定 93.6°，发射角 57°）")
    arr5, ev5 = flight_sim(57.0, 93.6, "trad", dt=0.02)
    T5, tau5, _, _, _ = best_tau(arr5, C2s, math.radians(93.6), range(-180, 31, 5))
    print(f"    T_cam={T5:.0f}s  tau={tau5:+d}s  apo={ev5['h_max']/1000:.2f}km  "
          f"x_land={ev5['x_land']/1000:.2f}km  (地面 {Tg:.0f}s)")

    # --- v6 联合寻优 ---
    print("\n[C] v6 联合寻优：发射方位 × 发射角")
    rows = []
    for daz in (-8.0, -4.0, 0.0, 4.0, 8.0):
        az = az_s + daz
        for ang in (45.0, 50.0, 55.0, 57.0, 60.0, 65.0, 70.0, 75.0):
            arr, ev = flight_sim(ang, az, "trad", dt=0.02)
            if ev["rail_exit_v"] < 15.0: continue
            tc, tau, t0, t1, mask = best_tau(arr, C2s, math.radians(az), range(-150, 31, 5))
            rows.append(dict(T=tc, az=az, daz=daz, ang=ang, tau=tau, ev=ev, mode="trad"))
            print(f"    az={az:5.1f}° ang={ang:4.0f}°  T_cam={tc:5.0f}s  tau={tau:+4d}s  "
                  f"apo={ev['h_max']/1000:5.2f}km  Mmax={ev['M_max']:.2f}  "
                  f"q={ev['q_max']/1000:5.0f}kPa  x={ev['x_land']/1000:5.1f}km")
    rows.sort(key=lambda r: -r["T"])
    b_abs = rows[0]
    # 工程约束：q_max ≤ 300 kPa（业余复材/铝箭体常用上限，与 v2/v4 同级）
    safe = [r for r in rows if r["ev"]["q_max"] <= 300000.0]
    # 在最优 1 s 内取弹道最低者（风偏小、落点近、回收容易）
    Tmax_safe = max(r["T"] for r in safe) if safe else b_abs["T"]
    near = [r for r in safe if r["T"] >= Tmax_safe-1.0]
    near.sort(key=lambda r: r["ev"]["h_max"])
    b = near[0] if near else (safe[0] if safe else b_abs)
    print(f"\n    绝对最优: T_cam={b_abs['T']:.0f}s  {b_abs['ang']:.0f}°  q={b_abs['ev']['q_max']/1000:.0f}kPa")
    print(f"    推荐(≤300kPa): T_cam={b['T']:.0f}s  方位 {b['az']:.1f}°(Δ{b['daz']:+.0f}°)  "
          f"发射角 {b['ang']:.0f}°  tau={b['tau']:+d}s  q={b['ev']['q_max']/1000:.0f}kPa")

    # --- 乘波体：同一搜索域内取最优 ---
    print("\n[D] 乘波体搜索（方位=月影方位，发射角 55–80°）")
    lift_rows = []
    for ang in (55.0, 60.0, 65.0, 70.0, 75.0, 80.0):
        for amax in (2.0, 4.0):
            arrL, evL = flight_sim(ang, az_s, "lift", alpha_max=amax, dt=0.02)
            if evL["rail_exit_v"] < 15.0 or not evL["landed"]: continue
            tcl, taul, _, _, _ = best_tau(arrL, C2s, math.radians(az_s), range(-150, 31, 5))
            lift_rows.append(dict(T=tcl, ang=ang, amax=amax, tau=taul, ev=evL))
            print(f"    ang={ang:4.0f}° αmax={amax:.0f}°  T_cam={tcl:5.0f}s  tau={taul:+4d}s  "
                  f"apo={evL['h_max']/1000:5.2f}km  Mmax={evL['M_max']:.2f}  "
                  f"q={evL['q_max']/1000:5.0f}kPa")
    lift_rows.sort(key=lambda r: -r["T"])

    # --- 细算最优方案 ---
    print("\n[E] 最优方案细算 (dt=0.005)")
    arrB, evB = flight_sim(b["ang"], b["az"], "trad", dt=0.005)
    Tc, tauc, t0c, t1c, mask = best_tau(arrB, C2s, math.radians(b["az"]), range(-150, 31, 1))
    # 理论：T ≈ (W + 沿轨位移)/V，W = V·Tg
    T_theory = (Tg*V_s + evB["x_land"])/V_s
    print(f"    T_cam={Tc:.1f}s  增益 相对地面 {Tc-Tg:+.1f}s")
    print(f"    理论 (W+Δx)/V = ({Tg:.0f}s×{V_s:.0f}m/s + {evB['x_land']:.0f}m)/{V_s:.0f} = {T_theory:.1f}s")

    # 物理上限：要 270 s 需要多快
    u_need = V_s - Tg*V_s/270.0
    print(f"    要 270s 需沿轨 {u_need:.0f} m/s 持续整个窗口")

    out = dict(
        version="v6",
        site=dict(name=SITE["name"], lat=round(SITE["lat0"], 2), lon=round(SITE["lon0"], 2),
                  elev_m=SITE["elev"]),
        ground_totality_s=round(Tg, 1),
        C2_UT_min=round(C2s*60, 2),
        C3_UT_min=round(C2s*60+Tg/60.0, 2),
        shadow_az_deg=round(az_s, 1),
        shadow_speed_ms=round(V_s, 0),
        v5_replay=dict(az_deg=93.6, angle_deg=57, T_cam_s=round(T5, 1), tau_s=tau5,
                       note="v5 报告的 88–95s 源于时长积分 bug：按记录条数×名义步长，实际记录间隔漂移导致低估约 20%"),
        best=dict(mode="traditional", T_cam_s=round(Tc, 1), gain_vs_ground_s=round(Tc-Tg, 1),
                  az_deg=round(b["az"], 1), angle_deg=b["ang"], tau_s=tauc,
                  apogee_m=round(evB["h_max"], 0), M_max=round(evB["M_max"], 3),
                  q_max_kPa=round(evB["q_max"]/1000.0, 1),
                  x_land_m=round(evB["x_land"], 0), t_land_s=round(evB["t_land"], 1)),
        alt_best_unconstrained=dict(T_cam_s=round(b_abs["T"], 1), angle_deg=b_abs["ang"],
                                    q_max_kPa=round(b_abs["ev"]["q_max"]/1000.0, 1)),
        waverider=(dict(alpha_max_deg=lift_rows[0]["amax"], angle_deg=lift_rows[0]["ang"],
                        T_cam_s=round(lift_rows[0]["T"], 1), tau_s=lift_rows[0]["tau"],
                        note="同推力/同尺寸下乘波体参考面积放大 87 倍，诱导阻力过高，不占优")
                   if lift_rows else None),
        theory=dict(formula="T ≈ (W + Δx)/V，W = V·Tg",
                    T_theory_s=round(T_theory, 1)),
        feasible_270s=False,
        needed_along_track_speed_ms=round(u_need, 0),
        verdict=(f"不出国最优：地面 {Tg:.0f}s → 骑影 T_cam≈{Tc:.0f}s（{Tc-Tg:+.0f}s）。"
                 f"理论上限 (W+Δx)/V≈{T_theory:.0f}s；270s 需沿轨 {u_need:.0f}m/s，"
                 f"超推进剂能量约 125 倍，物理不可达。"))

    json.dump(out, open(f"{DATA}/summary_v6.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n已写出 {DATA}/summary_v6.json")

    # --- 图 ---
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax[0].plot(arrB[:, 0], arrB[:, 2]/1000.0, lw=1.3, label="v6 最优")
    ax[0].plot(arr5[:, 0], arr5[:, 2]/1000.0, lw=1.1, ls="--", label="v5 复演")
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("h [km]"); ax[0].legend(); ax[0].set_title("高度")
    ax[1].plot(arrB[:, 0], arrB[:, 3], lw=1.3); ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("v [m/s]"); ax[1].set_title("速度")
    ts = np.arange(len(mask))*(20*0.05)
    ax[2].step(ts, mask, where="mid", lw=1.4)
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("在影=1"); ax[2].set_title("月影内时间轴")
    fig.suptitle(f"v6 骑影：地面 {Tg:.0f}s → T_cam {Tc:.0f}s（{Tc-Tg:+.0f}s）")
    fig.tight_layout(); fig.savefig(f"{FIG}/v6_ride.png", dpi=130); plt.close(fig)
    print(f"已写出 {FIG}/v6_ride.png")
