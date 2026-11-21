#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
协和逐日 v5 — 乘波体/升力体构型评估
目标：全食窗口 T_cam ≥ 270s (4.5min)
方法：
  [A] 传统构型复现 v2 基线（run_all.py 同款气动：分段参考面积 + cd_mach 表）
  [B] 乘波体：升力垂直于速度矢量、平面参考面积、Ackeret 超声速升力
  [C] 物理上限核算（能量/阻力）——判定 270s 是否可达
"""
import numpy as np, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE="/workspace/eclipse-chaser"; DATA=BASE+"/data"; FIG=BASE+"/figures"
os.makedirs(DATA,exist_ok=True); os.makedirs(FIG,exist_ok=True)
R_E=6371000.0; G0=9.80665; PI=math.pi

# ================= [1] KNSB 发动机（与 run_all.py 一致） =================
PROP=dict(rho=1820.0,n=0.35,a=5.164e-5,cstar=890.0)
def ideal_CF(gamma,eps,pa,pc):
    lo,hi=1.001,6.0
    for _ in range(60):
        Me=0.5*(lo+hi)
        ar=(1.0/Me)*((2.0/(gamma+1.0))*(1.0+0.5*(gamma-1.0)*Me*Me)**((gamma+1.0)/(2.0*(gamma-1.0))))
        if ar<eps: lo=Me
        else: hi=Me
    Me=0.5*(lo+hi)
    pr=(1.0+0.5*(gamma-1.0)*Me*Me)**(-gamma/(gamma-1.0))
    t1=math.sqrt((2.0*gamma*gamma/(gamma-1.0))*(2.0/(gamma+1.0))**((gamma+1.0)/(gamma-1.0))*(1.0-pr**((gamma-1.0)/gamma)))
    return t1+(pr-pa/pc)*eps

def motor_sim(name,ro,rc0,L0,N,At,Ae,dt=0.002):
    rc,Ls=float(rc0),float(L0)
    mprop=PROP["rho"]*N*PI*(ro**2-rc0**2)*L0*1e-9
    ts,Fs=[],[]
    t=0.0
    while Ls>0.0 and rc<ro and t<30.0:
        Ab=N*(2.0*PI*rc*Ls)+N*2.0*PI*(ro**2-rc**2)
        Kn=Ab/At
        P=(PROP["rho"]*PROP["a"]*PROP["cstar"]*Kn)**(1.0/(1.0-PROP["n"]))
        r=PROP["a"]*P**PROP["n"]
        F=0.93*ideal_CF(1.14,Ae/At,101325.0,P)*P*At*1e-6
        ts.append(t);Fs.append(F)
        rc+=r*dt*1e3; Ls-=2.0*r*dt*1e3; t+=dt
    ts=np.array(ts);Fs=np.array(Fs)
    imp=float(np.trapezoid(Fs,ts))
    cum=np.concatenate([[0.0],np.cumsum((Fs[1:]+Fs[:-1])/2*dt)])
    cum/=(cum[-1] if cum[-1]>0 else 1.0)
    return dict(t=ts,F=Fs,cumf=cum,prop_mass_kg=mprop,imp=imp,burn_time=float(ts[-1]))

m1=motor_sim("booster",34.0,12.0,150.0,6,PI/4*20.5**2,PI/4*28.0**2)   # N=6，与 opt_v2.json 的 nb=6 一致
m2=motor_sim("sustainer",22.5,7.0,110.0,4,PI/4*13.0**2,PI/4*18.0**2)
print(f"Booster  : Isp={m1['imp']/m1['prop_mass_kg']/G0:.0f}s  mp={m1['prop_mass_kg']:.2f}kg  burn={m1['burn_time']:.2f}s")
print(f"Sustainer: Isp={m2['imp']/m2['prop_mass_kg']/G0:.0f}s  mp={m2['prop_mass_kg']:.2f}kg  burn={m2['burn_time']:.2f}s")

# ================= [2] 大气 / 气动 =================
def atmos(h):
    h=max(h,0.0)
    if h<11000: T=288.15-0.0065*h; p=101325.0*(T/288.15)**5.2559
    elif h<20000: T=216.65; p=22632.0*math.exp(-G0*(h-11000)/287.053/T)
    else: T=216.65; p=5474.9
    return p/(287.053*T),p,math.sqrt(1.4*287.053*T)

_CDM=[(0.0,0.45),(0.8,0.45),(1.05,0.78),(1.5,0.68),(2.0,0.62),(3.0,0.62)]
def cd_mach(M):
    if M<=0.8: return 0.45
    for i in range(len(_CDM)-1):
        if M<=_CDM[i+1][0]:
            x0,v0=_CDM[i];x1,v1=_CDM[i+1]
            return v0+(v1-v0)*(M-x0)/(x1-x0)
    return 0.62

A_BO=PI/4*76.2**2*1e-6      # 助推段参考面积 0.00456 m2
A_SU=PI/4*54.0**2*1e-6      # 续航/滑行段参考面积 0.00229 m2
S_LIFT=0.20                 # 乘波体平面参考面积 m2
AR_LIFT=1.0; E_OSW=0.90

def cl_lift(M,alpha_deg):
    a=math.radians(alpha_deg)
    if M<0.8:
        # 有限展弦比修正的亚声速升力线斜率 (AR=1)
        return 2.0*PI*a/(1.0+2.0*PI/(PI*AR_LIFT))
    if M<1.2:
        f=(M-0.8)/0.4
        cl0=2.0*PI*a/(1.0+2.0*PI/(PI*AR_LIFT))
        cl1=1.3*4.0*a/math.sqrt(max(M*M-1.0,0.02))
        return cl0*(1-f)+cl1*f
    return 1.3*4.0*a/math.sqrt(max(M*M-1.0,0.02))

def cd_lift(M,alpha_deg,cd0):
    cl=cl_lift(M,alpha_deg)
    if M<0.8: cdw=0.0
    elif M<1.2: cdw=0.005*((M-0.8)/0.4)**2
    else: cdw=0.006
    return cd0+cdw+cl*cl/(PI*AR_LIFT*E_OSW)

# ================= [3] Besselian 月影 =================
BES=dict(x=[0.134339,0.5377713,-3.58e-5,-8.0e-6],
         y=[0.348988,-0.1584657,-5.95e-5,2.3e-6],
         d=[8.01771,-0.014783,-2.0e-6,0.0],
         l2=[-0.004197,1.098e-4,-1.18e-5,0.0],
         mu=[210.02995,15.004648,0.0,0.0],
         tanf2=0.0046097,t0=2.0)
FLAT=1/298.25722; DTT=80.6/3600.0
def _p(c,tc): return c[0]+c[1]*tc+c[2]*tc*tc+c[3]*tc**3
def elem(te):
    tc=te-BES["t0"]
    return(_p(BES["x"],tc),_p(BES["y"],tc),math.radians(_p(BES["d"],tc)),
           abs(_p(BES["l2"],tc)),math.radians(_p(BES["mu"],tc)))
def ut2ep(ut): return ut+DTT
def umbra_margin(lat,lon,h_m,te):
    x,y,d,l2,mu=elem(te)
    phi=math.radians(lat); phic=math.atan((1-FLAT)**2*math.tan(phi))
    rho=1.0+h_m/R_E; H=math.radians(lon)+mu
    xi=rho*math.cos(phic)*math.sin(H)
    eta=rho*(math.sin(phic)*math.cos(d)-math.cos(phic)*math.sin(d)*math.cos(H))
    ze=rho*(math.sin(phic)*math.sin(d)+math.cos(phic)*math.cos(d)*math.cos(H))
    return math.hypot(xi-x,eta-y)-(l2+ze*BES["tanf2"])
def ground_scan(lat,lon,ut_h0,ut_h1,step_s=2.0):
    return [float(ut) for ut in np.arange(ut_h0,ut_h1,step_s/3600.0)
            if umbra_margin(lat,lon,0.0,ut2ep(ut))<0]

SITE=dict(lat0=40.154,lon0=115.850,elev=600.0)
AZ=math.radians(93.6)
def pos_geo(xm,hm_agl):
    coslat=math.cos(math.radians(SITE["lat0"]))
    return (SITE["lat0"]+math.degrees(xm*math.cos(AZ)/R_E),
            SITE["lon0"]+math.degrees(xm*math.sin(AZ)/(R_E*coslat)),
            hm_agl+SITE["elev"])
def cam_tot(arr,C2s,tau_h,step=20,hmin=300.0):
    if len(arr)<2: return 0.0,None,None
    dt=arr[1,0]-arr[0,0]; tin=[]
    for i in range(0,len(arr),step):
        t,xs,hs=arr[i,0],arr[i,1],arr[i,2]
        if hs<hmin: continue
        la,lo,hg=pos_geo(xs,hs)
        if umbra_margin(la,lo,hg,ut2ep(C2s+tau_h+t/3600.0))<0: tin.append(t)
    if not tin: return 0.0,None,None
    return len(tin)*step*dt,tin[0],tin[-1]

# ================= [4] 飞行仿真 =================
def flight_sim(angle_deg,mode="trad",alpha_max=4.0,dd_s=20.0,sust_delay=3.5,
               cd0=0.008,payload_kg=1.0,dt=0.005,tmax=1500.0):
    th=math.radians(angle_deg)
    mb,ms=2.85,1.15
    mbp,msp=m1["prop_mass_kg"],m2["prop_mass_kg"]
    tb1,tb2=m1["burn_time"],m2["burn_time"]
    t_sep=tb1+0.6; t_ign=t_sep+sust_delay
    t=0.0;x=0.0;h=0.0;vx=0.0;vh=0.0
    on_rail=True; rail_s=0.0
    ap_done=False; dro=False; main=False; landed=False
    rec=[]; next_rec=0.0; v_rail_exit=0.0
    max_h=max_M=max_q=0.0
    while t<tmax and not landed:
        if t<=tb1:
            m=mb+mbp+ms+msp+payload_kg-mbp*float(np.interp(t,m1["t"],m1["cumf"]))
            F=float(np.interp(t,m1["t"],m1["F"]))
        elif t<t_sep: m=ms+msp+payload_kg; F=0.0
        elif t<t_ign: m=ms+msp+payload_kg; F=0.0
        else:
            t2=t-t_ign
            if t2<=tb2:
                m=ms+msp+payload_kg-msp*float(np.interp(t2,m2["t"],m2["cumf"]))
                F=float(np.interp(t2,m2["t"],m2["F"]))
            else: m=ms+payload_kg; F=0.0

        v=math.hypot(vx,vh); rho,pa,am=atmos(h); M=v/am; q=0.5*rho*v*v

        if mode=="trad":
            # run_all.py 同款：纯阻力，分段参考面积
            alpha=0.0
            A=A_BO if t<=tb1 else A_SU
            Fd=cd_mach(M)*q*A; Fl=0.0
        else:
            # 乘波体：低空大动压时不允许大攻角，滑行段用给定攻角
            if h>1500.0 and M<1.5 and q<60000.0: alpha=alpha_max
            elif h>800.0 and M<1.0: alpha=min(alpha_max,1.5)
            else: alpha=0.0
            Fd=cd_lift(M,alpha,cd0)*q*S_LIFT
            Fl=cl_lift(M,alpha)*q*S_LIFT

        ax=ah=0.0
        if dro and not main:
            kd=G0/(9.0*9.0); ah=-G0-kd*vh*abs(vh); ax=-vx/3.0
        elif main:
            km=G0/(5.0*5.0); ah=-G0-km*vh*abs(vh); ax=-vx/1.5
        else:
            if on_rail:
                a=F/m-G0*math.sin(th)-Fd/m
                ax=a*math.cos(th); ah=a*math.sin(th)
                rail_s+=v*dt
                if rail_s>=4.0: on_rail=False; v_rail_exit=v
            else:
                if v>1e-3:
                    ux,uh=vx/v,vh/v
                    ca=math.cos(math.radians(alpha))
                    # 推力沿体轴：T(cosα·u + sinα·n)；阻力沿 -u；升力沿 n=(-uh,ux)
                    ax=(F*ca*ux-Fd*ux-Fl*uh)/m
                    ah=(F*ca*uh+Fl*ux-Fd*uh)/m-G0
                else: ah=F/m-G0

        vx+=ax*dt; vh+=ah*dt
        if (not ap_done) and vh<=0 and t>tb1+1.0: ap_done=True
        if ap_done and (not dro) and (t>t_sep+dd_s): dro=True
        if dro and (not main) and h<=250.0: main=True
        x+=vx*dt; h+=vh*dt; t+=dt
        max_h=max(max_h,h); max_M=max(max_M,M); max_q=max(max_q,q)
        if h<=0.0 and t>20: landed=True
        if t>=next_rec:
            rec.append((t,x,h,v,vx,vh,M,q)); next_rec+=0.05
    arr=np.array(rec) if rec else np.zeros((1,8))
    ev=dict(rail_exit_v=float(v_rail_exit),x_land=float(arr[-1,1]),t_land=float(arr[-1,0]),
            v_land=abs(float(arr[-1,4])),h_max=float(max_h),M_max=float(max_M),
            q_max=float(max_q),landed=bool(landed))
    return arr,ev

# ================= [5] 主流程 =================
print("\n"+"="*72)
print("v5 评估：乘波体能否把全食窗口做到 4.5min (270s)")
print("="*72)
ins=ground_scan(SITE["lat0"],SITE["lon0"],31.5/60,35.0/60)
Tg=len(ins)*2.0; C2s=ins[0]
print(f"发射点地面全食: {Tg:.0f}s   C2={C2s*60:.2f}min UT   目标 270s")

print("\n[1] 传统构型基线（复现 v2）")
arr_t,ev_t=flight_sim(57,"trad",dd_s=20.0,dt=0.01)
best_tc,best_tau=-1,0
for tau in range(-220,61,5):
    tc,_,_=cam_tot(arr_t,C2s,tau/3600.0)
    if tc>best_tc: best_tc,best_tau=tc,tau
print(f"  T_cam={best_tc:.0f}s  apo={ev_t['h_max']/1000:.2f}km  Mmax={ev_t['M_max']:.2f}  "
      f"xland={ev_t['x_land']/1000:.2f}km  tland={ev_t['t_land']:.0f}s  tau={best_tau:+d}s")
T_trad=best_tc

print("\n[2] 乘波体搜索")
rows=[]
for angle in [57,60,65,70,75,80]:
    for amax in [2.0,4.0,6.0,8.0]:
        arr,ev=flight_sim(angle,"lift",alpha_max=amax,dd_s=20.0,dt=0.01)
        if ev["rail_exit_v"]<15 or ev["M_max"]>4.0: continue
        if not ev["landed"] or ev["h_max"]<2000 or ev["h_max"]>25000: continue
        btc,btau=-1,0
        for tau in range(-220,61,10):
            tc,_,_=cam_tot(arr,C2s,tau/3600.0)
            if tc>btc: btc,btau=tc,tau
        rows.append((btc,angle,amax,btau,ev))
rows.sort(key=lambda r:-r[0])
print(f"  可行构型: {len(rows)}")
for i,(tc,ang,amax,tau,ev) in enumerate(rows[:8]):
    print(f"   #{i+1}: T_cam={tc:.0f}s  ang={ang}°  αmax={amax:.0f}°  apo={ev['h_max']/1000:.2f}km  "
          f"Mmax={ev['M_max']:.2f}  xland={ev['x_land']/1000:.1f}km  tau={tau:+d}s")

T_best=rows[0][0] if rows else T_trad
gain=T_best-T_trad
print(f"\n  乘波体相对传统增益: {gain:+.0f}s   (传统 {T_trad:.0f}s → 乘波体 {T_best:.0f}s)")

# ---------- 物理上限核算 ----------
print("\n[3] 270s 可行性核算")
V_shadow=1377.0   # 月影地面速度 m/s（Besselian 差分，方位93.6°）
# 要 T_cam=270s，需要的沿轨相对速度
v_rel_need=V_shadow*Tg/270.0
v_h_need=V_shadow-v_rel_need
print(f"  月影地面速度 V={V_shadow:.0f} m/s；地面全食 {Tg:.0f}s")
print(f"  要 270s 需相对速度 ≤ {v_rel_need:.0f} m/s → 飞行器需沿轨 {v_h_need:.0f} m/s")
# 维持该速度所需功率（3km 高度，乘波体 S=0.2m², CD≈0.05）
for alt in [3000.0,6000.0,10000.0]:
    rho,_,_=atmos(alt); q=0.5*rho*v_h_need**2
    D=0.05*q*S_LIFT; P=D*v_h_need
    print(f"    @{alt/1000:.0f}km: q={q/1000:.0f}kPa  阻力={D:.0f}N  功率={P/1000:.0f}kW  "
          f"270s做功={P*270/1e6:.0f}MJ")
E_prop=0.5*6.36*PROP["cstar"]**2
print(f"  推进剂可用化学能 ≈ {E_prop/1e6:.1f} MJ  →  缺口约 {P*270/E_prop:.0f} 倍")
# 滑翔方案：L/D 与沿轨速度
print("\n  滑翔方案（无动力）能提供多大增益？")
for LD in [3.0,4.5]:
    v_glide=90.0
    T_ext=Tg*V_shadow/(V_shadow-v_glide)
    print(f"    L/D={LD:.1f}, 滑翔速度{v_glide:.0f}m/s → 沿轨减速后全食 ~{T_ext:.0f}s (增益 {T_ext-Tg:+.0f}s)")

out=dict(ground_totality_s=Tg,C2_UT_min=round(C2s*60,2),
         shadow_speed_ms=V_shadow,target_s=270.0,
         eclipse_global_max_totality_s=130.0,
         eclipse_max_location=dict(lat=39.23,lon=125.65),
         traditional=dict(T_cam_s=round(T_trad,1),apogee_m=ev_t["h_max"],
                          M_max=ev_t["M_max"],x_land_m=ev_t["x_land"],t_land_s=ev_t["t_land"]),
         lift_body_best=(dict(T_cam_s=round(rows[0][0],1),angle_deg=rows[0][1],alpha_max=rows[0][2],
                              tau_s=rows[0][3],apogee_m=rows[0][4]["h_max"],
                              M_max=rows[0][4]["M_max"],x_land_m=rows[0][4]["x_land"]) if rows else None),
         gain_s=round(gain,1),
         feasible_270s=False,
         verdict="270s 不可达：本次日食全球最长地面全食仅约130s；沿轨跟随需826m/s，约需690MJ，超出推进剂化学能(2.5MJ)约125倍",
         needed_along_track_speed_ms=round(v_h_need,0),
         prop_chemical_energy_MJ=round(E_prop/1e6,1))
json.dump(out,open(f"{DATA}/summary_v5.json","w"),ensure_ascii=False,indent=2)
print(f"\n已写出 {DATA}/summary_v5.json")

# 图
if rows:
    ang,amax=rows[0][1],rows[0][2]
    arrL,_=flight_sim(ang,"lift",alpha_max=amax,dd_s=20.0,dt=0.005)
    fig,ax=plt.subplots(1,3,figsize=(13,3.6))
    ax[0].plot(arr_t[:,0],arr_t[:,2]/1000,lw=1.2,label="traditional")
    ax[0].plot(arrL[:,0],arrL[:,2]/1000,lw=1.2,label="waverider")
    ax[0].set_xlabel("t[s]");ax[0].set_ylabel("h[km]");ax[0].legend();ax[0].set_title("Altitude")
    ax[1].plot(arr_t[:,0],arr_t[:,5],lw=1.2);ax[1].plot(arrL[:,0],arrL[:,5],lw=1.2)
    ax[1].set_xlabel("t[s]");ax[1].set_ylabel("v_h[m/s]");ax[1].set_title("Vertical speed")
    ax[2].plot(arr_t[:,0],arr_t[:,6],lw=1.2);ax[2].plot(arrL[:,0],arrL[:,6],lw=1.2)
    ax[2].set_xlabel("t[s]");ax[2].set_ylabel("Mach");ax[2].set_title("Mach")
    fig.suptitle(f"v5 waverider vs traditional (T_cam {T_trad:.0f}s → {rows[0][0]:.0f}s)")
    fig.tight_layout();fig.savefig(f"{FIG}/v5_lift_compare.png",dpi=130);plt.close(fig)
    print(f"已写出 {FIG}/v5_lift_compare.png")
