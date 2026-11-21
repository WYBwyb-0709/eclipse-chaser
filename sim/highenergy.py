#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高能配方对照：在同一药柱几何/壳体约束下，把 KNSB 换成高能复合推进剂，
自动匹配喉径把 MEOP 压回设计线，再做同一套弹道+月影寻优，给出"增益上限"。
"""
import sys, math, json
sys.path.insert(0,"/workspace/eclipse-chaser/sim")
import run_all as R

BASE="/workspace/eclipse-chaser"; DATA=BASE+"/data"
# 高能复合推进剂（APCP 类，含铝/两相流损失的等效 c*）——仅作对照，不作建造推荐
APCP=dict(rho=1770.0,n=0.35,a=3.60e-5,cstar=1400.0)
ADV =dict(rho=1780.0,n=0.35,a=3.20e-5,cstar=1520.0)   # 更激进的高能配方（理论上限对照）

def fit(cfg,name,prop,target=7.5):
    At,Ae=cfg["At"],cfg["Ae"]
    for _ in range(10):
        m=R.motor_sim(name,cfg["ro"],cfg["rc0"],cfg["L0"],cfg["N"],At,Ae,prop=prop,quiet=True)
        pm=m["summ"]["P_max_MPa"]
        if abs(pm-target)/target<0.004: break
        k=(pm/target)**(1.0-prop["n"]); At*=k; Ae*=k
    m["summ"]["casing_SF_at_MEOP"]=round(R.hoop_sf(pm,cfg["OD"],cfg["wall"]),2)
    m["summ"]["At_mm2"]=At; m["summ"]["Ae_mm2"]=Ae
    m["summ"]["At_mm"]=math.sqrt(4*At/math.pi); m["summ"]["Ae_mm"]=math.sqrt(4*Ae/math.pi)
    return m

lat0,lon0=R.SITE["lat0"],R.SITE["lon0"]; coslat=math.cos(math.radians(lat0))
ins=R.ground_scan(lat0,lon0,33.0/60.0-0.03,33.0/60.0+0.03)
C2s=ins[0]; Tg=len(ins)*2.0
def pos_geo(xm,hm):
    return (lat0+math.degrees(xm*math.cos(R.AZ)/R.R_E),
            lon0+math.degrees(xm*math.sin(R.AZ)/(R.R_E*coslat)), hm+R.SITE["elev"])
def cam_tot(arr,tau_h,step=20,hmin=300.0):
    dt=arr[1,0]-arr[0,0]; tin=[]
    for i in range(0,len(arr),step):
        t,xs,hs=arr[i,0],arr[i,1],arr[i,2]
        if hs<hmin: continue
        la,lo,hg=pos_geo(xs,hs)
        if R.umbra_margin(la,lo,hg,R.ut2ep(C2s+tau_h+t/3600.0))<0: tin.append(t)
    return (len(tin)*step*dt,tin[0],tin[-1]) if tin else (0.0,None,None)

def optimize(m1,m2,tag):
    best=None
    for angle in [25,30,35,40,45,50,55,60,70]:
        for dd in [5.0,20.0,60.0]:
            for sd in [2.0,3.5,5.0]:
                arr,ev=R.flight_sim(m1,m2,angle,dd,sust_delay=sd,dt=0.01)
                if ev.get("rail_exit_v",0)<20 or ev["M_max"]>2.85: continue
                if ev["h_max"]<3500 or ev["x_land"]>70000: continue
                bt=(-1e9,None,None)
                for tau in range(-220,41,10):
                    tc,t0_,t1_=cam_tot(arr,tau/3600.0)
                    if tc>bt[0]: bt=(tc,tau,t0_)
                if best is None or bt[0]>best[0][0]:
                    best=(bt,angle,dd,sd,ev)
    if best is None:
        arr,ev=R.flight_sim(m1,m2,45,20.0,sust_delay=3.5,dt=0.01)
        bt=(-1e9,None,None)
        for tau in range(-220,41,10):
            tc,t0_,t1_=cam_tot(arr,tau/3600.0)
            if tc>bt[0]: bt=(tc,tau,t0_)
        best=(bt,45,20.0,3.5,ev)
    (tc,tau,_),angle,dd,sd,ev=best
    arr,ev=R.flight_sim(m1,m2,angle,dd,sust_delay=sd,dt=0.005)
    for tau in range(tau-12,tau+13,1):
        c=cam_tot(arr,tau/3600.0)
        if c[0]>tc: tc,tau=c[0],tau
    c=cam_tot(arr,tau/3600.0); _,t0_,t1_=c
    msk=(arr[:,0]>=t0_)&(arr[:,0]<=t1_) if t0_ is not None else np.zeros(len(arr),bool)
    alt_lo=float(arr[msk,2].min()) if msk.any() else 0.0
    alt_hi=float(arr[msk,2].max()) if msk.any() else 0.0
    print(f"  [{tag}] ang={angle} dd={dd:.0f} sd={sd} tau={tau:+d}  T_cam={tc:.1f}s  "
          f"apo={ev['h_max']/1000:.1f}km xL={ev['x_land']/1000:.1f}km M={ev['M_max']:.2f} "
          f"alt={alt_lo/1000:.2f}-{alt_hi/1000:.2f}km")
    return dict(tag=tag,T_cam_s=tc,angle=angle,dd=dd,sd=sd,tau=tau,
                apogee_m=ev["h_max"],x_land_m=ev["x_land"],M_max=ev["M_max"],
                alt_min_m=alt_lo,alt_max_m=alt_hi)

print("[HE] fitting high-energy motors to MEOP 7.5 MPa (same grain geometry)")
res={}
for tag,prop in (("APCP",APCP),("ADV",ADV)):
    print(f"  prop {tag}: rho={prop['rho']} c*={prop['cstar']} a={prop['a']:.2e} n={prop['n']}")
    m1=fit(R.BO,f"booster_{tag}",prop); m2=fit(R.SU,f"sustainer_{tag}",prop)
    for nm,m in (("booster",m1),("sustainer",m2)):
        s=m["summ"]
        print(f"    {nm:9s} prop={s['prop_mass_kg']:.2f}kg imp={s['total_impulse_Ns']:.0f}Ns "
              f"Isp={s['Isp_eff_s']:.0f}s burn={s['burn_time_s']:.2f}s MEOP={s['P_max_MPa']:.2f}MPa "
              f"At=Ø{s['At_mm']:.1f} SF={s['casing_SF_at_MEOP']}")
    res[tag]=dict(booster=m1["summ"],sustainer=m2["summ"])

print("[HE] optimizing flight/eclipse for each propellant")
base=json.load(open(f"{DATA}/summary.json"))
res["KNSB"]=dict(T_cam_s=base["cam_totality_s"],angle=base["best_angle_deg"],
                 x_land_m=base["flight"]["x_land"],apogee_m=base["flight"]["h_max"])
for tag,prop in (("APCP",APCP),("ADV",ADV)):
    m1=fit(R.BO,f"booster_{tag}",prop); m2=fit(R.SU,f"sustainer_{tag}",prop)
    res[tag].update(optimize(m1,m2,tag))

ground=Tg; gmax=110.8
print("\n=== GAIN CEILING (ground=%.0fs, best ground site anywhere=%.1fs) ==="%(ground,gmax))
for k in ("KNSB","APCP","ADV"):
    d=res[k]; print(f"  {k:5s}: T_cam={d['T_cam_s']:6.1f}s  gain_vs_site={d['T_cam_s']-ground:+5.1f}s "
                    f"({100*(d['T_cam_s']/ground-1):+4.1f}%)  range={d['x_land_m']/1000:5.1f}km apo={d['apogee_m']/1000:4.1f}km")
json.dump(dict(ground_site_s=ground,ground_max_site_s=gmax,props={"KNSB":R.PROP,"APCP":APCP,"ADV":ADV},results=res),
          open(f"{DATA}/highenergy_summary.json","w"),indent=2,default=float)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ks=["KNSB","APCP","ADV"]; vals=[res[k]["T_cam_s"] for k in ks]
fig,ax=plt.subplots(figsize=(6.4,3.6))
ax.bar(ks,vals,color=["#2b6cb0","#dd6b20","#38a169"])
ax.axhline(ground,ls="--",c="k",lw=1,label=f"ground @site {ground:.0f}s")
ax.axhline(gmax,ls=":",c="gray",lw=1,label=f"best ground {gmax:.1f}s")
for i,v in enumerate(vals): ax.text(i,v+1,f"{v:.1f}s",ha="center",fontsize=9)
ax.set_ylabel("camera totality [s]"); ax.set_title("eclipse-chase gain vs propellant energy")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(BASE+"/figures/prop_compare.png",dpi=130); plt.close(fig)
print("[HE] wrote data/highenergy_summary.json, figures/prop_compare.png")
