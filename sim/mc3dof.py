#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-DOF（三分量平动）风场弹道 + 蒙特卡洛落点散布。
状态: 东向/北向/高度 与 三分量速度；风为随高度幂律剖面，含风向随机；
随机源: 风速/风向、两级推进剂冲量、发射仰角/方位、分离/点火延时、伞降终端速度、大气密度。
"""
import sys, math, json, csv
import numpy as np
sys.path.insert(0,"/workspace/eclipse-chaser/sim")
import run_all as R
BASE="/workspace/eclipse-chaser"; DATA=BASE+"/data"; FIG=BASE+"/figures"
G0=R.G0; PI=math.pi

m1=R.motor_sim("booster",R.BO["ro"],R.BO["rc0"],R.BO["L0"],R.BO["N"],R.BO["At"],R.BO["Ae"],quiet=True)
m2=R.motor_sim("sustainer",R.SU["ro"],R.SU["rc0"],R.SU["L0"],R.SU["N"],R.SU["At"],R.SU["Ae"],quiet=True)
A1=PI/4*R.BO["OD"]**2*1e-6; A2=PI/4*R.SU["OD"]**2*1e-6
mb,ms=R.BO["m_dry"],R.SU["m_dry"]
mbp=m1["summ"]["prop_mass_kg"]; msp=m2["summ"]["prop_mass_kg"]
tb1=m1["summ"]["burn_time_s"]; tb2=m2["summ"]["burn_time_s"]

def make_wind(W0,Ddeg):
    we0=-W0*math.sin(math.radians(Ddeg)); wn0=-W0*math.cos(math.radians(Ddeg))
    def f(h):
        k=(max(h,10.0)/10.0)**0.14
        return we0*k, wn0*k
    return f

def flight3d(m1,m2,elev,az,wind_fn,drogue_v,main_v,sd,dd,sep_delay=0.6,
             dt=0.02,tmax=600.0,scale_b=1.0,scale_s=1.0,rho_fac=1.0):
    ce=math.sin(az)*math.cos(elev); cn=math.cos(az)*math.cos(elev); ch=math.sin(elev)
    m0=mb+mbp*scale_b+ms+msp*scale_s
    t_sep=tb1+sep_delay; t_ign=t_sep+sd
    xe=xn=0.0; h=0.0; ve=vn=vh=0.0; t=0.0
    on_rail=True; rail_s=0.0; ap=False; t_ap=None; dro=False; main=False; landed=False
    apogee=0.0; vmax=0.0; Mmax=0.0
    while t<tmax and not landed:
        # mass & thrust
        if t<=tb1:
            fr=float(np.interp(t,m1["t"],m1["cumf"])); m=m0-mbp*scale_b*fr
            F=float(np.interp(t,m1["t"],m1["F"]))*scale_b; A=A1
        elif t<t_sep:
            m=ms+msp*scale_s; F=0.0; A=A2
        elif t<t_ign:
            m=ms+msp*scale_s; F=0.0; A=A2
        else:
            t2=t-t_ign
            if t2<=tb2:
                fr=float(np.interp(t2,m2["t"],m2["cumf"])); m=ms+msp*scale_s-msp*scale_s*fr
                F=float(np.interp(t2,m2["t"],m2["F"]))*scale_s
            else:
                m=ms; F=0.0
            A=A2
        we,wn=wind_fn(h)
        vre,vrn,vrh=ve-we,vn-wn,vh
        vrel=math.sqrt(vre*vre+vrn*vrn+vrh*vrh); vtot=math.sqrt(ve*ve+vn*vn+vh*vh)
        rho,pa,am=R.atmos(h); rho*=rho_fac
        M=vrel/am if am>0 else 0.0
        ae=an=ah=0.0
        if dro and not main:
            kd=G0/(drogue_v*drogue_v)
            ah=-G0-kd*vh*abs(vh); ae=(we-ve)/3.0; an=(wn-vn)/3.0
        elif main:
            km=G0/(main_v*main_v)
            ah=-G0-km*vh*abs(vh); ae=(we-ve)/1.5; an=(wn-vn)/1.5
        else:
            Fd=R.cd_mach(M)*0.5*rho*vrel*vrel*A
            if on_rail:
                a=F/m-G0*ch-Fd/m; ae=a*ce; an=a*cn; ah=a*ch
                rail_s+=vtot*dt
                if rail_s>=R.RAIL: on_rail=False
            elif vtot>1e-6 and vrel>1e-6:
                ae=(F/m)*(ve/vtot)-(Fd/m)*(vre/vrel)
                an=(F/m)*(vn/vtot)-(Fd/m)*(vrn/vrel)
                ah=(F/m)*(vh/vtot)-(Fd/m)*(vrh/vrel)-G0
            else:
                ah=F/m-G0
        ve+=ae*dt; vn+=an*dt; vh+=ah*dt
        vnow=math.sqrt(ve*ve+vn*vn+vh*vh)
        if vnow>vmax: vmax=vnow
        if M>Mmax: Mmax=M
        if (not ap) and vh<=0 and t>tb1+1.0: ap=True; t_ap=t; apogee=h
        if ap and (not dro) and (t-t_ap)>=dd and vh<0: dro=True
        if dro and (not main) and h<=400.0: main=True
        xe+=ve*dt; xn+=vn*dt; h+=vh*dt; t+=dt
        if h<=0.0 and t>20: landed=True
    rng_=math.hypot(xe,xn)
    bearing=(math.degrees(math.atan2(xe,xn))+360)%360
    return dict(xe=xe,xn=xn,range_m=rng_,bearing_deg=bearing,apogee_m=apogee,
                t_land=t,v_land=abs(vh),t_ap=t_ap,vmax=vmax,Mmax=Mmax)

# ---------- 标称（无风）----------
nom=flight3d(m1,m2,math.radians(45),math.radians(93.6),make_wind(0,0),13.0,5.0,2.0,20.0)
print(f"[MC] nominal no-wind: range={nom['range_m']/1000:.2f} km  bearing={nom['bearing_deg']:.1f}  apo={nom['apogee_m']/1000:.2f} km  "
      f"t_ap={nom['t_ap']:.1f}s vmax={nom['vmax']:.0f} m/s Mmax={nom['Mmax']:.2f}")

# ---------- 蒙特卡洛 ----------
rng=np.random.default_rng(20350902); N=1000
Ws=np.clip(rng.weibull(2.0,N)*5.5,0,16)          # 风速均值 ~4.9 m/s
Wd=rng.uniform(0,360,N)                           # 风向均匀
pb=np.clip(rng.normal(1.0,0.02,N),0.9,1.1)
ps=np.clip(rng.normal(1.0,0.02,N),0.9,1.1)
elev=math.radians(45)+rng.normal(0,math.radians(0.4),N)
az=math.radians(93.6)+rng.normal(0,math.radians(0.3),N)
sds=np.clip(rng.normal(2.0,0.08,N),1.6,2.4)
dds=np.clip(rng.normal(20.0,1.0,N),17,23)
dv=np.clip(rng.normal(13.0,0.8,N),10,16)
mv=np.clip(rng.normal(5.0,0.35,N),3.5,6.5)
rf=np.clip(rng.normal(1.0,0.02,N),0.94,1.06)

rows=[]
for i in range(N):
    r=flight3d(m1,m2,elev[i],az[i],make_wind(Ws[i],Wd[i]),dv[i],mv[i],sds[i],dds[i],
               dt=0.02,scale_b=pb[i],scale_s=ps[i],rho_fac=rf[i])
    rows.append((i,Ws[i],Wd[i],pb[i],ps[i],math.degrees(elev[i]),math.degrees(az[i]),
                 sds[i],dv[i],mv[i],r["xe"],r["xn"],r["range_m"],r["bearing_deg"],r["apogee_m"],r["t_land"],r["v_land"]))

E=np.array([r[10] for r in rows]); Nn=np.array([r[11] for r in rows])
Rg=np.array([r[12] for r in rows]); Bg=np.array([r[13] for r in rows])
Apo=np.array([r[14] for r in rows]); TL=np.array([r[15] for r in rows]); VL=np.array([r[16] for r in rows])
cep50=float(np.percentile(Rg,50)); r95=float(np.percentile(Rg,95))
dmean=np.hypot(E-E.mean(),Nn-Nn.mean())
cep50_mean=float(np.percentile(dmean,50)); p95_mean=float(np.percentile(dmean,95))
cov=np.cov(E,Nn); ev,evec=np.linalg.eigh(cov)
a95=2.4477*math.sqrt(max(ev[-1],0)); b95=2.4477*math.sqrt(max(ev[0],0))
ang95=math.degrees(math.atan2(evec[1,-1],evec[0,-1]))
print(f"[MC] N={N}  range mean={Rg.mean()/1000:.2f} km  sigma={Rg.std()/1000:.2f} km  "
      f"CEP50(mean)={cep50_mean/1000:.2f} km  P95(mean)={p95_mean/1000:.2f} km  max(from launch)={Rg.max()/1000:.2f} km")
print(f"[MC] 95% ellipse semi-axes {a95/1000:.2f} x {b95/1000:.2f} km @ {ang95:.0f} deg  "
      f"landing v mean={VL.mean():.2f} max={VL.max():.2f} m/s")

with open(f"{DATA}/mc_landing.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["sample","wind_ms","wind_dir_deg","prop_scale_booster","prop_scale_sustainer",
        "elev_deg","az_deg","sust_delay_s","drogue_v_ms","main_v_ms","east_m","north_m",
        "range_m","bearing_deg","apogee_m","t_land_s","v_land_ms"])
    for r in rows: w.writerow([f"{v:.3f}" if isinstance(v,float) else v for v in r])

summary=dict(N=N,nominal=nom,wind_speed_mean_ms=float(Ws.mean()),wind_dir_random=True,
    range_mean_m=float(Rg.mean()),range_sigma_m=float(Rg.std()),cep50_m=cep50_mean,p95_m=p95_mean,max_range_m=float(Rg.max()),
    ellipse95_semi_axis_m=[a95,b95],ellipse95_angle_deg=ang95,landing_v_mean_ms=float(VL.mean()),landing_v_max_ms=float(VL.max()),
    apogee_mean_m=float(Apo.mean()),apogee_sigma_m=float(Apo.std()))
json.dump(summary,open(f"{DATA}/mc_summary.json","w"),indent=2,default=float)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6.6,6.2))
ax.scatter(E/1000,Nn/1000,s=6,alpha=0.35,c=Rg/1000,cmap="viridis")
th=np.linspace(0,2*PI,200)
for Rk,st in ((5,"--"),(10,"--"),(15,":")):
    ax.plot(Rk*np.cos(th),Rk*np.sin(th),st,c="gray",lw=0.8)
    ax.text(0,Rk+0.3,f"{Rk} km",ha="center",fontsize=7,color="gray")
c=np.cos(math.radians(ang95)); s=math.sin(math.radians(ang95))
cx,cy=E.mean()/1000,Nn.mean()/1000
ex=cx+(a95*c*np.cos(th)-b95*s*np.sin(th))/1000
ny=cy+(a95*s*np.cos(th)+b95*c*np.sin(th))/1000
ax.plot(ex,ny,"r-",lw=1.4,label="95% ellipse")
ax.plot(cx,cy,"r+",ms=10,label="mean landing")
ax.plot(0,0,"k^",ms=9,label="launch")
ax.plot(nom["xe"]/1000,nom["xn"]/1000,"wx",ms=8,label="nominal (no wind)")
ax.set_aspect("equal"); ax.set_xlabel("East [km]"); ax.set_ylabel("North [km]")
ax.set_title(f"Monte-Carlo landing dispersion (N={N})\nCEP50={cep50_mean/1000:.2f} km, P95(mean)={p95_mean/1000:.2f} km")
ax.legend(fontsize=8,loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIG}/mc_landing.png",dpi=130); plt.close(fig)
print("[MC] wrote data/mc_landing.csv, data/mc_summary.json, figures/mc_landing.png")
