#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 优化：载荷 1.0 kg、降本 BOM、延长飞行时长，同时保持月影追逐增益。
输出: data/opt_v2.json, data/opt_v2_table.csv, data/descent_trade.csv,
      figures/opt_trade.png, figures/cost_breakdown.png, figures/mass_budget.png
"""
import sys, math, json, csv
sys.path.insert(0,"/workspace/eclipse-chaser/sim")
import numpy as np
import run_all as R
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE="/workspace/eclipse-chaser"; DATA=BASE+"/data"; FIG=BASE+"/figures"
lat0,lon0=R.SITE["lat0"],R.SITE["lon0"]; coslat=math.cos(math.radians(lat0))
ins=R.ground_scan(lat0,lon0,33.0/60.0-0.03,33.0/60.0+0.03)
C2s=ins[0]; Tg=len(ins)*2.0

# ---------- 降本 BOM（相对基线 4365 → 目标 <3100）----------
BOM_V2=[
 ("硝酸钾 KNO3 工业级≥99.5% 5kg",90,"化工原料"),("山梨醇 食品级 3kg",60,"化工原料"),
 ("助推机匣 6061-T6 Ø76.2×3.0×900",280,"商购铝管"),("续航机匣 6061-T6 Ø54×2.5×560",160,"商购铝管"),
 ("石墨喷管毛坯 2件",120,"石墨棒"),("喷管车削(喉Ø20.5/Ø13.0)",120,"外协车床"),
 ("铝堵头/隔板 3件",60,"商购+车削"),("鼻锥 PETG 打印",70,"3D打印"),
 ("级间尾锥 6061 76.2→54",90,"商购+车削"),("尾翼 桦木胶合板 4片",60,"板材切割"),
 ("尾翼对齐工装 PETG",30,"3D打印"),("相机楔块 PETG",20,"3D打印"),
 ("载荷/电子舱滑板 PETG",20,"3D打印"),("喷管保持环 6061",30,"车削"),
 ("飞行计算机(开源IMU/气压/GPS)",320,"商购电子"),("GPS/LoRa 遥测信标",150,"商购电子"),
 ("相机(二手运动相机+转接)",380,"二手市场"),("减速伞 24in 尼龙(9 m/s)",110,"商购伞降件"),
 ("主伞 36in 尼龙(5 m/s)",110,"商购伞降件"),("CO2 开伞器 2套",120,"商购"),
 ("减震绳/旋转环/五金",80,"商购"),("导轨按钮+1010导轨1.5m",100,"商购"),
 ("发射架 自制",80,"自制"),("点火具 e-match×6",60,"商购"),
 ("环氧/砂纸/紧固件",150,"商购"),("灭火器/警戒用品",120,"商购")]
BOM_TOTAL=sum(p for _,p,_ in BOM_V2)
print(f"[V2] 降本 BOM 合计 = ¥{BOM_TOTAL}  (基线 ¥4365, 降幅 {100*(1-BOM_TOTAL/4365):.1f}%)")

def cost(nb,ns):
    d_casing=280*(nb-5)*150/900 + 160*(ns-4)*110/560
    mbp=0.868*nb; msp=0.287*ns
    d_prop=max(0.0,(mbp+msp)-6.0)*25.0
    return BOM_TOTAL+d_casing+d_prop

def build(nb,ns):
    cB=dict(R.BO); cS=dict(R.SU); cB["N"]=nb; cS["N"]=ns
    m1=R.motor_sim("booster",cB["ro"],cB["rc0"],cB["L0"],nb,cB["At"],cB["Ae"],quiet=True)
    m2=R.motor_sim("sustainer",cS["ro"],cS["rc0"],cS["L0"],ns,cS["At"],cS["Ae"],quiet=True)
    return m1,m2

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

# ---------- 主扫描：级数 × 仰角 × 续航点火延时 ----------
print("[V2] scanning design space ...")
rows=[]; best=None
for nb in (4,5,6):
    for ns in (3,4):
        m1,m2=build(nb,ns)
        for ang in (42,45,48,51,54,57):
            for sd in (2.0,3.5,5.0):
                arr,ev=R.flight_sim(m1,m2,ang,5.0,sust_delay=sd,drogue_v=9.0,main_v=5.0,
                                    main_h=250.0,dt=0.02,payload_kg=R.PAYLOAD)
                if ev.get("rail_exit_v",0)<20 or ev["M_max"]>2.85: continue
                if ev["x_land"]>15000 or ev["h_max"]<3000: continue
                bt=(-1e9,None)
                for tau in range(-200,21,20):
                    tc,_,_=cam_tot(arr,tau/3600.0,step=40)
                    if tc>bt[0]: bt=(tc,tau)
                c=cost(nb,ns)
                rec=dict(nb=nb,ns=ns,angle=ang,sd=sd,T_cam=bt[0],tau=bt[1],
                         t_land=ev["t_land"],apogee=ev["h_max"],x_land=ev["x_land"],
                         M_max=ev["M_max"],v_land=ev["v_land"],cost=c)
                rows.append(rec)
                score=bt[0]+0.035*ev["t_land"]-0.0002*c
                if best is None or score>best[0]: best=(score,rec)
if best is None: raise SystemExit("no feasible config")
_,B=best
print(f"[V2] best coarse: nb={B['nb']} ns={B['ns']} ang={B['angle']} sd={B['sd']} "
      f"T_cam={B['T_cam']:.1f}s t_land={B['t_land']:.0f}s cost=¥{B['cost']:.0f}")

# ---------- 细化：精细积分 + tau ±20 步1 ----------
m1,m2=build(B["nb"],B["ns"])
arr,ev=R.flight_sim(m1,m2,B["angle"],5.0,sust_delay=B["sd"],drogue_v=9.0,main_v=5.0,
                    main_h=250.0,dt=0.005,payload_kg=R.PAYLOAD)
tc,tau=B["T_cam"],B["tau"]
for k in range(tau-20,tau+21):
    c=cam_tot(arr,k/3600.0,step=20)
    if c[0]>tc: tc,tau=c[0],k
c=cam_tot(arr,tau/3600.0,step=20); _,t0_,t1_=c
msk=(arr[:,0]>=t0_)&(arr[:,0]<=t1_) if t0_ is not None else np.zeros(len(arr),bool)
alt_lo=float(arr[msk,2].min()) if msk.any() else 0.0
alt_hi=float(arr[msk,2].max()) if msk.any() else 0.0
print(f"[V2] refined: T_cam={tc:.1f}s tau={tau:+d}s alt={alt_lo/1000:.2f}-{alt_hi/1000:.2f}km "
      f"t_land={ev['t_land']:.0f}s xL={ev['x_land']/1000:.2f}km")

# ---------- 下降段权衡：减速伞终端速度 → 飞行时长 ----------
print("[V2] descent trade ...")
des=[]
for dv in (7.0,8.0,9.0,10.0,11.0,13.0):
    _,e2=R.flight_sim(m1,m2,B["angle"],5.0,sust_delay=B["sd"],drogue_v=dv,main_v=5.0,
                      main_h=250.0,dt=0.02,payload_kg=R.PAYLOAD)
    des.append((dv,e2["t_land"],e2["x_land"],e2["v_land"]))
    print(f"   drogue_v={dv:4.1f} m/s → t_land={e2['t_land']:5.0f}s")
for mh in (150.0,250.0,400.0):
    _,e2=R.flight_sim(m1,m2,B["angle"],5.0,sust_delay=B["sd"],drogue_v=9.0,main_v=5.0,
                      main_h=mh,dt=0.02,payload_kg=R.PAYLOAD)
    print(f"   main_h={mh:4.0f} m → t_land={e2['t_land']:5.0f}s")

with open(f"{DATA}/opt_v2_table.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
with open(f"{DATA}/descent_trade.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["drogue_v_ms","t_land_s","x_land_m","v_land_ms"]); w.writerows(des)

out=dict(payload_kg=R.PAYLOAD,chosen=dict(nb=B["nb"],ns=B["ns"],angle_deg=B["angle"],
    sust_delay_s=B["sd"],tau_s=tau,T_cam_s=tc,ground_totality_s=Tg,alt_min_m=alt_lo,alt_max_m=alt_hi,
    t_land_s=ev["t_land"],apogee_m=ev["h_max"],x_land_m=ev["x_land"],M_max=ev["M_max"],
    q_max_kPa=ev["q_max"]/1000,v_max_ms=ev["v_max"],v_land_ms=ev["v_land"],
    drogue_v=9.0,main_v=5.0,main_h=250.0,cost_cny=B["cost"]),
    bom_total_cny=BOM_TOTAL,baseline_cost_cny=4365,descent_trade=des,
    mass=dict(booster_prop_kg=m1["summ"]["prop_mass_kg"],sustainer_prop_kg=m2["summ"]["prop_mass_kg"],
              booster_struct_kg=R.BO["m_dry"],sustainer_struct_kg=R.SU["m_dry"],payload_kg=R.PAYLOAD,
              liftoff_kg=R.BO["m_dry"]+R.SU["m_dry"]+R.PAYLOAD+m1["summ"]["prop_mass_kg"]+m2["summ"]["prop_mass_kg"]))
json.dump(out,open(f"{DATA}/opt_v2.json","w"),indent=2,default=float)

# ---------- 图 ----------
ns_l=sorted(set((r["nb"],r["ns"]) for r in rows))
fig,ax=plt.subplots(1,2,figsize=(10.5,3.8))
for nb in (4,5,6):
    y=[r["T_cam"] for r in rows if r["nb"]==nb]
    x=[r["t_land"] for r in rows if r["nb"]==nb]
    ax[0].scatter(x,y,s=8,alpha=0.5,label=f"boost {nb} seg")
ax[0].scatter([ev["t_land"]],[tc],marker="*",s=220,c="r",label="chosen")
ax[0].axhline(Tg,ls="--",c="k",lw=1,label=f"ground {Tg:.0f}s")
ax[0].set_xlabel("flight duration t_land [s]"); ax[0].set_ylabel("camera totality [s]")
ax[0].set_title("duration vs eclipse window"); ax[0].legend(fontsize=7)
dv=[d[0] for d in des]; tl=[d[1] for d in des]
ax[1].plot(dv,tl,"o-"); ax[1].set_xlabel("drogue terminal speed [m/s]")
ax[1].set_ylabel("flight duration [s]"); ax[1].set_title("descent-time trade (smaller = longer)")
ax[1].axvline(9.0,ls=":",c="r"); fig.tight_layout()
fig.savefig(f"{FIG}/opt_trade.png",dpi=130); plt.close(fig)

# 成本构成
labels=[n for n,_,_ in BOM_V2]; vals=[p for _,p,_ in BOM_V2]
order=np.argsort(vals)[::-1][:10]
fig,ax=plt.subplots(figsize=(7.2,3.8))
ax.barh([labels[i][:18] for i in order][::-1],[vals[i] for i in order][::-1],color="#2b6cb0")
ax.set_xlabel("CNY"); ax.set_title(f"cost breakdown — total ¥{BOM_TOTAL} (baseline ¥4365)")
fig.tight_layout(); fig.savefig(f"{FIG}/cost_breakdown.png",dpi=130); plt.close(fig)

# 质量预算
mbp=m1["summ"]["prop_mass_kg"]; msp=m2["summ"]["prop_mass_kg"]
parts=[("助推推进剂",mbp),("续航推进剂",msp),("助推结构",R.BO["m_dry"]),
       ("续航结构",R.SU["m_dry"]),("载荷",R.PAYLOAD)]
fig,ax=plt.subplots(figsize=(7.2,2.6))
left=0
for n,v in parts:
    ax.barh([0],[v],left=left,label=f"{n} {v:.2f}"); left+=v
ax.set_yticks([]); ax.set_xlabel("mass [kg]"); ax.set_title(f"liftoff mass budget = {left:.2f} kg (payload 1.00 kg)")
ax.legend(fontsize=7,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.35))
fig.tight_layout(); fig.savefig(f"{FIG}/mass_budget.png",dpi=130); plt.close(fig)
print(f"[V2] wrote opt_v2.json/opt_v2_table.csv/opt_v2_table + 3 figures")
