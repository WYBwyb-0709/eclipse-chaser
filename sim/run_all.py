#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
协和逐日·业余尺度 —— KNSB 两级探空火箭追逐 2035-09-02 日全食(北京段) 仿真链
[1] KNSB 内弹道(准稳态0D,BATES) [2] 两级飞行(2D垂直平面,沿月影轨迹方位93.6deg)
[3] Besselian 月影几何(NASA GSFC 根数,先用地面全食时长验证) [4] 发射时刻/仰角/开伞延迟寻优
数据出处: NASA GSFC/Espenak 2035 Sep02 Besselian & path table; KNSB 参数取公开业余火箭文献量级
"""
import numpy as np, json, math, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE="/workspace/eclipse-chaser"; DATA=BASE+"/data"; FIG=BASE+"/figures"
os.makedirs(DATA,exist_ok=True); os.makedirs(FIG,exist_ok=True)
R_E=6371000.0; G0=9.80665; PI=math.pi

# ================= [1] KNSB motor =================
PROP=dict(rho=1820.0,n=0.35,a=5.164e-5,cstar=890.0)   # r[m/s]=a*P[Pa]^n
def ideal_CF(gamma,eps,pa,pc):
    lo,hi=1.001,6.0
    for _ in range(60):
        Me=0.5*(lo+hi)
        ar=(1.0/Me)*((2.0/(gamma+1.0))*(1.0+0.5*(gamma-1.0)*Me*Me))**((gamma+1.0)/(2.0*(gamma-1.0)))
        if ar<eps: lo=Me
        else: hi=Me
    Me=0.5*(lo+hi)
    pr=(1.0+0.5*(gamma-1.0)*Me*Me)**(-gamma/(gamma-1.0))
    t1=math.sqrt((2.0*gamma*gamma/(gamma-1.0))*(2.0/(gamma+1.0))**((gamma+1.0)/(gamma-1.0))*(1.0-pr**((gamma-1.0)/gamma)))
    return t1+(pr-pa/pc)*eps

def motor_sim(name,ro,rc0,L0,N,At,Ae,eta=0.93,gamma=1.14,pa=101325.0,dt=0.002,prop_scale=1.0,prop=None,quiet=False):
    # ro,rc0,L0: grain outer radius, core radius, segment length (mm); OD inhibited BATES
    PP=prop or PROP
    rc,rc0_,Ls,Ns=float(rc0),float(rc0),float(L0),N*prop_scale
    mprop=PP["rho"]*Ns*PI*(ro**2-rc0**2)*L0*1e-9
    ts,Fs,Ps,Kns=[],[],[],[]
    t=0.0
    while Ls>0.0 and rc<ro and t<30.0:
        Ab=Ns*(2.0*PI*rc*Ls)+Ns*2.0*PI*(ro**2-rc**2)   # mm^2
        Kn=Ab/At
        P=(PP["rho"]*PP["a"]*PP["cstar"]*Kn)**(1.0/(1.0-PP["n"]))
        r=PP["a"]*P**PP["n"]
        CF=ideal_CF(gamma,Ae/At,pa,P)
        F=eta*CF*P*At*1e-6
        ts.append(t);Fs.append(F);Ps.append(P);Kns.append(Kn)
        rc+=r*dt*1e3; Ls-=2.0*r*dt*1e3; t+=dt
    ts=np.array(ts);Fs=np.array(Fs);Ps=np.array(Ps);Kns=np.array(Kns)
    imp=float(np.trapezoid(Fs,ts))
    cumf=np.concatenate([[0.0],np.cumsum((Fs[1:]+Fs[:-1])/2)*ts[1]])
    cumf=cumf/imp
    summ=dict(name=name,prop_mass_kg=mprop,total_impulse_Ns=imp,Isp_eff_s=imp/mprop/G0,
              burn_time_s=float(ts[-1]),P_max_MPa=float(Ps.max()/1e6),F_max_N=float(Fs.max()),
              F_mean_N=imp/float(ts[-1]),Kn_init=float(Kns[0]),Kn_max=float(Kns.max()),eps=Ae/At)
    if quiet:
        return dict(t=ts,F=Fs,P=Ps,cumf=cumf,summ=summ)
    with open(f"{DATA}/motor_{name}.csv","w",newline="") as f:
        w=csv.writer(f);w.writerow(["t_s","F_N","P_MPa","Kn"])
        for i in range(0,len(ts),5): w.writerow([f"{ts[i]:.4f}",f"{Fs[i]:.1f}",f"{Ps[i]/1e6:.3f}",f"{Kns[i]:.1f}"])
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    ax[0].plot(ts,Fs,lw=1.2);ax[0].set_xlabel("t [s]");ax[0].set_ylabel("thrust [N]");ax[0].set_title(f"KNSB {name}")
    ax[1].plot(ts,Ps/1e6,lw=1.2,color="crimson");ax[1].set_xlabel("t [s]");ax[1].set_ylabel("Pc [MPa]")
    ax[1].axhline(8.0,ls="--",c="k",lw=0.8)
    fig.suptitle(f"{name}: {imp:.0f} Ns | Isp {imp/mprop/G0:.0f} s | burn {ts[-1]:.2f} s | MEOP {Ps.max()/1e6:.2f} MPa")
    fig.tight_layout();fig.savefig(f"{FIG}/motor_{name}.png",dpi=130);plt.close(fig)
    return dict(t=ts,F=Fs,P=Ps,cumf=cumf,summ=summ)

def hoop_sf(P_MPa,OD,wall): return 276.0/(P_MPa*(OD/2.0-wall)/wall)  # 6061-T6 sy=276MPa

# ================= [2] atmosphere/aero/flight =================
def atmos(h):
    h=max(h,0.0)
    if h<11000: T=288.15-0.0065*h; p=101325.0*(T/288.15)**5.2559
    elif h<20000: T=216.65; p=22632.0*math.exp(-G0*(h-11000)/287.053/T)
    else: T=216.65; p=5474.9
    return p/(287.053*T),p,math.sqrt(1.4*287.053*T)
_CDM=[(0.0,0.45),(0.8,0.45),(1.05,0.78),(1.5,0.68),(2.0,0.62),(3.0,0.62)]
def cd_mach(M):
    if M<=0.8: return 0.45 if M<=0.8 else 0.78
    for i in range(len(_CDM)-1):
        if M<=_CDM[i+1][0]:
            x0,v0=_CDM[i];x1,v1=_CDM[i+1];return v0+(v1-v0)*(M-x0)/(x1-x0)
    return 0.62

BO=dict(ro=34.0,rc0=12.0,L0=150.0,N=6,At=PI/4*20.5**2,Ae=PI/4*28.0**2,OD=76.2,wall=9.0,m_dry=5.26)
SU=dict(ro=22.5,rc0=7.0,L0=110.0,N=4,At=PI/4*13.0**2,Ae=PI/4*18.0**2,OD=54.0,wall=6.0,m_dry=1.65)
PAYLOAD=1.0   # kg 相机+飞控+遥测+电源+伞舱
SITE=dict(lat0=40.154,lon0=115.850,elev=600.0)
AZ=math.radians(93.6); RAIL=4.0

def flight_sim(m1,m2,angle_deg,drogue_delay_s,wind_ms=0.0,two_stage=True,
               sep_delay=0.6,sust_delay=2.0,drogue_v=13.0,main_v=5.0,main_h=400.0,
               dt=0.01,single_mode=False,tmax=1500.0,payload_kg=1.0):
    th=math.radians(angle_deg)
    mb,ms=BO["m_dry"],SU["m_dry"]
    mbp=m1["summ"]["prop_mass_kg"]; msp=m2["summ"]["prop_mass_kg"]
    A1=PI/4*BO["OD"]**2*1e-6; A2=PI/4*SU["OD"]**2*1e-6
    if single_mode: m0=mb+mbp+payload_kg
    else: m0=mb+mbp+ms+msp+payload_kg
    tb1=m1["summ"]["burn_time_s"]; tb2=m2["summ"]["burn_time_s"]
    t_sep=tb1+sep_delay; t_ign=t_sep+sust_delay
    t=0.0;x=0.0;h=0.0;vx=0.0;vh=0.0
    on_rail=True;rail_s=0.0;phase=1
    ap_done=False;t_ap=None;dro=False;main=False;landed=False
    ev={};rec=[]
    rec_s=0.05;next_rec=0.0
    while t<tmax and not landed:
        if single_mode:
            if t<=tb1:
                frac=float(np.interp(t,m1["t"],m1["cumf"])); m=m0-mbp*frac; F=float(np.interp(t,m1["t"],m1["F"])); A=A1; phase=1
            else:
                m=mb+payload_kg; F=0.0; A=A1; phase=2
        else:
            if t<=tb1:
                frac=float(np.interp(t,m1["t"],m1["cumf"])); m=m0-mbp*frac; F=float(np.interp(t,m1["t"],m1["F"])); A=A1; phase=1
            elif t<t_sep:
                m=mb+ms+msp+payload_kg; F=0.0; A=A2; phase=2
                if t_sep-t<=dt: ev["sep"]=(t,x,h,vx,vh)
            elif t<t_ign:
                m=ms+msp+payload_kg; F=0.0; A=A2; phase=2
            else:
                t2=t-t_ign
                if t2<=tb2:
                    frac=float(np.interp(t2,m2["t"],m2["cumf"])); m=ms+msp+payload_kg-msp*frac; F=float(np.interp(t2,m2["t"],m2["F"]))
                else:
                    m=ms+payload_kg; F=0.0
                A=A2; phase=3
        v=math.hypot(vx,vh); rho,pa,am=atmos(h); M=v/am; q=0.5*rho*v*v
        ax=ah=0.0
        if dro and not main:      # drogue descent: k s.t. terminal v = drogue_v
            kd=G0/(drogue_v*drogue_v)
            ah=-G0-kd*vh*abs(vh)
            ax=-(vx-wind_ms)/3.0
            phase=4
        elif main:                # main chute: terminal v = main_v
            km=G0/(main_v*main_v)
            ah=-G0-km*vh*abs(vh)
            ax=-(vx-wind_ms)/1.5
            phase=5
        else:
            Fd=cd_mach(M)*q*A
            if on_rail:
                a=F/m-G0*math.sin(th)-Fd/m
                ax=a*math.cos(th); ah=a*math.sin(th)
                rail_s+=v*dt
                if rail_s>=RAIL:
                    on_rail=False; ev["rail_exit_v"]=v; ev["t_rail"]=t
            else:
                if v>1e-3:
                    ax=(-Fd+F)/m*(vx/v); ah=(-Fd+F)/m*(vh/v)-G0
                else:
                    ah=F/m-G0
        vx+=ax*dt; vh+=ah*dt
        if (not ap_done) and vh<=0 and t>tb1+1.0:
            ap_done=True; t_ap=t; ev["apogee_m"]=h; ev["t_apogee"]=t
        if ap_done and (not dro) and (t-t_ap)>=drogue_delay_s and vh<0:
            dro=True; ev["t_drogue"]=t
        if dro and (not main) and h<=main_h:
            main=True; ev["t_main"]=t
        x+=vx*dt; h+=vh*dt; t+=dt
        if h<=0.0 and t>20: landed=True
        if t>=next_rec:
            rec.append((t,x,h,v,vx,vh,M,q,phase)); next_rec+=rec_s
    arr=np.array(rec)
    ev.update(x_land=float(arr[-1,1]),t_land=float(arr[-1,0]),v_land=abs(float(arr[-1,5])),
              h_max=float(arr[:,2].max()),M_max=float(arr[:,6].max()),q_max=float(arr[:,7].max()),
              v_max=float(arr[:,3].max()),landed=bool(landed))
    return arr,ev

# ================= [3] Besselian umbra =================
BES=dict(x=[0.134339,0.5377713,-3.58e-5,-8.0e-6],y=[0.348988,-0.1584657,-5.95e-5,2.3e-6],
         d=[8.01771,-0.014783,-2.0e-6,0.0],l2=[-0.004197,1.098e-4,-1.18e-5,0.0],
         mu=[210.02995,15.004648,0.0,0.0],tanf2=0.0046097,t0=2.0)
FLAT=1/298.25722; DTT=80.6/3600.0
def _p(c,tc): return c[0]+c[1]*tc+c[2]*tc*tc+c[3]*tc**3
def elem(te):
    tc=te-BES["t0"]
    return (_p(BES["x"],tc),_p(BES["y"],tc),math.radians(_p(BES["d"],tc)),
            abs(_p(BES["l2"],tc)),math.radians(_p(BES["mu"],tc)))
def umbra_margin(lat,lon,h_m,te):
    x,y,d,l2,mu=elem(te)
    phi=math.radians(lat); phic=math.atan((1-FLAT)**2*math.tan(phi))
    rho=1.0+h_m/R_E; H=mu+math.radians(lon)
    xi=rho*math.cos(phic)*math.sin(H)
    eta=rho*(math.sin(phic)*math.cos(d)-math.cos(phic)*math.sin(d)*math.cos(H))
    ze=rho*(math.sin(phic)*math.sin(d)+math.cos(phic)*math.cos(d)*math.cos(H))
    return math.hypot(xi-x,eta-y)-(l2+ze*BES["tanf2"])
def ut2ep(ut_h): return ut_h+DTT
def ground_scan(lat,lon,ut_h0,ut_h1,step_s=2.0,h_m=0.0):
    ins=[]
    for ut in np.arange(ut_h0,ut_h1,step_s/3600.0):
        if umbra_margin(lat,lon,h_m,ut2ep(ut))<0: ins.append(float(ut))
    return ins
def axis_ground(te):
    x,y,d,l2,mu=elem(te)
    z=math.sqrt(max(1e-9,1-x*x-y*y))
    phic=math.asin(max(-1,min(1,y*math.cos(d)+z*math.sin(d))))
    H=math.atan2(x/math.cos(phic),(z-math.sin(phic)*math.sin(d))/(math.cos(phic)*math.cos(d)))
    return math.degrees(phic),((math.degrees(H)-math.degrees(mu)+540)%360-180)

# ================= MAIN =================
if __name__=="__main__":
    print("="*76)
    print("[1] KNSB motors (quasi-steady 0D, BATES grains, OD inhibited)")
    m1=motor_sim("booster",BO["ro"],BO["rc0"],BO["L0"],BO["N"],BO["At"],BO["Ae"])
    m2=motor_sim("sustainer",SU["ro"],SU["rc0"],SU["L0"],SU["N"],SU["At"],SU["Ae"])
    for m_,cfg in ((m1,BO),(m2,SU)):
        s=m_["summ"]; sf=hoop_sf(s["P_max_MPa"],cfg["OD"],cfg["wall"]); s["casing_SF_at_MEOP"]=round(sf,2)
        print(f"  {s['name']:10s}: prop={s['prop_mass_kg']:.2f}kg imp={s['total_impulse_Ns']:.0f}Ns Isp={s['Isp_eff_s']:.0f}s "
              f"burn={s['burn_time_s']:.2f}s MEOP={s['P_max_MPa']:.2f}MPa Kn0={s['Kn_init']:.0f} Knmax={s['Kn_max']:.0f} SF={sf:.1f}")

    print("[2] Besselian validation vs NASA path table")
    ut34=34.0/60.0
    ins=ground_scan(40.105,116.8583,ut34-0.025,ut34+0.025)
    dur=len(ins)*2.0; c2cl=ins[0] if ins else 0; c3cl=ins[-1] if ins else 0
    print(f"  ground totality @central line (40.105N,116.858E): {dur:.1f}s  [NASA: 110.8s] C2={c2cl*3600:.0f} C3={c3cl*3600:.0f} (UT s-of-day)")
    p1=axis_ground(ut2ep(ut34)); p2=axis_ground(ut2ep(ut34+30/3600.0))
    dN=(p2[0]-p1[0])*111320.0; dE=(p2[1]-p1[1])*111320.0*math.cos(math.radians(40.1))
    vs=math.hypot(dN,dE)/30.0
    print(f"  umbra ground speed near Beijing: {vs:.0f} m/s [path-table cross-check ~1430 m/s]")
    ok=(abs(dur-110.8)<=8.0) and (1200<=vs<=1700)
    print(f"  VALIDATION {'PASS' if ok else 'FAIL'}")

    lat0,lon0=SITE["lat0"],SITE["lon0"]
    ins_s=ground_scan(lat0,lon0,33.0/60.0-0.03,33.0/60.0+0.03)
    C2s=ins_s[0]; Tg=len(ins_s)*2.0
    print(f"  launch site ({lat0}N,{lon0}E): ground totality {Tg:.0f}s, C2={C2s*3600:.0f} UT s-of-day")

    coslat=math.cos(math.radians(lat0))
    def pos_geo(xm,hm_agl):
        return (lat0+math.degrees(xm*math.cos(AZ)/R_E),
                lon0+math.degrees(xm*math.sin(AZ)/(R_E*coslat)),hm_agl+SITE["elev"])
    def cam_tot(arr,tau_h,step=10,hmin=300.0):
        dt=arr[1,0]-arr[0,0]; tin=[]
        for i in range(0,len(arr),step):
            t,xs,hs=arr[i,0],arr[i,1],arr[i,2]
            if hs<hmin: continue
            la,lo,hg=pos_geo(xs,hs)
            if umbra_margin(la,lo,hg,ut2ep(C2s+tau_h+t/3600.0))<0: tin.append(t)
        if not tin: return 0.0,None,None
        return len(tin)*step*dt,tin[0],tin[-1]

    print("[3] sweep: angle x drogueDelay x sustDelay x launch-offset")
    results=[]
    for angle in [25,30,35,40,45,50,55,60,70,80]:
        for dd in [5.0,20.0,60.0,120.0]:
            for sd in [2.0,3.5,5.0]:
                arr,ev=flight_sim(m1,m2,angle,dd,sust_delay=sd,dt=0.01)
                if ev.get("rail_exit_v",0)<20 or ev["M_max"]>2.85: continue
                best=(-1e9,None,None)
                for tau in range(-220,41,10):
                    tc,t0_,t1_=cam_tot(arr,tau/3600.0,step=20)
                    if tc>best[0]: best=(tc,tau,t0_)
                if ev["h_max"]>=3500 and ev["x_land"]<=60000:
                    results.append([best[0],angle,dd,sd,best[1],ev])
    results.sort(key=lambda r:-r[0])
    print(f"  {len(results)} feasible configs (apo>=3.5km, xland<=60km, M<=2.85)")
    if not results:
        print("  !! NO feasible config -- relax to max apogee")
        for angle in [65,75,85]:
            arr,ev=flight_sim(m1,m2,angle,20.0,sust_delay=4.0,dt=0.03)
            results.append([0.0,angle,20.0,4.0,-20,ev])
        results.sort(key=lambda r:-r[5]["h_max"])
    for r in results[:6]:
        ev=r[5]
        print(f"   T_cam~{r[0]:6.1f}s ang={r[1]} dd={r[2]:.0f}s sd={r[3]:.0f}s tau={r[4]:+4d}s "
              f"apo={ev['h_max']/1000:4.1f}km xL={ev['x_land']/1000:5.1f}km M={ev['M_max']:.2f}")
    print("[4] refine top candidates (dt=0.005, fine tau)")
    refined=[]
    for tc_c,angle,dd,sd,tau_c,ev_c in results[:10]:
        arr,ev=flight_sim(m1,m2,angle,dd,sust_delay=sd,dt=0.005)
        best=(-1e9,None,None)
        for tau in np.arange(tau_c-15,tau_c+10.1,1):
            tc,t0_,t1_=cam_tot(arr,tau/3600.0,step=20)
            if tc>best[0]: best=(tc,tau,t0_)
        refined.append([best[0],angle,dd,sd,best[1],best[2],arr,ev])
    refined.sort(key=lambda r:-r[0])
    T_cam,ang,dd,sd0,tau_best,t0_,arrB,evB=refined[0]
    r_cam=cam_tot(arrB,tau_best/3600.0,step=20)
    if r_cam[0]>0.0 and r_cam[1] is not None:
        _,t0_,t1_=r_cam
        msk=(arrB[:,0]>=t0_)&(arrB[:,0]<=t1_)
        hmin,hmaxv=float(arrB[msk,2].min()),float(arrB[msk,2].max())
        x_during=(float(arrB[msk,1].max())-float(arrB[msk,1].min()))
    else:
        T_cam,t0_,t1_=0.0,0.0,0.0
        hmin=hmaxv=float(arrB[:,2].max()); x_during=0.0
    print(f"[4] BEST: ang={ang} dd={dd}s sd={sd0}s tau={tau_best:+.0f}s")
    print(f"  camera totality {T_cam:.1f}s vs ground {Tg:.0f}s  -> gain {T_cam-Tg:+.1f}s ({100*(T_cam/Tg-1):+.0f}%)")
    print(f"  cam C2={((C2s+(tau_best+t0_)/3600)*3600):.0f} -> C3={((C2s+(tau_best+t1_)/3600)*3600):.0f} UT s-of-day (BJT +8h: {(C2s*3600+(tau_best+t0_)+8*3600)%(24*3600):.0f})")
    print(f"  altitude during totality: {hmin:.0f}..{hmaxv:.0f} m | downtrack sweep {x_during/1000:.1f} km")

    np.save(f"{DATA}/flight_opt.npy",arrB)
    with open(f"{DATA}/flight_opt.csv","w",newline="") as f:
        w=csv.writer(f);w.writerow(["t_s","x_downtrack_m","h_m","v_ms","vx_ms","vh_ms","Mach","q_Pa","phase"])
        for row in arrB[::2]: w.writerow([f"{v:.2f}" for v in row])
    mr=[];mg=[];tt=[]
    with open(f"{DATA}/umbra_scan.csv","w",newline="") as f:
        w=csv.writer(f);w.writerow(["t_s","rocket_margin_km","ground_margin_km"])
        for i in range(0,len(arrB),20):
            t,xs,hs=arrB[i,0],arrB[i,1],arrB[i,2]
            la,lo,hg=pos_geo(xs,hs)
            a=umbra_margin(la,lo,hg,ut2ep(C2s+tau_best/3600.0+t/3600.0))*R_E/1000
            b=umbra_margin(lat0,lon0,SITE["elev"],ut2ep(C2s+tau_best/3600.0+t/3600.0))*R_E/1000
            tt.append(t);mr.append(a);mg.append(b);w.writerow([f"{t:.2f}",f"{a:.3f}",f"{b:.3f}"])

    m1d=motor_sim("booster_d15",BO["ro"],BO["rc0"],BO["L0"],BO["N"],BO["At"],BO["Ae"],prop_scale=0.85)
    arrS,evS=flight_sim(m1d,m2,ang,dd,sust_delay=sd0,dt=0.005)
    tcS,_,_=cam_tot(arrS,tau_best/3600.0,step=20)
    arrW,evW=flight_sim(m1,m2,ang,dd,sust_delay=sd0,wind_ms=10.0,dt=0.005)
    arr1,ev1=flight_sim(m1,m2,ang,0.0,single_mode=True,dt=0.005)
    tc1,_,_=cam_tot(arr1,tau_best/3600.0,step=20)
    print(f"  sensitivity: -15% booster prop -> apo={evS['h_max']/1000:.1f}km T_cam={tcS:.1f}s | +10m/s wind -> xland={evW['x_land']/1000:.1f}km")
    print(f"  single-stage fallback: apo={ev1['h_max']/1000:.1f}km T_cam={tc1:.1f}s")

    summ=dict(validation_besselian_pass=ok,ground_vs_NASA_s=[dur,110.8],umbra_ground_speed_ms=vs,
        site_ground_totality_s=Tg,site_C2_UT_s=C2s*3600,best_angle_deg=ang,best_drogue_delay_s=dd,
        best_sust_delay_s=sd0,tau_launch_s=float(tau_best),cam_totality_s=T_cam,cam_gain_s=T_cam-Tg,cam_C2_UT_s=(C2s+(tau_best+t0_)/3600)*3600,
        cam_C3_UT_s=(C2s+(tau_best+t1_)/3600)*3600,alt_in_totality_m=[hmin,hmaxv],
        downtrack_in_totality_m=x_during,
        flight=evB,motor_booster=m1["summ"],motor_sustainer=m2["summ"],
        sens_prop15=dict(apogee_m=evS["h_max"],T_cam_s=tcS),
        sens_wind10_xland_m=evW["x_land"],single_stage=dict(apogee_m=ev1["h_max"],T_cam_s=tc1))
    summ["flight"].pop("sep",None)
    with open(f"{DATA}/summary.json","w") as f: json.dump(summ,f,indent=2,default=float)

    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(arrB[:,0],arrB[:,2]/1000,lw=1.3);ax[0].axvspan(t0_,t1_,color="gold",alpha=0.35,label="camera totality")
    ax[0].set_xlabel("t [s]");ax[0].set_ylabel("altitude [km]");ax[0].legend(fontsize=8)
    ax[0].set_title(f"ascent, apogee {evB['h_max']/1000:.1f} km")
    ax[1].plot(arrB[:,0],arrB[:,1]/1000,lw=1.3);ax[1].axvspan(t0_,t1_,color="gold",alpha=0.35)
    ax[1].set_xlabel("t [s]");ax[1].set_ylabel("downtrack [km]");ax[1].set_title(f"landing x={evB['x_land']/1000:.1f} km")
    fig.tight_layout();fig.savefig(f"{FIG}/trajectory.png",dpi=130);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(arrB[:,0],arrB[:,6],lw=1.2);ax[0].axhline(1,ls=":",c="k");ax[0].set_xlabel("t [s]");ax[0].set_ylabel("Mach")
    ax[0].set_title(f"M_max={evB['M_max']:.2f}")
    ax[1].plot(arrB[:,0],arrB[:,7]/1000,lw=1.2,c="crimson");ax[1].set_xlabel("t [s]");ax[1].set_ylabel("q [kPa]")
    ax[1].set_title(f"q_max={evB['q_max']/1000:.1f} kPa")
    fig.tight_layout();fig.savefig(f"{FIG}/mach_q.png",dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(tt,mr,lw=1.4,label="rocket/camera (3D margin to umbra)")
    ax.plot(tt,mg,lw=1.0,ls="--",label="ground observer @ launch site")
    ax.axhline(0,c="k",lw=0.8);ax.axvspan(t0_,t1_,color="gold",alpha=0.3)
    ax.set_xlabel("t after liftoff [s]");ax.set_ylabel("dist. to umbra edge [km] (<0 inside)")
    ax.set_title(f"umbra chase: camera {T_cam:.1f}s vs ground {Tg:.0f}s (+{T_cam-Tg:.0f}s, {100*(T_cam/Tg-1):+.0f}%)")
    ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(f"{FIG}/umbra_chase.png",dpi=130);plt.close(fig)
    print("[5] artifacts: data/*.csv, data/summary.json, figures/*.png")
    print("DONE")
