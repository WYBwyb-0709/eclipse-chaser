#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由飞行仿真结果导出地理轨迹：CSV + KML（Google Earth 可开），并给出射界/落点包围。"""
import csv, math, json, sys, os
sys.path.insert(0,"/workspace/eclipse-chaser/sim")
import run_all as R

BASE="/workspace/eclipse-chaser"; DATA=BASE+"/data"
summ=json.load(open(f"{DATA}/summary.json"))
lat0,lon0=R.SITE["lat0"],R.SITE["lon0"]; coslat=math.cos(math.radians(lat0))
def pos_geo(xm,hm):
    return (lat0+math.degrees(xm*math.cos(R.AZ)/R.R_E),
            lon0+math.degrees(xm*math.sin(R.AZ)/(R.R_E*coslat)), hm+R.SITE["elev"])

rows=[]
with open(f"{DATA}/flight_opt.csv") as f:
    rd=csv.DictReader(f)
    for r in rd:
        rows.append((float(r["t_s"]),float(r["x_downtrack_m"]),float(r["h_m"]),float(r["Mach"]),float(r["q_Pa"]),int(float(r["phase"]))))
C2=summ["site_C2_UT_s"]; tau=summ["tau_launch_s"]

with open(f"{DATA}/flight_geo.csv","w",newline="") as f:
    w=csv.writer(f);w.writerow(["t_s","UT_s","BJT_hhmmss","lat_deg","lon_deg","alt_m","x_downtrack_m","Mach","q_kPa","phase"])
    for t,x,h,M,q,ph in rows:
        la,lo,hg=pos_geo(x,h); ut=C2+tau+t
        s=int(ut)%(24*3600); bj=(s+8*3600)%(24*3600)
        w.writerow([f"{t:.2f}",f"{s}",f"{bj//3600:02d}:{(bj%3600)//60:02d}:{bj%60:02d}",
                    f"{la:.6f}",f"{lo:.6f}",f"{hg:.1f}",f"{x:.1f}",f"{M:.3f}",f"{q/1000:.2f}",ph])

coords=" ".join(f"{pos_geo(x,h)[1]:.6f},{pos_geo(x,h)[0]:.6f},{pos_geo(x,h)[2]:.1f}" for _,x,h,_,_,_ in rows[::4])
kml=f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<name>Eclipse Chaser 2035-09-02 trajectory</name>
<Placemark><name>flight track (UTC={int(C2+tau)//3600:02d}:{(int(C2+tau)%3600)//60:02d}:{int(C2+tau)%60:02d})</name>
<LineString><altitudeMode>absolute</altitudeMode><coordinates>{coords}</coordinates></LineString></Placemark>
<Placemark><name>launch site</name><Point><coordinates>{lon0:.6f},{lat0:.6f},{R.SITE['elev']:.0f}</coordinates></Point></Placemark>
</Document></kml>"""
open(f"{DATA}/flight_path.kml","w").write(kml)

# 落点/射界
x_land=summ["flight"]["x_land"]; la_l,lo_l,hl=pos_geo(x_land,0)
print(f"launch  : {lat0:.5f}N {lon0:.5f}E  elev {R.SITE['elev']:.0f} m  az {math.degrees(R.AZ):.1f} deg")
print(f"landing : {la_l:.5f}N {lo_l:.5f}E  range {x_land/1000:.2f} km")
print(f"apogee  : {summ['flight']['h_max']/1000:.2f} km  M_max {summ['flight']['M_max']:.2f}  q_max {summ['flight']['q_max']/1000:.0f} kPa")
print("geo artifacts: data/flight_geo.csv, data/flight_path.kml")
