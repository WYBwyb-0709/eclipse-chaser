# eclipse-chaser

KNSB 两级探空火箭，追逐 **2035-09-02 日全食**（中国段）。仓库包含内弹道、飞行弹道、Besselian 月影几何、结构 FEA 与乘波体构型评估的完整仿真链。

## 1. 飞行器概要

| 项目 | 参数 |
|---|---|
| 推进剂 | KNSB（r = 5.164e-5·P^0.35, c* = 890 m/s） |
| 助推 | BATES 6 段，κ=20.5 mm，φ28 mm 喷管，mp = 5.21 kg，Isp≈127 s，燃时 1.58 s |
| 续航 | BATES 4 段，κ=13 mm，φ18 mm 喷管，mp = 1.15 kg，Isp≈126 s，燃时 1.27 s |
| 结构质量 | 助推 2.85 kg / 续航 1.15 kg / 载荷 1.0 kg |
| 起飞质量 | 11.36 kg |
| BOM | ¥2990 |

## 2. 关键结论：4.5 min 全食不可达

用 NASA GSFC Besselian 根数（ΔT=80.6 s）独立复算 2035-09-02 月影：

| 地点 | 地面全食 |
|---|---|
| 北京 (40.15N, 115.85E) | 109 s（NASA 站值 110.8 s，偏差 2.5%） |
| 山海关 | 110 s |
| 秦皇岛 / 北戴河 | 115 / 116 s |
| 长兴岛 (39.65N, 121.25E) | **119 s（国内陆地最长）** |
| 黄海西部 (39.2N, 125.6E) | ~130 s（本次日食全球最长） |

- 月影地面速度 **1377 m/s**，全食带宽度约 160 km。
- 要把窗口做到 270 s，飞行器须沿轨以 **826 m/s** 跟随月影 270 s；3 km 高度下需 2564 kW、692 MJ，而全箭推进剂化学能仅 **2.5 MJ**，缺口约 **125 倍**。
- 乘波体按正确模型（升力垂直于速度、动压限幅、Ackeret 超声速升力、L/D≈7）评估：T_cam 88 s → 89 s，增益 **+1 s**；无动力滑翔理论上限 +8 s。

**结论**：不出国时全食窗口上限约 119 s（长兴岛一带），火箭无法显著延长；4.5 min 只能通过更换日食（如 2027-08-02 埃及，地面 6m23s）或喷气平台实现。

## 3. 目录

```
sim/run_all.py        # v2 基线：内弹道 + 两自由度弹道 + Besselian + 寻优
sim/opt_v5_lift.py    # v5：乘波体气动模型 + 全食窗口评估 + 物理上限核算
sim/opt_v2.py         # v2 参数寻优
sim/fea_v2.py         # PyTorch 有限元（机匣/螺纹/螺栓）
sim/stl*.py           # 零件 STL 生成
docs/                 # FEA 报告、零件清单、v2 优化报告
data/                 # 求解结果 JSON/CSV（opt_v2、summary_v5 等）
figures/              # 弹道、应力、成本等图
```

## 4. 复现

```bash
python3 sim/run_all.py        # v2 基线
python3 sim/opt_v5_lift.py    # 乘波体评估，输出 data/summary_v5.json
python3 sim/fea_v2.py         # FEA，输出 data/fea_results.json
```

依赖：numpy、matplotlib、torch（CPU）。
