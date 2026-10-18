# A1a：存档测量引导方向的阶段变化

日期2026-10-18，工作线A，状态completed（用户回传生产结果；原始目录待归档审计）。
完整定义：../GUIDANCE_STABILITY_DESIGN_20261018.md。

## 1. 问题、父run与假设

目标：实际修正q的相邻方向、连续回摆和幅度变化是否能作为可读状态。
父源为A_state_trajectory已完成的FFHQ00000/seed1000/unit-null末轮存档；
A_cached_direction的用户回报重构误差0、末RNG一致。原始绝对Records未回传，
执行前必须inspect绑定目录、状态complete、source/张量/源码hash，不杜撰路径。
原最终指标26.701292650989096/.8201591372489929/.16083703935146332。
已有clean/mean/priorGuideCos不等于相邻q；这是新增信息，不重跑优化。

H1：至少24/30个q有可分辨非零方向，几何读数能按原邻接计算。
H2（后续）：该读数包含超出噪声阶段与残差的辅助信息；本单图不能认证。
允许mechanism_only，不允许state_gain、作者实现认证、跨图泛化或步数收益结论。

## 2. 唯一变化及固定项

唯一变化是离线观察对象：q与g_z各自的方向序列。g_z由实际A.H的VJP计算。
原模型、观测、PRE参数、Karras rho3网格30步×10轮、lr=.01、joint ζ/D、
unit sigma=.05、legacy_bicubic经验T、db4 level3、DDPM fixed-small eta1、
原clamp/输出口径全部继承父run；不改A、不用真实伴随替换实际q中的经验T。
GT不加载选输出；质量沿用父记录，只作离线解释。没有独立验证集。
固定三方向因果窗口；不同时增加EMA、频域投影或权重。

## 3. 预算、门控及失败路径

新增方法NFE=0，新增诊断UNet NFE=0，历史父方法300、A0回放90单列。
不加载checkpoint、不优化、不产生新测量，不覆盖源；算子VJP/30次重构成本记时。
绑定父completed、网格/索引/alpha/u/hash；数值及末RNG沿用A0b固定容差：
逐步输出relative error≤1e-6，残差≤1e-6+2e-5*norm，clean cosine误差≤2e-4。
不允许通过放宽阈值或重新训练制造新父源。失配保存failed，停止解释。

合成核验须覆盖：常方向、不等u间距、逐步小角度转向、A→B→A、近零/零q、
原邻接缺失、double范数和三角不等式范围。微小回摆O3可高，但幅度必须同时展示。

## 4. 判定与分支

正向：H1成立，合成/缓存门控通过，支持此轨迹读数可测；不以正余弦为门槛。
负向：有效q不足24或变化低于重复数值误差，降级为不可读，不扫描窗口。
不确定：读数可测但与clip/ζ/D/u纠缠，转A1b固定时间快照，不能叫真实敏感度。
门控失败只修接口，保留原基线。完整输出c_q、omega、Omega3、O3、RMS(q/b)、
ζ、M_q、g_z同类读数、clip及原三质量指标，不筛最好阶段。

## 5. 执行与归档

入口utils/diag_cached_guidance_geometry.py已实现。本机6项几何/合同检查通过，
4项张量检查因缺PyTorch待服务器；不是GPU通过。写新results子目录，原张量只读。
运行前保存本卡和resolved_config，运行后仅追加结果，不修改原假设。

```bash
cd /home/lzy/ZAPS/projects
python -m unittest utils.test_diag_cached_guidance_geometry utils.test_diag_cached_direction -v
python -u utils/diag_cached_guidance_geometry.py --inspect
set -o pipefail
python -u utils/diag_cached_guidance_geometry.py --device cuda | tee ../guidance_geometry_20261018.log
```

测试应15项全过且无skip。--inspect只校验身份，打印原A0 source及SHA。
可加--trace-dir <原A0目录>冻结来源；自动选择只依据兼容协议和目录时间，不按质量。
保存run.json、geometry.csv、geometry.pt（实际q和真实g_z序列）、PNG/SVG及本卡。
报告原输出三指标、完整逐点表、全部门控及描述性关联，不输出控制增益。
若缺matplotlib，JSON/CSV/张量仍完整；补图用--plot-only <本次A1a目录>，不重算。

## 6. 2026-10-18用户回传（只追加结果）

用户已回传全部门控passed，逐步输出/q重复误差0、RNG末状态一致。
30/30方向有效，分类READABLE_GUIDANCE_DIRECTION_CHANGE；高/中/低相邻qCos
中位数-.711957/-.096962/.985711。质量关联未认证控制方向。
详见../GUIDANCE_GEOMETRY_RESULT_20261018.md。回贴缺实际运行commit、Source/SHA、
Records和测试输出，完整服务器目录身份仍待下一入口绑定；本机没有GPU执行。

2026-10-18后续A1b补充了A1a实际路径及SHA（见GUIDANCE_LOCAL_PROBE_RESULT_20261018.md），
其五个中心重算clean/q/loss误差均0；A1a原始目录尚待阶段归档下载。
