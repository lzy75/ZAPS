# A60：当前方向刷新频率与真实反向成本

状态planned。固定原30采样点，只改方向u的更新频率，不调步长、步数、采样位置或倍率阈值。
父A55r2十例完成，旧5FFHQ图×SR/IP，seed1000、unit sigma=.05、checkpoint/算子/原PRE/29创新/IEEE/clamp/指标均不变。
源 `/data0/lzy/ZAPS-results/diag_current_direction_value_r2/A55r2_20261026`；不依赖A56—A59的新运行结果。

## 问题与唯一因素

A55r2支持pullback方向U相对物理V和原Q，但未证明每步都要重算当前方向。
本包比较CURRENT每步、LAG2每2步、LAG4每4步。k0与k29必须刷新，分别30/16/9个预定刷新点。
每一步都重新前向、读取当前raw m、R和v，保持当前物理长度K||v||。
刷新步使用原U目标K||v||u/||u||；非刷新步用最近一次成功的u方向，仍按**当前**K||v||校准长度。
称“方向滞后”，不是在当前v上用精确冻结Jacobian：旧方向同时含旧x、t和旧v信息。
禁止在滞后步计算当前u、当前u夹角或用它做监控后声称省反向；不会按GT或未公开状态触发额外刷新。

所有forward保持原graph-enabled/IEEE算术，省的是输入VJP尝试，不改变前向图策略。
CG未解决用原q；预定刷新若方向未解决则清空缓存，直到下一预定成功刷新；不加非预定VJP。
滞后步CG未解决只当步回退，保留已有方向缓存。数值失败/回退完整报告，不静默剔图。

## 预算与门控

10例×3臂×30=900新诊断前向；输入VJP上限10×(30+16+9)=550，0优化更新。
每臂仍30forward，不宣称NFE节省；原300方法预算中前270复用，不重复计新实验调用。
同运行CURRENT重新回放，与缓存NORM_ONLY端点相对误差≤1e-6核对，作为同场计时/质量参考。
检查每个非预定步backward_calls=0、总调用序列、原参数/权重grad/RNG不变、目标长度误差≤2e-6。
真实Torch预检明确禁止滞后步VJP；缺Torch直接失败，不skip。全部30端点冻结后才评价GT。

## 判据与分支

工程低成本候选：所有路径完全解决；该lag臂相对CURRENT每任务mean PSNR≥−.05、SSIM≥−.002、LPIPS≤+.005，
至少4/5图PSNR损失≤.10；同时相对C_IEEE mean PSNR≥+.05且SSIM/LPIPS保护线通过。
不通过就保留CURRENT，不扩大刷新周期或扫门槛追分。
每图端点、PSNR/SSIM/LPIPS/物理loss、回退数、输入VJP、CG算子调用、峰值显存、每臂秒数都报告。
计时是同运行但服务器共享负载、固定顺序和逐步诊断写盘和日志会混入开销，不能据此保证生产墙钟加速。
即使lag保持质量，也仅支持旧开发图的方向复用；不能证明当前J/曲率实时控制或跨数据集泛化。

## 输出

新根 `/data0/lzy/ZAPS-results/diag_direction_refresh_value/A60_20261027`，
结果分支 `results/direction-refresh-value-20261027`。
入口 `python -u utils/a60_session.py`；断线安全、失败不自动重跑，不改任何旧SHA绑定源文件。
