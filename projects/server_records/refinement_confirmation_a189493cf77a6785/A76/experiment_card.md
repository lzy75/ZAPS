# A76 DPS 固定表与错配表消融
状态planned，2026-11-01，A状态机制验证。

## 目标与证据
A75原三图SR/Motion实时SHRINK较DPS平均+3.03/+4.11dB；A72在ZAPS上固定/错配近实时。
本次检验DPS是否需要当前图像状态匹配。父run `/data0/lzy/ZAPS-results/diag_dps_multiplier_transfer/A75_20261031`，18/18完成；结果SHA `dcd224d5e215051806e9f30d307f1dba77a0a524`。
复用原DPS/SHRINK共12run，不重新采样。

## 唯一变量与固定条件
新增FIXED_LOO：另外两图逐步倍率中位数；MISMATCH：按ID排序循环后继图完整倍率表。
均1000项连续值；同任务同sigma，排除本图供体；无质量标签参与，不选最优供体。
三FFHQ图00226/00443/00963、SR4/Motion、sigma=.05模型坐标；原H/y/seed/checkpoint完全继承并SHA绑定。
原DPS1000 DDPM learned_range/clipTrue/scale=.3，时间999→0，FP32 IEEE，原输入梯度、随机更新及q_sample原样保留。
只改gamma来源；无ZAPS学习/倍率迭代/新超参数。GT仅离线原口径PSNR/SSIM/LPIPS。
固定表来自付费开发轨迹，不称无需校准的独立验证。

## 预算、门控、计时
新6case×2臂×1000=12000F/12000原生输入VJP，0优化反向，每模型加载1F/1VJP探针另记。
A75既有18kF/B来源成本保留。固定实现保留未用于决策的实时统计计算，不代表优化后的固定策略速度。
服务器必过Torch原DPS小模型：gamma1与原DPS误差<=1e-6、自身实时表回放逐位等价、init/RNG/grid/F/B/网络参数不变。
完整新run计时含输入加载/算子/init/采样/CPU输出同步，排除模型加载/探针/指标/输出写盘；周期进度记录包含在时钟内。
复用时钟是历史记录，非同场随机交错，不能用小差异宣称加速。
GPU只用分配资源，>=6GiB空闲；不足在attempt前暂停，失败不自动重试。

## 预注册解释
每任务实时对两控制均平均PSNR>=+.1dB、各至少2/3图胜、平均SSIM不降/LPIPS不升，才标匹配收益候选。
相近则不继续搜更差对照，保留无需离线表的工程价值；指标权衡如实报告。非有限失败不改上限救分。

## 命令与归档
`bash /home/lzy/ZAPS/projects/run_a76_20261101.sh --gpus 1`
根目录 `/data0/lzy/ZAPS-results/diag_dps_feedback_ablation/A76_20261101`，逐图PNG/PT/trace、cost、comparison.csv/md。
断线安全并跟随至summary，轻量结果自动推 `results/dps-feedback-ablation-20261101`；不修改历史源。
