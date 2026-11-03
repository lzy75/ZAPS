# A81：官方DAPS像素空间倍率迁移

状态planned；2026-11-04；A跨求解框架开发验证。

## 1. 问题、来源与假设
用户要求以官方代码为基线迁移至DAPS。DPS已有部分收益，DiffPIR的A79R1/A80直接及近端适配均退化；不继续DiffPIR调参。
可否定假设：同一个噪声校正投影模块插入DAPS的观测梯度漂移，能在固定FFHQ面板上改善至少一个任务的三项重建指标，且额外成本主要是算子运算。
原文 https://arxiv.org/html/2407.01521v3 §3、附录F；官方 https://github.com/zhangbingliang2019/DAPS ，固定提交0b182726f84df1ad26fde4dccebefda4747b82d8。
23个上游文件原字节保留并验证Git blob，原许可与各文件头保留；从中选择无关第三方导入之外的原AST类，不重写采样循环。
父A71 `/data0/lzy/ZAPS-results/diag_shrink_multitask/A71_20261029` completed72/72仅提供已用图像路径，不读取其参数、观测或核；不是复用旧模型结果。
本轮是官方核心与官方pixel参数的本地配对参考，不是作者100图benchmark复现。

## 2. 固定面板与强基线
FFHQ三图00226/00443/00963；官方down_sampling超分4×与motion_blur61/intensity.5。两臂DAPS/SHRINK。
采用commands/pixel_ffhq.sh对应配置：200轮退火，每轮5步Euler PF-ODE和100步Langevin；Gaussian prior。
官方Scheduler内部num_steps+1，因此实际200×5=1000次NN，每run200×100=20000次观测梯度。不是只有200次前向。
退火EDM poly7 sigma_max100/min.1；内层扩散poly7 min.01。原get_lr比例与最后一次零sigma随机抽样保留，不修正或删除。
原MCMC tau=.01；SR lr=1e-4，Motion lr=5e-5；lr_min_ratio=.01，Gaussian prior，Langevin噪声和条件先验不改。tau不是观测sigma，不自行改成.05。
使用现有ffhq_10m.pt，sha256=81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa；官方DDPM架构与连续VPPrecond映射，无裁剪/精度替换。
官方UNet工厂有载入失败后返回随机权重的行为；独立入口在其后严格再次load_state_dict，失败即停，不能把随机权重当基线。
官方图像预处理ToTensor/Resize/CenterCrop及[-1,1]坐标；观测sigma=.05（物理[0,1]等效.025）。
原SR抗混叠Resizer；原Motion reflection convolution、原随机核构造（含原本两次核构造），使用冻结kernel_seed保证复现。新共享y，不与旧零边界或圆周y混用。
每图采样seed=1000+ID，测量seed=2000+ID，算子seed=3000+ID；两臂init、Gaussian随机创新与时序一致。图像/权重/源码hash在plan冻结。

## 3. 唯一变量与插入公式
原数据漂移 b_y(x)=−gradient(||H(x)−y||²)/tau²，数据校正q=lr*b_y(x)。
当前x已经在干净图坐标，不使用1/sqrt(alpha)转换。r=y−H(x)，d=H(q)。
omega=(1−M*sigma_y²/||r||²)_+；gamma=omega*positive(<r,d>)/||d||²，完全复用A71 noise_signal_shrink.select，无beta、上限2或新系数。
原score_fn通过只替换其operator.gradient返回值，实现gamma*b_y + 原xt_term + 原prior_term；原mc_update继续原lr与sqrt(2lr)随机项。
gamma1分支返回原梯度对象，保留原算术顺序。原MCMC、PF-ODE、退火循环均直接执行。
使用原梯度计算中已经产生的H(x)，每次仅额外一个H(q)，没有新去噪器调用/输入VJP/下一步响应探测。
这会改变Langevin漂移及其目标；只验证重建效果，不继承原文精确后验保证，也不宣称它是无偏似然梯度。
正的全局数据漂移幅度仍可能被投影倍率抵消；不把保留lr配置等同于完整保留数据项幅度。

## 4. 预算、门控与成本
3图×2任务×2臂=12次完整运行，12000NN前向、240000原生观测梯度，0 UNet输入VJP、0优化反向。
候选额外120000次H(q)，不将观测反向计为零成本；每次模型加载1真实网络探针另列。实际计数含失败尝试，突然中断标下界。
纯检查：上游字节、原参数/预算、父计划冻结、新CLI与零成本摘要。
服务器真实Torch必过：原生DAPS对象直接运行与观测封装/强制gamma1输出逐位一致、init/末尾CPU/CUDA RNG一致、调用时序一致、只允许原operator梯度、网络参数不变；错误/NaN零输出不得伪作成功。
小型torch测试同时实际调用原SR和原Motion构造，核对梯度有限、相同seed算子状态复现。
真实模型载入及探针验证架构/权重完整。模型前强制torchdiffeq0.2.5；入口仅在需要时安装该小包，不更换Torch/CUDA，不下载模型。
每个task shard单独运行三图，按图交替臂顺序。指定GPU>=6GiB并要求真实前向峰值+1GiB；不抢占或结束其他进程。
完整clock包含y/算子状态加载、原算子重建、初始化、全部PF-ODE/MCMC/注噪、CPU输出同步及周期进度；模型载入/探针、生成共享y、指标、输出写盘另列。
两臂计时都包含计数/非有限监测封装；这是同场受观测的参考计时，不冒充未经封装的作者benchmark时间。
无模型输入反向但原算子autograd必须保留；禁止误用全局no_grad导致原operator.gradient失效。
GT仅生成测量与离线PSNR/SSIM/LPIPS，不参与倍率、早停或输出选择；返回原xt，不额外clamp/挑选/采样。

## 5. 判据与分支
每任务平均PSNR>=+.1dB、至少2/3图提升、SSIM不降且LPIPS不升，并通过两臂init/末尾RNG/NN时序指纹配对，才列正向开发候选。
质量权衡单列，不按PSNR单指标宣称有效。真实失败保留，不自动重试/切换倍率/修正tau或噪声。
若正向，下一阶段再做固定表/错配消融及ImageNet；若两任务都退化，停止本DAPS插入方式，不扫描lr/tau/gamma系数救分。
本轮不证明后验正确、独立新图泛化、在线匹配必要性或NFE节省。

## 6. 入口与产物
`bash /home/lzy/ZAPS/projects/run_a81_20261104.sh --gpus 1`
根 `/data0/lzy/ZAPS-results/diag_daps_multiplier_transfer/A81_20261104`，断线安全并跟随到最终摘要。
每图质量/clock/计数、共享y与原算子状态、原输出PNG/PT；逐退火倍率统计trace.json，全部20000点dense_trace.npy及列说明。
二进制仅留服务器并在归档清单记录hash；不下载PT或大量中间轨迹。轻量结果分支 `results/daps-multiplier-transfer-20261104`，不合入master。
