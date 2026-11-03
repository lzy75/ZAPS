# A79：迁移至 DiffPIR 的原生干净图数据校正

状态planned；2026-11-03；A模块跨求解框架验证。

## 1. 问题与父证据
A75/A77在DPS两先验上平均质量改善；A76/A78表明实时匹配收益具有任务依赖性。
本次必须切换求解框架，而非在DPS循环内换权重。选择DiffPIR原生解析数据子问题。
官方仓库 https://github.com/yuanzhi-zhu/DiffPIR ，提交 `2a9898129a1b274131b98746e5b364bc20adc1e1`。
vendored10个文件保持上游Git blob字节一致，MIT授权与UPSTREAM清单保留。
原文 https://arxiv.org/html/2305.08995 ：§3.2—3.3数据子问题/采样，附录B.1表3参数，B.2圆周卷积。
共享核父run `/data0/lzy/ZAPS-results/diag_shrink_multitask/A71_20261029`，不读取学习参数/旧y，不重跑ZAPS。

## 2. 唯一变量及公式
每case两臂：DIFFPIR、SHRINK；均直接执行官方main_ddpir.py中同一个未改动的采样For AST。
原数据解 z=argmin_z ||y-Hz||²+rho_t||z-m||²，rho_t=lambda*sigma_y²/(sqrt(1-alpha_bar)/sqrt(alpha_bar))²。
干净图[0,1]坐标：q=z-m，r=y-Hm，d=Hq；omega=(1-M*sigma_y²/||r||²)_+；gamma=omega*<r,d>_+/||d||²。
只将原校正m+q换成m+gamma*q，随后继续原DiffPIR有效epsilon及重加噪。
**q已经在干净图坐标，不再除sqrt(alpha_bar)**；这与DPS潜变量校正映射的区别属于坐标适配，不新增目标或系数。
原rho/lambda的频率方向作用与zeta随机注噪保留；与原模块相同，正的全局幅度会被投影倍率抵消，不能声称原幅度全部保留。
不加入beta/上限2/有限差分/下一步网络探测/输入反向。

## 3. 固定条件与强基线
FFHQ三图00226/00443/00963、ImageNet三图00067/00151/00193，已有两套checkpoint及SHA沿用。
Gaussian61/std3；A71冻结Motion61核；**原生FFT圆周卷积**，新生成同case共享y，不能复用旧零边界观测。
物理[0,1] sigma=.05，等效模型[-1,1] sigma=.10；不能把这轮与A75/A77的sigma_model=.05平均数直接相减。
100点官方quad序列，U=1，eta=0，guidance_scale=1，原预测裁剪/learned_range/float32 IEEE。
不用demo默认lambda=1追求弱基线；预先固定论文表3 NFE100/sigma_y=.05：
FFHQ Gaussian(lambda7,zeta.3)、Motion(7,.4)；ImageNet Gaussian(8,.3)、Motion(8,.7)。不扫描、不按质量选参。
模型eval，同一case种子1000+ID；观测种子2000+ID；GT仅生成测量及离线PSNR/SSIM/LPIPS。
不修改最终输出：保留原代码x_0及最后一个虽未替换输出但已发生的网络前向。不补采样或选epoch。
这是论文参数、原生循环和共享核的本地对照，不声称完整原论文均值复现。

## 4. 预算、门控、成本
6图×2任务×2臂×100=2400新前向，0输入VJP/0优化反向；每次模型加载1次真实前向探针另计。
原FFT求解不包含NN，记录99次数据校正与模块额外H动作；缓存、观测、求解计量0NN。
每case先用真实核/y及rho的早中晚值做float32原生FFT正规方程相对残差<=1e-3准入，失败在模型加载前停止。
必过真实Torch门控：FFT解对小型dense方程一致、H伴随关系、原AST循环与gamma1逐位等价、相同init/最终RNG/时序/前向次数；禁止输入/参数反向。
完整clock含y/核/初始化/解析解/采样/CPU同步/周期进度，不含模型加载/探针/观测生成/指标/输出写盘。
每run计100F/0B，不因最后一次pred_xstart未用于输出少计该调用。
只用分配GPU，初始>=6GiB并按真实forward峰值+1GiB复查。资源不足attempt前暂停可续接；真实失败保留不自动重跑。
Mac无Torch/CUDA，只做纯协议/语法核对；模型及Torch数值门控交服务器，不跳过。

## 5. 可否定假设和分支
每数据集/任务平均PSNR>=+.1dB、至少2/3图胜、平均SSIM不降且LPIPS不升，视该格跨框架正向候选。
权衡或退化如实报告；不改论文参数/噪声/核/公式上限救分。整体效果与实时匹配必要性分开，不用这两臂证明后者。
不保证正向，更不以少量旧图推断普适有效。若所有格无优势，停止第三方法上的同类调参。

## 6. 命令与输出
`bash /home/lzy/ZAPS/projects/run_a79_20261103.sh --gpus 1`
根 `/data0/lzy/ZAPS-results/diag_diffpir_multiplier_transfer/A79_20261103`。
包含冻结plan、新y/核、0NN计量、逐图PNG/PT/trace、完整clock/cost、comparison.csv/md、summary。
断线安全session，原命令跟随到总结并自动轻量上传 `results/diffpir-multiplier-transfer-20261103`。
不覆盖A71—A78任何SHA绑定代码、配置或旧结果。
