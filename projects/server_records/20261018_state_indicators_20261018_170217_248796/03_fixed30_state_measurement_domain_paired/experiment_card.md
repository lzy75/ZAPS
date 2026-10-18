# A：新测量协议下的状态收益，以及FFHQ值域×状态交互

日期2026-10-14；状态implemented / GPU pending。用户已授权这两组单图实验。
新增6项本机契约测试通过，3项张量/实际核心测试因无PyTorch跳过；
服务器必须先执行测试模块，通过后才启动预训练模型数值实验。
本轮只实现独立入口，不改共享核心、历史入口、默认值或批量分支。

## 问题、去重与参照

ImageNet Gaussian的新unit协议paper21.7256/.5387/.3958，uniform
21.2757/.5571/.4440。paper相对uniform的PSNR/LPIPS更好，但SSIM较低；
这是固定调度比较，不是状态增益。父run：
`/home/lzy/ZAPS/projects/results/diag_zaps_paper_task_schedule/imagenet_20261008_172549/optimized_trace_gaussian_deblur_20261008_181857/gaussian_unit_domain_20261014_133157_715747`。
本次的新信息是**原初值、unit观测**上的fresh null/state；不重复改初值旁支。

FFHQ既有Karras rho3状态10图PSNR+.1398（9/10），LPIPS+.0067。
历史model测量域σ=.05，经验legacy_bicubic引导，并无可复用的完整unit配对。
本次冻结该候选和经验映射，用两个测量域各自null/state；不是重新搜网格。
两组不同任务/模型，不将交互外推为跨数据集统一因果解释。

## 预注册实验臂与固定条件

| 组 | task/grid | 测量域与状态 | 固定参数 |
| --- | --- | --- | --- |
| ImageNet | Gaussian/paper15/10/5 | unit-null、unit-state | seed1001，全ζ=.2、D=.2、lr=.001 |
| FFHQ | SR×4/Karras rho3 | model-null、model-state、unit-null、unit-state | 00000.png/seed1000，全ζ=.1、D=.2、lr=.01 |

每臂30×10、joint Adam(.9,.999)/eps1e-8、fixed-small DDPM eta1、
db4 level3、原clamp/score no_grad、surrogate OFF、MSE mean、clipnorm1、
ζ>=1e-6、last_opt第10轮更新前输出；不使用额外final sample/早停/GT挑选。
ImageNet完全复用父unit y；旧21.7256缓存仅检查null漂移，不代替本次null。
FFHQ同一个标准噪声场xi，用y_m=H(x_gt)+.05xi，
y_u=H((x_gt+1)/2)+.05xi；不同物理SNR，不是同y。
H不变；旧经验引导T=legacy_bicubic，新unit经验引导=.5T；不声称是精确伴随。
FFHQ物理域与引导/loss单位一起改变，不单独隔离σ或梯度缩放。
UNet始终使用model状态[-1,1]；不把观测clip到[0,1]。

状态策略冻结（没有扫描入口）：第一轮固定网格作pilot，计入300NFE；
残差定方向、clean全局余弦veto置信门控，λ=.20、β=.8、s_r=.15、s_c=.20、
步长调制[.8,1.2]，不额外调制ζ。ImageNet沿用已实现log-sigma u参考对齐、
u-scaled EMA、gamma=.25；FFHQ沿用历史候选步序reference、gamma=.35。
FFHQ本轮不同时改成u-reference，否则无法与历史候选关联；两域控制器相同。
两套策略差异显式记录，不声称跨模型对照只差权重。
30组ζ/D按逆向调用位置29→0映射，动态网格不转移已训练参数。
首两点余弦无效时中性；各轮完整999→0，固定30次调用。
GT只用于FFHQ合成测量与最终评价；ImageNet所有训练完成后才加载GT；
optimize均x0_gt=None。状态标量detach，不优化score权重。

## 门控与完整预算

ImageNet：90诊断NFE（fixedA/fixedB/adaptive-null各30），600方法NFE。
FFHQ：每域90诊断NFE，共180；四臂1200方法NFE。
全部合计2070新UNet NFE，不把pilot当免费，不重复已完成固定调度训练。
门控失败停止该组，不重试到通过；保存失败记录和已用成本。

0UNet身份/域/网格/初值检查；ImageNet父completed、所有gate、源码/权重/
图片/y/输出/PRE快照hash和runtime检查，noise preflight确保原x_T可重建。
FFHQ新测量、标准噪声和初始noise均存档，无旧结果兼容缓存假设。
Gaussian unit实际affine/VJP伴随门控2e-5；FFHQ仅检查仿射与.5T缩放合同，
不得把经验T与真实VJP不相等报告成接口失败或精确伴随PASS。

fixed-repeat与null单unroll同时比输出和ζ/D梯度，零响应元数据/噪声末状态相同。
沿用现有parity_decision：输出硬cap1e-6，梯度cap1e-4；cross限制为
max(输出1e-7/梯度1e-6,3×repeat误差)，不超过hard cap，不事后放宽。
实际训练验证同x_T、每臂300、每轮30端点严格下降、同标准正态步序draws、
score不消费RNG、10轮联合学习、finite loss/grad/output；不是同物理Brownian路径。
每轮被动记录r/c/u（适用时）、信号/置信度/h/t/参数索引，保存PRE快照和输出hash。

## 决策与后续

每域只用自己的state-null计算三指标增益，跨域state-old-null不叫状态收益。
FFHQ另报告域效应（unit-null减model-null）和状态交互
`(unit-state−unit-null)−(model-state−model-null)`，不可直接比跨域MSE大小。

单图候选门槛：dPSNR>=.1dB、dSSIM>=-.002、dLPIPS<=.002，且有效网格确有改变。
只有PSNR改善且辅助指标明显恶化为PSNR_ONLY_TRADEOFF；其余无综合收益。
未实际改变步点为STATE_NOT_ACTIVATED，不将CUDA漂移当调度增益。
ImageNet先核对fresh null对缓存漂移：PSNR<=.05、SSIM/LPIPS<=.005绝对误差，
超限记INCONCLUSIVE_NULL_DRIFT；不把漂移解释为状态效果。
候选成立再讨论固定10图；感知取舍则只改余弦表达/组合一项；负结果结合trace
定位信号饱和/反弹/预算末段补偿，不删除两个指标、不回到初值/降噪扫描。
单图工程门槛不是统计显著性、作者实现认证或原文均值复现，不自动扩大范围。

## 运行与存档

独立入口`utils/diag_state_domain_paired.py`，--dataset显式选择。
ImageNet用--unit-run-dir，FFHQ用--image和--seed1000。
每次新目录，保存card/run.json/summary.csv、trace.csv、观测/输出/PRE及post参数、
RNG和源码hash、诊断门控和完整成本；后续恢复需使用PRE，而非post参数。
本机无PyTorch时数值测试明确skip；服务器先跑测试，再跑两条命令。
