# A78：ImageNet DPS 实时匹配消融与缓存审计

状态planned；2026-11-02；A状态机制线。固定原1000点，不改变采样器或公式。

## 问题与父证据
A77在ImageNet预训练先验的SR4/Motion平均PSNR提升+.959/+2.955dB。
A76在FFHQ上，固定/错配1000项倍率表接近实时效果。本次检验这一现象是否跨先验成立。
父run `/data0/lzy/ZAPS-results/diag_dps_imagenet_transfer/A77_20261101`，12/12 complete，
结果commit `017f6bbedd938dd1be48aed90e0caf1b7360b6dd`。
允许开发面板机制对照，不将旧三图结果写成独立大样本检验。

## 唯一变量、固定条件与复用
原DPS及实时SHRINK12run只读复用；新增FIXED_LOO/MISMATCH两控制。
固定表：另外两图每个t倍率的中位数；错配表：image_id排序循环后继图的原倍率轨迹。
每张图均排除自身供体，同任务/sigma；只读gamma与时间，GT质量不参与供体选择。
所有1000项表在GPU新run前冻结；保留连续值、无倍率上限或新系数。
图00067/00151/00193，SR4/Motion，sigma=.05模型坐标；H/y/seed/weights与A77逐项SHA绑定。
ImageNet无条件256模型、原DPS native1000 DDPM learned_range/clipTrue/PS scale=.3；FP32 IEEE。
原修正方向和随机循环、q_sample调用不变；只替换gamma来源。无ZAPS学习、无新网络反向。
零NN缓存审计报告逐图质量变化、最终观测误差/噪声方差和10个时间段的倍率/收缩权重/响应代理统计。
离线标签只作审计，不能改变表、参数、执行顺序或输出选择。00193超分是PSNR下降而SSIM/LPIPS改善，不能称全面失败。

## 预算、门控、计时与资源
新6case×2臂×1000=12000F/12000原生输入VJP，0优化反向；每加载1F/1VJP真实探针另记。
A77既有12000F/VJP及2F/2VJP探针另列，不计为本轮新调用；缓存审计0NN。
强制服务器Torch门控沿用已验证的原循环：gamma1对原DPS误差<=1e-6、自身倍率回放逐位等价、init/RNG/grid/F/B/权重不变。
新run固定1000F/1000VJP/0优化反向；缺Torch不得skip，Mac仅纯协议和语法检查。
完整时钟含输入加载/算子/init/采样/CPU输出同步/周期进度，模型加载/探针/指标/输出写盘分列。
A77复用时钟非同场，不能用微小时差解释为控制策略加速；固定实现仍计算不用于决策的实时统计。
只选显式分配且>=10GiB空闲GPU，再检查真实探针峰值+1GiB；不足在attempt前暂停，原命令可续。
真实失败保留不自动重复；不杀其他任务、不降精度、不改图/观测/倍率范围。

## 预注册解释
每任务实时对两控制均PSNR平均>=+.1dB且至少2/3图胜，平均SSIM不降/LPIPS不升，标为匹配收益候选。
否则分别报告近似等效、控组更优或指标权衡；不事后找更差供体/阈值救分。
A76/A78若均接近，论文应区分自动构造幅度规律的收益与在线逐图匹配收益，不继续原样重复。

## 执行与归档
`bash /home/lzy/ZAPS/projects/run_a78_20261102.sh --gpus 1`
根 `/data0/lzy/ZAPS-results/diag_dps_imagenet_feedback_ablation/A78_20261102`。
保存plan、冻结表、cached_diagnosis.json、逐图PNG/PT/trace、质量/完整时钟CSV与summary。
独立断线安全session跟随到summary，自动轻量归档push `results/dps-imagenet-feedback-ablation-20261102`。
不覆盖A75—A77源码、配置、记录、已有判定；只更新独立新包。
