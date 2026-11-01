# A77 冻结模块迁移至 ImageNet 扩散模型
状态planned，2026-11-01，A迁移验证。

## 问题与证据
A75仅FFHQ模型；本次换已下载ImageNet256无条件模型，检验另一权重和网络及数据上的相同模块。
父A75完成18/18，结果SHA `dcd224d5e215051806e9f30d307f1dba77a0a524`；物理算子来自其A71父目录。
仍是DPS求解器，模型/数据一起改变，不单独归因其中一项。三图00067/00151/00193为既有固定面板，不调参或挑图。

## 固定协议
两臂原DPS/同一实时SHRINK，SR4/Motion，sigma=.05模型[-1,1]坐标，原算子及冻结Motion核。
y=H(gt)+sigma*epsilon，每case只生成一次并hash；seed1000+ID，观测seed2000+ID。
原1000点DDPM learned_range/scale.3/clipTrue，原噪声及q_sample RNG不变，无新增NN/VJP。
ImageNet网络256通道/2残块/attention32,16,8，FP32/learn_sigma；无精度/分辨率更改。
checkpoint `modules/models/256x256_diffusion_uncond.pt` SHA `a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1`。
沿用A71/A75公式，不增加系数/上限2/逐任务调参。无ZAPS学习，GT仅生成观测与离线评价。

## 预算与门控
3图×2任务×2臂×1000=12000F/12000原生VJP，0优化反向；每加载1F/1VJP真模型探针另记。
服务器必须Torch原DPS门控，Mac缺Torch不得skip。GPU初始>=10GiB空闲，实测VJP峰值+1GiB逐run准入。
资源不足在attempt前暂停，可原命令续；真实失败保留不自动重试。不得下载/换模型救分。
完整时钟含输入加载/算子/init/1000采样/CPU同步；观测生成、模型加载、探针、指标、输出写盘分记；周期进度记录计入时钟。
输出最后一步，无GT筛选。报告逐图PSNR/SSIM/LPIPS/残差、完整秒数、NN/VJP、峰值。

## 可否定判据与后续
每任务平均PSNR>=+.1dB且至少2/3图胜、SSIM不降/LPIPS不升：该面板跨先验候选。
失败报告任务边界，不再改scale/图/噪声救分。多指标权衡不得声称全面优越。

## 命令与归档
`bash /home/lzy/ZAPS/projects/run_a77_20261101.sh --gpus 1`
结果 `/data0/lzy/ZAPS-results/diag_dps_imagenet_transfer/A77_20261101`，完整逐图PNG/PT/trace及CSV/summary。
断线安全并跟随到结束，自动轻量推 `results/dps-imagenet-transfer-20261101`。
与A76不同GPU可并行，同GPU顺序执行。
