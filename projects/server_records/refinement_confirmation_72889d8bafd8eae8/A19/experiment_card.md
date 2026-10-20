# A19：末端动作失败归因，新增UNet NFE=0

状态 planned；2026-10-20，mechanism_only。A18新图12例F2的平均PSNR+0.295160，
但LPIPS+0.007182，超过用户原保护线0.005，原分类未通过，禁止改门槛救回。
同成本F1在该新图名单上平均PSNR+0.156078、SSIM+0.004000、LPIPS+0.004233，通过原平均线。
停止固定F2扩系数及A16方向cap；本轮不生成新策略或扫描系数。

## 问题与可否定解释

PSNR与物理残差下降没有保证LPIPS改善。本轮检验退化主要在额外clean预测中产生，
还是在其后的测量修正中增加。另用精确的测量空间代数分解，检查观测loss下降中
真实测量误差变化与沿实际观测噪声移动各自的贡献。不预设“过拟合噪声”就是原因。
不把测量空间分解当作LPIPS退化的因果证明，也不直接称为曲率/轨迹稳定性证据。

## 源与固定内容

复用A16/A17/A18全部24例（12图、各2seed）的outputs.pt、fields.clean/q、原y和GT。
三个父run必须completed、plan/run SHA及张量SHA通过，config/grid/checkpoint一致。
新图A18单独报告，历史选择图单独报告；不以合并均值救回A18失败。
只恢复F0=clean；C/F1/F2已有，验证F1=clean+q、F2=clean+2q的相对误差≤1e−7。
GT仅在本离线归因中使用，不进入控制器；F0/系数集合固定，无GT输出选择。
任务与指标沿用原FFHQ SR4 unit_operator+legacy_bicubic、sigma=.05、uint8 PSNR/SSIM11、Alex LPIPS。
以CPU读取张量、计算H和指标，不加载/调用扩散模型；LPIPS评价网络不属于UNet NFE。

## 精确分解与门控

令 b=H(GT)，n=y−b，e_C=H(C)−b，a=H(X)−H(C)。
归一化真实测量误差变化 T=(mean((e_C+a)^2)−mean(e_C^2))/(2 sigma²)，
沿观测噪声移动项 K=mean(n*a)/sigma²。
则观测loss变化必须满足 delta_loss=T−K；K正只表示该运动沿本次噪声方向，
不能据此独立判定有害，因为T也可能下降。代数误差≤1e−9+1e−7|delta_loss|。
报告C/F0/F1/F2的loss/PSNR/SSIM/LPIPS、越界比例、RMS位移；F0−C/F1−F0/F2−F1
为固定臂之间的差值，不宣称非线性LPIPS的可加因果归因。
重算父C/F1/F2指标漂移≤.01dB/.001/.001；物理loss≤1e−7+2e−5|loss|。
保留轻微CPU/GPU算术差异，不改变原A18质量判据。

## 输出与处理分支

新增method/diagnostic UNet NFE均0，不优化、不采样、不加载UNet权重。
导出完整24例CSV/JSON；四张固定名单的对照板，每例GT/C/F0/F1/F2，全部seed均展示；
不按最好/最坏效果挑图，不改变显示范围，不把差分放大图当实际图像。
报告各父队列均值与逐图LPIPS退化；分解若显示观测拟合改善却真实误差变差，
仅作为后续噪声约束假设的依据。若真实测量误差也改善但LPIPS退化，
下一阶段需针对图像先验/纹理代价立题，不能继续把残差下降当充分条件。
本轮不拟合分类器、不扫描阈值，不把已见数据的描述性关联计为状态控制收益。
门控失败停止解释并保留已写结果；没有解锁自动在线控制或正式核心修改。

## 执行与归档

`bash run_a19_20261020.sh`，CPU顺序计算，保留GPU给其他任务。
新目录 `/data0/lzy/ZAPS-results/diag_terminal_failure_attribution/A19_20261020`，
张量复用不复制；JSON/CSV/PNG提交results/refinement-confirmation-20261020。
本机快速NumPy/语法检查，服务器执行实际张量与LPIPS检查。
