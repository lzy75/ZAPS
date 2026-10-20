# A20：固定q，去掉额外先验位移p的缓存消融

状态 planned；2026-10-20，mechanism_only。A19已完成24例，新增UNet NFE=0。
新图A18上F0=clean相对C：PSNR+0.005234、LPIPS+0.004218；F1相对F0：
PSNR+0.150844、LPIPS+0.000015。F2的LPIPS失败保留，不继续扩系数或恢复A16方向cap。

## 目标与可否定假设

已见结果提出新假设：额外先验位移p=clean−C可能带来较多感知代价而PSNR收益很小。
只改一个分量，保持同一个缓存q，将F1=C+p+q改为B=C+q。
注意q仍由额外网络预测clean计算，不能称为在C上重算的梯度或零调用部署方法。
若部署，B和F1均为原300+1；本次回用缓存所以新增UNet NFE=0。
不扫描p比例或q系数，不用GT选择每图输出。F2只作历史背景，不加入新动作。

## 固定源与条件

源A19 `/data0/lzy/ZAPS-results/diag_terminal_failure_attribution/A19_20261020`：
24例/12图completed、report_complete、plan/run SHA和原源张量/图像/测量SHA全部通过。
复用相同C、clean、q、y、GT，F1=clean+q、B=C+q；q固定，单独检验p是否有用。
原FFHQ SR4、unit_operator+legacy_bicubic、sigma=.05、uint8 PSNR/SSIM11、Alex LPIPS。
GT只作离线指标；B在指标计算前构造，不加载/调用扩散模型，只使用CPU。
历史A16/A17分别报告，新图A18结果作为本次主要探索读数；由于结果已见，
本次不称前瞻独立确认，不用历史均值救回新图失败。

## 预算/门控

新增method NFE=0、diagnostic NFE=0，无训练/采样。仅LPIPS评价网络需加载。
原字段/输出关系相对误差≤1e−7，B有限且形状float32[1,3,256,256]。
重新读取C/F1/GT质量指标相对A19漂移≤.01dB/.001/.001；物理loss容差1e−7+2e−5|loss|。
源文件不可变；输出对照图固定列GT/C/F1/B，24例全部展示，不挑图/种子。
失败/中断保留，不覆盖原run；无新控制器门槛拟合。

## 事先读数与分支

先独立判断B是否满足用户原平均保护线：PSNR相对C超过数值保护线、SSIM降幅≤.002、
LPIPS升幅≤.005、物理loss非增、图像簇PSNR正向比例至少2/3（A18需4/6）。
数值保护线沿用10×repeat/指标漂移与底限PSNR1e−6、SSIM/LPIPS1e−7、loss1e−10。
消融支持的额外工程读数事先固定为：B保留至少90%的F1平均PSNR增益，
并至少消除50%的F1平均LPIPS退化（F1退化必须大于数值保护线才计算该比例）。
90%/50%是本次机制消融的工程判据，不是修改A18原质量容差或文献标准。
报告B−C、B−F1均值/逐例/图像簇结果和最差退化，不以单个指标替代联合读数。

若B普遍更好，优先保留简单固定动作，不能据此宣称需要状态开关。
若不同图存在明确取舍，仅为下一阶段可观测状态选择提出目标，不能直接用GT开关。
若不支持，停止该p移除假设。本轮后先阶段总结，不自动开p/q系数或门控阈值扫描。

## 执行

`bash run_a20_20261020.sh`，CPU缓存分析，新增UNet NFE=0。
新目录 `/data0/lzy/ZAPS-results/diag_cached_prior_ablation/A20_20261020`，
CSV/JSON/全部对照板归档到results/refinement-confirmation-20261020，原张量留原服务器路径。
本机只做快速判据/语法测试，真实缓存指标与图像结果待服务器。
