# A52：完整几何反馈获得正向结构结果，实时夹角贡献仍待验证

本轮4/4例完成，480新增UNet前向、240输入反向，无中断。两张旧开发图、一个seed，SR/IP，
不属于独立确认或四任务成功。原前270调用和最后30步位置固定，完整目标改变引导方向。

## 1. 来源与复核

结果分支`results/ffhq-feedback-vector-constraint-20261026`，提交
`de6af1fece6ba6745f4c9066bc581b8662ce6624`。
仅fetch，未把结果分支合入master，未下载PT。
抽取根`/Users/liziyue.22/ZAPS-deliverables/analysis/A52_20261026/archive`；
清单31个文件（含4幅PNG）全部核对字节数/SHA，清单本身另读，JSON签名侧文件逐一核对。
由独立evaluation重算84个均值对比和120个投影丢弃分量的范数恒等式；四例原记录门控均通过，
Q重放误差至多约9.24e-8。Mac没有重跑Torch/GPU门控或原张量指标。
机器记录：`A52_cached_audit.json`；可复算脚本：`projects/utils/audit_feedback_vector_archive.py`。
已实际查看00446的SR/IP两幅完整对比图；另外两幅已取回，未声称全部视觉审阅。

## 2. 正向效果有多大

FULL_ALIGN相对匹配本地基线C_IEEE：

| 任务 | C_IEEE均值PSNR | FULL_ALIGN均值PSNR | ΔPSNR | ΔSSIM | ΔLPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4×SR | 20.963 | 28.721 | +7.758 | +.19230 | −.08031 |
| IP | 23.354 | 25.235 | +1.881 | +.06875 | −.01319 |

每个任务两图均同时改善这三项指标；所有臂每图30/30已解析。
相对不带Jacobian几何投影的FULL_PROX，FULL_ALIGN的PSNR还分别提高.976/1.272dB，
SSIM提高.07533/.06740，LPIPS降低.08552/.07059，均为同两图上的成对均值。

这些是A40继承的本地协议（unit观测sigma=.05、LR=.01等）下的结果，不是作者1000图均值复现；
也不能把大幅改善解释为相对所有充分调优固定方法的增益。未直接对齐的旧F不能混成同一基线。
科学前向均300并不消除额外VJP/CG成本。

## 3. 这轮支持了什么

FULL_PROX去掉q方向约束后，SR相对PROX_Q仅+.072dB，IP约−.001dB；
FULL_ALIGN去掉同一约束后，SR相对ALIGN_Q为+1.568dB，IP为+10.976dB。
后者包含从旧失败映射恢复的收益，不应拿+10.976当作胜原基线的标题。
效果取决于目标与约束的相互作用，不能说“去掉任何投影都会更好”。

沿FULL_ALIGN自己的轨迹，把完整目标重新投回q时，被丢弃分量的范数占完整目标范数的逐步中位数：
SR约.719/.745，IP约.895/.896。这是向量投影损失，不是丢失了90%的图像或有用信息。
结合Q重放和2×2对照，结果支持：将当前几何目标压回原q方向会严重限制这套映射，尤其IP。

IP的机制具有可解释性：测量目标v受mask约束；J_m(x)^T v可以通过去噪器的空间耦合传播到
不可观测区域。再限制到q会丢弃这部分方向的潜在作用。当前数据与这个解释相符，
并非已经完成所有空间分量的因果隔离，也不是后验正确性定理。

区域GT误差也说明单看物理拟合不够：FULL_PROX两图观测区unit-MSE均比FULL_ALIGN低，
但FULL_ALIGN的不可观测区误差更低，且整体PSNR/SSIM/LPIPS更好。
这些区域GT只作事后评价，未参与控制。

## 4. 当前状态到底是什么

```text
测量残差 r -> 含噪声尺度的测量目标 v
当前去噪器Jacobian -> u = J_m(x)^T v
当前幅度 s = ||v||/||u||，当前夹角 c = cos(v,u)
完整位移 T = K * s * c * u = K * <v,u>/||u||² * u
```

当前方向u、响应幅度s、夹角c都随图像和时间变化。
这轮比较支持完整Jacobian几何指导的质量候选，但其变化同时涉及方向与幅度，
还没有隔离“当前夹角优于纯时间夹角”。旧TIME/SHUFFLE用的是q方向，不能替代同机制消融。
这个c是同一状态的几何兼容角，不能改称相邻步余弦、整条轨迹曲率或局部积分误差。
这里也没有新的投影定理；作为论文方法贡献仍需明确文献差异和独立控制证据。

## 5. A53为何能更直接回答状态反馈价值

扩到同批5张旧图/两任务，共10例；复用A52已有4条STATE轨迹，只补6条。
对照全部保留各自当前u方向和当前||v||/||u||，只更换c：

- STATE：当前夹角；TIME_COS：其他4图同一步的中位数；
- SHUFFLE_COS：循环另一图的夹角轨迹；NORM_ONLY：c=1。

新增1080前向、至多1080输入反向，无训练，双卡可用则并行。
先生成全部STATE，再排除自身图像、冻结系数，不用GT选规则；全部控制输出完成后才评价。
源STATE若有未解析夹角，先停，不插值补值或改阈值。

若STATE稳定胜TIME_COS/SHUFFLE_COS，可获得局部角度匹配的开发证据；
若只是所有几何臂胜C，则保留几何质量结果，不能宣称当前角度具有独立贡献。
若NORM_ONLY更好，则停止追这个夹角的衰减规则，保留响应方向/范数体系待单独验证。
五旧图LOO仍没有独立泛化或置信保证；blur数值准入、第二seed/噪声、新图和h控制仍未补齐。

## 图片

- [SR 00446完整对比](/Users/liziyue.22/ZAPS-deliverables/analysis/A52_20261026/archive/ffhq_super_resolution_00446_s1000/constraint_comparison.png)
- [IP 00446完整对比](/Users/liziyue.22/ZAPS-deliverables/analysis/A52_20261026/archive/ffhq_inpainting_00446_s1000/constraint_comparison.png)

复核命令（Mac，从projects运行；输出文件需不存在）：

```bash
python3 -m utils.audit_feedback_vector_archive \
  --archive /Users/liziyue.22/ZAPS-deliverables/analysis/A52_20261026/archive \
  --output /tmp/zaps_a52_cached_audit.json
```
