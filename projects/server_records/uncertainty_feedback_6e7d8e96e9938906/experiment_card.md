# A42：条件不确定度驱动的引导选择，四任务最小验证

状态 planned；2026-10-25，A工作线。实现不是实验结果，不承诺最终PSNR必然提高。

## 问题与新增信息

A41真实倍率响应可测，但P/I→ridge→最终质量没有四任务增量。
本轮停止旧系数/特征回归，测试新的机制：未裁剪Tweedie均值的方向Jacobian提供
条件协方差近似，结合已知观测噪声计算局部条件均值，据此选择原ZAPS的引导倍率。
依据TMPD Proposition1/Eq6–10：https://arxiv.org/html/2310.06721v3#S3.SS1。
文献的理想恒等式不直接保证本模型、clamp、有限差分、低秩近似的正确性/收益。

父目录 /data0/lzy/ZAPS-results/diag_ffhq_action_geometry/A41_20261025。
取父plan图像ID最小的一张（看结果前固定选择规则），原四任务、seed1000、sigma_y=.05。
复用4个completed case的C、PRE、原测量、17个native单脉冲终局结果，不新训练。
该图已用于开发；四任务只含1个图像簇，不构成确认、泛化或论文成功证据。

## 固定条件、状态与动作

- A41 FFHQ native本地参考全部固定：30×10、lr=.01、原前270调用、ζ/D、db4、
  clamp、DDPM eta1、29个噪声、单位观测/精确伴随、原四算子/运动核/mask、checkpoint。
  不把它称作者实现认证，不改步数/网格/方差/优化器/任何历史源码。
- 固定原k19/t641及k24/t319，分别进行一次干预，不叠加两个动作。
- 当前原C状态在下一原时间s的五个候选clean：gamma∈{0,.5,1,1.5,2}，同一DDPM噪声。
  五点库用于识别，不代表理论最佳范围。IEEE探测与native生产终局分开。
- 以gamma0下一状态为固定Gaussian参考中心，m为其**未裁剪**Tweedie均值。
- 控制子空间V：clean(2)−clean(0)的单位方向，及L^T[y−A(m)]经Gram-Schmidt后的方向，
  最多2列。列Euclidean范数1；相关残量≤1e-3不伪造第二方向，降为rank1。
  此为控制子空间，不声称覆盖全部图像/后验不确定度。
- 对每列，在IEEE执行中心差分，输入RMS扰动尺度 .002/.004，
  h=eta*max(RMS(x_s0),sqrt(1-alpha_bar_s))*sqrt(numel)。
  B=V^T C V≈(1-alpha_bar_s)/sqrt(alpha_bar_s) * V^T J_m V。
- A(x)=Lx+b，其中L=.5H。W=LV直接用.5*原线性H(V)，避免仿射常量作差抵消。
  只统计有效观测；补洞使用mask=1，原观测/训练loss不变。
- 用rank≤2近似C=VBV^T，在小矩阵内求Gaussian更新：
  a=B W^T(sigma_y² I+W B W^T)^(-1)[y−A(m)]；target=m+Va。
  内部readout为B特征值和tr(B W^TW)/(m_active*sigma_y²)，
  物理readout为r^T(sigma_y²I+WBW^T)^(-1)r/m_active。
- FULL：在下一步测量loss≤gamma1、到target预测MSE≤gamma1的候选中选预测MSE最小者。
  ISO：B替换为trace(B)/rank * I，作为相同总方差、无方向性消融。
  PHYSICAL：只最小化实际下一步测量loss；KEEP1、FIXED2均保留。
  并列优先保持1。无GT、无任务分支、无学习的质量预测、无gamma→1惩罚项或权重扫描。
- 固定参考中心/低秩/非Gaussian都是近似，预测风险不增和局部loss不增不保证最终PSNR/LPIPS。
  未分辨/无效协方差明确标为UNRESOLVED，FULL/ISO保持1，不借此宣布控制成功。

## 预算、门控、记录

- 新方法训练NFE=0；每case C回放30+两位置各(5候选+1重复+4*rank)≤58。
  四case总新诊断≤232；rank1时少于此数，失败/中断尝试另累加，不隐瞒。
- 单位置部署传感器最高14额外调用，科学臂原300，带传感器最高314；
  FULL/ISO/PHYSICAL/KEEP可共享同一付费查询池作消融；高效PHYSICAL或固定倍率
  实际可更便宜，单列此差异。不宣称免费300 NFE或NFE节省，不把本轮当在线闭环。
- C相对原last_opt误差≤1e-6；模型/图像/原观察/PRE/终局张量/源码SHA核验；
  外部RNG、模型无梯度、native精度、ζ/D/tau不变；算子门≤2e-5。
- IEEE重复raw中心误差≤1e-7；实际输入中心差分跨度相对误差≤1e-3；
  两尺度B相对差≤.10，非对称度≤.05，最小特征值≥−1e-6*max_eigen_scale。
  容差内微小负值截0并报告；不截大负值强造PSD，不做多尺度/阈值事后搜索。
- raw/clean候选、差分导数、V、两尺度B保存CPU张量；指标、选择、异常及实际calls完整保存。
- 在序列化并冻结所有策略选择后，才读取父run的最终metrics作评价。
  父GT未参与新决策；图像只在原算子mask构造/后续固定展示中读取。

## 预注册判定与处理分支

1. 协方差两尺度及结构门通过率先报告；失败是当前估计不可用，不选择有利状态删除失败。
2. 四任务t319的FULL均可用、最终PSNR>1e-6dB、SSIM≥−.002、LPIPS≤+.005，
   仅叫FOUR_TASK_LOCAL_POSITIVE_ACTION_CANDIDATE；不是泛化/状态增量成功。
3. 单独报告每任务FULL−PHYSICAL、FULL−ISO、FULL−FIXED2。
   只有实际PSNR相对C≥+.05dB，并相对PHYSICAL和ISO各≥+.02dB、两者画质保护通过，
   才记该点STATE_INCREMENT_CANDIDATE。1图无统计确信、无全任务成功声称。
4. 若FULL统一选择2/与P相同：保留正向动作，明确额外状态价值未建立，不包装为创新。
5. metrology失败则先停止此近似；画质失败不通过GT回退挑输出。
   仅正向候选才解锁新图/seed与完整闭环；不自动追加大规模扫描。

## 执行

`bash run_a42_20261025.sh`；双卡两shard，或单卡串行；每分钟进度，完成统一摘要和PNG。
新根 /data0/lzy/ZAPS-results/diag_uncertainty_feedback/A42_20261025。
独立分支 results/ffhq-uncertainty-feedback-20261025；不写父目录、不依赖A41已推送。
