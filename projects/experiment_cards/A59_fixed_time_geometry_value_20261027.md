# A59：固定t方向场几何与引导剂量终局响应

状态planned，独立诊断。固定原30点，不调步长、步数或位置，不部署新控制器。
父A55r2十例complete：`/data0/lzy/ZAPS-results/diag_current_direction_value_r2/A55r2_20261026`。
旧5FFHQ图00446/00559/00608/00873/00951×SR/IP，seed1000，unit sigma=.05，模型/算子/IEEE/clamp/指标不变。
A53夹角衰减、A54当前g无独立增量，A55r2支持方向U；新问题是同t局部响应能否解释剂量的最终画质差异。

## 付费状态与唯一干预

完整回放U30点，与缓存NORM_ONLY端点相对误差≤1e-6核对，保存k19/t641、k24/t319、k27/t54当前状态。
u=J_mᵀv，d=u/RMS(u)，RMS(d)=1；x±eps d，eps=s max(1,RMS(x))，s=.001/.002，两尺度各正负，t/y固定。
每次重新计算当前v/u，是同t方向场响应，绝非相邻随机轨迹差分。
Omega_eps=||u_plus/||u_plus||−u_minus/||u_minus||||/(2eps)，单位每输入RMS；
kappa_eps=Omega_eps/sqrt(n)是每欧氏单位输入方向的变化率；
S_eps=||u_plus−u_minus||/(2eps||u0||)，含幅值和方向；
B_eps=||u_plus+u_minus−2u0||/(eps²||u0||)，有限尺度非线性，不是Hessian范数。
两尺度Omega/S相对差均≤.25才可用；失败保留，不更换尺度追分。

每中心仅该步位移乘γ=.5或1.5，再按原网格/相同创新续推U到终点；γ1参照为已回放U端点。
未解决几何点也保留端点/标签。所有干预输出冻结后才GT，标签为final metric(γ1.5)−metric(γ.5)。

## 固定预测诊断与判据

leave-one-image-out，排除同图全部任务记录。STAGE=其余图同任务/同t响应均值；
PHYSICAL=log1p(R_raw)、log(g=||v||/||u||)的1近邻；
GEOMETRY=PHYSICAL加log1p(Omega均值)、log1p(S均值)的1近邻。
折内标准化，平局按image_id；不扫k、尺度、特征或模型，几何未解决确定性回退PHYSICAL，全部30标签保留。
这是新付费几何诊断，不恢复旧R/cos ridge控制；预测关联不是在线控制增益。
候选判据：≥24/30几何可用；PSNR剂量对比MAE相对STAGE/PHYSICAL均降低≥10%，且相对PHYSICAL≥4/5图聚类误差更低。
SSIM/LPIPS/loss误差、逐图剂量曲线均报告。若无增量，停止将此读数直接转权重，不再扫描阈值/倍率。
即使通过也只支持旧开发几何信息，未来需冻结新图；不能称曲率控制/泛化成功。

## 预算、门控和产物

300F回放＋30状态×4=120F探针＋10例×2剂量×(10+5+2)=340F续推，上限760新诊断前向/≤760输入VJP，0优化。
科学臂原300方法NFE、付费重放/探针/续推分账；不把探针当免费。未解决点可能少于上限，实际调用/失败完整列出。
源指纹、同噪声、端点回放、参数/模型grad/RNG不变、调用预算门控；失败保留，不自动重算，无Torch不skip。
新根 `/data0/lzy/ZAPS-results/diag_fixed_time_geometry_value/A59_20261027`；
分支 `results/fixed-time-geometry-value-20261027`。入口 `python -u utils/a59_session.py`；断线同命令跟随，不自动重做失败。代码检查/推送不等于GPU结果。
