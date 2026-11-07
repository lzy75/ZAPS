# A99：超分退化是否集中在真实观测零空间——终局反事实诊断

状态planned；A工作线。用户明确要求验证“超分观测少、自由度多，观测目标无法约束细节”的猜测。
不重复A98全系数/LL2限制的在线轨迹实验；先用已完成A97输出做0新增扩散网络调用的精确物理分解。

## 身份、父记录与假设

父 `/data0/lzy/ZAPS-results/diag_full_unroll_response/A97_20261106`。
只要求原3个SR4案例00226/00443/00963、sigma_model=.05的REAL_SCALAR/REAL_DIAG六条记录completed，所有输出PT/SHA匹配。
不要求A97全部72完成，不读其仍会变化的summary/session作为绑定输入，不改父记录，不占GPU或结束A97/A98。
沿用父图像/观测/模型坐标与实际DPS Resizer前向算子；GT/y/源/PT及完成记录逐项SHA绑定。
可否定假设：矩阵相对真实响应标量的终局rawMSE增加，主要来自H不可见方向；去掉这部分终局差异，保持矩阵原观测的同时会恢复部分整图质量。
A98的LL2对照只测试频率结构；本轮分解真实H的行空间/零空间，两者不混用。

## 精确H的构建与门控

实际forward H(X)=Ay X Ax^T，Ay/Ax为原Resizer的64×256一维加权矩阵。
直接读取其固定weights/field_of_view，镜像边界重复输入索引必须累加；不重新定义核、边界或缩放，不使用历史bicubic upsample当H伴随。
float64薄SVD得到两个轴的正交行空间基Qy/Qx：

P_R(X)=Qy(Qy^T X Qx)Qx^T，P_N=I−P_R。

此处P_R=H†H是真实线性H的欧氏正交投影；3通道行空间维度12,288、零空间维度184,320（须先通过轴满行秩门控）。
SVD截断仅机器eps×矩阵最大维×最大奇异值的数值rank判据；不做正则/CG/噪声阈值调分。
强制门控：原H float64前向与分离矩阵相对L2<=1e-10，原float32<=3e-6，投影幂等与H(P_N v)相对误差<=1e-10。
原Torch/Resizer真实测试禁止调用legacy transpose；不通过则停止，不用近似投影兜底。

## 五个全部报告的反事实输出

S为原真实响应标量最终图，D为矩阵最终图，v=D−S。
构造v_R=P_Rv，v_N=P_Nv。

| 输出 | 构造 | raw观测关系 |
| --- | --- | --- |
| SCALAR_FINAL | S | 原标量 |
| DIAG_FINAL | D | 原矩阵 |
| ROW_ONLY_DIFFERENCE | S+v_R | H(S+v_R)=H(D) |
| NULL_ONLY_DIFFERENCE | S+v_N | H(S+v_N)=H(S) |
| LL2_ONLY_DIFFERENCE | S+W^T mask_LL2 Wv | 频率结构参照，无观测等价保证 |

所有输出先构造和检查观测等价，再读取GT。GT不进入v、投影、阈值或输出选择。
这是最终图像的反事实组合，不是可直接在线获得的算法；S和D来自不同完整轨迹，不称同一中间状态的控制因果实验。
全部五臂保留并输出PNG/指标，不按GT挑最优组合当本文方法。

## 误差恒等式与判定

对每个最终图像x：
MSE_raw(x)=||P_R(x−GT)||²/N+||P_N(x−GT)||²/N。
必须满足分解误差<=1e-10相对容差。
矩阵相对标量的误差增加：ΔE=ΔE_R+ΔE_N，逐图和总和全部报告。
预先固定支持门槛：三图总ΔE>0、ΔE_N>0、ΔE_N/ΔE>0.5，且ROW_ONLY_DIFFERENCE相对D的rawMSE在至少2/3图降低。
若可见部分改善，零空间贡献比可大于1；这个比不是概率。总ΔE<=0则不计算该占比，不强行写支持。
同时报告原PSNR/SSIM/LPIPS、raw/clampedMSE、clamp比例、真实/带噪观测误差、显示clamp后的观测误差。
**观测等价只保证clamp前原坐标**；显示clamp/uint8后不强行宣称仍等价。
两条缓存原输出质量复算必须满足PSNR1e-4/SSIM1e-5/LPIPS1e-5容差。

支持：说明本组终局差异中存在实质性观测不可见风险贡献；结合A98判断在线限制是否有用。
不支持：查看可见误差或clamp/感知取舍，不继续用“自由度过多”作为既定解释。
即使支持，也不单独证明高维优化造成漂移、不保证完整30位置限制后改善、不把H零空间等同高频或去噪器响应Jacobian零空间。
下一网络的非线性影响由A98实际轨迹验证，本轮不拼凑该保证。

## 预算、实现与交付

0 diffusionUNet、0输入VJP/JVP、0优化反向、0重建重跑；两次64×256小SVD、三次离线DWT分析/合成。
15次LPIPS评价另计，它们是评价网络而非0NN方法成本；沿用服务器已有AlexNet缓存，缺失即停、不下载。
CPU2线程，CUDA隐藏；模型不加载，原采样RNG不变。CPU诊断耗时单列，不作采样加速benchmark。
本机只做NumPy/CLI/语法检查；服务器强制真实Torch原H/重复边界/投影测试与实际数值门控，无Torch不skip。
失败/中断记录保留，不自动失败重跑；成功案例跳过。detached安全，多终端仅跟随日志，结束后汇总并轻量上传独立results分支。

入口：`bash projects/run_a99_20261107.sh`，无需等A97/A98或分配GPU。
根 `/data0/lzy/ZAPS-results/diag_sr_observable_counterfactual/A99_20261107`。
分支 `results/zaps-sr-observable-counterfactual-20261107`。
`--status`只读；`--archive-only`空闲时仅重试汇总上传。输出summary.json、comparison.csv、逐图run.json及五臂PNG；父PT留服务器。
