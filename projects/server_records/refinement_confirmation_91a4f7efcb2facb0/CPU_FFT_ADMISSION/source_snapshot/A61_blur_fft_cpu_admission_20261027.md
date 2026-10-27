# A61：原零边界模糊算子的CPU数值准入

状态implemented，未运行服务器Torch/真实算子，A工作线的mechanism_only数值资格。

## 问题与可否定假设

A51/51b/A58在高噪声t999未通过原真实残差1e-5；不是GPU数目不足。
本轮检验：能否用严格零填充FFT加速同一个空间卷积动作，在事先从条件数上界导出的
有限求解预算内通过Gaussian/Motion准入。FFT不是把裁剪零边界系统当循环系统直接求逆。
没有GT、去噪网络、图像质量标签或Nyström/正则/阈值调参。不是重跑旧128步方案。

父run：/data0/lzy/ZAPS-results/diag_blur_pcg_admission/A58_20261027；只复用其签名plan、
metrology_spec、seed580027。原核与eta的身份由A51 source_spec重新核对A50r2完成记录。
两任务各t999/t319×WHITE/RANGE=4，合计8个double CPU求解。RHS生成保留原float32舍入再promote。

## 唯一机制与固定条件

机制为原有限卷积的零填充FFT动作以及事先条件预算；有两个执行因素，不能声称单独识别其贡献。
B=.25 HH^T+eta I与原eta完全相同，核系数原float32promote64，不重生成更光滑核。
H/H^T在完整256图及边缘上对照原double空间实现，相对误差/伴随配对门控1e-10。
最终验收额外调用原空间H/H^T，必须真实相对残差<=1e-5，FFT内部残差不能替代。

由Young不等式||H||2<=||k||1，kappa(B)<=1+.25||k||1²/eta。
标准精确算术CG误差界给出||r_j||/||r0||<=2 sqrt(kappa) q^j，q=(sqrt(kappa)-1)/(sqrt(kappa)+1)。
预算j=ceil(log(1e-5/(2sqrt(kappa)))/log(q))，上限2048；超过则不求解。
该界只设预算，浮点环境不靠理论界宣布收敛；仍用原空间真实残差判定。
与A58的128迭代不是同算子调用成本，明确可能增加迭代；不覆写A58失败结论。
不加正则，不改tol，不以GT选solver/预算。

## 预算、资源与停止

NN forward/backward均0；两任务CPU各2线程，可与GPU1 ImageNet并行，不初始化CUDA，不终止进程。
每solve至多2048迭代、4097次FFT目标动作（含极端每次表观收敛都需真实检查与末次检查），
300秒求解时间上限（不含初始/最后空间对照）。
超时/预算耗尽不算通过；记录每项H/HT动作、时间、真实残差。内部FFT setup为2个kernel FFT，
每个H或HT为一对FFT，空间门控与RHS生成另列，不能把0NN说成零计算。
签名plan/worker源/核身份固定，partial/failure保留且不自动重试。

正向：8/8原空间真实残差通过，只解锁实际去噪残差的下一阶段资格；不自动启动重建。
负向：任何失败保留并停止该数值候选，不放宽误差或继续扫描。部分通过逐任务报告。
不确定：合成RHS通过不等于全30点覆盖，更不等于PSNR、四任务或曲率控制成功。

新目录/data0/lzy/ZAPS-results/diag_blur_fft_admission/A61_20261027；
结果分支results/blur-fft-cpu-admission-20261027，不合入master。
命令：python -u utils/a61_blur_cpu_session.py；一次等待两个CPU任务，30秒进度，结束汇总/归档/push。
Mac只做小矩阵NumPy卷积/伴随/残差三项检查；真实Torch原算子门控在服务器，不skip。
