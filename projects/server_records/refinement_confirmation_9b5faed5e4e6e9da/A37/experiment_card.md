# A37：噪声能量正则的状态倍率与噪声知情时间对照

状态planned；2026-10-25；A线。用户授权继续噪声水平调整，并核对论文/开题契合性。
可否定假设：把已知测量噪声直接纳入倍率目标，能在保留相对C正向PSNR的同时减轻A36感知退化。
同时检验噪声信息、实时图像状态是否分别有价值，不把所有改善归为状态感知。

## 父run和新增信息

A36 /data0/lzy/ZAPS-results/diag_frozen_response_transfer/A36_20261023，18/18完成；
六图00339/00447/00561/00609/00875/00952，三sigma=.025/.05/.10，seed1000。
原STATE在低/原/高噪声相对C PSNR+1.467/+1.291/+.657，而高噪声LPIPS+.07719、SSIM−.02474。
STATE没有比TIME更强的迁移证据。原I_cont2中sigma仅归一化R，没有直接约束倍率。
A34无上限噪声目标已失败，本轮不重复强制R趋近1；A35曲率无增量，不加入曲率调权。
复用A36完成、SHA/预算/门控通过的BASELINE PRE、观测、C/STATE/TIME终局；不重训C。
这些六图已暴露A36结果，本轮仅机制开发诊断，不能再称独立确认。

## 正则目标及冻结参数

r=y-H(clean)，a=H(clean+q/sqrt(alpha_bar))-H(clean)，m为测量标量数。

    J(gamma)=.5||r-gamma*a||² + .5*lambda*m*sigma²*(gamma-1)²
    gamma_raw=(dot(r,a)+lambda*m*sigma²)/(||a||²+lambda*m*sigma²)
    gamma=clip(gamma_raw,0,2)

lambda固定1，无参数扫描；保留相对J(1)的1%改善回退规则，与旧规则同一数值地板。
lambda0必须精确退化到原I_cont2决策。响应不可分辨时保持1，不伪造方向或输出。
等价的raw_gamma=1+w*(gamma_LS-1)，w=RMS(a)²/(RMS(a)²+lambda*sigma²)。
w是无量纲响应/噪声尺度权重，不是已校准统计置信度、真实SNR或画质概率。
用当前状态R=RMS(r)/sigma、c=cos(r,a)、b=||a||/||r||表示，
gamma_raw=(c*b*R²+lambda)/(b²*R²+lambda)，w=b²*R²/(b²*R²+lambda)。
这里c比较同一步的残差与响应，并非相邻轨迹方向余弦；w也不是轨迹曲率。
增大sigma使raw倍率向原倍率1回缩，不保证总是减小倍率（原raw<1时会提高）。
局部凸二次目标的解析最优不代表最终画质最优、精确后验采样或全程收敛。

## 三条新臂及对照解释

- REG：读取当前r/a与真实sigma，在线选择gamma。
- BLIND：同一正则、同一当前r/a，但策略始终假定sigma=.05；R记录仍用真实sigma。
  REG−BLIND检验噪声水平信息。sigma=.05时两臂数学完全相同，直接复用REG输出/轨迹，新增调用0。
- TIME_NOISE：用A33旧开发图00020–25×两seed的I_cont2标量记录，恢复每元素r²、r·a、a²，
  在每个slot取十二条二次目标的算术均值（保持半正定），按实际sigma计算同一正则最优倍率。
  冻结三档30项表，不读取A36目标图状态，不用GT。它是旧轨迹上的参考响应表，
  不是新REG轨迹的精确匹配时间表；负结果可否定当前状态增量，正结果仍须更强对照确认。

A36 C、旧STATE(I_cont2)、旧TIME全部缓存保留。REG−TIME_NOISE检验当前图状态信息，
两者都知道sigma；REG−旧STATE检验新目标的实际取舍。
不修改30个位置、原ζ/D学习、forward/transpose、native精度、clamp或模型。
继承local_reference/legacy transpose与值域边界，不能当作作者实现完全一致。

## 预算与数值契约

新方法NFE0；新诊断6*(90+60+90)=1440。每候选科学总270+30=300。
只复用PRE和原29次噪声，所有新轨迹每个后续位置真实调用UNet；没有复用旧受控轨迹替代新输出。
REG/BLIND每步计算H响应，无额外UNet；额外算子和仪表时间单列，不声称等NFE等耗时。
验证父完成记录及SHA、原PRE/观测、模型/网格/参数、运行时、逐臂实际30调用、外部RNG不变。
首个中心用原sensor重算gamma/R/c/b/qRMS，与A36旧STATE第一行相对误差≤1e−6；这是标量指纹，
不是新做全向量重放。原C全向量回放门控复用A36已完成证据；新规则lambda0/二次最优由数值测试约束。
输出全部保存后才评价GT；GT还用于原合成观测，不进入倍率、stop、输出选择。
完成case复用，失败attempt保留计费。源A36的231额外中断计数只作历史成本，不计作本轮新消耗。

## 预先读数和判断

主读数：REG相对C在每档均PSNR≥+.05 dB、≥4/6图正向、平均SSIM≥−.002、LPIPS≤+.005，
才记NOISE_REGULARIZED_QUALITY_CANDIDATE_ONLY，否则混合/未支持。仍不作独立确认。
分别报告REG−BLIND及REG−TIME_NOISE；两种工程候选读数满足任一即可：
1) PSNR≥+.02、≥4/6图PSNR正向、SSIM≥−.002、LPIPS≤+.005；
2) LPIPS≤−.005、≥4/6图LPIPS改善、PSNR≥−.02、SSIM≥−.002。
上述比较还需该档REG−C主保护线通过。同sigma=.05的BLIND别名不计信息增量。
完整显示每图、三档、全部参考、最差退化；不以单一平均值遮盖画质风险。
bootstrap只按六图配对，区间描述性、未校正、不作p值或精确置信保证。
实际终局R从每case终局loss计算sqrt(2*loss)再均值，另列最后次调权前clean的R，避免混淆。
任一图/条件失败保留，不看结果扩lambda/换锚点。若REG≈TIME_NOISE，结论是噪声知情规则有效、
当前图状态独立价值未建立；只有后者有正证据才解锁新图冻结确认。

## 运行产物

/data0/lzy/ZAPS-results/diag_noise_regularized_feedback/A37_20261025。
保存三新臂终局、逐步R/c/b/w/gamma、各对照、固定ROI图、逐图摘要和SHA。
新入口独立，旧A36所有哈希绑定文件字节不变。启动器自动在允许GPU中择卡，一卡时两分片串行；
所有子进程关闭launcher锁FD，避免遗留子进程长持总锁。完成后一次输出并归档。
运行：bash run_a37_20261025.sh。Mac仅NumPy/标量检查；Torch数值测试由服务器强制执行，不能跳过。
