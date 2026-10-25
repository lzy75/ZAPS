# A48：用局部实测可靠性约束动作外推

planned；2026-10-26；A线，mechanism_only / development。

## 1. 问题和可否定假设

A46b中SR前三图局部实际/预测改善rho约.52/.61/.53，后3约.83/.93/.95；
低rho与部分过强倍率并存，但motion有反例。不能由此推断最终画质或因果。
有限已执行臂的guarded oracle空间SR/Gauss约.003dB，IP/motion约.117/.078dB；
不是全局上限。新问题是可靠性约束能否比仅降低同一局部目标更稳健。

可否定：相同补测动作下，rho门控改变至少2/5图的倍率，TRUST-LOCAL平均
PSNR>=.02且SSIM>=-.002、LPIPS<=+.005；相对LEGACY不低于-.05dB且守同画质线。
每任务独立判，不用SR抵消其他任务。TRUST-C另判均>=.05dB、4/5图正向并守画质线。
固定倍率可略好于TRUST，不作主目标硬否决；差距和逐图最差结果照报。

旧5开发图00446/00559/00608/00873/00951、四任务、seed1000、sigma_y=.05，20例。
父A45 `/data0/lzy/ZAPS-results/diag_balanced_state_calibration/A45_20261025`，
父A44 `/data0/lzy/ZAPS-results/diag_frozen_state_matching/A44_20261025`；必须全完成且
签名/张量/源码/模型SHA兼容。A46六图已经曝光，不作本轮独立确认或调参图。
rho阈值.75是受A46b启发的预先锁定工程假设，不是文献保证或校准置信概率。

## 2. 唯一因子、指标及固定条件

冻结A45权重SR=.5/Gauss=0/IP=1/motion=0，原P物理残差与U低秩条件目标不改。
J=(1-lambda)*(L-L1)/range(L)+lambda*(R-R1)/range(R)，range只取原0..2五点。
PHYSICAL任务lambda0不要求协方差可用，不声称协方差对其控制有贡献。
在LEGACY实际接受倍率g_old与1之间，新增1+f*(g_old-1)，f=.25/.5/.75；
与两端组成同噪声动作路径，.001量化。不是固定倍率，更不在终局挑最优图。

|臂|规则|
|--|--|
|C|已完成原生本地ZAPS参考，原300NFE|
|LEGACY|冻结A45选中规则，复用其原输出|
|LOCAL|相同五点中，物理L不劣于g1且J下降的最小实测J；无可行点回g1|
|TRUST|LOCAL再要求rho=[J(1)-J(g)]/[Jfit(1)-Jfit(g)]>=.75；无可靠点回g1|
|MATCHED_FIXED|A45已锁的任务中位倍率152.699/9.178/30.209/9.61，开发参照|

LOCAL/TRUST看到完全相同的付费探测；唯一消融因子是rho可靠性门控。
不扫描阈值。rho只限制模型乐观程度，不保证终局PSNR。
记录五点路径的粗/细两尺度方向余弦、按真实delta-gamma归一化的斜率变化，
及qRMS*delta-gamma实际输入位移。几何仅描述，尚不参与动作；不冒称它已被验证
为控制指标或真实时间轨迹曲率。若rho独立无益，先停止这一门控，不叠加调参追分。

只改k24/t319->218的q倍率；30个采样点、方差、注噪、前270调用、最后一轮参数、
任务算子和值域A(x)=H((x+1)/2)、mask/kernel/measurement/seed全部复用；不训UNet。
精度策略：诊断IEEE、续程原native；最终原uint8 PSNR/SSIM+alex LPIPS。协议完整
继承父plan并检查运行库指纹。每case动作写签名decision_before_labels再读取终局标签。

## 3. 成本、门控和失败处理

新方法训练0，新增诊断最多400=20例*20：每例重复g1一次、SR补缓存g_old最多一次、
三个路径探测最多三次、LOCAL/TRUST/MATCHED_FIXED最多3*5续程；别名只执行一次。
缓存旧输出仅同倍率、同身份复用。已支付旧传感器不能称免费；部署保守<=322NFE，
固定300；此轮400是复用缓存后的新增研究成本，不是完整在线成本。
旧训练6000及旧诊断不重复计新增。失败尝试仍占20/case预算，running保留中断成本下界。
预算耗尽停止，不擅自重置；非finite探测保留并拒绝该动作，不改输出选择。

advance锚点重放<=1e-7，g1 next-clean重复<=1e-7，原风险目标回放<=1e-6；
精度恢复、RNG不变、参数不变、无grad、调用t序列、所有身份/签名/张量SHA门控。
不改任何历史绑定文件，不覆盖父run。所有拒绝尝试计调用，决策不接收GT参数。

## 4. 处理分支和运行

质量与可靠性增量分别报告；开发阳性才安排未用图确认，不由5旧图宣称可迁移。
rho与几何可以诊断局部失配，终局未改进则否定其当前控制映射；不同任务反例保留。
门控失败不解释算法收益。无步长/选时/全程反馈/NFE节省/收敛或全后验正确性结论。

`bash /home/lzy/ZAPS/projects/run_a48_20261026.sh`，两分片/单卡可串行，前台最终摘要，
每分钟进度；20张C/LEGACY/LOCAL/TRUST/MATCHED_FIXED对照图及逐点读数归档Git，PT留服务器。
新目录 `/data0/lzy/ZAPS-results/diag_response_reliability/A48_20261026`。
Mac仅必要NumPy/语法检查，真实Torch检查服务器执行且不允许skip；不启动A47噪声试验。
