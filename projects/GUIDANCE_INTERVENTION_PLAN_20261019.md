# 下一轮：先验证实际引导剂量，再验证最终重建收益

日期2026-10-19。目标是尽快回答可干预的问题，保持正式调度不变。
不再扫描余弦窗口/分母，不扩展无明确假设的相关性表。

2026-10-19完成更新：A2c六例完成，新增0方法+120诊断NFE，全部回传生产门控通过。
分类LOCAL_HALF_DOSE_NOT_SUPPORTED，t903/641各仅1/3图双seed联合受益，
低于原H3至少2/3图标准。按负向分支收束简单缩放，第二/第三阶段不解锁；
不再执行下面保留的历史命令，不重跑六例。详见[A2c完成报告](GUIDANCE_ONE_STEP_SCALE_RESULT_20261019.md)。

## 第一阶段A2c：原快照的受控单步干预（已完成，原设计保留）

复用A2a六条完成轨迹及原五快照，保持同一输入、参数、先验更新及原同一次随机噪声，
只将当前实际q乘以gamma=1/.5/0。状态构造为
archived_x_next+(gamma-1)*archived_q；不在每个分支重抽噪声。
在相同下一时间点s真实运行UNet，比较clean估计的测量损失、PSNR/SSIM/LPIPS和clip。
全部分支先确定，GT只做离线评价。原λ1必须复现旧clean/q/loss/指标后才能解读效果。

该设计回答：完整修正是否在某些状态过强？缩小修正是否能改善下一状态？关闭修正时
是否也同样改善？原残差、q幅度、S_dir/kappa/Omega保留为解释上下文。
它不回答终局质量或增加步数的收益，不把局部均值改善直接升级为控制策略。

每点两次原分支和两次干预=4调用；每例20诊断NFE，6例总120，新增方法0。
先00020_s1000冒烟20，通过技术门控后自动补五例100；正/负结果均完整报告。
如果首例失败则停止，不运行其他例、不自动重试、不重训；已完成例按checksum复用。
数值约定、候选判据及结果分支已写入冻结卡：
[A2c实验卡](experiment_cards/A2c_one_step_guidance_scale_20261019.md)。

## 第二阶段：候选修正的最终图像验证（后置，另卡）

若A2c出现可重复的局部联合收益，先在明确的固定快照做配对后续轨迹，沿原网格继续到
原最终输出，保持学习参数及公共随机数。先确认最终三项质量，避免继续只优化中间量。
使用原缓存共享前缀，按真实新增模型调用计成本，不重复整段训练。
候选点/规则只能在开发数据上确定，确认图像/seed须另行审计冻结，不能把旧六例当独立测试。
具体映射、剩余步数预算和配对门控届时立卡，不在本轮自动启动。

若局部效果只改善残差却损害质量、效果不一致或没有可分辨收益，则停止当前简单缩放路线，
保留原基线；不扫描更多gamma/时间层/阈值追正结果。先解释方向/先验/裁剪机制再另立新假设。

## 第三阶段：固定网格的状态方法对照（后置，另卡）

确认终局候选后，再比较原方法、固定缩放、残差only、物理+方向双状态。
固定缩放臂隔离“只减小引导”的收益；双状态相对残差only用于判断方向状态的增量作用。
保持30步、10轮及同观测/随机协议；额外探针若进入在线计算须计方法成本，不能冒称仍300 NFE。
同时报告三项质量、失败率及实际成本，独立样本用于最终确认。自适应调度继续后置。

## 历史服务器一次运行方式（已完成，不再执行）

新代码已同步GitHub后，正常fetch/merge即可。Windows Clash与现有SSH反向转发保持运行。
下面首例与剩余串联，任一步技术失败即停；没有全局代理修改或离线包。

```bash
cd /home/lzy/ZAPS &&
git fetch origin refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
cd projects &&
python -m unittest utils.test_diag_one_step_guidance_scale -v &&
python -u utils/diag_one_step_guidance_scale.py --source-plan-dir results/diag_guidance_replication/A2a_20261018 --run-dir /data0/lzy/ZAPS-results/diag_one_step_guidance_scale/A2c_20261019 --inspect &&
python -u utils/diag_one_step_guidance_scale.py --source-plan-dir results/diag_guidance_replication/A2a_20261018 --run-dir /data0/lzy/ZAPS-results/diag_one_step_guidance_scale/A2c_20261019 --case 00020_s1000 &&
python -u utils/diag_one_step_guidance_scale.py --source-plan-dir results/diag_guidance_replication/A2a_20261018 --run-dir /data0/lzy/ZAPS-results/diag_one_step_guidance_scale/A2c_20261019 --remaining
```

服务器单测应8项全过、无skip。本机8项中6项通过、2项因缺Torch跳过；未代替GPU验证。
回贴全部POINT/CASE Gates/A2c Summary及Records即可，一阶段结束后统一归档。
原A2a、恢复记录、A2b0与共享算法不修改。本轮没有启动第二/第三阶段。
