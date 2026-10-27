# A56 FFHQ 已完成队列提前汇总

状态：待服务器CPU评价；不是新的去噪实验。用户在ImageNet持续资源等待后要求先输出FFHQ。

## 目的、范围和协议时间变更

原20例计划与科学源不动。仅在固定的12例FFHQ全部通过原 finished/baseline_done
门控后评价所有六臂；6张新图 × SR/IP、seed1000、sigma=.05，原30点。
参照：selection_r2_20261027/A56_selection_r2_20261027。不得用完成子集替换缺失的FFHQ案例。

按用户本次明确要求，将评价时机改为“FFHQ12例输出全部冻结后”，不再等待ImageNet8例。
这是评价时间顺序的显式修订，**不能再声称完整双数据集一直保持20例全部输出后才首次看GT**。
原计划、源记录、PT、队列源码与既有标志不覆写；原根目录新增一次性early_FFHQ_evaluation.json
旁注，未来完整归档必须连同该旁注解释历史all20标志。ImageNet方法/样本/规则已在GT前冻结，
不得依据本次FFHQ指标修改ImageNet后仍称原冻结确认。

## 指标、成本和判定

- CPU读取已存outputs.pt；使用原 utils.metrics 的uint8 PSNR/SSIM和alex LPIPS。
- 新UNet前向0、新输入VJP0、训练更新0。LPIPS是CPU指标网络，72次指标评价另列。
- 保留原A56工程线：U对C_IEEE平均PSNR≥+.05dB、SSIM≥−.002、LPIPS≤+.005，
  ≥4/6图PSNR正向，U/V/Q/TIME四路径全部30/30解决。方向迁移对V/Q分别同样检查。
- TIME完整列出，不事后加新阈值；C_REPLAY/C_IEEE仍是本地实现参照，不称作者基准。
- 全量每图数值、最差值、原NN/VJP计数、CPU评价成本、基线与U/V/Q/TIME对比图归档。
- 仅支持或否定FFHQ两任务迁移；不代表四FFHQ任务、多数据集、曲率控制或NFE节省已成立。

独立输出：`/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/ffhq_readout_20261027`。
轻量结果分支：`results/a56-ffhq-early-readout-20261027`；PT留原位置，原队列可继续等待。
入口：`python -u utils/a56_ffhq_readout.py`。重复调用校验并复用已完成评价，不调用去噪器。
任一原case/签名/字节/样本/门控不兼容即停止，不修数据、不挑图、不自动重跑GPU。

