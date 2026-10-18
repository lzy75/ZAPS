# 状态指标下一阶段：三图、双种子确认

状态：六条已全部回传complete，第四条独立恢复及所有生产门控通过，H1分类PILOT_READABILITY_SUPPORTED。
实际1800方法+840诊断=2640 NFE，包含原中断与额外恢复成本。
当前不再续跑GPU；见[完整结果](GUIDANCE_REPLICATION_RESULT_20261018.md)及
[一次性轻量归档](A2A_ARCHIVE_AND_NEXT_20261018.md)。原故障及恢复过程保留在
[迁移与恢复命令](A2A_STORAGE_RECOVERY_20261018.md)。
[首条结果](GUIDANCE_REPLICATION_SMOKE_RESULT_20261018.md)保留历史观察。本机32项检查中23项通过、
9项因缺PyTorch/PyWavelets跳过；服务器32项单测输出未回贴，不认证其已全部通过。
以下初始同步/prepare/旧remaining命令保留作历史操作记录，当前不要重复执行。
完整预注册条件：[A2a实验卡](experiment_cards/A_guidance_replication_20261018.md)。

## 当前结论

1. 物理残差可作为过程主读数；单图与PSNR负相关强，但不能用残差单独保证感知质量。
2. 原clean增量余弦与后续质量改善关联弱；余弦转正不等于重建更好，也不能把负值一律当成有害震荡。
3. 实际引导q的方向序列更贴近当前研究问题。A1a高/中/低段相邻余弦中位数约
   -.712/-.097/.986，Omega3约109.63/50.53/3.22，说明该轨迹早期回摆、后期连贯。
4. A1b五点两尺度均可读；十组小幅正向扰动均降低实际测量损失。后期当前q跨度内的
   响应小，但S_dir/RMS(q)不单调，五点中t319最高，不能称末期场本身最平坦。
5. 尚未证明这些方向读数在残差之外提供稳定辅助信息，或能指导多分配步数、改善画质。
   最近测量阶段新增方法NFE=0、诊断NFE=120；最终图仍为26.7013/.8202/.1608。
   [阶段报告](STATE_INDICATOR_STAGE_REPORT_20261018.md)及[归档审计](STATE_INDICATOR_ARCHIVE_AUDIT_20261018.md)。

## 本轮要解决的问题

固定三张候选图00020/00021/00022，采样种子1000/2000，每图共用同一观测。
先审计可访问历史记录，命中用过的ID或缺图即停止，不自动换图追正结果。
不称最终独立测试集。原00000–00019已被反复探索，不再用于独立性声称。

每条运行同一300 NFE固定基线，再用90 NFE验证末轮回放、30 NFE测原五快照。
首条420 NFE；六条合计1800方法+720诊断=2520 UNet调用。
固定调度、30步×10轮、lr=.01、原ζ/D训练、eta1、原clamp及unit观测协议。
观测seed=1020/1021/1022，采样seed=1000/2000；不同时改变两个随机来源。

主判据：至少5/6条轨迹各有>=24/30有效q，且>=4/5快照可读。Phi为正也照常报告，
不把局部上升当作可删除的“坏样本”。这只证明可测性，不证明正向干预有效。
固定t分别给出Omega3及qRMS与S_dir、单位位移响应的两尺度描述性秩相关；
只有3个图像簇，不做显著性或增量预测认证，不使用全时间点混合相关追结论。

## 服务器命令

先同步代码并运行检查：

```bash
cd /home/lzy/ZAPS
git pull --ff-only origin master
cd /home/lzy/ZAPS/projects
python -m unittest \
  utils.test_diag_guidance_replication \
  utils.test_diag_guidance_local_probe \
  utils.test_diag_cached_guidance_geometry \
  utils.test_diag_cached_direction -v
```

32项全部通过、无skip后，冻结计划（0 UNet NFE；仅首次执行）：

```bash
python -u utils/diag_guidance_replication.py \
  --prepare \
  --plan-dir results/diag_guidance_replication/A2a_20261018 \
  --anchor-dir /home/lzy/ZAPS/projects/results/diag_guidance_local_probe/20261018_162902_899938 \
  --data-dir /home/lzy/FFHQ/00000/00000
```

Prepared成功且无历史冲突后，首条冒烟运行：

```bash
set -o pipefail
python -u utils/diag_guidance_replication.py \
  --plan-dir results/diag_guidance_replication/A2a_20261018 \
  --case 00020_s1000 --device cuda \
  | tee ../guidance_replication_A2a_first_20261018.log
```

回贴测试结果、Prepared、CASE、A2a Summary及五点表。首条完整门控通过、status complete
后可补齐剩余五条；可读性不理想或Phi为正也不换图，仍按原卡披露。若技术门控失败则停止修接口。

```bash
python -u utils/diag_guidance_replication.py \
  --plan-dir results/diag_guidance_replication/A2a_20261018 \
  --remaining --device cuda \
  | tee ../guidance_replication_A2a_remaining_20261018.log
```

`--remaining`核对已完成case的计划、源码、观测和文件hash后复用，不重跑首条；
failed/running/文件损坏时停止，不覆盖。不重复执行--prepare。阶段结束后统一轻量归档一次，
本轮先回贴摘要，不逐case提交Git。run.json保留完整轨迹/探针/每次UNet调用，
summary.json保留同t关联，原张量留服务器。

## 后续实验的顺序

| 阶段 | 固定比较 | 允许结论与停止条件 |
| --- | --- | --- |
| A2a（现在） | 同一协议、3图×2采样seed，原指标与五点探针 | 可测性和阶段结构是否迁移；若失败不扫描窗口/η补通过率 |
| A2b（另卡） | 更多预先固定图，按图留出；残差/噪声/clip/q幅度基础模型，与加方向状态模型比较 | 同阶段增量信息；需同时看S_dir及S_dir/RMS(q)，不把幅度效应全叫曲率 |
| A3（另卡） | 继续固定30步；原方法、残差only、方向only、双状态，固定总300方法NFE及同y/noise | 验证指标是否改善重建；同时报告PSNR/SSIM/LPIPS，未建立触发依据前不选权重做扫描 |
| A4（后置） | 同起终状态粗细步反事实，预先固定累计ζ预算、D映射与DDPM噪声耦合 | 判断高状态读数处是否更受益于细步；通过后才讨论改调度 |

A3优先研究固定网格下引导幅度的信任域/冲突约束，具体公式和对照须由A2证据再固定。
这是研究候选，当前没有实现或授权自动启动这些后续干预，也没有预设它们必定提升指标。
