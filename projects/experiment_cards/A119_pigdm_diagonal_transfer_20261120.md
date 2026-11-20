# A119 · 含噪 ΠGDM 的等预算方向细化

**状态：等待服务器执行。** 比较同一个观测收缩目标上的标量与对角更新，检验对角收益能否离开 ZAPS。冻结 A85R1 的 FFHQ 三张开发图、运动核、`sigma_model=.05`、逐图观测、权重和各随机种子。任何模型或观测缺失均停止。该实验不改变 DPS 或 DAPS。

- 三组独立运行 A85R1 原更新、原闭式标量、从同一标量解出发沿三级 db4 非零支撑做一步对角更新。只影响加性观测修正。采样网格、去噪、原引导方向的构造、噪声注入、最终输出保持原流程。
- 每图每组 100 次前向及 100 次基线输入 VJP，合计 900 次前向及 900 次输入 VJP。原实现数值测试单列合成小模型调用。额外操作仅为物理前向、伴随和小波变换，分开记录。报告含加载观测、建立算子、完整采样及 CPU 同步的时间。
- 比对原版和标量的历史输出，验证输入和终点随机数指纹、去噪调用和网格完全配对。每个对角步骤核对同目标代理不增及标量锚点；离线记录干净观测误差差值，真值既不参与更新也不筛选结果。
- 对角相对标量平均 PSNR 至少 +0.10 dB、至少 2/3 图胜、平均 SSIM 不降且 LPIPS 不升，才记为开发集正向。标量相对原更新独立判定。仅有代理降幅或指标取舍时，报告机制与终局的差别。数值或历史重放门控失败时保留记录，停止，不自动重试。
- 下一步：通过后在 DPS 做同结构的等预算试验，再评估另一任务。未通过时先检查逐步代理与干净观测误差一致性，先保留全部结果再决定是否调整目标。

```bash
cd /home/lzy/ZAPS/projects
env -u MKL_SERVICE_FORCE_INTEL MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=0 \
  /home/lzy/anaconda3/bin/python -u -m utils.diag_pigdm_diagonal_transfer
```

结果路径为 `/data0/lzy/ZAPS-results/diag_pigdm_noisy_transfer/A119_ffhq_20261120`。加 `--archive` 同步完整轻量记录到 `results/pigdm-diagonal-transfer-20261120`。
