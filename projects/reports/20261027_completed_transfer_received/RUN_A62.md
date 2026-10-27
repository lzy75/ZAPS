# A62实际blur质量：服务器一段命令

先确认A56M1/A61均已结束（本轮结果已完成）；不重跑它们。固定GPU1；其他用户进程保持不动。

```bash
cd /home/lzy/ZAPS &&
env -u NO_PROXY -u no_proxy git -c http.proxy=http://127.0.0.1:10809 fetch https://github.com/lzy75/ZAPS.git refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
cd projects &&
/home/lzy/anaconda3/bin/python -u utils/a62_session.py --gpu GPU-e21d8c3c-cde6-c10f-878d-b20a4ded1742
```

真实Torch toy物理目标/输入VJP、原checkpoint门控由入口强制执行，无Torch不skip。
10旧例×4臂×30=1200新诊断前向，最多600输入VJP；CPU求解+单GPU串行模型。
全部输出冻结后集中PSNR/SSIM/LPIPS、30点覆盖、算子与时间；结束归档/push。
断线同命令重新跟随；30秒进度，FFT当前求解每100迭代更新状态。
资源不足且无新case attempt可显式再次运行；科学失败/partial不自动重试。

状态：

```bash
/home/lzy/anaconda3/bin/python -u /home/lzy/ZAPS/projects/utils/a62_session.py --status
```

卡片：projects/experiment_cards/A62_blur_fft_quality_20261027.md。
根/data0/lzy/ZAPS-results/diag_blur_fft_quality/A62_20261027，分支results/blur-fft-quality-20261027。
