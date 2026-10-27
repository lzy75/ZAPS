# A56 FFHQ独立CPU结果读取

先读完全部12个已完成的冻结FFHQ案例；不等待ImageNet，不启动UNet，不重跑GPU。
CPU评价后输出summary、REPORT.md、per_image.csv和12张对比图，并推独立结果分支。
原20例计划及现有队列源码不动；按用户授权注明提前评价时序。

```bash
cd /home/lzy/ZAPS &&
env -u NO_PROXY -u no_proxy \
git -c http.proxy=http://127.0.0.1:10809 \
fetch https://github.com/lzy75/ZAPS.git \
refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
git merge-base --is-ancestor dfd3ea7 HEAD &&
cd projects &&
/home/lzy/anaconda3/bin/python -m unittest utils.test_a56_ffhq_readout -v &&
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/lzy/anaconda3/bin/python -u utils/a56_ffhq_readout.py
```

输出目录：`/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/ffhq_readout_20261027`
结果分支：`results/a56-ffhq-early-readout-20261027`
ImageNet下一步设计见`projects/experiment_cards/A56M_imagenet_memory_admission_20261027.md`。
该计量入口尚未实现，本轮只运行上述真实CPU报告入口。
