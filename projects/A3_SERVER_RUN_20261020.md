# A3服务器运行

新入口已实现，GPU实验尚未运行。六例×五原区间×两条预定桥路径，共180诊断NFE，方法0。
只检验同起终点的局部细化，无重训、不改变正式网格。质量判断同时看PSNR/SSIM/LPIPS及loss，
至少一个时间层须2/3图的两seed、两桥全部受益；不选最好桥或事后调整阈值。

更新代码后在服务器执行：

```bash
cd /home/lzy/ZAPS &&
git fetch origin refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
mkdir -p /data0/lzy/ZAPS-results/diag_coupled_refinement &&
cd projects &&
bash -c 'nohup bash run_a3_20261020.sh >> /data0/lzy/ZAPS-results/diag_coupled_refinement/A3_20261020.log 2>&1 < /dev/null & echo "A3 PID=$!"'
```

使用已修好的SSH/Clash链路和当前Python环境，不更改认证配置。启动器要求14项检查全部通过、
不得skip，再核对六例来源和冻结文件、绑定新plan。首例技术通过后自动补五例，科学负结果不删余例。
GPU默认cuda；需要指定时可在启动前`export ZAPS_A3_DEVICE=cuda:0`，不自动占用另一张卡。

```bash
tail -n 12 /data0/lzy/ZAPS-results/diag_coupled_refinement/A3_20261020.log
```

完成后只回贴`A3 Summary`及任何报错即可。摘要也可只读获取：

```bash
cd /home/lzy/ZAPS/projects
python -u utils/diag_coupled_refinement.py --summary-only
```

输出目录`/data0/lzy/ZAPS-results/diag_coupled_refinement/A3_20261020`。
nohup可跨SSH终端断开保留进程，但不能抵御服务器重启或进程被终止。两级锁避免重复运行；
完整兼容case复用，failed保留成本并停止，不自动覆盖或重训。运行期间不要更新此冻结入口/卡。
GT在全部分支及网络调用确定后才加载。真实raw Z、辅助draw、中间输入、F端点逐点持久化，
每例先核对29次原draw及冻结场端点，技术失败先处理门控，不解读画质。
本机9项通过、5项Torch未执行；所有真实Torch/CUDA门控由服务器完成。
