# A50 下一轮服务器命令

代码已推送master，最低提交01522b8。尚未运行服务器Torch或UNet；本机6项NumPy/设计检查和语法通过。

本轮保持原最后30个采样位置，用当前真实网络的局部Jacobian与观测残差决定每步引导倍率，
对照原C、同IEEE的C、无Jacobian、无对齐以及纯时间/固定/错配。区别于只在t319放大。
只用5张旧图×4任务做机制开发，共4872新前向、最多1212输入反向；没有新增训练。
这不是新理论或NFE节省证明，当前也没有画质收益结论。

在学校服务器执行一次，前台等待最终输出；GPU可用性检查失败不修改实验计划。

```bash
cd /home/lzy/ZAPS &&
env -u NO_PROXY -u no_proxy git -c http.proxy=http://127.0.0.1:10809 fetch https://github.com/lzy75/ZAPS.git refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
git merge-base --is-ancestor 01522b8 HEAD &&
cd projects &&
bash run_a50_20261026.sh
```

日志父目录：`/data0/lzy/ZAPS-results/diag_current_score_geometry`；结果目录：`A50_20261026`。
每60秒输出进度。若中断/数值门控/OOM失败，保留日志，不删除目录盲目重跑。
若仅结果push失败，计算完成，重试归档或push即可。完整新结果分支为
`results/ffhq-current-score-geometry-20261026`，不会合入master。

本次相对C的正反馈只说明方法候选；相对TIME/SHUFFLE的额外价值才是状态匹配证据。
五张复用图不提供独立泛化结论；能否转入新图确认由各任务独立判断。
