# A50r2 下一轮服务器命令

最低提交a479de4。修正A50新增上下文对旧customcheckpoint的不兼容；r1文件保持原字节。
数学动作、20旧例、4872新前向/最多1212输入反向预算不变。r1若已产生实耗，另加不隐藏。

Mac6项NumPy/设计及语法/CLI检查通过。6项Torch与真实FFHQ均未跑；服务器先严格准入，
没有网络权重更新或GT在线选择。检查失败保留日志并停止，不改容差追结果。

```bash
cd /home/lzy/ZAPS &&
env -u NO_PROXY -u no_proxy git -c http.proxy=http://127.0.0.1:10809 fetch https://github.com/lzy75/ZAPS.git refs/heads/master:refs/remotes/origin/master &&
git merge --ff-only origin/master &&
git merge-base --is-ancestor a479de4 HEAD &&
cd projects &&
bash run_a50r2_20261026.sh
```

根目录：/data0/lzy/ZAPS-results/diag_current_score_geometry_r2；run=A50r2_20261026。
前台每60秒进度，终局集中输出并生成所有对比面板，推送轻量结果到
results/ffhq-current-score-geometry-r2-20261026。仅归档/push失败时不要重跑GPU。

本轮检验全30步当前响应对gamma的价值，采样点位置不变；相对TIME/SHUFFLE的额外改善
才能支持新增状态匹配。五旧图仍是开发数据，不是独立确认，也不证明原曲率/步长/NFE主张。
