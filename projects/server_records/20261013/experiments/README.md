# 实验归档

每次运行 `main_single.py` 自动写入：

```
experiments/
  experiments.csv          # 主索引：一行一次运行（参数 + 任务 + 时间 + 指标 + 代码版本）
  <exp_id>/
    gt.png  observed.png  recon.png   # 重建图归档（git 忽略）
    run.json                          # 完整参数 + 指标 + git 信息
    conclusion.md                     # 预填指标，结论/观察待人工补充
```

`exp_id` = `YYYYmmdd_HHMMSS_<task>_<dataset>_<imgstem>`。

## 代码版本 ↔ 实验关联

CSV 每行含 `git_commit`（运行时的 HEAD 短 hash）与 `git_dirty`（工作区是否有未提交改动）。

推荐工作流：
1. 改完代码先 `git commit`（保证 `git_dirty=False`，结果对应纯净 HEAD）；
2. 跑实验 → logger 自动把当前 commit 写进 CSV 行与 `run.json`；
3. 可选 `git add experiments/experiments.csv experiments/<exp_id>/run.json experiments/<exp_id>/conclusion.md && git commit`，把索引与结论并入历史。

回溯：拿任一 CSV 行的 `git_commit` 执行 `git checkout <commit>` 即可回到产生该结果的确切代码。
`git_dirty=true` 表示当时有未提交改动，结果不完全对应该 commit。
