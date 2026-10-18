# A1b回传与单图指标测量阶段结论

日期2026-10-18。证据：用户回贴五个Snapshot、十组Pair、全部Gates及Summary。
初稿依据服务器回传；现已拉取结果提交711da42并独立核对文本归档。
实际运行commit为bdc165a6aa524cb58d891cfafd733d187a2560e1，本run.json SHA256为
7e60a041cc53f0766a1b657389ead84820a038f7a70daef97088e5674260f2c6。
完整五点十组rows、gates、summary均与回传一致；未在本机重跑GPU或读取服务器张量。
[归档审计及CSV换行说明](STATE_INDICATOR_ARCHIVE_AUDIT_20261018.md)
[原始回传转录](experiment_cards/evidence/A_guidance_local_probe_20261018_user_report.json)
[事后单位分解](experiment_cards/evidence/A_guidance_local_probe_20261018_units.json)

## 1. 来源、门控和预注册判定

- A1a父run：`/home/lzy/ZAPS/projects/results/diag_cached_guidance_geometry/20261018_160358_781525/run.json`
- 父SHA256：`aafc669728477d1ba241b1e4dcaed7ed05e605cb120ba10f0669b6789dd191e9`
- A1b Records：`/home/lzy/ZAPS/projects/results/diag_guidance_local_probe/20261018_162902_899938`
- 中心存档、固定参数/网格、RNG、原文件身份、30次预算门控全部passed；
  五点各两次中心clean/q/loss误差均为0，重复噪声地板也均为0。
- 5/5点方向读数可分辨且两尺度一致，5/5点为有限尺度下降、0个上升。
  超过原计划至少4/5可用点要求，分类PROBE_READABLE_CANDIDATE。
- 新增诊断UNet NFE=30，新增方法NFE=0；没有优化、新采样轨迹或GT读取。
  原输出26.70129265/.82015914/.16083704不变。

预定目标u=.1/.3/.5/.7/.9映射到实际u=.095565/.307959/.491358/.695879/.814627。
最后一个是t14，不是实际u=.9；本实验没有探测t999或t0的局部响应。

## 2. 两尺度结果：当前修正跨度内的方向响应

S_dir=||normalize(q_plus)-normalize(q_minus)||/effective_span，
其中x_plus/minus=x±ηq0，η=.01/.02，t及原ζ/D固定；不把η=1处行为外推出来。
Phi=[L(F_t(x_plus))-L(F_t(x_minus))]/effective_span。
损失是原clamp之后的实际unit测量损失mean(r²)/(2*sigma_y²)。

| t（k） | 原q RMS | S_dir (.01 / .02) | Phi (.01 / .02) | 原裁剪比例 |
| --- | ---: | ---: | ---: | ---: |
| 903（8） | .719562 | 1.056334 / .991327 | -8.211106 / -8.143481 | .990946 |
| 641（19） | .273398 | .706957 / .706541 | -8.420268 / -8.419678 | .897497 |
| 319（24） | .009788 | .059581 / .059024 | -.068772 / -.068946 | .076752 |
| 54（27） | .005027 | .019593 / .019382 | -.027442 / -.027431 | .023158 |
| 14（28） | .004802 | .017641 / .017568 | -.030958 / -.030930 | .010198 |

方向两尺度相对差最大6.1541%，损失斜率相对差最大.8236%，均小于预注册20%。
所有十组实际浮点跨度/方向表示均通过1%精度检查。
十个正向扰动的损失均低于各自中心，负向扰动均高于中心；本次不只看到
负中央差分，也直接看到测试小幅正向修正的测量一致性改善。

这支持：在这五个存档输入及所测小尺度上，实际近似q具有物理损失下降性质。
不认证完整似然score、无限小梯度、参数近似的全面正确性或实际完整一步的下降。
两个早期点原裁剪比例很高，探针会改变裁剪活动集合；这是原clamp系统的局部
响应，不能称“去除裁剪后的纯网络曲率”。末期clip变化小，也不能忽略模型/任务依赖。

## 3. 必须拆开修正幅度与单位位移敏感度

当前S_dir沿ηq测量，包含q的实际幅度。令a=RMS(q0)，在局部线性近似下：

    S_dir ≈ a * k_dir
    k_dir = ||D_x(normalize(q))[q0/a]||.

q0/a的RMS为1，所以k_dir是沿当前引导方向“每单位状态RMS位移”的角度响应，
不是全方向Jacobian范数、Hessian曲率或全局Lipschitz常数。
用已保存S_dir/a作单位分解，不调用网络，也没有重新挑尺度/窗口/阈值。
此分解是事后解释，不算新的独立验证或性能成果。

| t | k_dir (.01 / .02) |
| --- | ---: |
| 903 | 1.468025 / 1.377681 |
| 641 | 2.585813 / 2.584291 |
| 319 | 6.087489 / 6.030553 |
| 54 | 3.897454 / 3.855371 |
| 14 | 3.673671 / 3.658356 |

S_dir随这五个快照显著降低，但单位位移响应不单调降低，所测点中t319最高。
因此准确结论是“后期当前修正跨度内方向变化较小”，不是“后期场本身最平坦”。
若只看κ或只看角度，都会漏掉实际修正幅度；若只看S_dir下降，也会把幅度变小
误解释成所有局部几何都变简单。对当前研究，幅度×方向敏感性的解释更贴近
“该次修正是否粗大”的问题，但仍不能据此认证应该增加采样步数。

## 4. 与A1a合并：本阶段解决了什么

| 问题 | 当前证据 |
| --- | --- |
| 相邻引导方向是否有阶段结构 | A1a支持：早期回摆、后期连贯且幅度变小 |
| 固定t后，沿q的方向响应是否可测 | A1b五点、两尺度均支持 |
| 实际q是否在测试尺度降低真实复合物理损失 | 五点十组正向扰动均下降 |
| S_dir变小是否就是单位位移敏感度下降 | 不成立，单位分解显示不单调 |
| 当前方向指标能否补充残差预测重建风险 | 未建立，需要同阶段跨图/seed检验 |
| 高读数位置是否更受益于细步进 | 未验证，需要明确ζ/D分配和噪声耦合 |
| PSNR/SSIM/LPIPS是否提高 | 本轮没有新重建，原输出不变 |

本阶段新增诊断UNet成本为A0的90+A1b的30=120，A0b/A1a及事后统计均为0。
历史父方法300 NFE单列，不将离线探针写成最终算法免费组成。

单图指标测量阶段现可收束。保留引导方向变化、修正幅度及物理残差的分工；
停止在同图上继续扫描窗口/频段/阈值。下一阶段先固定少量其他图像和第二seed，
核查其是否参与过调参，再验证阶段条件下的增量信息；只有建立候选机制才做
局部细步反事实。当前不改ζ、调度、采样噪声，不新增GPU扫描。

## 5. 一次性服务器归档

2026-10-18更新：轻量文本/曲线归档已同步并完成审计，结果提交711da42。
两次导出内容重复，只计一份实验；六个CSV经可逆换行恢复后精确匹配服务器hash。
原张量/权重仍在服务器，未作二进制备份。以下命令保留为历史操作记录，当前无需重跑。
独立归档入口utils/archive_state_indicator_stage.py沿A1b→A1a→A0→父run验证
status和source SHA，并附带同一A0来源、已完成且门控通过的A0b记录。
复制JSON/CSV/实验卡/曲线；binary张量不上传，清单保留其原路径、大小及可用hash。
这与此前20261013结果分支的轻量归档约定一致，不称完整二进制备份。

```bash
cd /home/lzy/ZAPS/projects
python -m unittest utils.test_archive_state_indicator_stage -v
python -u utils/archive_state_indicator_stage.py \
  --probe-dir /home/lzy/ZAPS/projects/results/diag_guidance_local_probe/20261018_162902_899938 \
  --inspect
```

四项测试通过、来源检查成功后执行：

```bash
python -u utils/archive_state_indicator_stage.py \
  --probe-dir /home/lzy/ZAPS/projects/results/diag_guidance_local_probe/20261018_162902_899938 \
  --results-branch results/state-indicators-20261018 &&
git push origin results/state-indicators-20261018
```

脚本用临时Git index和compare-and-swap只更新results分支，不切换当前工作分支，
不包含其他已暂存改动，不推送网络、不覆盖原结果。以上git push由用户手动执行。
回传Archive、Manifest SHA256、Results commit和push结果后，再拉取审计完整文本归档。
