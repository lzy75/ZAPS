# A2a首条回传：00020_s1000

状态：用户回传生产完成，1/6条；完整服务器文件尚未下载审计。
[结构化转录](experiment_cards/evidence/A_guidance_replication_00020_s1000_20261018_user_report.json)。
这是新的FFHQ00020固定方法输出，不与00000的分数作跨图收益比较。

## 已确认的回传内容

- status complete，30/30有效q，5/5可读快照，5个有限尺度下降分类、0个上升分类。
- training、observer、archive、缓存转移、probe和身份/预算门控全部passed。
  两次重复及观察器/原输出相对误差均0；10项中心clean/q/loss误差均0；参数/网格/RNG门控通过。
- 方法300 NFE、诊断120 NFE，总420。PSNR=25.123948614331034、SSIM=.7211341261863708、LPIPS=.2285727858543396。
- 总阶段仍为INCOMPLETE_PILOT；此回传未包含服务器32项单测输出、plan/run SHA或完整相邻q序列。
  不把生产门控等同于本机重放或完整文件审计，也不能据此判定完整H1通过。

## 读数的含义

| t | S_dir (.01 / .02) | 每单位状态RMS响应 (.01 / .02) | 原clip比例 |
| --- | ---: | ---: | ---: |
| 903 | 1.00911 / .934419 | 1.34208 / 1.24274 | .993118 |
| 641 | 1.39368 / 1.39070 | 5.67711 / 5.66497 | .698629 |
| 319 | .0616357 / .0611757 | 6.19793 / 6.15167 | .00575765 |
| 54 | .0185442 / .0184626 | 3.63198 / 3.61600 | .0000254313 |
| 14 | .0167048 / .0166474 | 3.42535 / 3.41358 | .0000203451 |

本表为终端舍入值。t641的当前q跨度响应高于t903，不能要求敏感度随去噪单调下降。
同时，单位位移响应在所测五点中仍以t319最高，与开发图00000的五点排序峰值位置一致；
这只是两图的有限快照观察，不认证普遍的高曲率区或应增加步数。

该图t319裁剪比例约.00576，明显低于开发图同点约.07675，单位位移响应仍较大；
这提示应同时保留clip、q幅度和方向响应，不能仅用一个阶段或裁剪读数替代全部局部几何。
目前没有隔离这些因素的因果效应。

十个Phi均负，支持所测两尺度的中央差分下降分类。此次回传没有plus/center/minus
三个实际loss，不能进一步声称十个正向扰动各自都低于中心；完整文件归档后再核实。
本轮无状态干预，三质量指标只是新图的固定参考，不是创新方法增益。

## 后续执行

按原冻结卡补齐剩余五条，不改源码、尺度、窗口、阈值、调度或样本。
剩余1500方法+600诊断=2100 NFE，完整阶段仍为1800方法+720诊断=2520。
继续条件是本次技术门控通过，不要求后续读数符合已见曲线或全部Phi为负。

```bash
cd /home/lzy/ZAPS/projects
set -o pipefail
python -u utils/diag_guidance_replication.py \
  --plan-dir results/diag_guidance_replication/A2a_20261018 \
  --remaining --device cuda \
  | tee ../guidance_replication_A2a_remaining_20261018.log
```

首条完成记录会核对hash后复用；不重新prepare，不改已冻结实验卡。
回传最终A2a Summary、六条Gates和五点表。还需同t的描述性关联，用于分开比较
Omega3与q幅度对S_dir及单位位移响应的关系；以下只读取JSON，新增NFE=0：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('results/diag_guidance_replication/A2a_20261018/summary.json')
s = json.loads(p.read_text())
assert s['completed_cases'] == 6, 'Wait for all six cases; no partial association interpretation'
print('t eta predictor target n spearman')
for r in s['same_t_descriptive_associations']:
    print(r['t'], r['eta'], r['predictor'], r['target'], r['n'], r['spearman'])
PY
```

每个固定t最多6条记录，但只有3张图，不做独立6样本显著性或控制收益结论。
阶段结束后一次性归档原始文本、表格和曲线；当前不追加GPU扫描。
