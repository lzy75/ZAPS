# ZAPS 基线与时间步趋势核对

## 目的与已有证据

论文 ImageNet 定量表报告 Gaussian Deblur 和 4x SR，ZAPS 分别为
22.45/23.82 dB、SSIM 0.682/0.718、LPIPS 0.225/0.186。
这不是 ImageNet 随机修复的参照。已有 ImageNet SR 10 张图均值为
19.3007 dB、SSIM 0.4350、LPIPS 0.4732；图像集合不同，不能当成
同样本的严格差值，但这组结果尚不足以宣布论文基线复现成功。

论文 FFHQ SR 的 300 NFE 调度对照为：

| 调度 | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Uniform | 26.29 | 0.729 | 0.119 |
| Irregular | 26.63 | 0.768 | 0.104 |

本地已报告的 10 张 FFHQ SR 对照为：

| 调度 | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Uniform | 28.3043 | 0.8438 | 0.1303 |
| Irregular | 28.1905 | 0.8360 | 0.1265 |

本地对照使用 lr=0.01、legacy_bicubic 映射、固定 skip 方差和最后一轮
优化输出。均匀 PSNR 高 0.1138 dB，但 LPIPS 差 0.0038。这个排序没有
复现论文的 PSNR 趋势，不能解释成“所有指标下均匀更好”。本地子集与
论文 1000 张均值不同，绝对值更高也不能说明已经超过论文。

## 第一阶段：ImageNet 单图两个任务

```bash
cd ~/ZAPS/projects
python -u utils/diag_zaps_paper_task_schedule.py \
  --dataset imagenet \
  --image /home/lzy/imagenet/256x256/00001.png \
  --tasks gaussian_deblur super_resolution \
  --learning-rates 0.001 \
  --sr-transpose exact \
  --device cuda \
  --seed 1001 \
  | tee ../imagenet_two_task_schedule_audit.log
```

各任务仅比较 15/10/5 不规则调度与全局均匀 30 点调度。固定每轮 30 次
模型调用、10 轮、300 NFE，共同学习 zeta 和 D，固定 DDPM skip 方差。
同一任务下 y、x_T 和按步位置生成的 DDPM 随机数相同。高斯任务 zeta
初值为 0.2，SR 为 0.1，D 初值都是 0.2。sigma=0.05 保持当前 [-1,1]
域约定。这是现有实现的基线审计，不宣称所有实现细节已经与原文一致。
不运行状态调度，不改变核心算法，不依据单图数值选择论文基线。

## 第二阶段：核对 FFHQ 学习率与调度交互

```bash
cd ~/ZAPS/projects
python -u utils/diag_zaps_paper_task_schedule.py \
  --dataset ffhq \
  --image /home/lzy/FFHQ/00000/00000/00000.png \
  --tasks super_resolution \
  --learning-rates 0.001 0.01 \
  --sr-transpose legacy_bicubic \
  --device cuda \
  --seed 1000 \
  | tee ../ffhq_lr_schedule_audit.log
```

保留先前 FFHQ 调度比较的算子设置，只做学习率 x 调度 2x2 配对。若两个
学习率下排序不同，支持“调参改变调度排序”的解释；若均匀在两个学习率
下都更好，不能把排序差异归因于学习率，下一步需核对前向算子、引导映射、
输出策略和图像子集。legacy_bicubic 是历史引导映射，不称为精确 H^T。

## 输出和判读

脚本每完成一条轨迹就保存 run.json、metrics.csv、PNG，且保存每个任务
实际使用的 measurement.pt。run.json 包含提交号、完整 ZAPS 参数、实际
时间步、各轮 loss、最终 zeta、模型路径和任务参数。PSNR/SSIM 沿用已有
uint8 评估方式，同时记录未量化的 float_psnr，避免日志 PSNR 与最终指标
的细小差异被误判为算法变化。

最终表的差值统一为 Irregular - Uniform：PSNR/SSIM 正值、LPIPS 负值
支持不规则调度。单图用于检查流程与方向；只有在参数和算子约定确定后，
才用固定的 10--20 张图验证平均趋势。这里没有 ImageNet 不规则一定胜过
均匀的预设，因为论文展示的直接调度消融是在 FFHQ 上。

当前 ImageNet 随机修复的 10/15 步增益仅作为探索记录，不能替代这两个
正式任务的复现审计，也不能证明 ImageNet 论文基线已经复现。
