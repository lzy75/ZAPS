# A56M1 原FP32显存资格与冻结ImageNet连续运行

状态：独立实现，真实模型结果待服务器。本卡落实此前A56M设计，不改旧卡或科学源。

## 目标与协议

不再等未经测量的12GiB门槛。只在已分配卡空闲≥8GiB时，尝试一次原ImageNet256/FP32
的有上限计量，测出原优化unroll及当前U方向的实际峰值。8GiB只是允许一次计量，非质量准入。
先验证并安全接管idle且无ImageNet attempt的旧队列；FFHQ12例原完成记录必须全通过旧门控。
若旧任务已开算则不打断。旧队列日志/状态保存；只停止经uid、命令、PID/PGID与子进程检查的自有等待器。
原20例计划、6/4图像、两个任务、sigma=.05、seed、30点网格、FP32、算子、基线与判定均不改。

计量案例固定为冻结ImageNet第一个SR案例00067，不能按GT或已见画质换图。
第一条：原SnapshotZAPS优化num_epochs=1、30个前向（只测资源，不当原300调用质量基线）。
第二条：固定初值zeta=.1/D=.2的独立U路径，原30点、独立seed配对9×29创新，30个中心前向。
在t319/t14调用既有实际VJP计量，各6个前向/1个输入VJP。
总上限72诊断前向、32输入VJP；优化损失反向/Adam步数单列实际计数，不与输入VJP混计。
记录各阶段peak allocated/reserved、CUDA空闲、耗时、OOM位置及所有尝试；参数权重不更新、不填.grad。
不读取GT质量、不用已学zeta/D驱动U、无新超参扫描。

## 通过、负向和资源门控

两条路径有限、U全部30点解决，原真实VJP/grad-mode门控全过；模型权重grad/RNG/未学习参数不变。
计量峰值reserved加至少1GiB余量必须≤当前可供本进程的free+reserved空间。
一epoch原优化不是全十epoch峰值保证：接着先做唯一固定原300调用SR案例，原全部门控过后
才继续其余7例。均串行单卡，沿用原training/sample函数、原attempt目录，完整结果可复用。
任何case有失败/残缺attempt，或profile OOM/数值失败即停止；不自动撞OOM、不改阈值追分。
当前空闲不足或安全余量不够，生成资源受限报告并返回，不无期限等待。
计量通过但后续资源下降时不启动下一案例；完整已有案例保留，余下可用同命令恢复且不重做计量。

## 预算与范围

新计量≤72前向+32输入VJP；原ImageNet8例剩余3852前向+≤482输入VJP。
本次最大新费用3924前向/514输入VJP，参数损失反向及算子成本另计。
原FFHQ5772前向不重跑。若原IN已有完整案例，仅复用原finished/baseline_done验证后扣减剩余费用。
原FFHQ已按用户要求提前评价：完整输出的all20-blinded标志须显式改为不宣称，原SHA文件不改。
所有IN输出完成后才评价IN；原FFHQ评价不用于调ImageNet参数。原完整汇总通过适配器附加评价时序修订。
不静默fp16、缩图、换checkpoint或启动CPU卸载。资源失败后，CPU卸载需下一独立等价资格实验。

独立计量目录：`/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/A56M1_20261027`。
原ImageNet重建仍存原冻结A56S2目录；附加本资源实现/资格manifest引用，保持原科学身份。
轻量结果分支：`results/a56m1-imagenet-qualification-20261027`。
入口：`python -u utils/a56m_imagenet_session.py`，detached且前台跟随；--status只查看。

