# A1b：固定时间快照的引导方向局部敏感度

日期2026-10-18，工作线A，状态completed（生产结果已回传，轻量归档已审计）。
A1a门控已由用户回传passed，读数可分辨，但控制有效性未建立。
完整定义：../GUIDANCE_STABILITY_DESIGN_20261018.md §3。

## 1. 问题与身份

目标：区分轨迹上方向变化与固定t的输入敏感度，并核验q的实际局部下降性质。
复用A1a同一completed父run的x、y、PRE ζ/D、模型和所有hash，不按质量挑快照。
当前绝对Records待inspect绑定；历史终端摘要不代替原始张量和权重身份。
假设：沿当前q的方向响应可由两尺度中央差分分辨，且能指出阶段差异。
若可靠Phi>0，否定该点q是复合物理损失局部下降方向的解释。
只允许mechanism_only，不允许更优重建、完整Hessian或已建立自适应步长结论。

## 2. 唯一因子与固定项

唯一因子是快照输入x→x±ηq，η=.01/.02，四点均评价；不挑有利尺度。
5个u目标(.1,.3,.5,.7,.9)，各取最近原网格点，平局取较早k，须唯一。
每快照固定t、i、ζ_i、D_i、y、A/T、clamp和真实冻结权重；每次重新预测epsilon，
不能用旧epsilon构造假探针。其他协议全继承A1a，eval网络，不执行随机转移。
GT不参与，不训练、不形成新完整轨迹，也不改变原输出。

## 3. 预算和门控

每点2次原输入重复+4次扰动，共30新增诊断UNet NFE；方法新增0，父历史300。
所有失败/重复额外调用计数，超出30不自动追加。无需UNet输入反传。
原输入clean/q必须匹配存档；沿用父回放校准的精度界限，不能事后放宽。
实现固定原输入clean/q的relative ceiling=1e-6；loss绝对误差≤1e-7+2e-5*abs(loss)，
clip比例差≤1e-5。记录浮点扰动实际投影跨度；相对请求跨度误差或正交量化误差
超过1%则该点记输入表示不确定。这些是新探针执行前固定的数值准则，不是画质阈值。
记录实际浮点扰动跨度、所有clip活动集合变化、S_dir、L_path及Phi。
信号需>10倍原输入重复噪声；可分辨两尺度相对差≤20%，Phi符号一致；
不足分辨率或尺度不一致记inconclusive，不扩大η追信号。
q/扰动后q近零时方向NA；原参数、RNG、y及输出文件hash前后不变。

## 4. 判定和下一步

≥4/5点有可分辨且尺度一致读数：支持探针可用，不证明指标有重建收益。
可靠Phi>0：记录局部上升，不按更大ζ/更小步直接纠正；先讨论近似与clamp机制。
多数点尺度依赖/不可分辨：否定当前探针分辨能力，不将omega代称局部敏感度。
全部点和失败原因均报告；高低敏感度只作相对描述，不由单图建立控制阈值。
跨图/seed确认及粗细步对照另立卡；不同噪声区域不能直接以同一绝对阈值比较。

## 5. 执行与归档

入口utils/diag_guidance_local_probe.py独立实现，不改共享核心；A1a源码保持不动以
保留父hash。先完成源身份绑定，结果写新目录，保存卡片、
所有配置/源码/模型/张量hash、逐调用NFE、全快照CSV及failed/complete状态。
本机7项合同检查通过，2项张量/真实ZAPS二次函数检查因缺PyTorch待服务器。
探针读数为有限尺度局部响应；记录所有裁剪活动集合变化，不认证无限小导数。
下降与上升响应都保留为可读结果，不因方向为上升而丢弃该快照。

```bash
cd /home/lzy/ZAPS/projects
python -m unittest utils.test_diag_guidance_local_probe -v
python -u utils/diag_guidance_local_probe.py --inspect
set -o pipefail
python -u utils/diag_guidance_local_probe.py --device cuda | tee ../guidance_local_probe_20261018.log
```

测试应9项全部通过、无skip。--geometry-dir可固定A1a目录；默认只按兼容协议/
完成状态/目录时间选择，不按质量挑选。--inspect不加载Torch/权重，打印五个原点。
输出results/diag_guidance_local_probe/<时间戳>/，含run.json、probes.csv及实验卡。
回贴Snapshot/Pair、Gates、Summary、Report source/Source SHA256、Records。

## 6. 2026-10-18用户回传（只追加结果）

五点、两尺度全部可用，LOCAL_FINITE_SCALE_DESCENT=5，ASCENT=0；
全部中心/参数/RNG/文件身份/30调用预算门控passed，中心clean/q/loss误差0。
方向相对差最大6.1541%、loss相对差最大.8236%，达到原≥4/5点可用要求。
来源、单位分解和证据边界见../GUIDANCE_LOCAL_PROBE_RESULT_20261018.md。
当前为用户回传生产运行完成；服务器单测输出和本run.json SHA尚未回贴。
不将本结果升级为控制收益，不增加GPU扫描；进入阶段性结果统一归档。

## 7. 归档审计追加

结果提交711da42已拉取；实际运行bdc165a，run.json SHA256为
7e60a041cc53f0766a1b657389ead84820a038f7a70daef97088e5674260f2c6。
五点分类/十组算式本机复核，全部完整rows与此前回传一致，30次累计调用编号完整。
见../STATE_INDICATOR_ARCHIVE_AUDIT_20261018.md。未在本机读取服务器张量或重新执行GPU门控。
