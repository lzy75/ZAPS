# A56S2：冻结前选样扫描修订

状态：用户回传原A56在prepare/history_selection阶段报Need6 unused unique images; available=0；尚无原plan或病例调用记录。
这是A56的选样接口修订，不是第六个科学idea。原A56/A57及既有修复源、失败状态/日志均不改。

## 目标、原因与唯一变化

原A36等计划将所有候选保存在selection_audit.pool；原A56递归将pool中每个image/image_sha256都误计为已用。
新扫描只将明确候选库存容器标为非使用；cases、selected_images、普通/未知路径中的image引用及failed/planned病例仍保守排除。
同内容SHA跨数据集去重、PNG范围32..999、原种子20261026/20261027、6新FFHQ＋4新ImageNet不变。
每数据集打印修正前后unused计数及实际使用数；保存逐历史文件hash和被忽略的候选容器位置。
如果修正后真实未用池仍不足，停止并显示目录/计数，不扩大范围、不使用旧图、不按结果换图。

## 零调用资格及冻结

新入口先检查原A56没有plan.json、没有病例run.json，且原日志对应这条冻结前选样异常；
取得原session/launcher互斥，复制原日志和状态至独立快照。若已有原plan或任何病例记录，拒绝重新选样。
新根/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/selection_r2_20261027/A56_selection_r2_20261027。
新session/log和原入口分离，原launcher锁共享。原失败目录不删除、不重置。
新plan标注implementation_revision=A56_SELECTION_R2，绑定新适配源/卡/测试/launcher/session与保留快照。
原checkpoints、算子/伴随、unit sigma=.05、lr、原30点、10轮、原6臂、冻结旧FFHQ时间g表、源A55r2均不变。
GT只用于既定观测构建与全20例输出冻结后的评价；候选扫描只读取路径/文件hash，不读质量标签。

## 预算、原判据与分支

选样修复0UNet；原A56科学预算仍6000方法＋3624诊断=9624新前向、最多1204输入VJP。
原原生C回放、IEEE、实际VJP、RNG/参数、CG资格和每组PSNR/SSIM/LPIPS阈值完全保留。
原13项Torch预检与新选样回归由服务器入口运行；Mac仅运行纯Python/NumPy与shell/导入检查。
新规则只声明在可访问实验历史中未用、字节内容去重，不声明全球未见或网络训练集未见。
A57已确认旧图30前向等价，不自动降低本新图确认协议的已冻结成本记账。

正向：元数据修正后池恢复、固定种子选出6+4并先冻结，再正常运行原A56确认。
不足/未知历史损坏：0NN停止，保留诊断计数；不通过放宽“已用”定义追求开跑。
数值或质量失败：按原A56保留，不修改阈值/seed/模型。所有新病例失败仍不自动重跑。

归档分支results/frozen-direction-transfer-selection-r2-20261027；PT留服务器。
入口python -u utils/a56_selection_r2_session.py；断线同入口跟随，已完成病例复用。
