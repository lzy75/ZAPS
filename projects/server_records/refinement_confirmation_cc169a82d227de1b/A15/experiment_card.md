# A15：固定时间方向响应的增量收益预测，冻结后随A14自动执行

planned，2026-10-20；A14完整数据后自动执行，新增方法/诊断NFE均0。
与A14分开判定：方向响应确实不同，不等于它能区分哪里值得额外计算。
问题是IEEE可靠量测的相对方向敏感度是否在物理状态和通用响应之外包含最终细化收益信息。

## 固定设计

开发00020–22三图拟合，转移00023–25三图评价，各双seed；t14、t54各自独立模型。
现有终局早已见过，A13还有一个转移例已见字段，本实验属于探索性转移检验，不能称前瞻盲测。
不修改A7/A10余弦/Ω/三点差分，不调整窗口、lambda或目标；本轮唯一新状态是固定时间干预响应。
Gq、Go分别为两固定eta所得clean gain的算术均值；只有两个方向均usable、正gain且有限时才可进入拟合。

    B = [log R, log qRMS]
    U = B + [log Go]
    D = B + [log(Gq/Go)]

U与D同特征数，用来排除仅是总体平滑/响应幅度的信息。
ridge lambda=1，开发组均值/标准差归一化，常数列scale=1，截距不惩罚；全部沿用A7求解/逐图MAE函数。
目标是原A7保存的两桥平均F-C最终收益：PSNR/SSIM取正差，LPIPS/物理loss取负差。
这些标签属于原native生产候选；IEEE仅用于测量原输入的局部响应，不声称新IEEE生产重建已验证。
GT仅为已有离线标签，不进入特征、方向、模型选择、输出、时间或停止规则。
保存所有拟合系数、预测、逐图误差和阶段均值参照，不挑最好目标/层/seed/桥。

## 判定

若某层任何一个预定案例未完成或量测不usable，整层标UNRESOLVED_METROLOGY_NO_FIT，不删除/插补该例。
H15：D的三质量图像平均MAE均不高于B、U、开发阶段均值；至少两项质量MAE相对B和U均降低≥10%，
且各有≥2/3转移图比B/U误差低。loss完整报告，但不替代质量门槛。
通过只标IEEE_DIRECTION_INCREMENT_EXPLORATORY_CANDIDATE_ONLY；否则IEEE_DIRECTION_INCREMENT_NOT_SUPPORTED。
负结果不改分母、映射、正则或模型追胜。通过只解锁另卡的独立前瞻/成本试验，不能自动部署控制器。
此量测目前每点10次诊断网络调用，远非免费代理；预测通过仍须解决在线成本并做同成本最终重建对照。
本轮不进行新输出生成，不计新的重建PSNR提升，进度工作包仍45/100，不按运行轮数增加。

## 来源、身份和运行

父A7完整run/sha/rows/原冻结分析、A14完整plan/所有case run/fields SHA、全部新代码/卡/测试绑定。
新目录/data0/lzy/ZAPS-results/diag_ieee_response_value/A15_20261020；A14两worker结束后由父进程运行。
单独plan/run/SHA/points.csv/summary，精确复用完成结果；失败保留不覆盖。原数据不修改。
最后A14/A15一起轻量归档到results/refinement-confirmation-20261020，二进制留服务器；上传失败不重跑GPU。
可重复命令bash run_a14_a15_20261020.sh，完成A14只复核复用；未完成/失败不自动重训父实验。
