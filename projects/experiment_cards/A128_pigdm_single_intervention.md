# A128：对角的局部收益能否保留到终局

状态 planned。工作线 A。A126 完成后，A127 在 FFHQ 图 00108 的两项模糊任务中显示
从闭式标量出发，两种真实响应目标的一步对角更新在六个固定状态均改善局部全图 MSE。
真实响应射线本身是否已获得这部分收益，以及额外方向收益能否延续到终局，有待对照。

父记录 /data0/lzy/ZAPS-results/diag_a126_full_response/A126_pigdm_ffhq。使用其完成的
RESPONSE_RAY 配置和归档输出。固定 FFHQ 00108、Gaussian/Motion、sigma_model .05、
原观测/图像/模型/核/网格/种子。每任务5条完整重建：原真实响应射线复核1条，
在 t=920 分别多做一次射线或一次全系数对角，在 t=90 分别多做同样两组。
同一时点两臂均从同一真实响应射线解出发，保持 A126 的下一时刻目标，后续照常
执行 A126 射线控制。唯一改变是额外一步可更新的系数空间，每步同样一次 VJP、
最多一次 JVP 和三次实测前向。因为接受次数和梯度终止可能不同，报告实际成本。

先检查原真实响应射线重建相对 L2<=1e-4，同一对照内核对状态/原方向/射线解哈希及
原网格和最终随机数。每对照保存局部损失、下一次干净图像及干净观测 MSE，终局
PSNR/SSIM/LPIPS及原始MSE、计时和实际 NFE/VJP/JVP。每个干预臂额外离线评价2次
前向，GT不参与在线选择。10条完整轨迹及离线评价合计至多2796 NFE；
模型加载和离线指标不进重建计时。任一门控失败保留摘要并停止。

若对角相对额外射线没有局部改善，则已有射线控制吸收收益或对角目标不够匹配；
若局部改善后终局消失，则检查后续的传播或抵消；若某个位置的局部及终局同时改善，
再用同位置和规则对其他原图确认。单图两任务只用于归因，不作迁移结论或时间选择训练。

在 /home/lzy/ZAPS/projects 执行
`/home/lzy/anaconda3/bin/python -u -m utils.diag_pigdm_single_intervention`，指定一张GPU。
摘要 /data0/lzy/ZAPS-results/diag_a126_full_response/audit_a128_pigdm_single_intervention/summary.json。
