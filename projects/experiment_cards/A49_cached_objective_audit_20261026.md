# A49：局部目标与终局质量错位的缓存审计

planned；2026-10-26；A线，descriptive audit，新增method/diagnostic NFE均0。
父run /data0/lzy/ZAPS-results/diag_response_reliability/A48_20261026 必须20/20完成；
5旧开发图四任务，原调用与PT不改、不覆盖。A48报告195新增诊断、0重训。

rho门控改变0/20动作，不能证明其控制收益；SR/IP相对LEGACY约-.398/-.422dB。
核查退化是否伴随更低的实测局部J，以及门控未干预是所有候选都可靠还是
被选中的点恰好通过。区分无干预与无收益，不扫描.75阈值追分。

新增信息仅来自缓存读数和已有终局标签；不选择参数或生成新策略。
重放无GT选择函数，要求与签名决策相同；重算父summary.groups并精确核对。
每图报告倍率变化、rho及通过数、局部J/下一时刻loss/终局loss/质量差异，
统计局部J改善但PSNR恶化的次数。双尺度几何仅描述，不称控制效果或时间曲率。

唯一因子为审计计算，无新采样和GPU前置；模型/算子/观测/采样不变。
签名、来源、worker源码、父artifact SHA全部校验；失败保留，不重跑父实验。
不把事后相关解释为可迁移或在线收益。不设追认成功的阈值，不提供p值。

新目录 /data0/lzy/ZAPS-results/diag_response_reliability/A49_cached_20261026：
签名report、父20例run/决策/对照图快照及来源SHA；张量留服务器，不称重放GPU。
bash /home/lzy/ZAPS/projects/run_a49_20261026.sh，前台摘要后推独立结果分支。
