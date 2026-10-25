# A41 JSON 类型修复

2026-10-25。错误是 NumPy scalar（包括 bool_、float32）进入原严格 JSON writer。
原纯数值测试未覆盖真实 probe 对象的 JSON 落盘，这是本次遗漏。

不改已经被 A41 plan SHA 绑定的原 driver/helper/launcher、实验卡、指标或参数。
新入口 `utils/diag_ffhq_action_geometry_json_fix.py` 只在计数和签名写入的两个
WRITE alias 上把 NumPy scalar 转为值相同的 Python scalar；不修改内存中的响应对象。
不接受 NaN/Inf，不泛化为任意对象转字符串，不引入额外模型计算。

新文件的源码 SHA 及原 plan SHA 写入独立签名 `serialization_compat_v1.json`。
新 run.json 附加 compatibility SHA。所有原始 completed/failed/partial 尝试保留；
原 plan、源码及已有记录不刷新 hash、不覆盖。已完成 case 和 C 基线按原门控复用。
中断 case 另建新 attempt；原中断调用仍由原汇总累加，若中断记录不完整保留成本下界标记。

回归检查：真实 features 对象（含 NumPy scalar）严格 JSON roundtrip；
实际计数 writer 和 signed writer 的落盘/读回、SHA、一致性元数据；冻结 plan 字节不变；
NaN 仍拒绝。真实 Torch 原回放检查仍在服务器强制执行。

续跑入口 `bash run_a41_json_fix_20261025.sh`，与旧 launcher 共用锁，拒绝重复 GPU 作业。
新日志为 launcher_jsonfix_20261025.log / shard*_jsonfix.log / final_jsonfix.log，旧日志保留。
科学假设与样本、门槛、grid、预算保持原卡。失败重试的实际累计调用可能超过最初7812计划。
