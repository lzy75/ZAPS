# A95R2：源码检查的封装对象边界修复

R1在源码指纹检查中把FFHQDiffusionModel当nn.Module调用.modules()，尚未进入真实AD探测或案例。
本轮只将源码检查传参规范为内部model.model；实际预测/AD探测继续传完整扩散封装。
同时兼容直接nn.Module，避免重复解包。新增execute->probe->源码检查整条调用边界回归测试。

A95/R1全部源码、失败记录、计划及probe成本保持原SHA；只有确认R1为指定modules错误且0案例/0前缀才运行新目录。
原正常噪声、四臂、共同响应目标、LM/CG标准、预算、固定干预、真实权重primal/伴随门控和所有结果判据不变。
R1两路径识别修复继续生效，实际加载源码SHA仍记录。数值/模型未通过不跳过、不放宽；失败不自动重试。
本机无Torch，运行封装对象/整条调用边界/失败记录守护三项测试；服务器仍强制全部真实Torch与实模型门控。

bash projects/run_a95r2_20261106.sh --gpus 0,1
/data0/lzy/ZAPS-results/diag_response_target/A95R2_20261106
results/zaps-response-target-wrapper-r2-20261106
断线安全、逐30F日志、完整摘要和自动上传继承A95。--status只读，--archive-only仅汇总上传。
