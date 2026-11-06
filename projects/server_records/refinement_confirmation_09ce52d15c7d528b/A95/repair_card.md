# A95R1：实际加载检查点路径识别修复

用户回传A95失败：实际网络来自/home/lzy/ZAPS/guided_diffusion，旧前向AD兼容处理只识别DPS/guided_diffusion。
真实权重JVP在旧自定义autograd.Function报setup_context错误，尚无案例/前缀尝试。
本修复独立目录A95R1，A95全部源码、plan、失败记录、probe成本不改。

仅改识别范围：从实际layer.forward的globals找到checkpoint，允许仓库根guided_diffusion/nn.py或DPS副本。
只在JVP内直接执行相同函数，退出/异常均恢复；不修改两份nn.py或unet.py。
真实模型载入后记录实际checkpoint与unet源码SHA，续跑必须一致；保留原primal/VJP-JVP伴随、实际显存门控。
新增根路径别名回归测试，防止只用DPS测试替代实际加载路径。
原A95四臂、正常噪声、数据、原ζ/D、优化目标、LM/CG门槛、预算、固定单次干预及判据全部继承。
旧失败必须是本次setup_context路径错误且0案例/0前缀；不同错误或已有正式尝试停止，不自动重跑。
原probe成本由原loads记录绑定保留，新probe/正式调用另计；不把失败probe漏计为零。

本机仅无Torch检查；服务器强制旧9项门控及新增路径识别/作用域恢复/真实AD回归。
Mac无Torch/CUDA，不能声称实模型通过。无数值/质量策略回退、不增加图/seed/高噪/步数。

bash projects/run_a95r1_20261106.sh --gpus 0,1
/data0/lzy/ZAPS-results/diag_response_target/A95R1_20261106
results/zaps-response-target-checkpoint-r1-20261106
断线安全，沿用逐30F进度与自动上传；--status只读，--archive-only仅汇总上传。
