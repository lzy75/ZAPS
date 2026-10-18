# A2b0：缓存状态路径尺度与观察方向分解（0 NFE探索性诊断）

状态：planned，2026-10-18，A工作线。仅CPU缓存分析；没有新GPU实验或控制器。

## 1. 问题、身份与假设

A2a六条生产run均complete并通过身份/回放/探针门控；同t的Omega3–kappa_dir相关在
五个快照均为负。当前不支持把相邻引导方向转动直接当作固定时间的输入敏感度。
父目录：/home/lzy/ZAPS/projects/results/diag_guidance_replication/A2a_20261018，
已迁至/data0/lzy/ZAPS-results/A2a_20261018；采用恢复注册表中的有效第四条。
本地结果归档提交8f58702867f6227dabad87d3fecff0fc9eb9a08e尚未推送或独立拉取。

新增信息：原Omega3按噪声坐标跨度归一化，但kappa_dir按状态RMS位移归一化，
且后者沿当前q方向探测。利用原张量显式拆分路径移动速度并测实际位移与q的轴向关系。
本分析是在看过A2a关联后提出，不计为A2a预注册H1的独立确认。

探索性H2：在五个固定t、两个尺度下，按状态路径长度归一化的方向差商与kappa_dir
均为正秩关联。任一零/负结果不支持这个全阶段正向代理；全正也仅为候选描述，
不能认证增量预测/控制/泛化。只有三个图像簇，不做p值或独立样本显著性声明。

## 2. 唯一改动与固定条件

唯一新增的是对已完成缓存的坐标及方向分解；不修改方法、不重采样、不训练。
完整模型/图像/观测/源码/原卡/参数/网格/算子/随机协议身份继承并核验父plan及注册表。
六例00020–00022 × seed1000/2000；每图共享观测1020/1021/1022；
FFHQ SR×4、unit sigma_y=.05、Karras rho3、30×10、lr=.01、eta1、
原clamp、joint zeta/D、legacy_bicubic经验T与last_opt均保留。
本次不接触GT图像；原PSNR/SSIM/LPIPS只从已保存final_metrics携带，不再计算或挑输出。
不存在新校准/验证集；同一三图只作机制探索。

仍用原五点k8/19/24/27/28（t903/641/319/54/14），禁止选择窗口或替换点。
n_k=q_k/||q_k||，d_k=x_k-x_(k-1)，均取原latent_inputs和guidance。

    B3 = ||n_k-n_(k-1)|| + ||n_(k-1)-n_(k-2)||
    X3 = RMS(d_k) + RMS(d_(k-1))
    U3 = u_k-u_(k-2)
    K_path = B3/X3
    V_state = X3/U3
    Omega3 = K_path * V_state
    axis_cos = cosine(d_k, q_k)

K_path是有限混合路径差商，仍含时间、D变化和随机历史，不是冻结时间的方向导数。
Phi和kappa_dir沿当前q轴探测；axis_cos同时报告符号和绝对值，仅绝对值用于范数敏感度
的轴向比较，负号不直接解释为有害冲突。三个方向保持原相邻关系，不跨零向量缺口。

保持父u和单位向量定义。X3<=1e-12时K_path记NA、保留点；状态位移方向使用父
unit_direction阈值1e-8，不能为凑相关删除/替换点。时间跨度必须正。
按同t分别完整报告Omega3/K_path/V_state/|axis_cos|与两尺度kappa_dir的40项秩关联，
报告有效case与图像簇数；不混时间，不反转指标符号追相关，不增加候选窗口/频带/权重。

## 3. 预算、门控与停止

新增方法NFE=0、诊断NFE=0；父全部尝试1800方法+840诊断另列。仅torch.load CPU
读取六份原trajectory.pt，不载模型、不调用UNet、不初始化新轨迹、不用CUDA。
原观测/参数不改，CPU RNG须不变。输出写/data0/lzy/ZAPS-results/diag_cached_guidance_state_path/<timestamp>，
不写回原A2a目录或已归档结果分支。

生产前核对父冻结源码/卡/输入及六条工件hash；中心q幅度及原Omega3须与缓存匹配：
Omega rel_tol=1e-6/abs_tol=1e-8，qRMS rel_tol=1e-6/abs_tol=1e-12。
原Omega3–kappa的10项同t关联须逐值复现父摘要。
分解恒等式误差<=1e-10*max(1,|Omega3|)，源SHA及CPU RNG不变。
原错误/缺文件/过期摘要/损坏张量/零引导方向均停止，不重训、不补采样。

## 4. 结果分支与允许结论

| 情形 | 判断及处理 |
| --- | --- |
| K_path五点两尺度均正 | 仅EXPLORATORY_POSITIVE_ORDERING_ONLY；后续独立图验证，不直接启用控制 |
| 任一K_path同t关联零/负 | UNIFORM_POSITIVE_PROXY_NOT_SUPPORTED；停止简单替代指标的扫描 |
| 常数/零路径/NA导致无法比较 | INCONCLUSIVE_ORDERING；保留全部点，不扩大窗口追可读性 |
| 轴向差异明显 | 只描述实际状态路径与原q探针不一致，不唯一归因为噪声或解释全部负相关 |
| 恒等式/身份/RNG门控失败 | 技术失败，停止解释；原A2a六例及归档不变 |

所有情形都不解锁自动调度或宣称重建改善。若仍需验证纯状态贡献，另卡采用固定t/参数的
回看或匹配位移方向探针，额外UNet成本另计；本卡没有自动启动该后续任务。

## 5. 运行与归档

先服务器单测（本机缺Torch时张量两项仅skip，不能当服务器通过）：

```bash
cd /home/lzy/ZAPS/projects
python -m unittest utils.test_diag_cached_guidance_state_path -v
python -u utils/diag_cached_guidance_state_path.py --plan-dir results/diag_guidance_replication/A2a_20261018 --inspect
python -u utils/diag_cached_guidance_state_path.py --plan-dir results/diag_guidance_replication/A2a_20261018
```

新目录记录run.json、points.csv和本卡，保存源SHA、配置、每点轴向/速度/差商、所有关联、
门控及NA。回贴POINT、Gates、Summary和Records。该分析尚未在服务器执行；
不重新导出A2a，也不将未来分析塞进已完成结果提交。
