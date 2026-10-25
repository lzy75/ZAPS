# A36：冻结状态公式与时间表的跨图、跨测量噪声迁移

状态planned；2026-10-23设计、2026-10-25交付；A线。用户要求继续验证跨图适用性和正向趋势。
可否定假设：同一I_cont2公式在未用图、不同sigma上仍稳定改善原ZAPS；另报告是否比
冻结时间规则更稳健。不要求每图PSNR超过TIME，但基线收益不自动归为状态独立贡献。

## 父证据与唯一新增问题

A33六张开发图的I_cont2相对C均提高PSNR，均值+1.2356 dB；TIME也约+1.2368 dB。
A35路径弯曲/耦合未建立额外调权价值。A35b后三个位置的有限动作PSNR oracle空间为零；
不能在这个集合上继续叠加指标期待明显提升。现在冻结控制器，改测试未见图和物理噪声条件。
父run：/data0/lzy/ZAPS-results/diag_continuous_response_feedback/A33_20261021，complete，
I_cont2的十二条轨迹与输出SHA/门控必须通过。此前A26/A27测试另一状态规则，不能充作本公式验证。

## 数据、冻结规则与科学臂

六张图，从同一FFHQ目录32–999的本机服务器可访问run.json历史未使用、SHA未重复图片池，
用numpy.default_rng(20261023)无放回抽样，首次prepare先于输出冻结。审计含失败run；
少于六张或历史不可读则停止，不凭结果换图。仅声明可访问历史内未用。
每图sigma_y=.025/.05/.10，采样seed1000，测量seed1000+image_id。
三噪声条件共用一张图的同一个标准Gaussian观测扰动，只改变缩放sigma；
相同seed的18个case只有六个图像簇，不作18独立样本。

- C：原本地ZAPS，原10轮优化，最后更新前last_opt输出。
- TIME：旧六张图×两seed的I_cont2每slot倍率中位数，生成一个30项固定表，
  所有新图/新sigma完全共用，不针对新条件重估，不用GT。
- STATE：原I_cont2连续c/b、上限2、1%回退；不加曲率项，不扫参数。
  老sensor仅R读数写死.05，本轮将读数按实际sigma正确归一化；c/b与gamma算法原样保留。
  这不引入额外按sigma缩放gamma的规则。

三臂原前270调用共用，最后30步分别C/TIME/STATE。每个科学臂300 NFE，
反馈只在最后unroll；不能声明已在全部十轮部署自适应。
冻结A33的config/grid/checkpoint/native precision，FFHQ256、SR4、DDPM eta1、
unit H(x)=H_base((x+1)/2)、legacy bicubic transpose、clamp和PSNR/SSIM量化/LPIPS alex口径。
ζ/D仍为每个测量case正常优化得到的原ZAPS参数，不是冻结整个图像的学习参数。
噪声为unit测量域sigma，不是模型[-1,1]域或扩散sigma。原共享核心/旧父文件不修改。

## 预算、门控和重启

18case×原300方法=5400方法NFE；每case诊断C回放30+TIME30+STATE30=90，合1620诊断；
总7020。两卡按图分片，同图三sigma在同一worker。先验证完整18case，不依均值早停/追加参数。
训练300与诊断90分别保存attempt；已完成训练可复用，失败attempt保留并计成本。
原PRE参数和初始latent、29次噪声由PRE保存的RNG恢复；C回放相对训练last_opt<=1e-6，
逐臂实际30点、总NFE、RNG不变、ζ/D/网格不变、输入/观测SHA、native runtime门控。
GT仅用于合成观测和所有候选输出冻结后的评价，不进入调权、停止或选择输出。

## 预先判断规则

逐sigma、逐图报告STATE-C、TIME-C、STATE-TIME的PSNR/SSIM/LPIPS/loss，最低PSNR收益、
最大LPIPS退化、逐图风险和gamma相对TIME差异。物理loss按实际sigma白化。
STATE在三档均平均PSNR增益≥.05 dB、至少4/6图正向、均值SSIM降幅≤.002且LPIPS升幅≤.005，
记FROZEN_STATE_TRANSFER_CANDIDATE_ONLY；TIME使用同标准另判。任一条件不满足则保留混合/负结论。
STATE-TIME单独以平均PSNR≥.02 dB、≥4/6正向和相同保护线报告候选条件；
不要求所有条件都胜TIME，也不把仅C改善称为状态增量。上述为工程初筛阈值，不是文献标准。
10,000次配对图像簇bootstrap百分位95%区间仅描述性，六图、小样本、跨条件相关且未校正多重比较，
不作确认性显著性、精确覆盖或跨任务泛化声明。若STATE/TIME同样迁移，归因为共享阶段规则更合适；
若STATE在噪声变化下更少失效，才解锁更大冻结确认。失败不现场调整上限/表/图像名单。

## 产物与运行

/data0/lzy/ZAPS-results/diag_frozen_response_transfer/A36_20261023，所有训练PRE/last_opt、
三臂实际终局、30步R/c/b/gamma轨迹、GT/C/TIME/STATE同ROI图、每图指标、summary与SHA。
轻量结果归档Git，完整tensor留data0。一次命令等待结束打印总结；Mac无Torch/CUDA，
仅纯逻辑和元数据测试，服务器必须通过Torch sensor测试（不允许跳过）才启动。
运行：bash run_a36_20261023.sh。
