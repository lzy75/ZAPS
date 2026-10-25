# AdamDPS：可引用的时间状态依据及不可直接移植部分

核查2026-10-26；正式ICLR2026，当前arXiv v2日期2026-04-22。
[主文](https://arxiv.org/html/2603.16797v2)、[正式PDF](https://openreview.net/pdf?id=qYDObsHldZ)。
OpenReview论坛页面有浏览器验证，未据此认证评审等级；PDF正式会议页眉可核实。

方法维护引导梯度的一阶/二阶矩，经偏差校正后逐元素缩放：

    m_k=beta1*m_(k-1)+(1-beta1)*g_k
    v_k=beta2*v_(k-1)+(1-beta2)*g_k^2
    g_adam=m_hat/(sqrt(v_hat)+eps)

它支持“过去引导信息可能帮助当前引导”的动机。二阶矩不是居中方差、后验协方差或
真实曲率；相邻余弦是分析读数，不提供本项目按余弦缩步的理论保证。

实际读完作者仓库[adam_dps.py](https://github.com/christianbelardi/adam-guidance/blob/main/methods/adam_dps.py)、
[dps.py](https://github.com/christianbelardi/adam-guidance/blob/main/methods/dps.py)、
[base.py](https://github.com/christianbelardi/adam-guidance/blob/main/methods/base.py)及
[采样循环](https://github.com/christianbelardi/adam-guidance/blob/main/diffusion/ddim.py)。
每步仍需真实denoiser输入反向；Adam矩本身不增加DPS前向，但相对stop-gradient的ZAPS
不能称无额外反向成本。代码在每个新batch轨迹前reset矩，未跨独立轨迹共享历史。
更新另含有效skip alpha的缩放，与本项目q/ζ的单位不能直接等同。

还读了tasks/base、utils/configs、cat调参配置和pipeline。仓库默认观测噪声为0，参数可覆盖；
不能未核任务就外推到当前unit sigma=.05。作者公开150次搜索历史及32样本开发配置，
不能把其效果归为通用免调参定理。当前仓库文件修改日期2026-08-12，与4月论文时间分开记录。

对本课题：它是历史平滑/自适应预条件的强先行工作和可选参照，不是本夜新增第三机制。
把相同EMA改名为双状态不产生新颖性。若以后引入，须另立原q/真实梯度、reset、单位及
匹配算力协议；目前只读源码，未运行模型，也没有本地重建效果。
