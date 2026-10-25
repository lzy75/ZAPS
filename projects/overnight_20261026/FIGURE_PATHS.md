# 已核对的原图与论文可用范围
本轮视觉查看7个本地原文件并记录SHA；不从张量重生成，不声称看过全量图片。

- [original_passive_six_panel](/Users/liziyue.22/ZAPS-deliverables/state-indicator-archive-20261018/20261018_state_indicators_20261018_170217_248796/02_passive_fixed_parent_state_trajectory/trajectory.png)：最早单图30步状态轨迹六联图；与本对话用户图片视觉对应。限制：clean/latent增量方向，不是q余弦；单图被动，不证明控制价值。
- [A44_super_resolution_first_sorted_image](/Users/liziyue.22/ZAPS-deliverables/analysis/A44_20261025/archive/ffhq_super_resolution_00446_s1000/finish_000/policy_comparison.png)：四任务预定第一图00446全臂并排，非按最好结果筛选。限制：五旧图开发组中的一图；付费传感，固定t319单脉冲；本例PHYSICAL可优于STATE。
- [A44_gaussian_deblur_first_sorted_image](/Users/liziyue.22/ZAPS-deliverables/analysis/A44_20261025/archive/ffhq_gaussian_deblur_00446_s1000/finish_000/policy_comparison.png)：四任务预定第一图00446全臂并排，非按最好结果筛选。限制：五旧图开发组中的一图；付费传感，固定t319单脉冲；本例PHYSICAL可优于STATE。
- [A44_inpainting_first_sorted_image](/Users/liziyue.22/ZAPS-deliverables/analysis/A44_20261025/archive/ffhq_inpainting_00446_s1000/finish_000/policy_comparison.png)：四任务预定第一图00446全臂并排，非按最好结果筛选。限制：五旧图开发组中的一图；付费传感，固定t319单脉冲；本例PHYSICAL可优于STATE。
- [A44_motion_deblur_first_sorted_image](/Users/liziyue.22/ZAPS-deliverables/analysis/A44_20261025/archive/ffhq_motion_deblur_00446_s1000/finish_000/policy_comparison.png)：四任务预定第一图00446全臂并排，非按最好结果筛选。限制：五旧图开发组中的一图；付费传感，固定t319单脉冲；本例PHYSICAL可优于STATE。
- [A31_equal_forward_action_heatmap](/Users/liziyue.22/ZAPS-deliverables/analysis/A31_20261021/fixed_guidance_final_psnr.png)：六图双seed均值；已完成单脉冲终局收益矩阵，每科学臂300forward。限制：动作因果效应，不是状态选择；旧legacy协议，有限动作库。
- [A37_high_noise_tradeoff_first_sorted_image](/Users/liziyue.22/ZAPS-deliverables/analysis/A37_20261025/A37/00339_n100_s1000/attempt_000/comparison.png)：预定第一图00339 sigma.10显示失真/感知取舍与固定ROI。限制：REG改善OLD颗粒但仍未普遍过C感知保护；路线已停止，不作新方法主结果。

最早六联图的SVG原件：[trajectory.svg](/Users/liziyue.22/ZAPS-deliverables/state-indicator-archive-20261018/20261018_state_indicators_20261018_170217_248796/02_passive_fixed_parent_state_trajectory/trajectory.svg)。
它对应FFHQ00000、seed1000、原30点、末轮pre-correction clean读数；父run为/home/lzy/ZAPS/projects/results/diag_state_domain_paired/ffhq/state_domain_20261014_155809_691106/run.json。
最终修正输出PSNR26.7013单列，不等于曲线上最后clean约26.51。实际网格以run.json保存的grid为准，不能只看配置中的spacing标签。
A48/A49的原图尚未取回；不借A44图冒充A48修正结果。
