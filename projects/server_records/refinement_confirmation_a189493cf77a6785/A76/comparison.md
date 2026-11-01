# A76：dps_feedback_ablation

Same A75 three FFHQ images x SR4/Motion, sigma=.05, same H/y/seed/model/native1000 sampler.

|任务|方法|PSNR|SSIM|LPIPS|完整秒数|NFE|输入VJP|
|---|---|---:|---:|---:|---:|---:|---:|
|super_resolution|DPS|22.9980|0.63442|0.25785|126.898|1000|1000|
|super_resolution|SHRINK|26.0304|0.79772|0.10590|127.873|1000|1000|
|super_resolution|FIXED_LOO|26.0124|0.79686|0.10694|130.958|1000|1000|
|super_resolution|MISMATCH|26.0071|0.79664|0.10761|131.169|1000|1000|
|motion_deblur|DPS|21.2903|0.57125|0.25666|129.109|1000|1000|
|motion_deblur|SHRINK|25.3956|0.76944|0.15315|129.260|1000|1000|
|motion_deblur|FIXED_LOO|25.4180|0.76522|0.15307|138.272|1000|1000|
|motion_deblur|MISMATCH|25.4171|0.76407|0.15296|135.289|1000|1000|
