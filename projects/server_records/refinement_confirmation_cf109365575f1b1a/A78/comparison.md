# A78：dps_imagenet_feedback_ablation

Same A77 three ImageNet images x SR4/Motion, sigma=.05, same H/y/seed/model/native1000 sampler.

|任务|方法|PSNR|SSIM|LPIPS|完整秒数|NFE|输入VJP|
|---|---|---:|---:|---:|---:|---:|---:|
|super_resolution|DPS|24.6517|0.61978|0.32057|610.226|1000|1000|
|super_resolution|SHRINK|25.6105|0.70049|0.16911|619.977|1000|1000|
|super_resolution|FIXED_LOO|24.3340|0.67247|0.32441|621.730|1000|1000|
|super_resolution|MISMATCH|24.8115|0.68420|0.30804|629.246|1000|1000|
|motion_deblur|DPS|22.7205|0.55362|0.34424|604.090|1000|1000|
|motion_deblur|SHRINK|25.6751|0.73591|0.28215|611.692|1000|1000|
|motion_deblur|FIXED_LOO|26.8005|0.71706|0.26360|636.025|1000|1000|
|motion_deblur|MISMATCH|26.1856|0.69140|0.25717|643.205|1000|1000|
