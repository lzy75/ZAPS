# A77：dps_imagenet_transfer

Three existing ImageNet images x SR4/Motion; ImageNet256 unconditional pretrained prior; two arms DPS/SHRINK.

|任务|方法|PSNR|SSIM|LPIPS|完整秒数|NFE|输入VJP|
|---|---|---:|---:|---:|---:|---:|---:|
|super_resolution|DPS|24.6517|0.61978|0.32057|610.226|1000|1000|
|super_resolution|SHRINK|25.6105|0.70049|0.16911|619.977|1000|1000|
|motion_deblur|DPS|22.7205|0.55362|0.34424|604.090|1000|1000|
|motion_deblur|SHRINK|25.6751|0.73591|0.28215|611.692|1000|1000|
