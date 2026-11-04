# A88：dps_missing_tasks

imagenet: three existing images x Gaussian61/sigma3 zero boundary and random70% missing; DPS/SHRINK only. Completes missing A75/A77 task cells.

|任务|方法|PSNR|SSIM|LPIPS|完整秒数|NFE|输入VJP|
|---|---|---:|---:|---:|---:|---:|---:|
|gaussian_deblur|DPS|24.3828|0.61901|0.28752|636.199|1000|1000|
|gaussian_deblur|SHRINK|25.6068|0.69258|0.25237|664.438|1000|1000|
|inpainting|DPS|27.8876|0.72864|0.26014|636.791|1000|1000|
|inpainting|SHRINK|28.0431|0.79727|0.09137|643.741|1000|1000|
