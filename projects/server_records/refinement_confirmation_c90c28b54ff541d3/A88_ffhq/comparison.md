# A88：dps_missing_tasks

ffhq: three existing images x Gaussian61/sigma3 zero boundary and random70% missing; DPS/SHRINK only. Completes missing A75/A77 task cells.

|任务|方法|PSNR|SSIM|LPIPS|完整秒数|NFE|输入VJP|
|---|---|---:|---:|---:|---:|---:|---:|
|gaussian_deblur|DPS|24.4427|0.71075|0.19296|109.997|1000|1000|
|gaussian_deblur|SHRINK|26.2729|0.79940|0.12785|113.840|1000|1000|
|inpainting|DPS|27.1972|0.81748|0.15518|103.304|1000|1000|
|inpainting|SHRINK|29.8018|0.90803|0.03821|107.049|1000|1000|
