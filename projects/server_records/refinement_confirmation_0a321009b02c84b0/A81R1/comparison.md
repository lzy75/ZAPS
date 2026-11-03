# A81R1 官方像素DAPS倍率迁移

两臂均完整1000F/20000观测梯度，同场clock。未改变随机项；候选不继承原后验保证。

|任务|方法|PSNR|SSIM|LPIPS|完整秒数|
|---|---|---:|---:|---:|---:|
|super_resolution|DAPS|27.505|0.8134|0.1683|102.445|
|super_resolution|SHRINK|27.335|0.8098|0.1686|139.144|
|motion_deblur|DAPS|27.906|0.8133|0.1314|147.614|
|motion_deblur|SHRINK|26.856|0.7868|0.1776|201.609|
