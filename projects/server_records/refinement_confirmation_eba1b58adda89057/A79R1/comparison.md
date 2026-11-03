# A79R1 DiffPIR 原版与噪声感知倍率

新圆周卷积观测，[0,1]噪声σ=.05，原文100NFE参数；不能与旧DPS观测平均值直接相减。

|数据集|任务|方法|PSNR|SSIM|LPIPS|完整秒数|
|---|---|---|---:|---:|---:|---:|
|ffhq|gaussian_deblur|DIFFPIR|26.208|0.7871|0.1560|6.195|
|ffhq|gaussian_deblur|SHRINK|25.046|0.7384|0.2229|6.155|
|ffhq|gaussian_deblur|STABLE_DIFFPIR|26.208|0.7871|0.1560|6.311|
|ffhq|gaussian_deblur|STABLE_SHRINK|25.046|0.7384|0.2229|7.013|
|ffhq|motion_deblur|DIFFPIR|25.463|0.7534|0.1628|6.855|
|ffhq|motion_deblur|SHRINK|23.299|0.6596|0.2645|7.176|
|ffhq|motion_deblur|STABLE_DIFFPIR|25.463|0.7534|0.1628|6.961|
|ffhq|motion_deblur|STABLE_SHRINK|23.299|0.6596|0.2645|7.062|
|imagenet|gaussian_deblur|DIFFPIR|25.153|0.6484|0.2952|29.003|
|imagenet|gaussian_deblur|SHRINK|24.586|0.6481|0.3051|31.654|
|imagenet|gaussian_deblur|STABLE_DIFFPIR|25.153|0.6484|0.2952|30.242|
|imagenet|gaussian_deblur|STABLE_SHRINK|24.586|0.6481|0.3050|29.952|
|imagenet|motion_deblur|DIFFPIR|26.203|0.7143|0.3166|32.209|
|imagenet|motion_deblur|SHRINK|23.365|0.5998|0.5107|32.290|
|imagenet|motion_deblur|STABLE_DIFFPIR|26.203|0.7143|0.3166|33.100|
|imagenet|motion_deblur|STABLE_SHRINK|23.365|0.5999|0.5107|33.853|
