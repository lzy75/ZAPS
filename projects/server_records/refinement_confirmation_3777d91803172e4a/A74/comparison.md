# A74：10/15/30步，固定10轮，末轮缩放

S30原样复用A71，计时非同场；S10/15为本轮完整独立测时。总NFE分别100/150/300。

|任务|每轮步数|方法|PSNR|SSIM|LPIPS|完整秒数|NFE|复用|
|---|---:|---|---:|---:|---:|---:|---:|---|
|super_resolution|10|BASE|26.1113|0.79925|0.19639|7.468|100|False|
|super_resolution|10|SHRINK|27.2370|0.82782|0.23204|7.342|100|False|
|super_resolution|15|BASE|26.5414|0.81312|0.17733|10.794|150|False|
|super_resolution|15|SHRINK|27.1514|0.83074|0.19507|10.901|150|False|
|super_resolution|30|BASE|26.8775|0.82200|0.16128|22.248|300|True|
|super_resolution|30|SHRINK|27.0435|0.82797|0.16844|23.697|300|True|
|gaussian_deblur|10|BASE|21.7676|0.60652|0.33042|7.745|100|False|
|gaussian_deblur|10|SHRINK|25.0636|0.74983|0.33369|7.863|100|False|
|gaussian_deblur|15|BASE|21.2360|0.57090|0.34481|11.394|150|False|
|gaussian_deblur|15|SHRINK|25.1527|0.75676|0.28788|11.815|150|False|
|gaussian_deblur|30|BASE|24.6105|0.73645|0.20429|24.804|300|True|
|gaussian_deblur|30|SHRINK|25.3853|0.76563|0.23174|25.239|300|True|
|inpainting|10|BASE|22.1726|0.66067|0.26818|7.647|100|False|
|inpainting|10|SHRINK|21.1897|0.64226|0.40256|7.552|100|False|
|inpainting|15|BASE|25.3245|0.78285|0.16205|10.743|150|False|
|inpainting|15|SHRINK|25.4261|0.79629|0.16731|11.573|150|False|
|inpainting|30|BASE|26.5519|0.83196|0.12368|24.815|300|True|
|inpainting|30|SHRINK|26.8903|0.85517|0.10411|25.694|300|True|
|motion_deblur|10|BASE|18.6724|0.46365|0.44515|7.871|100|False|
|motion_deblur|10|SHRINK|20.8279|0.57612|0.38883|7.582|100|False|
|motion_deblur|15|BASE|19.0392|0.47832|0.39348|11.325|150|False|
|motion_deblur|15|SHRINK|20.9376|0.57388|0.34326|11.392|150|False|
|motion_deblur|30|BASE|19.7519|0.48941|0.41757|25.274|300|True|
|motion_deblur|30|SHRINK|21.6599|0.62035|0.30618|26.135|300|True|
