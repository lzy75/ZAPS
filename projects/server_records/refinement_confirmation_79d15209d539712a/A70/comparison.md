# A70 strong-ZAPS noise-aware scalar policies

|sigma|method|n|PSNR|SSIM|LPIPS|full seconds|NN|
|---|---|---:|---:|---:|---:|---:|---:|
|0.05|BASE|4|26.5327|0.81627|0.16642|22.848|300|
|0.05|FIRST_NOISE|4|26.6870|0.83118|0.17677|22.733|300|
|0.05|SHRINK|4|26.6837|0.82928|0.17067|24.347|300|
|0.05|FIXED_FIRST|4|26.2085|0.81923|0.16974|23.041|300|
|0.05|MISMATCH_FIRST|4|26.6755|0.82927|0.17734|22.548|300|
|0.05|FIXED_SHRINK|4|26.3452|0.82314|0.16416|23.401|300|
|0.05|MISMATCH_SHRINK|4|26.6623|0.82609|0.17232|22.660|300|
|0.1|BASE|4|23.5487|0.59485|0.42163|23.415|300|
|0.1|FIRST_NOISE|4|25.4284|0.77909|0.21604|23.787|300|
|0.1|SHRINK|4|25.3738|0.77334|0.21316|23.963|300|
|0.1|FIXED_FIRST|4|24.9229|0.76522|0.19862|23.278|300|
|0.1|MISMATCH_FIRST|4|25.4232|0.77553|0.21772|23.892|300|
|0.1|FIXED_SHRINK|4|24.9644|0.76506|0.19824|24.219|300|
|0.1|MISMATCH_SHRINK|4|25.3505|0.76863|0.21645|23.263|300|

Paired quality on identical BASE270 prefix; full clock is an independent300-forward execution. Model load/warmup/metrics/file writes excluded.

Four unused-in-accessible-history FFHQ SR images, two noise levels, one seed/image. Fixed30, final unroll only. No new optimizer/gradient/trajectory-curvature/step-size claim. Shrinkage Gaussian residual model is an engineering proxy, not an exact posterior or SURE certificate. No gamma upper cap for SHRINK; FIRST retains historical0..2.
