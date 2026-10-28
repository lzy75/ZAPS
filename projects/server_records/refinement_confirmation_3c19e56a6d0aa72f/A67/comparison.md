# A67 strong ZAPS: paired quality and independent full clock

|sigma|arm|n|PSNR|SSIM|LPIPS|delta PSNR|full seconds|
|---|---|---:|---:|---:|---:|---:|---:|
|0.025|BASE|3|30.2680|0.88627|0.09474|+0.0000|20.173|
|0.025|A33|3|30.8429|0.89169|0.09860|+0.5749|20.877|
|0.025|FIRST_NOISE|3|30.6874|0.89258|0.09737|+0.4193|21.129|
|0.025|NEAREST_NOISE|3|30.7336|0.89355|0.09576|+0.4656|20.878|
|0.025|TIME_NEAREST|3|30.7317|0.89344|0.09590|+0.4637|20.748|
|0.05|BASE|6|28.1862|0.83357|0.13085|+0.0000|20.155|
|0.05|A33|6|27.8532|0.78719|0.20747|-0.3329|21.326|
|0.05|FIRST_NOISE|6|28.2872|0.84553|0.13091|+0.1011|21.078|
|0.05|NEAREST_NOISE|6|28.3888|0.83796|0.13535|+0.2026|20.759|
|0.05|TIME_NEAREST|6|28.3862|0.83793|0.13530|+0.2000|20.003|
|0.1|BASE|3|24.7982|0.60809|0.41146|+0.0000|20.661|
|0.1|A33|3|23.2572|0.51403|0.56830|-1.5411|20.743|
|0.1|FIRST_NOISE|3|27.8004|0.80752|0.13927|+3.0022|21.018|
|0.1|NEAREST_NOISE|3|25.0936|0.62144|0.41432|+0.2954|20.655|
|0.1|TIME_NEAREST|3|25.0942|0.62153|0.41417|+0.2960|21.322|

Quality uses the identical learned270-prefix replay. Clock is a separately measured complete independent300 execution; independent output quality is also in summary.json. One timing per case, not three-repeat median.

Six OLD FFHQ SR development images, noise boundaries first3 only, one seed/image; no independent confirmation, new theory, curve/step-size or NFE-saving claim. Noise-energy target is a discrepancy heuristic, not calibrated confidence coverage. State derivative uses A33 identity-response surrogate, not actual next-step network response.

R1 only freezes/restores NN flags. Prior failed cost: {'method_forward_attempts': 600, 'diagnostic_forward_attempts': 0, 'optimizer_backward_attempts': 20, 'input_vjp': 0, 'warmup_forward_attempts': 2}; cumulative cost: {'method_forward_attempts': 18600, 'diagnostic_forward_attempts': 1800, 'optimizer_backward_attempts': 620, 'input_vjp': 0, 'warmup_forward_attempts': 6}.
