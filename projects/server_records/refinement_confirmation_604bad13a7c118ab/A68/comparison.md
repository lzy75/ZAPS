# A68 next-response scalar control

|sigma|method|n|PSNR|SSIM|LPIPS|full seconds|full NN|
|---|---|---:|---:|---:|---:|---:|---:|
|0.05|BASE|4|29.2795|0.85329|0.10870|23.192|300.0|
|0.05|FIRST_NOISE|4|29.3364|0.86412|0.10473|22.738|300.0|
|0.05|REAL_ISO|4|29.3925|0.86169|0.10358|35.182|471.0|
|0.05|REAL_STATE|4|29.4001|0.86178|0.10348|39.875|530.8|
|0.1|BASE|4|24.7988|0.61628|0.38595|24.809|300.0|
|0.1|FIRST_NOISE|4|27.6734|0.81201|0.13095|25.112|300.0|
|0.1|REAL_ISO|4|25.5648|0.66717|0.33493|37.215|468.0|
|0.1|REAL_STATE|4|25.5649|0.66739|0.33440|42.552|528.0|

Quality paired on the same270 prefix. Clock measured independently from initialization through all learning/probes/output CPU sync. Additional queries are paid.

Four OLD FFHQ SR images, two noise levels, one seed/image; scalar local Gaussian approximation, not full posterior/curvature, no new-theory or speedup claim. Additional NN calls fully paid. Continuous remapping can cancel current positive zeta amplitude; learned path and D direction retained.
