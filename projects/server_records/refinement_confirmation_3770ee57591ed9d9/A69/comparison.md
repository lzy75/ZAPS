# A69 next-response scalar control

|sigma|method|n|PSNR|SSIM|LPIPS|full seconds|full NN|
|---|---|---:|---:|---:|---:|---:|---:|
|0.05|BASE|4|29.2478|0.85199|0.10905|22.439|300.0|
|0.05|FIRST_NOISE|4|29.3363|0.86412|0.10473|23.396|300.0|
|0.05|REAL_ISO|4|29.3925|0.86169|0.10357|36.246|471.0|
|0.05|MODEL_NEWTON|4|28.2229|0.77957|0.22539|29.669|351.2|
|0.05|AUTO_NEWTON|4|29.2942|0.84369|0.13906|33.692|350.8|
|0.1|BASE|4|24.7994|0.61576|0.38655|24.696|300.0|
|0.1|FIRST_NOISE|4|27.6734|0.81201|0.13095|25.046|300.0|
|0.1|REAL_ISO|4|25.5650|0.66718|0.33492|37.614|468.0|
|0.1|MODEL_NEWTON|4|23.1545|0.52408|0.54313|28.784|350.5|
|0.1|AUTO_NEWTON|4|26.3095|0.70454|0.29931|33.393|349.2|

Quality paired on the same270 prefix. Clock measured independently from initialization through all learning/probes/output CPU sync. Additional queries are paid.

Four OLD FFHQ SR images, two noise levels, one seed/image; scalar local Gaussian approximation, same A68 REAL_ISO objective with beta1; no global-minimum or speedup claim. Input backwards and NN calls fully paid. Continuous remapping can cancel current positive zeta amplitude; learned path and D direction retained.
