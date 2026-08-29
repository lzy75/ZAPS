"""
诊断21: ε 预测相关性——测模型是否真在做去噪(比单步PSNR更本质)。

背景(权重已证完整):
  · 权重文件大小精确==官方2.211GB, 无NaN/全零, std正常 → 权重没坏、下载完整。
  · 但1000步无条件生成仍噪声。怀疑转向"模型调用方式"而非权重本身。
  · 单步PSNR 38dB 不够本质(可能恒等映射蒙的)。更本质判据:
    模型预测的 ε_θ(x_t,t) 和【我们加进去的已知噪声 ε】的相关性。
    正常工作: 预测ε与真实ε高度相关(余弦>0.9), 尤其中低噪步。
    调用有问题(t传错/值域错/schedule错): 相关性低。

本脚本: 干净图x0, 用已知噪声ε做 x_t=√ᾱ·x0+√(1-ᾱ)·ε, 让模型预测 ε̂=_predict_eps(x_t,t),
  比 cos(ε̂, ε) 和 ‖ε̂-ε‖/‖ε‖ 相对误差, 扫多个t。对照 imagenet vs ffhq。
  判据:
   · ImageNet cos 明显低于FFHQ(尤其中低噪t) ⇒ 模型调用/输入有问题(非权重), 查t传参/值域/embedding
   · 两者cos都高(>0.9) ⇒ 模型调用对、去噪正常 → 噪声问题在采样累积/schedule而非单步预测
   · 都低 ⇒ 加噪/schedule口径本身错(两模型共用_precompute故都受影响)

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_eps_corr.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from configs.config import IMG_SIZE
from modules.main_single import load_diffusion_model, load_image_as_tensor

SEED = 1000
T_LIST = [50, 100, 200, 400, 600, 800, 999]
IMAGES = {
    "imagenet": "/home/lzy/imagenet/256x256/00000.png",
    "ffhq":     "/home/lzy/FFHQ/00000/00000/00000.png",
}


@torch.no_grad()
def run(dataset, device):
    print(f"\n{'='*66}\n  {dataset}  ε预测相关性 cos(ε̂,ε) (越接近1=模型越在正确去噪)\n{'='*66}", flush=True)
    x0 = load_image_as_tensor(IMAGES[dataset]).to(device)   # [-1,1]
    dm = load_diffusion_model(dataset, device)
    ab = dm.alphas_cumprod
    print(f"  {'t':>5s}{'cos(ε̂,ε)':>12s}{'‖ε̂-ε‖/‖ε‖':>13s}{'‖ε̂‖':>9s}", flush=True)
    for t in T_LIST:
        torch.manual_seed(SEED + t)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + t)
        eps = torch.randn_like(x0)                          # 已知真实噪声
        ab_t = ab[t]
        x_t = ab_t.sqrt() * x0 + (1 - ab_t).sqrt() * eps    # 标准前向加噪
        t_b = torch.full((1,), t, device=device, dtype=torch.long)
        eps_hat = dm._predict_eps(x_t, t_b)                 # 模型预测ε
        cos = F.cosine_similarity(eps_hat.flatten(), eps.flatten(), dim=0).item()
        rel = (eps_hat - eps).flatten().norm().item() / eps.flatten().norm().item()
        print(f"  {t:5d}{cos:12.4f}{rel:13.4f}{eps_hat.flatten().norm().item():9.1f}", flush=True)
    del dm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds in IMAGES:
        if os.path.exists(IMAGES[ds]):
            run(ds, device)
    print("\n判读:", flush=True)
    print("  · 正常模型 cos 应>0.9(中低噪t尤其高)。ImageNet明显低于FFHQ ⇒ 模型调用有问题(非权重),", flush=True)
    print("    查t传参(是否需rescale)/值域/timestep embedding。", flush=True)
    print("  · 两者cos都高 ⇒ 单步去噪本质正常, 噪声问题在采样累积/schedule。", flush=True)
    print("  · 两者cos都低 ⇒ 加噪/schedule口径错(两模型共用故都中招)。", flush=True)


if __name__ == "__main__":
    main()
