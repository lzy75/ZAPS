"""
诊断22: 引导有效性量化——引导到底有没有推动轨迹/降低残差(ImageNet vs FFHQ)。

背景(终局定论: 是bug, 根因在引导未生效):
  · 铁证: DPS用同一无条件先验+朴素后验引导到21.77。ZAPS=DPS+ζ/D,理论下限就是DPS。
    我们16<DPS 5.8dB,先验对两者相同→不可能"先验弱",一定是【我们的引导没把先验拉住】。
  · 组件全好: 单步ε cos>0.92、权重完整、采样公式对。是引导退化到近乎无效(16=MCG档)。
  · 1000步无条件噪声=无条件ImageNet正常现象(无锚点会漂),不是停止理由——超分靠y约束不靠自由生成。
  · 引导 correction=ζ·(1/√ᾱ)·(I+(1-ᾱ)WDWᵀ)·Aᵀ(y-Ax̂0)。公式对、伴随非根因、ζ大小无用。
    → 问题不在公式形式, 而在它有没有真的【改变轨迹+降低残差】。

本脚本逐步量化(用初始ζ=0.1/D=0.2, 复刻采样循环):
  每步记录: ‖correction‖/‖x_uncond‖(引导相对无条件步的幅度) 、
            引导前残差‖y-Ax̂0‖ vs 下一步残差(引导有没有让残差下降)、
            x̂0 PSNR。对照 imagenet vs ffhq。
  判据:
   · ImageNet ‖correction‖/‖x‖ 远小于FFHQ ⇒ 引导幅度太小推不动轨迹(量级问题)
   · ImageNet 残差不随步下降(FFHQ下降) ⇒ 引导方向对残差无效
   · 两者correction幅度相近但ImageNet PSNR不升 ⇒ 引导动了但方向对宽先验无益(更深)

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_guidance_eff.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
from configs.config import ZAPS_CONFIG, ZETA_INIT_BY_TASK, TASK_CONFIGS, IMG_SIZE
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.degradations import get_operator
from modules.zaps_algorithm import ZAPS, ddpm_posterior_step

TASK = "super_resolution"
SEED = 1000
IMAGES = {
    "imagenet": "/home/lzy/imagenet/256x256/00000.png",
    "ffhq":     "/home/lzy/FFHQ/00000/00000/00000.png",
}


def to_psnr(mse):
    return 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()


@torch.no_grad()
def trace(dataset, device):
    print(f"\n{'='*76}\n  {dataset}  引导有效性 (初始ζ=0.1/D=0.2, eta=1)\n{'='*76}", flush=True)
    x0_gt = load_image_as_tensor(IMAGES[dataset]).to(device)
    operator = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    y = operator(x0_gt)

    dm = load_diffusion_model(dataset, device)
    cfg = {**ZAPS_CONFIG, "zeta_init": ZETA_INIT_BY_TASK.get(TASK, 0.1),
           "use_learned_var": False, "sampler_mode": "ddpm"}
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    zaps = ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg)

    ab = dm.alphas_cumprod
    tau = zaps.tau
    S = len(tau)
    torch.manual_seed(SEED + 7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + 7)
    x = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[0], device=device)

    print(f"  {'步':>3s}{'t':>5s}{'‖corr‖/‖unc‖':>14s}{'残差‖y-Ax̂0‖':>13s}{'x̂0PSNR':>9s}", flush=True)
    for i in range(S - 1, -1, -1):
        t_curr = tau[i].item()
        t_prev = tau[i - 1].item() if i > 0 else -1
        t_b = torch.full((1,), t_curr, device=device, dtype=torch.long)
        eps = dm._predict_eps(x, t_b)
        ab_t = ab[t_curr]; sqrt_ab_t = ab_t.sqrt(); sqrt_1mab = (1 - ab_t).sqrt()
        x0 = ((x - sqrt_1mab * eps) / sqrt_ab_t.clamp(min=1e-8)).clamp(-1, 1)
        resid = (y - operator.H(x0)).flatten().norm().item()
        p_x0 = to_psnr(((x0 - x0_gt) ** 2).mean().item())

        x_unc = ddpm_posterior_step(x, x0, t_curr, t_prev, ab, eta=1.0, mode="ddpm")
        v = operator.transpose(y - operator.H(x0))
        Hv = zaps.dwt.synthesis(zaps.D[i] * zaps.dwt.analysis(v))
        guided = (v + (1 - ab_t) * Hv) / sqrt_ab_t.clamp(min=1e-8)
        corr = zaps.zeta[i] * guided
        ratio = corr.flatten().norm().item() / (x_unc.flatten().norm().item() + 1e-8)
        x = x_unc + corr
        if (S - 1 - i) < 8 or i <= 6 or i % 5 == 0:
            print(f"  {S-1-i:3d}{t_curr:5d}{ratio:14.4f}{resid:13.2f}{p_x0:9.2f}", flush=True)

    rec = to_psnr(((x.clamp(-1, 1) - x0_gt) ** 2).mean().item())
    print(f"  → 最终recon={rec:.2f}", flush=True)
    del dm, zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds in IMAGES:
        if os.path.exists(IMAGES[ds]):
            trace(ds, device)
    print("\n判读(‖corr‖/‖unc‖ = 引导修正相对无条件步的幅度):", flush=True)
    print("  · ImageNet 该比值 << FFHQ ⇒ 引导幅度太小推不动轨迹(量级/尺度问题, 可调)", flush=True)
    print("  · 两者比值相近但ImageNet x̂0 PSNR不升 ⇒ 引导动了但方向对宽先验无益(需改引导方向)", flush=True)
    print("  · 对比FFHQ: 看FFHQ引导幅度多大、残差怎么随步降、我们ImageNet差在哪一环", flush=True)


if __name__ == "__main__":
    main()
