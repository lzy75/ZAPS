"""
诊断11: 逐步 dump 采样轨迹——对比 ImageNet vs FFHQ 每步的注入噪声量级与 x̂0 质量。

背景(逐步排除后的定位):
  · 原文=eta=1+固定β̃ DDPM祖先采样(已对齐), D/Hessian项消融无效(16.4→16.9),
    learned variance更差, 单步去噪38dB正常, 小波无损。
  · 唯一让ImageNet变好的是eta=0(不注噪,22.5)。→ 问题在"每步注入的那点噪声,
    注了却没在后续步被消化", 高噪步嫌疑最大。
  · 新假设: 跳步采样下 β̃ 的量级在高噪步(大t、跳步跨度大)可能异常,
    ImageNet依赖高噪步→被过量注入噪声冲垮; FFHQ低噪步为主→几乎无感。

本脚本(纯诊断, 复刻真实采样循环, 不改核心代码): 对 imagenet+ffhq 各跑一次固定调度
  eta=1 采样, 逐步记录:
    t, √ᾱ_t, β̃(注噪方差), √β̃(注噪尺度),
    ‖mean‖(后验均值范数), ‖√β̃·z‖(实际注入噪声范数), 注噪占比=‖noise‖/‖mean‖,
    x̂0-vs-GT PSNR(该步Tweedie估计质量)
  看:
   · 高噪步(前几步,大t) ImageNet 注噪占比 >> FFHQ, 或 √β̃ 量级异常大 ⇒ 坐实高噪步过量注入
   · x̂0 PSNR 在ImageNet某步后不升反降 ⇒ 定位崩在哪几步
   · 若两者注噪占比相近 ⇒ 不是注入量级, 排除该假设

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_step_dump.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
from configs.config import ZAPS_CONFIG, ZETA_INIT_BY_TASK, TASK_CONFIGS, IMG_SIZE
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.degradations import get_operator
from modules.zaps_algorithm import ZAPS, build_irregular_timesteps

TASK = "super_resolution"
SEED = 1000
IMAGES = {
    "imagenet": "/home/lzy/imagenet/256x256/00000.png",
    "ffhq":     "/home/lzy/FFHQ/00000/00000/00000.png",
}


def to_psnr(mse):
    return 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()


@torch.no_grad()
def dump_traj(dataset, device):
    print(f"\n{'='*78}\n  {dataset}  逐步采样 dump (eta=1, 固定β̃, 初始ζ=0.1/D=0.2)\n{'='*78}", flush=True)
    print("  (diag_optim_curve已证ζ/D优化几乎不动、epoch1即崩,故直接用初始参数dump)", flush=True)
    x0_gt = load_image_as_tensor(IMAGES[dataset]).to(device)
    operator = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    y_obs = operator(x0_gt)

    dm = load_diffusion_model(dataset, device)
    cfg = {**ZAPS_CONFIG,
           "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"]),
           "use_learned_var": False}
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    zaps = ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg)
    # 用初始 ζ/D 直接 dump 采样轨迹(不 optimize,因已证优化不改善且会触发 backward)

    ab = dm.alphas_cumprod
    tau = zaps.tau
    S = len(tau)
    B = 1
    torch.manual_seed(SEED + 777)   # 采样噪声独立固定
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + 777)
    x = torch.randn(B, 3, IMG_SIZE[0], IMG_SIZE[0], device=device)

    print(f"  {'步':>3s}{'t':>5s}{'√ᾱ_t':>8s}{'√β̃':>9s}{'‖mean‖':>10s}{'‖noise‖':>10s}"
          f"{'噪占比':>8s}{'x̂0 PSNR':>9s}", flush=True)
    for i in range(S - 1, -1, -1):
        t_curr = tau[i].item()
        t_prev = tau[i - 1].item() if i > 0 else -1
        t_batch = torch.full((B,), t_curr, device=device, dtype=torch.long)
        eps = dm._predict_eps(x, t_batch)

        ab_t = ab[t_curr]; sqrt_ab_t = ab_t.sqrt(); sqrt_1mab = (1.0 - ab_t).sqrt()
        x0_pred = ((x - sqrt_1mab * eps) / sqrt_ab_t.clamp(min=1e-8)).clamp(-1.0, 1.0)
        p_x0 = to_psnr(((x0_pred - x0_gt) ** 2).mean().item())

        if t_prev < 0:
            print(f"  {S-1-i:3d}{t_curr:5d}{sqrt_ab_t.item():8.4f}{'--':>9s}"
                  f"{'--':>10s}{'--':>10s}{'--':>8s}{p_x0:9.2f}  (末步→x̂0)", flush=True)
            x = x0_pred
            break

        ab_prev = ab[t_prev]
        c1 = ab_prev.sqrt() * (1.0 - ab_t / ab_prev) / (1.0 - ab_t).clamp(min=1e-8)
        c2 = (ab_t / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8)
        mean = c1 * x0_pred + c2 * x
        beta_tilde = ((1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8) * (1.0 - ab_t / ab_prev)).clamp(min=0.0)
        z = torch.randn_like(x)
        noise = beta_tilde.sqrt() * z

        # 引导项
        residual = y_obs - operator.H(x0_pred)
        v = operator.transpose(residual)
        Hv = zaps.dwt.synthesis(zaps.D[i] * zaps.dwt.analysis(v))
        guided = (v + (1.0 - ab_t) * Hv) / sqrt_ab_t.clamp(min=1e-8)
        correction = zaps.zeta[i] * guided

        x = mean + noise + correction

        nm = mean.flatten().norm().item()
        nn = noise.flatten().norm().item()
        print(f"  {S-1-i:3d}{t_curr:5d}{sqrt_ab_t.item():8.4f}{beta_tilde.sqrt().item():9.4f}"
              f"{nm:10.2f}{nn:10.2f}{(nn/max(nm,1e-8)):8.3f}{p_x0:9.2f}", flush=True)

    p_final = to_psnr(((x.clamp(-1, 1) - x0_gt) ** 2).mean().item())
    print(f"  → 最终 recon PSNR = {p_final:.2f}", flush=True)
    del dm, zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds in IMAGES:
        if os.path.exists(IMAGES[ds]):
            dump_traj(ds, device)
    print("\n判读:", flush=True)
    print("  · ImageNet 高噪步(前几步,√ᾱ小) 噪占比 >> FFHQ ⇒ 高噪步过量注入是根因", flush=True)
    print("  · x̂0 PSNR 在某步骤后崩(不升反降) ⇒ 定位崩溃起点", flush=True)
    print("  · 两者噪占比/√β̃ 量级相近 ⇒ 排除注入量级, 另找方向", flush=True)


if __name__ == "__main__":
    main()
