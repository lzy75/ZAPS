"""
诊断19: ImageNet 标准 DDPM 无条件生成——1000/250/30步对照,验证"30步噪声"是步数不足还是模型/框架坏。

背景(第0层最该验证的伏笔):
  · diag_uncond: ImageNet 30步跳步(15/10/5)无条件生成 = 纯噪声(TV51.7); FFHQ正常(TV6.6)。
    当时归因"宽分布难", 但四任务全崩后重新审视: 这条线索可能是关键。
  · 单步去噪38dB正常、架构=官方 → 模型能用。那"30步噪声"到底是步数不够, 还是深层问题?
  · 标准guided-diffusion ImageNet无条件生成用250或1000步。我们(和ZAPS)用30步。
    宽分布ImageNet 30步可能天生不够收敛 → 必须靠强ζ/D引导补偿, 而我们ζ/D没学起来。

本脚本(脱离ZAPS跳步/引导框架, 纯教科书DDPM): 用 dm._predict_eps + alphas_cumprod,
  t从N-1逐步到0, 标准后验均值+方差注噪, 跑 num_steps ∈ {1000, 250, 30(均匀)} 无条件生成,
  存图+TV。对照ImageNet vs FFHQ。判据:
   · ImageNet 1000步TV回落到自然图水平(15-25)、出连贯图 ⇒ 模型完全正常, 30步噪声=步数不足,
     坐实"30步宽先验不够, 需强引导补偿", 方向转为"让ζ/D优化生效"
   · ImageNet 1000步仍高TV/噪声 ⇒ 模型或调度在多步下深层问题, 与ZAPS无关(更严重)
   · FFHQ 30步就好、1000步也好 ⇒ 窄先验步数不敏感(对照组)

标准DDPM(逐步, 步长1): 从ᾱ序列反推单步 α_t=ᾱ_t/ᾱ_{t-1}, β_t=1-α_t。
用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_ddpm_uncond.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import numpy as np
from configs.config import IMG_SIZE, RESULTS_DIR
from modules.main_single import load_diffusion_model

SEED = 1000
OUT_DIR = os.path.join(RESULTS_DIR, "diag_ddpm_uncond")
STEP_SETS = [1000, 250, 30]
DATASETS = ["imagenet", "ffhq"]


def tv255(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return ((dh + dw) * 127.5).item()


def stats(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    return f"min={x.min():.2f} max={x.max():.2f} mean={x.mean():.2f} std={x.std():.2f}"


def save_png(x, path):
    x = x.detach().float().cpu().clamp(-1, 1)[0]
    arr = ((x + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except Exception:
        np.save(path.replace(".png", ".npy"), arr)


@torch.no_grad()
def ddpm_sample(dm, num_steps, device):
    """标准教科书 DDPM 无条件采样。ᾱ 全序列(1000)已在dm; 按 num_steps 均匀取子序列时间步。
    每步: ε=model(x,t); x̂0=(x-√(1-ᾱ_t)ε)/√ᾱ_t clamp; 后验均值+√β̃·z。"""
    ab = dm.alphas_cumprod                       # [T] 累积, T=1000
    T = dm.num_steps
    # 均匀取 num_steps 个时间步(含端点), 降序
    ts = torch.linspace(T - 1, 0, num_steps).long().tolist()
    x = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[0], device=device)
    for idx, t in enumerate(ts):
        t_next = ts[idx + 1] if idx + 1 < len(ts) else -1
        t_b = torch.full((1,), t, device=device, dtype=torch.long)
        eps = dm._predict_eps(x, t_b)
        ab_t = ab[t]
        x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)).clamp(-1, 1)
        if t_next < 0:
            x = x0
            break
        ab_prev = ab[t_next]
        # 后验均值(跳步推广) + β̃ 方差
        c1 = ab_prev.sqrt() * (1 - ab_t / ab_prev) / (1 - ab_t).clamp(min=1e-8)
        c2 = (ab_t / ab_prev).sqrt() * (1 - ab_prev) / (1 - ab_t).clamp(min=1e-8)
        mean = c1 * x0 + c2 * x
        beta_tilde = ((1 - ab_prev) / (1 - ab_t).clamp(min=1e-8) * (1 - ab_t / ab_prev)).clamp(min=0)
        x = mean + beta_tilde.sqrt() * torch.randn_like(x)
    return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*64}\n  标准DDPM无条件生成 步数对照 (SEED={SEED})\n{'='*64}", flush=True)
    for ds in DATASETS:
        dm = load_diffusion_model(ds, device)
        print(f"\n── {ds} ──", flush=True)
        print(f"  {'步数':>6s}{'TV':>9s}{'  幅度统计':>10s}", flush=True)
        for N in STEP_SETS:
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            x0 = ddpm_sample(dm, N, device)
            print(f"  {N:6d}{tv255(x0):9.1f}   {stats(x0)}", flush=True)
            save_png(x0, os.path.join(OUT_DIR, f"{ds}_ddpm{N}.png"))
        del dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\n判读(自然图TV约15-25, 纯噪声>40):", flush=True)
    print("  · ImageNet 1000步TV落到15-25且出连贯图 ⇒ 模型正常,30步噪声纯粹步数不足,", flush=True)
    print("    坐实'宽先验需更多步/需强引导补偿', 方向转为让ζ/D优化生效。", flush=True)
    print("  · ImageNet 1000步仍高TV ⇒ 模型/调度多步深层问题(与ZAPS无关,更严重)。", flush=True)
    print("  · FFHQ 各步数都好 ⇒ 窄先验步数不敏感(对照)。", flush=True)
    print(f"\n  存图: {OUT_DIR}/  ({{ds}}_ddpm{{步数}}.png) —— 肉眼看1000步是否出真实自然图", flush=True)


if __name__ == "__main__":
    main()
