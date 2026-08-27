"""
诊断12: optimize 每 epoch PSNR 曲线——验证"每步随机z导致宽分布轨迹不稳、last_opt撞特定z"。

背景(逐步排除):
  · 原文eta=1+固定β̃(已对齐), 跳步β̃正确(fork数值证==guided-diffusion respaced),
    D/Hessian消融无效, learned variance更差, 单步去噪正常, 小波无损。
  · 代码事实: optimize每epoch初始噪声fixed_noise固定, 但ddpm_posterior_step每步
    z=randn_like每次调用重新随机 → 10个epoch的per-step z序列各不同。
    _last_opt_x0 = 最后一个epoch那次特定z序列的轨迹终点(line521)。
  · 假设(fork指出): ζ/D在"每次不同噪声"上做梯度平均, 宽分布ImageNet轨迹对z极敏感→
    梯度信号被噪声污染+last_opt撞最后那次特定z → 崩; 窄分布FFHQ对z不敏感→稳。
    eta=0好正因不注z。

本脚本(零改核心, verbose=True 拿每epoch PSNR): imagenet+ffhq 各跑一次 optimize,
  打印10个epoch的 PSNR 序列 + 标准差。判据:
   · ImageNet PSNR 剧烈跳动(std大)、FFHQ平稳(std小) ⇒ 坐实随机z致宽分布轨迹不稳,
     修复方向: optimize固定per-step z序列 / last_opt改多次平均 / 或最终用确定性重采样
   · 两者PSNR都平稳 ⇒ 不是随机z波动, 假设错, 另查
   · ImageNet PSNR平稳但低(都~16) ⇒ 系统性低而非波动, 也排除本假设

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_optim_curve.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
from configs.config import ZAPS_CONFIG, ZETA_INIT_BY_TASK, TASK_CONFIGS, IMG_SIZE
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.degradations import get_operator
from modules.zaps_algorithm import ZAPS

TASK = "super_resolution"
SEED = 1000
IMAGES = {
    "imagenet": "/home/lzy/imagenet/256x256/00000.png",
    "ffhq":     "/home/lzy/FFHQ/00000/00000/00000.png",
}


def run(dataset, device):
    print(f"\n{'='*70}\n  {dataset}  optimize 每epoch PSNR (eta=1, 每步z随机=现状)\n{'='*70}", flush=True)
    x0_gt = load_image_as_tensor(IMAGES[dataset]).to(device)
    operator = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    with torch.no_grad():
        y_obs = operator(x0_gt)

    dm = load_diffusion_model(dataset, device)
    cfg = {**ZAPS_CONFIG,
           "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"]),
           "use_learned_var": False}
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    zaps = ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg)

    # verbose=True 打印每 epoch PSNR(内部已算好,用同一 x0_est=last_opt那个量)
    zaps.optimize(y_obs, verbose=True, x0_gt=x0_gt)

    del dm, zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds in IMAGES:
        if os.path.exists(IMAGES[ds]):
            run(ds, device)
    print("\n判读(看上面每数据集10个epoch的 PSNR= 那列):", flush=True)
    print("  · ImageNet PSNR 在epoch间剧烈跳动(如22→15→19→16)、FFHQ平稳 ⇒ 随机z致宽分布轨迹不稳,", flush=True)
    print("    last_opt撞最后epoch特定z→崩。修复: 固定per-step z / 最终确定性重采样 / 多次平均。", flush=True)
    print("  · 两者PSNR都平稳 ⇒ 排除本假设。ImageNet平稳但都~16 ⇒ 系统性低,非波动,也排除。", flush=True)


if __name__ == "__main__":
    main()
