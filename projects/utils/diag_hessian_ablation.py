"""
诊断10: Hessian 项(W·D·Wᵀ)消融——定位 ImageNet 高频噪声是否来自 D 引导项。

背景(逐步逼近的根因):
  · 原文确认 eta=1+固定β̃ DDPM祖先采样, 我们已对齐 → 采样噪声项不是根因。
  · 后验均值系数 c1/c2/β̃ 逐字核对=原文Alg1第8行(跳步展开一致) → 均值不是根因。
  · 独立代码审查指认引导项(Alg1第9行):
      guided = (v + (1−ᾱ_t)·W·D·Wᵀ·v) / √ᾱ_t,  v = Aᵀ(y−A·x̂0)
    - D 初始化全子带均匀0.2(含HH高频), loss=‖y−A·x̂0‖²只约束低频→HH子带D不受惩罚、
      优化中梯度≈0停在0.2(diag1实测|D|max≈0.21佐证, D没被推大也没被压到0)。
    - 高噪步 ᾱ_t≈0.006 → 1/√ᾱ_t≈12, 把 W·D·Wᵀ·v 放大~12×。
    - ImageNet宽先验高噪步 x̂0 高频误差大 → v 高频分量大 → 被 D(0.2)×12 灌进零空间高频 → 噪点。
      FFHQ 平滑人脸高频少, 同增益下不显著。解释 TV_recon≈2×GT 且只砸ImageNet。

本实验(对 imagenet+ffhq 各跑):
  A. full   : D 原样(可学), = 现状基线
  B. D_off  : 构造后 self.D 置零并冻结 → 引导退化为 guided=v/√ᾱ_t(无Hessian项)
  C. D_clamp: 优化后把 |D| 限幅到 0.05 → 压制高频增益但保留低频引导
  比 recon/TV, 存图。判据:
   · ImageNet B/C recon 从16.4跳回~22+ 且 TV降到接近GT ⇒ 坐实Hessian高频项是根因
   · FFHQ B/C 不明显掉分 ⇒ 关/限 D 是安全修复方向
   · B/C 仍崩 ⇒ 不是D, 回到score符号/τ索引继续查

顺带先验证小波无损(self_check, 应<1e-5, 排除DWT重构有损)。
用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_hessian_ablation.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
import numpy as np
from configs.config import ZAPS_CONFIG, ZETA_INIT_BY_TASK, TASK_CONFIGS, IMG_SIZE, RESULTS_DIR
from modules.main_single import load_diffusion_model, load_image_as_tensor
from modules.degradations import get_operator
from modules.zaps_algorithm import ZAPS

TASK = "super_resolution"
SEED = 1000
OUT_DIR = os.path.join(RESULTS_DIR, "diag_hessian")
IMAGES = {
    "imagenet": "/home/lzy/imagenet/256x256/00000.png",
    "ffhq":     "/home/lzy/FFHQ/00000/00000/00000.png",
}
D_CLAMP = 0.05


def tv255(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return ((dh + dw) * 127.5).item()


def to_psnr(mse):
    return 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()


def save_png(x, path):
    x = x.detach().float().cpu().clamp(-1, 1)[0]
    arr = ((x + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except Exception:
        np.save(path.replace(".png", ".npy"), arr)


def build(dm, operator):
    cfg = {**ZAPS_CONFIG,
           "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"]),
           "use_learned_var": False}
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    return ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg)


def run_mode(mode, dm, operator, x0_gt, y_obs, device):
    zaps = build(dm, operator)
    if mode == "D_off":
        # 置零并冻结 D → 引导退化为 guided = v/√ᾱ (无 Hessian 项)
        with torch.no_grad():
            zaps.D.zero_()
        zaps.D.requires_grad_(False)
    result = zaps.run(y_obs, verbose=False, x0_gt=x0_gt, final_mode="last_opt")
    x0f = result["x0_final"]
    rec = to_psnr(((x0f.clamp(-1, 1) - x0_gt) ** 2).mean().item())
    tvr = tv255(x0f)
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec, tvr, result["loss_history"][-1], x0f


def run_clamp(dm, operator, x0_gt, y_obs, device):
    """D_clamp: 优化中每个 epoch 后把 |D| 限幅到 D_CLAMP。手动展开 run。"""
    zaps = build(dm, operator)
    # 借用 optimize 的钩子不方便,改为:优化后对 D 限幅,再用固定调度重采样一次
    zaps.run(y_obs, verbose=False, x0_gt=x0_gt, final_mode="last_opt")
    with torch.no_grad():
        zaps.D.clamp_(-D_CLAMP, D_CLAMP)
    # 用限幅后的 D/ζ 重新采样(固定调度, eta=1)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    x0f, _, _ = zaps.sample(y_obs, eta_override=None, init_noise=None, scheduler=None)
    rec = to_psnr(((x0f.clamp(-1, 1) - x0_gt) ** 2).mean().item())
    tvr = tv255(x0f)
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec, tvr, x0f


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    # 0. 小波无损自检
    from modules.wavelet import OrthogonalDWT2D
    dwt = OrthogonalDWT2D(wave="db4", level=3).to(device)
    err = dwt.self_check()
    print(f"[小波自检] db4 重构误差 = {err:.2e}  ({'✓无损' if err < 1e-5 else '✗有损!这可能就是根因'})", flush=True)

    print(f"\n{'='*66}\n  Hessian(D)项消融 (SEED={SEED}, SR, last_opt, eta=1固定β̃)\n{'='*66}", flush=True)
    rows = []
    for ds, img in IMAGES.items():
        if not os.path.exists(img):
            print(f"[跳过] {ds}: {img} 不存在", flush=True)
            continue
        x0_gt = load_image_as_tensor(img).to(device)
        operator = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            y_obs = operator(x0_gt)
        y_up = F.interpolate(y_obs, size=x0_gt.shape[-2:], mode="bicubic", align_corners=False)
        obs = to_psnr(((y_up.clamp(-1, 1) - x0_gt) ** 2).mean().item())
        tv_gt = tv255(x0_gt)

        dm = load_diffusion_model(ds, device)
        print(f"\n── {ds}  (bicubic={obs:.2f}dB, GT_TV={tv_gt:.1f}) ──", flush=True)
        print(f"  {'模式':>10s}{'recon':>10s}{'TV_recon':>11s}", flush=True)

        rec_a, tv_a, _, x_a = run_mode("full", dm, operator, x0_gt, y_obs, device)
        save_png(x_a, os.path.join(OUT_DIR, f"{ds}_full.png"))
        print(f"  {'full(基线)':>10s}{rec_a:10.2f}{tv_a:11.1f}", flush=True)

        rec_b, tv_b, _, x_b = run_mode("D_off", dm, operator, x0_gt, y_obs, device)
        save_png(x_b, os.path.join(OUT_DIR, f"{ds}_Doff.png"))
        print(f"  {'D_off(无Hessian)':>10s}{rec_b:10.2f}{tv_b:11.1f}", flush=True)

        rec_c, tv_c, x_c = run_clamp(dm, operator, x0_gt, y_obs, device)
        save_png(x_c, os.path.join(OUT_DIR, f"{ds}_Dclamp.png"))
        print(f"  {'D_clamp(限|D|≤%.2f)' % D_CLAMP:>10s}{rec_c:10.2f}{tv_c:11.1f}", flush=True)

        rows.append((ds, obs, tv_gt, rec_a, tv_a, rec_b, tv_b, rec_c, tv_c))
        del dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*66}\n  汇总\n{'='*66}", flush=True)
    print(f"  {'数据集':10s}{'bicubic':>8s}{'full':>7s}{'D_off':>8s}{'D_clamp':>9s}{'GT_TV':>7s}", flush=True)
    for ds, obs, tv_gt, ra, ta, rb, tb, rc, tc in rows:
        print(f"  {ds:10s}{obs:8.2f}{ra:7.2f}{rb:8.2f}{rc:9.2f}{tv_gt:7.1f}", flush=True)
    print("\n判读:", flush=True)
    print("  · ImageNet D_off/D_clamp recon 从16.4跳回~22+、TV降 ⇒ 坐实Hessian高频项是根因", flush=True)
    print("  · FFHQ D_off/D_clamp 不明显掉分 ⇒ 关/限D是安全修复", flush=True)
    print("  · 都没救回 ⇒ 不是D, 回查 score符号/τ索引", flush=True)
    print(f"\n  存图: {OUT_DIR}/  ({{ds}}_full/_Doff/_Dclamp.png)", flush=True)


if __name__ == "__main__":
    main()
