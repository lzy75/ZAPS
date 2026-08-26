"""
诊断4: 单步去噪能力测试(去掉"从零生成难"的混淆项) —— ImageNet vs FFHQ。

背景链条:
  诊断1: ζ/D 与 FFHQ 量级相同 → 排除引导过冲。
  诊断2: fp16=fp32(16.54 vs 16.51) → 排除精度。
  诊断3: 同一30步采样代码, FFHQ无条件出连贯图(TV6.6), ImageNet出纯噪声(TV51.7,
         肉眼完全没规律) → 指向 ImageNet 前向本身没在去噪。
  但混淆项: ImageNet是1000类宽分布, 30步无条件生成本就难, 高噪区仅5步可能天生不够,
           不能仅凭"无条件出噪声"断定模型坏。

本实验(去混淆): 拿干净GT, 用 q_sample 加【已知量】噪声到时间步 t, 再用模型做
  单步 Tweedie 去噪 predict_x0, 比 PSNR(x̂₀, GT)。这直接测模型核心去噪能力,
  与"从零生成"无关。对每个 t 同时给出 x_t 自身的 PSNR(去噪前基线),
  正常模型 x̂₀ 应远高于 x_t。
  · ImageNet 随 t 优雅退化(小t高PSNR>30, 像FFHQ) ⇒ 先验前向OK,
    诊断3的噪声只是"宽分布+步数少", 问题在SR引导/算子, 下一步dump逐步x̂₀
  · ImageNet 连小t(t=10/25)都出垃圾(<20dB) 而FFHQ正常 ⇒ 前向确实坏,
    查noise_schedule(betas端点)/attention配置与ckpt训练设定是否一致

不改任何磁盘代码。用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_denoise.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
from configs.config import IMG_SIZE
from modules.main_single import load_diffusion_model, load_image_as_tensor

SEED = 1000
T_LIST = [10, 25, 50, 100, 200, 400, 700]
IMAGES = {
    "ffhq":     "/home/lzy/FFHQ/00000/00000/00000.png",
    "imagenet": "/home/lzy/imagenet/256x256/00000.png",
}


def psnr(a, b):
    mse = ((a.clamp(-1, 1) - b.clamp(-1, 1)) ** 2).mean().item()
    return 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()  # 值域[-1,1]幅度2→峰4


def run_one(dataset, image_path, device):
    print(f"\n{'='*60}\n  {dataset}  单步去噪 x̂₀=Tweedie(x_t,t)\n{'='*60}", flush=True)
    if not os.path.exists(image_path):
        print(f"  [跳过] 图不存在 {image_path}", flush=True)
        return None
    x0_gt = load_image_as_tensor(image_path).to(device)
    dm = load_diffusion_model(dataset, device)

    print(f"  {'t':>5s}{'x_t PSNR':>11s}{'x̂₀ PSNR':>11s}{'增益(应>0)':>12s}", flush=True)
    rows = []
    for t in T_LIST:
        torch.manual_seed(SEED + t)                       # 每个t固定但不同的噪声
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + t)
        noise = torch.randn_like(x0_gt)
        t_b = torch.full((x0_gt.shape[0],), t, device=device, dtype=torch.long)
        with torch.no_grad():
            x_t = dm.q_sample(x0_gt, t_b, noise=noise)    # 加已知噪声
            x0_hat = dm.predict_x0(x_t, t_b)              # 真实 Tweedie 去噪
        p_xt = psnr(x_t, x0_gt)
        p_hat = psnr(x0_hat, x0_gt)
        rows.append((t, p_xt, p_hat))
        print(f"  {t:5d}{p_xt:11.2f}{p_hat:11.2f}{p_hat - p_xt:12.2f}", flush=True)

    del dm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for ds, img in IMAGES.items():
        r = run_one(ds, img, device)
        if r:
            out[ds] = r

    print(f"\n{'='*60}\n  对比(x̂₀ PSNR, 越高=去噪越好)\n{'='*60}", flush=True)
    header = f"  {'t':>5s}" + "".join(f"{ds:>12s}" for ds in out)
    print(header, flush=True)
    for i, t in enumerate(T_LIST):
        line = f"  {t:5d}" + "".join(f"{out[ds][i][2]:12.2f}" for ds in out)
        print(line, flush=True)

    print("\n判读:", flush=True)
    print("  · ImageNet 小t(10/25) x̂₀>30dB 且随t优雅下降(跟FFHQ形态一致) ⇒ 先验前向OK,"
          " 诊断3噪声=宽分布+步数少, 下一步dump SR逐步x̂₀查引导/算子", flush=True)
    print("  · ImageNet 连小t都<20dB(远低于FFHQ) ⇒ 前向确实坏, 查betas端点/attention配置"
          "与该ckpt训练设定, 或换用guided-diffusion原生create_model+官方FLAGS核对", flush=True)


if __name__ == "__main__":
    main()
