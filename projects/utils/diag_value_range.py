"""
诊断18(终结验证): [0,1] 值域协议 —— 原文标定值域, 看 ImageNet 是否跳升。

背景(fork终结排查后, 唯一未验证的具体差异):
  · 官方码不存在(stub)。同类无条件ImageNet SR×4基线(原文Tab.6同页): DPS21.77/ΠGDM22.94/
    DDRM24.33/MCG15.86/Score-SDE14.94。我们16正落MCG/Score-SDE档 = ζ/D引导对ImageNet
    未进入有效工作区(非随机bug, 与ζ优化不动/ζ=0掉到7.3/加大ζ反降 三条吻合)。
  · 原文明写: 图像normalize到[0,1], σ=0.05相对[0,1]标定。我们全程[-1,1](幅度2)。
    σ/ζ/D的相对尺度在[0,1]标定→我们[-1,1]下失配(幅度差2×): 引导残差幅度2×而ζ用原文0.1。
    宽频谱ImageNet对尺度失配敏感, 窄频谱FFHQ容差大(解释只砸ImageNet, 之前排早了)。

本脚本: 完整复刻 SR+ZAPS 但【全程[0,1]语义】——GT/y/噪声/loss/引导都在[0,1],
  仅在调用扩散模型前 [0,1]→[-1,1]、模型出来 [-1,1]→[0,1](模型必须吃[-1,1])。
  不改核心代码, 用一个包装 forward_operator 和一个值域转换的 diffusion_model 代理。
  对 imagenet+ffhq 各跑, 比 [0,1]版 vs 现状[-1,1]版 recon。判据:
   · ImageNet [0,1]版 跳向22+ ⇒ 值域协议是根因! 修法=全pipeline改[0,1]标定
   · ImageNet [0,1]版 仍~16 ⇒ 16dB是该图源/无条件先验真实水平(对齐MCG/Score-SDE),
     停止找bug, 改用整1000张val平均对齐23.82; FFHQ四任务+创新点才是正事。

实现: 把 x0_gt 存成[0,1], 定义 op 在[0,1]算H/transpose(线性算子值域无关),
  噪声σ=0.05加在[0,1] y上; ZAPS内部模型调用需[-1,1], 故包一层 DiffusionModel 代理:
  _predict_eps(x01)= 原模型(_2to1(x01)) 但ZAPS的x轨迹在[0,1]——需同步转换Tweedie。
  ⚠️ 复杂, 为降风险: 本脚本用"等效补偿"近似——在[-1,1]下把 y 的噪声和 ζ 都按 2× 尺度
  调整来模拟[0,1]标定(σ_eff=0.1, 残差按/2 喂引导), 快速看趋势; 若趋势正确再做完整[0,1]改造。

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_value_range.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
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


def to_psnr(mse):
    return 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()


def to_psnr01(mse):
    # [0,1]值域峰值1
    return 10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()


def tv255(x, amp2=True):
    x = x.detach().float().cpu()
    x = x.clamp(-1, 1) if amp2 else x.clamp(0, 1)
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    scale = 127.5 if amp2 else 255.0
    return ((dh + dw) * scale).item()


def run(ds, dm, x0_gt_2, device, mode):
    """mode='baseline'([-1,1]现状) | 'sigma2x'(σ=0.1补偿) | 'guide_half'(引导残差/2)"""
    op = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
    if mode == "sigma2x":
        op.noise.sigma = 0.10           # [0,1]的σ=0.05 等效 [-1,1]的σ=0.10
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    with torch.no_grad():
        y = op(x0_gt_2)
    cfg = {**ZAPS_CONFIG,
           "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"]),
           "use_learned_var": False, "sampler_mode": "ddim"}
    if mode == "guide_half":
        cfg["zeta_init"] = ZETA_INIT_BY_TASK.get(TASK, 0.1) * 0.5   # 补偿残差2×→ζ减半
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    zaps = ZAPS(diffusion_model=dm, forward_operator=op, img_size=IMG_SIZE[0], **cfg)
    result = zaps.run(y, verbose=False, x0_gt=x0_gt_2, final_mode="last_opt")
    x0f = result["x0_final"]
    rec = to_psnr(((x0f.clamp(-1, 1) - x0_gt_2) ** 2).mean().item())
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec, tv255(x0f)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*62}\n  值域协议验证: 现状 vs σ补偿 vs 引导补偿 (SEED={SEED})\n{'='*62}", flush=True)
    print("  (原文[0,1]标定σ=0.05/ζ=0.1; 我们[-1,1]幅度2→尺度失配, 宽谱ImageNet敏感)", flush=True)
    for ds, img in IMAGES.items():
        if not os.path.exists(img):
            continue
        x0_gt = load_image_as_tensor(img).to(device)   # [-1,1]
        dm = load_diffusion_model(ds, device)
        print(f"\n── {ds} ──", flush=True)
        for mode, label in [("baseline", "现状[-1,1]"),
                            ("sigma2x", "σ=0.10(补偿噪声)"),
                            ("guide_half", "ζ×0.5(补偿引导2×)")]:
            rec, tv = run(ds, dm, x0_gt, device, mode)
            print(f"  {label:>20s}  recon={rec:.2f}  TV={tv:.1f}", flush=True)
        del dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\n判读:", flush=True)
    print("  · 某补偿让ImageNet明显升 ⇒ 值域尺度失配是根因, 做完整[0,1]改造", flush=True)
    print("  · 三者ImageNet都~16纹丝不动 ⇒ 16dB是无条件先验真实水平(对齐MCG15.86/Score-SDE14.94),", flush=True)
    print("    非bug。停止找bug: 改用整1000张val平均对齐23.82, 精力转FFHQ四任务+创新点。", flush=True)


if __name__ == "__main__":
    main()
