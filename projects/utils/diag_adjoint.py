"""
诊断14: 超分算子伴随性检验 + ζ=0二分 + transpose修复试验。

背景(采样侧已全排除):
  · eta/learned var/D消融/ddim(Eq.25) 全没救 ImageNet(16.4). 采样步不是根因。
  · epoch1即崩、ζ/D优化不动 → 问题在优化前的"引导方向/观测"侧, 不在采样。
  · 代码发现: SuperResolutionOperator.H 用 antialias=True 下采样,
    transpose 用【无antialias】bicubic 上采样(注释自认"伪逆近似") → 非真伴随。
    引导 v=Aᵀ(y−A·x̂0) 方向在高频段错 → ImageNet高频丰富被错误引导灌进零空间→噪点;
    FFHQ平滑高频少→无感。符合"只砸ImageNet"。

本脚本三件事:
  A. 伴随性数值检验: 随机a,b, 比 ⟨H(a),b⟩ vs ⟨a,Aᵀ(b)⟩。真伴随应几乎相等(相对误差<1%)。
     误差大 ⇒ 坐实 transpose 非伴随。
  B. ζ=0 二分: 关掉全部引导(zeta置零冻结)跑ImageNet+FFHQ, 看纯无条件采样recon。
     隔离"引导错" vs "采样/先验错"。
  C. transpose修复试验: 临时把 operator.transpose 换成 antialias=True 的一致版本,
     重跑ImageNet ZAPS, 看recon是否回升。回升⇒伴随不匹配是根因。

用法(服务器): /home/lzy/anaconda3/bin/python3 projects/utils/diag_adjoint.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
import types
from configs.config import ZAPS_CONFIG, ZETA_INIT_BY_TASK, TASK_CONFIGS, IMG_SIZE, RESULTS_DIR
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


def tv255(x):
    x = x.detach().float().cpu().clamp(-1, 1)
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return ((dh + dw) * 127.5).item()


def adjoint_test(operator, device):
    """⟨H(a),b⟩ 应 == ⟨a,Aᵀ(b)⟩ (真伴随)。返回相对误差。"""
    torch.manual_seed(SEED)
    a = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[0], device=device)   # 图像空间
    Ha = operator.H(a)                                               # 下采样空间
    b = torch.randn_like(Ha)                                         # 下采样空间
    Atb = operator.transpose(b, output_size=(IMG_SIZE[0], IMG_SIZE[0]))
    lhs = (Ha * b).sum().item()          # ⟨H a, b⟩
    rhs = (a * Atb).sum().item()         # ⟨a, Aᵀ b⟩
    rel = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12) * 2
    return lhs, rhs, rel


def make_antialias_transpose(operator):
    """把 transpose 换成 antialias=True 的一致上采样(试验用)。"""
    def transpose(self, y, output_size=None):
        scale = self.scale_factor
        H_out = y.shape[2] * scale if output_size is None else output_size[0]
        W_out = y.shape[3] * scale if output_size is None else output_size[1]
        return F.interpolate(y, size=(H_out, W_out), mode="bicubic",
                             align_corners=False, antialias=True)
    operator.transpose = types.MethodType(transpose, operator)


def run_zaps(ds, dm, operator, x0_gt, y_obs, device, zeta_zero=False):
    cfg = {**ZAPS_CONFIG,
           "zeta_init": ZETA_INIT_BY_TASK.get(TASK, ZAPS_CONFIG["zeta_init"]),
           "use_learned_var": False, "sampler_mode": "ddpm"}
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    zaps = ZAPS(diffusion_model=dm, forward_operator=operator, img_size=IMG_SIZE[0], **cfg)
    if zeta_zero:
        with torch.no_grad():
            zaps.zeta.zero_()
        zaps.zeta.requires_grad_(False)
    result = zaps.run(y_obs, verbose=False, x0_gt=x0_gt, final_mode="last_opt")
    x0f = result["x0_final"]
    rec = to_psnr(((x0f.clamp(-1, 1) - x0_gt) ** 2).mean().item())
    del zaps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec, tv255(x0f)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── A. 伴随性检验(不需模型, 先跑) ──
    print(f"\n{'='*66}\n  A. 超分算子伴随性检验 ⟨H(a),b⟩ vs ⟨a,Aᵀ(b)⟩\n{'='*66}", flush=True)
    op = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
    lhs, rhs, rel = adjoint_test(op, device)
    print(f"  ⟨H(a),b⟩={lhs:.3f}  ⟨a,Aᵀ(b)⟩={rhs:.3f}  相对误差={rel*100:.1f}%", flush=True)
    print(f"  → {'✓近似伴随(误差<5%)' if rel < 0.05 else '✗非伴随!引导方向系统性偏,高频最受影响'}", flush=True)

    # ── B & C: 每数据集 ──
    for ds, img in IMAGES.items():
        if not os.path.exists(img):
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

        dm = load_diffusion_model(ds, device)
        print(f"\n{'='*66}\n  {ds}  (bicubic={obs:.2f}dB)\n{'='*66}", flush=True)

        # 基线(现状 transpose)
        rec0, tv0 = run_zaps(ds, dm, operator, x0_gt, y_obs, device, zeta_zero=False)
        print(f"  [基线 现状transpose]         recon={rec0:.2f}  TV={tv0:.1f}", flush=True)

        # B. ζ=0 纯无条件(隔离引导)
        recz, tvz = run_zaps(ds, dm, operator, x0_gt, y_obs, device, zeta_zero=True)
        print(f"  [B ζ=0 关引导 纯无条件]      recon={recz:.2f}  TV={tvz:.1f}", flush=True)

        # C. transpose换antialias一致版
        op2 = get_operator(TASK, device=device, **TASK_CONFIGS[TASK])
        make_antialias_transpose(op2)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            y_obs2 = op2(x0_gt)
        rec2, tv2 = run_zaps(ds, dm, op2, x0_gt, y_obs2, device, zeta_zero=False)
        print(f"  [C transpose改antialias一致]  recon={rec2:.2f}  TV={tv2:.1f}", flush=True)

        del dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n判读:", flush=True)
    print("  · A 相对误差大(>5%) ⇒ transpose非伴随, 引导方向偏。", flush=True)
    print("  · B ζ=0纯无条件: ImageNet若也就~16 ⇒ 引导无关,是无条件先验/采样本身;", flush=True)
    print("      若ζ=0反而不崩(接近bicubic或更高) ⇒ 是引导把它带崩的。", flush=True)
    print("  · C transpose改antialias后ImageNet回升 ⇒ 伴随不匹配是根因, 修transpose即可。", flush=True)


if __name__ == "__main__":
    main()
