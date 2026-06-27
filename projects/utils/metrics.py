"""
定量评估指标：PSNR、SSIM、LPIPS
输入均为 torch.Tensor，值域 [-1, 1]，形状 [B, C, H, W] 或 [C, H, W]

依赖：
  - scikit-image  (SSIM)
  - lpips==0.1.4  (LPIPS)
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage
import lpips as lpips_lib


# ═══════════════════════════════════════════════════════
# 可调参数
# ═══════════════════════════════════════════════════════
LPIPS_NET    = "alex"   # ← 可调："alex"（论文常用）| "vgg" | "squeeze"
SSIM_WIN_SIZE = 11      # ← 可调：SSIM 窗口尺寸，scikit-image 默认 7，论文惯例 11
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
# 内部工具
# ───────────────────────────────────────────────────────

def _to_uint8_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    将 [-1,1] 张量转为 [0,255] uint8 numpy 数组 [H, W, C]
    接受 [B,C,H,W] 或 [C,H,W] 或 [H,W,C]
    """
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t[0]                        # 取第一张
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)          # [C,H,W] → [H,W,C]
    arr = ((t.float().clamp(-1, 1) + 1.0) / 2.0 * 255.0).numpy().astype(np.uint8)
    return arr


def _to_float_numpy(tensor: torch.Tensor) -> np.ndarray:
    """将 [-1,1] 张量转为 [0,1] float32 numpy [H, W, C]"""
    return _to_uint8_numpy(tensor).astype(np.float32) / 255.0


# ───────────────────────────────────────────────────────
# PSNR
# ───────────────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 max_val: float = 1.0) -> float:
    """
    峰值信噪比 PSNR（dB），越高越好

    参数:
        pred   : 预测图像，值域 [-1, 1]
        target : 参考图像，值域 [-1, 1]
        max_val: 像素最大值（归一化到 [0,1] 后为 1.0）  ← 可调
    返回:
        psnr (float): dB 值；+inf 表示完全相同
    """
    pred_f   = _to_float_numpy(pred)        # [H,W,C] in [0,1]
    target_f = _to_float_numpy(target)

    mse = np.mean((pred_f - target_f) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


# ───────────────────────────────────────────────────────
# SSIM
# ───────────────────────────────────────────────────────

def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 win_size: int = SSIM_WIN_SIZE) -> float:
    """
    结构相似性指数 SSIM，越高越好（范围 [0, 1]）

    参数:
        pred    : 预测图像，值域 [-1, 1]
        target  : 参考图像，值域 [-1, 1]
        win_size: 滑动窗口尺寸（奇数），默认 11  ← 可调
    返回:
        ssim (float): [0, 1]
    """
    pred_f   = _to_float_numpy(pred)        # [H,W,C] in [0,1]
    target_f = _to_float_numpy(target)

    is_color = pred_f.ndim == 3 and pred_f.shape[2] == 3
    val = ssim_skimage(
        target_f, pred_f,
        data_range=1.0,
        win_size=win_size,          # ← 可调
        channel_axis=2 if is_color else None,
    )
    return float(val)


# ───────────────────────────────────────────────────────
# LPIPS
# ───────────────────────────────────────────────────────

# 延迟初始化，避免每次调用都重新加载网络
_lpips_fn: dict = {}


def _get_lpips_fn(net: str, device: torch.device) -> lpips_lib.LPIPS:
    key = (net, str(device))
    if key not in _lpips_fn:
        _lpips_fn[key] = lpips_lib.LPIPS(net=net).to(device)
        _lpips_fn[key].eval()
    return _lpips_fn[key]


def compute_lpips(pred: torch.Tensor, target: torch.Tensor,
                  net: str = LPIPS_NET) -> float:
    """
    感知相似度 LPIPS，越低越好

    参数:
        pred   : 预测图像，值域 [-1, 1]，[B,C,H,W] 或 [C,H,W]
        target : 参考图像，值域 [-1, 1]，形状与 pred 相同
        net    : 感知网络，"alex"（默认）| "vgg" | "squeeze"  ← 可调
    返回:
        lpips (float): ≥ 0，越小越相似
    """
    # 统一到 [1, C, H, W]
    def _prep(t):
        t = t.detach().cpu().float().clamp(-1, 1)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        return t

    pred_t   = _prep(pred)
    target_t = _prep(target)

    device = pred_t.device
    fn = _get_lpips_fn(net, device)

    with torch.no_grad():
        val = fn(pred_t, target_t)
    return float(val.mean().item())


# ───────────────────────────────────────────────────────
# 批量评估（一次性计算三项指标）
# ───────────────────────────────────────────────────────

def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor,
                        lpips_net: str = LPIPS_NET) -> dict:
    """
    同时计算 PSNR / SSIM / LPIPS

    参数:
        pred      : 预测图像，值域 [-1, 1]
        target    : 参考图像，值域 [-1, 1]
        lpips_net : LPIPS 感知网络  ← 可调
    返回:
        {"psnr": float, "ssim": float, "lpips": float}
    """
    return {
        "psnr":  compute_psnr(pred, target),
        "ssim":  compute_ssim(pred, target),
        "lpips": compute_lpips(pred, target, net=lpips_net),
    }
