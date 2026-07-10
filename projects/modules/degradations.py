"""
退化算子（前向模型 H）
每个算子封装为 nn.Module，支持 autograd，可直接用于 ZAPS 梯度计算

使用约定：
  - 输入 x       : [B, C, H, W]，值域 [-1, 1]
  - 输出 y       : [B, C, H', W']，值域与 x 相同（超分时 H'<H）
  - 观测模型     : y = forward(x) = H(x) + noise
  - ZAPS 梯度用  : H(x)  ← 纯算子，无噪声
  - 伴随算子     : transpose(y) = H^T(y)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════
# 可调参数汇总（修改此处即可全局生效）
# ═══════════════════════════════════════════════════════
# 高斯模糊
GAUSSIAN_KERNEL_SIZE = 61       # ← 可调：核尺寸（奇数），论文值 61
GAUSSIAN_SIGMA       = 3.0      # ← 可调：高斯标准差，论文值 3.0

# 修复（Inpainting）
INPAINT_RANDOM_RATIO = 0.7      # ← 可调：随机掩码遮挡比例，论文值 0.7（70%）
INPAINT_BOX_SIZE     = 128      # ← 可调：方形掩码边长（像素），论文值 128

# 运动模糊
MOTION_KERNEL_SIZE   = 61       # ← 可调：运动核尺寸（奇数）
MOTION_INTENSITY     = 0.5      # ← 可调：论文/DPS 使用 motionblur intensity=0.5
MOTION_SEED          = 0        # ← 可调：固定核，保证复现实验可比
MOTION_ANGLE_DEG     = None     # ← 可调：设为角度时退回线性核；None 使用随机方向核

# 超分辨率
SR_SCALE_FACTOR      = 4        # ← 可调：下采样倍率，论文通常为 4

# 观测噪声
NOISE_SIGMA          = 0.05     # ← 可调：加性高斯噪声标准差，论文值 0.05
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
# 加性高斯噪声
# ───────────────────────────────────────────────────────

class GaussianNoise(nn.Module):
    """
    加性高斯噪声 n ~ N(0, sigma^2·I)
    论文所有退化任务统一叠加此噪声

    可调参数:
        sigma (float): 噪声标准差，论文值 0.05
    """
    def __init__(self, sigma: float = NOISE_SIGMA):
        super().__init__()
        self.sigma = sigma  # ← 可调

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma == 0:
            return x
        return x + torch.randn_like(x) * self.sigma


# ───────────────────────────────────────────────────────
# 1. 高斯去模糊
# ───────────────────────────────────────────────────────

def _gaussian_kernel_2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """生成归一化二维高斯卷积核 [1, 1, K, K]"""
    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32) - half
    g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g2d = g1d[:, None] * g1d[None, :]
    g2d = g2d / g2d.sum()
    return g2d.view(1, 1, kernel_size, kernel_size)


class GaussianBlurOperator(nn.Module):
    """
    各向同性高斯模糊  H(x) = x * k_gaussian

    可调参数:
        kernel_size (int)  : 卷积核尺寸（奇数），论文值 61
        sigma       (float): 高斯标准差，论文值 3.0
        noise_sigma (float): 叠加观测噪声标准差，论文值 0.05
    """
    def __init__(
        self,
        kernel_size: int   = GAUSSIAN_KERNEL_SIZE,
        sigma:       float = GAUSSIAN_SIGMA,
        noise_sigma: float = NOISE_SIGMA,
    ):
        super().__init__()
        self.kernel_size = kernel_size  # ← 可调
        self.sigma       = sigma        # ← 可调
        self.noise       = GaussianNoise(noise_sigma)

        # 卷积核注册为 buffer（不参与梯度，随模型移动设备）
        kernel = _gaussian_kernel_2d(kernel_size, sigma)
        self.register_buffer("kernel", kernel)

    def H(self, x: torch.Tensor) -> torch.Tensor:
        """纯高斯模糊，无噪声（供 ZAPS 梯度计算使用）"""
        B, C, Hh, W = x.shape
        pad = self.kernel_size // 2
        kernel_c = self.kernel.expand(C, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(x, kernel_c, padding=pad, groups=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """H(x) + 噪声，用于生成观测 y"""
        return self.noise(self.H(x))

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """H^T：高斯核对称，H^T = H"""
        return self.H(y)


# ───────────────────────────────────────────────────────
# 2. 修复（Inpainting）
# ───────────────────────────────────────────────────────

class InpaintingOperator(nn.Module):
    """
    图像修复退化：H(x) = M ⊙ x，M 为二值掩码（1=保留，0=遮挡）

    支持两种掩码模式（可同时启用）：
      - random : 随机像素级掩码，遮挡比例 random_ratio
      - box    : 中心方形掩码，边长 box_size

    可调参数:
        random_ratio (float) : 随机遮挡比例，论文值 0.7（70% 像素被遮挡）
        box_size     (int)   : 方形掩码边长（像素），论文值 128；0 表示不使用
        mode         (str)   : "random" | "box" | "both"
        noise_sigma  (float) : 叠加观测噪声标准差，论文值 0.05
        seed         (int)   : 随机种子；-1 表示每次随机
    """
    def __init__(
        self,
        random_ratio: float = INPAINT_RANDOM_RATIO,
        box_size:     int   = INPAINT_BOX_SIZE,
        mode:         str   = "random",     # ← 可调："random" | "box" | "both"
        noise_sigma:  float = NOISE_SIGMA,
        seed:         int   = -1,           # ← 可调：固定掩码用正整数
    ):
        super().__init__()
        self.random_ratio = random_ratio    # ← 可调
        self.box_size     = box_size        # ← 可调
        self.mode         = mode            # ← 可调
        self.seed         = seed            # ← 可调
        self.noise        = GaussianNoise(noise_sigma)
        self._mask_cache  = None            # 缓存掩码，同一实例内复用

    def _make_mask(self, B: int, C: int, H: int, W: int,
                   device: torch.device) -> torch.Tensor:
        """生成 [B, 1, H, W] 二值掩码（1=保留）"""
        if self.seed >= 0:
            torch.manual_seed(self.seed)

        mask = torch.ones(B, 1, H, W, device=device)

        if self.mode in ("random", "both"):
            # 随机丢弃 random_ratio 比例的像素
            drop = torch.rand(B, 1, H, W, device=device) < self.random_ratio
            mask[drop] = 0.0

        if self.mode in ("box", "both"):
            # 中心方形遮挡
            s = self.box_size
            h0 = max(0, (H - s) // 2)
            w0 = max(0, (W - s) // 2)
            mask[:, :, h0:h0 + s, w0:w0 + s] = 0.0

        return mask  # [B, 1, H, W]

    def H(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """纯掩码遮挡，无噪声（供 ZAPS 梯度计算使用）"""
        B, C, Hh, W = x.shape
        if mask is None:
            if self._mask_cache is None or self._mask_cache.shape != (B, 1, Hh, W):
                self._mask_cache = self._make_mask(B, C, Hh, W, x.device)
            mask = self._mask_cache
        return x * mask

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """H(x) + 噪声，用于生成观测 y"""
        return self.noise(self.H(x, mask))

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """H^T：掩码操作自伴随，H^T = H"""
        return self.H(y)

    def get_mask(self, B: int, C: int, H: int, W: int,
                 device: torch.device) -> torch.Tensor:
        """显式获取掩码（供 ZAPS 算法使用）"""
        return self._make_mask(B, C, H, W, device)


# ───────────────────────────────────────────────────────
# 3. 运动模糊
# ───────────────────────────────────────────────────────

def _motion_kernel_2d(
    kernel_size: int,
    intensity: float = MOTION_INTENSITY,
    seed: int = MOTION_SEED,
    angle_deg: float = MOTION_ANGLE_DEG,
) -> torch.Tensor:
    """
    生成归一化运动模糊核 [1, 1, K, K]。
    默认使用固定随机方向核，对齐论文/DPS 的 motionblur intensity 设置；
    若显式传入 angle_deg，则使用可控线性核便于消融。
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2.0
    intensity = float(np.clip(intensity, 1e-3, 1.0))

    if angle_deg is None:
        rng = np.random.default_rng(seed)
        angle_rad = rng.uniform(0.0, np.pi)
    else:
        angle_rad = np.deg2rad(angle_deg)

    length = max(3, int(round(kernel_size * intensity)))
    if length % 2 == 0:
        length += 1

    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    half = (length - 1) / 2.0
    for i in np.linspace(-half, half, length):
        x = int(round(center + i * cos_a))
        y = int(round(center + i * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    # 轻微扩散离散线条，避免极稀疏核造成方向栅格化过强。
    kernel_t = torch.from_numpy(kernel).view(1, 1, kernel_size, kernel_size)
    if angle_deg is None:
        kernel_t = F.avg_pool2d(kernel_t, kernel_size=3, stride=1, padding=1)
    kernel_t = kernel_t / kernel_t.sum().clamp(min=1e-8)
    return kernel_t


class MotionBlurOperator(nn.Module):
    """
    线性运动模糊  H(x) = x * k_motion

    可调参数:
        kernel_size (int)  : 运动核尺寸（奇数），论文/DPS 值 61
        intensity   (float): 运动强度，论文/DPS 值 0.5
        seed        (int)  : 固定核随机种子，保证同一批实验可复现
        angle_deg   (float): 可选线性核角度；None 时使用随机方向核
        noise_sigma (float): 叠加观测噪声标准差，论文值 0.05
    """
    def __init__(
        self,
        kernel_size: int   = MOTION_KERNEL_SIZE,
        intensity:   float = MOTION_INTENSITY,
        seed:        int   = MOTION_SEED,
        angle_deg:   float = MOTION_ANGLE_DEG,
        noise_sigma: float = NOISE_SIGMA,
    ):
        super().__init__()
        self.kernel_size = kernel_size  # ← 可调
        self.intensity   = intensity    # ← 可调
        self.seed        = seed         # ← 可调
        self.angle_deg   = angle_deg    # ← 可调
        self.noise       = GaussianNoise(noise_sigma)

        kernel = _motion_kernel_2d(kernel_size, intensity=intensity, seed=seed, angle_deg=angle_deg)
        self.register_buffer("kernel", kernel)

    def H(self, x: torch.Tensor) -> torch.Tensor:
        """纯运动模糊，无噪声（供 ZAPS 梯度计算使用）"""
        B, C, Hh, W = x.shape
        pad = self.kernel_size // 2
        kernel_c = self.kernel.expand(C, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(x, kernel_c, padding=pad, groups=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """H(x) + 噪声，用于生成观测 y"""
        return self.noise(self.H(x))

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """H^T：翻转核做卷积"""
        B, C, Hh, W = y.shape
        pad = self.kernel_size // 2
        kernel_c = self.kernel.flip([2, 3]).expand(C, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(y, kernel_c, padding=pad, groups=C)


# ───────────────────────────────────────────────────────
# 4. 超分辨率
# ───────────────────────────────────────────────────────

class SuperResolutionOperator(nn.Module):
    """
    双三次下采样超分辨率退化  H(x) = downsample(x, scale)

    可调参数:
        scale_factor (int) : 下采样倍率，论文值 4  ← 可调
        noise_sigma  (float): 叠加观测噪声标准差，论文值 0.05
    """
    def __init__(
        self,
        scale_factor: int   = SR_SCALE_FACTOR,
        noise_sigma:  float = NOISE_SIGMA,
    ):
        super().__init__()
        self.scale_factor = scale_factor    # ← 可调
        self.noise        = GaussianNoise(noise_sigma)

    def H(self, x: torch.Tensor) -> torch.Tensor:
        """纯双三次下采样，无噪声（供 ZAPS 梯度计算使用）"""
        return F.interpolate(
            x,
            scale_factor=1.0 / self.scale_factor,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """H(x) + 噪声，用于生成观测 y"""
        return self.noise(self.H(x))

    def transpose(self, y: torch.Tensor, output_size=None) -> torch.Tensor:
        """H^T：双三次上采样（伪逆近似）"""
        scale = self.scale_factor
        H_out = y.shape[2] * scale if output_size is None else output_size[0]
        W_out = y.shape[3] * scale if output_size is None else output_size[1]
        return F.interpolate(y, size=(H_out, W_out), mode="bicubic", align_corners=False)


# ───────────────────────────────────────────────────────
# 工厂函数
# ───────────────────────────────────────────────────────

def get_operator(task: str, device: str = None, **kwargs) -> nn.Module:
    """
    按任务名称创建退化算子并移到指定设备

    参数:
        task   : "gaussian_deblur" | "inpainting" | "motion_deblur" | "super_resolution"
        device : 目标设备；None 时自动选择
        kwargs : 透传给对应算子的可调参数
    返回:
        operator : nn.Module，已移至 device
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    operators = {
        "gaussian_deblur":   GaussianBlurOperator,
        "inpainting":        InpaintingOperator,
        "motion_deblur":     MotionBlurOperator,
        "super_resolution":  SuperResolutionOperator,
    }
    if task not in operators:
        raise ValueError(f"未知任务: {task}，可选: {list(operators.keys())}")

    return operators[task](**kwargs).to(device)
