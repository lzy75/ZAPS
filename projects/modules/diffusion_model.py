import os
import torch
import numpy as np
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────
# 噪声调度工具（线性 beta 调度，与 guided-diffusion 一致）
# ─────────────────────────────────────────────

def _linear_beta_schedule(num_steps=1000, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
    return betas


def _precompute_diffusion_params(num_steps=1000):
    """预计算扩散过程所需的 alphas_cumprod 等参数"""
    betas = _linear_beta_schedule(num_steps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).float()           # ᾱ_t
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)                # √ᾱ_t
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)  # √(1-ᾱ_t)
    return {
        "betas": betas.float(),
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }


def _extract(arr, t, x_shape):
    """从一维参数数组中按时间步 t 取值，广播到 x_shape"""
    batch = t.shape[0]
    out = arr.to(t.device)[t]
    return out.view(batch, *([1] * (len(x_shape) - 1)))

class BaseDiffusionModel(ABC):
    """
    扩散模型接口基类
    所有子类需实现 _load_model()，并可直接使用 score / predict_x0 / q_sample
    """

    def __init__(self, ckpt_path: str, device: str = None, num_steps: int = 1000):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_steps = num_steps
        self.ckpt_path = ckpt_path

        # 预计算噪声调度参数
        params = _precompute_diffusion_params(num_steps)
        self.alphas_cumprod = params["alphas_cumprod"].to(self.device)
        self.sqrt_alphas_cumprod = params["sqrt_alphas_cumprod"].to(self.device)
        self.sqrt_one_minus_alphas_cumprod = params["sqrt_one_minus_alphas_cumprod"].to(self.device)

        # 加载模型（子类实现）
        self.model = self._load_model()
        self.model.eval()

    @abstractmethod
    def _load_model(self):
        """加载并返回 nn.Module，权重已就位"""
        pass

    # ── 核心接口 ──────────────────────────────

    @torch.no_grad()
    def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        计算得分函数 ∇_x log p(x_t)
        score = -ε_θ(x_t, t) / √(1 - ᾱ_t)

        参数:
            x_t : [B, C, H, W]  当前噪声图像
            t   : [B]            整数时间步索引 (0 ~ num_steps-1)
        返回:
            score : [B, C, H, W]
        """
        t = t.to(self.device).long()
        x_t = x_t.to(self.device)
        eps = self._predict_eps(x_t, t)
        sigma = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return -eps / sigma.clamp(min=1e-8)

    @torch.no_grad()
    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Tweedie 估计：x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t

        参数:
            x_t : [B, C, H, W]
            t   : [B]  整数时间步索引
        返回:
            x0_pred : [B, C, H, W]  预测的干净图像
        """
        t = t.to(self.device).long()
        x_t = x_t.to(self.device)
        eps = self._predict_eps(x_t, t)
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1mab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x0_pred = (x_t - sqrt_1mab * eps) / sqrt_ab.clamp(min=1e-8)
        return x0_pred.clamp(-1.0, 1.0)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """
        前向加噪：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

        参数:
            x_0   : [B, C, H, W]  干净图像，值域 [-1, 1]
            t     : [B]            整数时间步索引
            noise : 可选，与 x_0 同形状；为 None 时随机生成
        返回:
            x_t : [B, C, H, W]
        """
        t = t.to(self.device).long()
        x_0 = x_0.to(self.device)
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_1mab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_ab * x_0 + sqrt_1mab * noise

    def _predict_eps(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        调用 guided-diffusion 模型，返回 ε 预测。
        learn_sigma=True 时输出通道翻倍，前半为 ε，后半为方差参数。
        """
        out = self.model(x_t, t)                 # [B, 2C, H, W] 或 [B, C, H, W]
        C = x_t.shape[1]
        if out.shape[1] == 2 * C:                # learn_sigma=True
            eps = out[:, :C]
        else:
            eps = out
        return eps

    def _predict_eps_var(self, x_t: torch.Tensor, t: torch.Tensor):
        """
        同 _predict_eps，但额外返回学出的方差参数（后半通道）。
        learn_sigma=True（LEARNED_RANGE）时后半是 v∈[-1,1]，供采样步按
        model_log_var = frac·log β + (1-frac)·log β̃（frac=(v+1)/2）插值出真实方差。

        返回:
            (eps, var_values)  var_values 为 None 表示模型非 learn_sigma。
        """
        out = self.model(x_t, t)
        C = x_t.shape[1]
        if out.shape[1] == 2 * C:
            return out[:, :C], out[:, C:]
        return out, None

# ─────────────────────────────────────────────
# FFHQ 模型封装
# ─────────────────────────────────────────────
class FFHQDiffusionModel(BaseDiffusionModel):
    """
    FFHQ 256×256 无条件扩散模型
    checkpoint : modules/models/ffhq_10m.pt
    来源       : OpenAI guided-diffusion，ILVR / DPS 均引用同一权重

    模型参数（与 DPS configs/model_config.yaml 一致）:
        image_size=256, num_channels=128, num_res_blocks=1,
        attention_resolutions="16", num_heads=4, num_head_channels=64,
        learn_sigma=True, use_fp16=False, dropout=0.0
    """

    MODEL_CONFIG = dict(
        image_size=256,
        num_channels=128,
        num_res_blocks=1,
        channel_mult="",          # 默认值：guided-diffusion 按 image_size 自动推断
        attention_resolutions="16",
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
    )

    def _load_model(self):
        from guided_diffusion.script_util import create_model

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"FFHQ 模型文件不存在: {self.ckpt_path}")

        cfg = self.MODEL_CONFIG
        model = create_model(
            image_size=cfg["image_size"],
            num_channels=cfg["num_channels"],
            num_res_blocks=cfg["num_res_blocks"],
            channel_mult=cfg["channel_mult"],
            learn_sigma=cfg["learn_sigma"],
            class_cond=cfg["class_cond"],
            use_checkpoint=cfg["use_checkpoint"],
            attention_resolutions=cfg["attention_resolutions"],
            num_heads=cfg["num_heads"],
            num_head_channels=cfg["num_head_channels"],
            num_heads_upsample=cfg["num_heads_upsample"],
            use_scale_shift_norm=cfg["use_scale_shift_norm"],
            dropout=cfg["dropout"],
            resblock_updown=cfg["resblock_updown"],
            use_fp16=cfg["use_fp16"],
            use_new_attention_order=cfg["use_new_attention_order"],
        )

        state_dict = torch.load(self.ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(self.device)
        print(f"[FFHQ] 模型加载成功: {self.ckpt_path}  device={self.device}")
        return model

# ─────────────────────────────────────────────
# ImageNet 模型封装
# ─────────────────────────────────────────────
class ImageNetDiffusionModel(BaseDiffusionModel):
    """
    ImageNet 256×256 无条件扩散模型
    checkpoint : modules/models/256x256_diffusion_uncond.pt
    来源       : OpenAI guided-diffusion（Dhariwal & Nichol, NeurIPS 2021）

    模型参数（与 guided-diffusion 官方 256x256_diffusion_uncond.pt 一致）:
        image_size=256, num_channels=256, num_res_blocks=2,
        attention_resolutions="32,16,8", num_heads=4, num_head_channels=64,
        learn_sigma=True, use_fp16=True, dropout=0.0
    """

    MODEL_CONFIG = dict(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        channel_mult="",
        attention_resolutions="32,16,8",
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
    )

    def _load_model(self):
        from guided_diffusion.script_util import create_model

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"ImageNet 模型文件不存在: {self.ckpt_path}")

        cfg = self.MODEL_CONFIG
        model = create_model(
            image_size=cfg["image_size"],
            num_channels=cfg["num_channels"],
            num_res_blocks=cfg["num_res_blocks"],
            channel_mult=cfg["channel_mult"],
            learn_sigma=cfg["learn_sigma"],
            class_cond=cfg["class_cond"],
            use_checkpoint=cfg["use_checkpoint"],
            attention_resolutions=cfg["attention_resolutions"],
            num_heads=cfg["num_heads"],
            num_head_channels=cfg["num_head_channels"],
            num_heads_upsample=cfg["num_heads_upsample"],
            use_scale_shift_norm=cfg["use_scale_shift_norm"],
            dropout=cfg["dropout"],
            resblock_updown=cfg["resblock_updown"],
            use_fp16=cfg["use_fp16"],
            use_new_attention_order=cfg["use_new_attention_order"],
        )

        state_dict = torch.load(self.ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(self.device)
        # fp16 推理
        if cfg["use_fp16"]:
            model.convert_to_fp16()
        print(f"[ImageNet] 模型加载成功: {self.ckpt_path}  device={self.device}")
        return model

def load_ffhq_model(model_dir: str = "modules/models", device: str = None) -> FFHQDiffusionModel:
    ckpt = os.path.join(model_dir, "ffhq_10m.pt")
    return FFHQDiffusionModel(ckpt_path=ckpt, device=device)


def load_imagenet_model(model_dir: str = "modules/models", device: str = None) -> ImageNetDiffusionModel:
    ckpt = os.path.join(model_dir, "256x256_diffusion_uncond.pt")
    return ImageNetDiffusionModel(ckpt_path=ckpt, device=device)
