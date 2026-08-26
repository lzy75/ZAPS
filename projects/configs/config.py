"""
ZAPS 实验配置
集中管理所有可调参数，修改此文件即可控制全局行为
"""

import os

# ═══════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR     = os.path.join(BASE_DIR, "modules", "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")   # 实验归档与 CSV 索引根目录
DATASET_DIR   = os.path.join(BASE_DIR, "datasets")

# 预训练模型路径
FFHQ_CKPT       = os.path.join(MODEL_DIR, "ffhq_10m.pt")
IMAGENET_CKPT   = os.path.join(MODEL_DIR, "256x256_diffusion_uncond.pt")


# ═══════════════════════════════════════════════════════
# 数据集配置
# ═══════════════════════════════════════════════════════
IMG_SIZE = (256, 256)   # ← 可调：输入图像尺寸


# ═══════════════════════════════════════════════════════
# ZAPS 算法超参数（对应论文 Section 4.1 / Algorithm 1）
# ═══════════════════════════════════════════════════════
ZAPS_CONFIG = dict(
    num_steps    = 30,          # ← 可调：总采样步数，论文值 30
    schedule     = (15, 10, 5), # ← 可调：低/中/高噪声区步数，论文 "15,10,5"
    timestep_spacing = "linear", # ← 可调：linear | quadratic
    schedule_power   = 2.0,      # ← 可调：非线性取点指数
    num_epochs   = 10,          # ← 可调：零样本优化轮数，论文值 10
    lr           = 1e-3,        # ← 可调：Adam 学习率（论文用 Adam 默认 1e-3）
    zeta_init    = 0.2,         # ← 可调：ζ 初始值（缺省 0.2，按任务见下表覆盖）
    d_init       = 0.2,         # ← 可调：D_t 对角初值，论文统一 0.2
    eta          = 1.0,         # ← 可调：采样随机性（1.0=DDPM，0.0=DDIM）
    use_learned_var = False,    # 原文用固定β̃(Eq.10),learned variance验证更差,默认关
    wave         = "db4",       # ← 可调：正交小波，论文用 db4
    level        = 3,           # ← 可调：DWT 分解级数
)

# ζ 初始值按任务区分（论文 4.1）：高斯/运动模糊 0.2，随机修复/超分 0.1
ZETA_INIT_BY_TASK = {
    "gaussian_deblur":  0.2,
    "motion_deblur":    0.2,
    "inpainting":       0.1,
    "super_resolution": 0.1,
}


# ═══════════════════════════════════════════════════════
# 退化任务参数（对应论文 Table 1 实验设置）
# ═══════════════════════════════════════════════════════
TASK_CONFIGS = {
    "gaussian_deblur": dict(
        kernel_size  = 61,      # ← 可调：论文值 61
        sigma        = 3.0,     # ← 可调：论文值 3.0
        noise_sigma  = 0.05,    # ← 可调：论文值 0.05
    ),
    "inpainting": dict(
        random_ratio = 0.7,     # ← 可调：论文值 0.7（70% 遮挡）
        box_size     = 128,     # ← 可调：论文值 128（方形中心遮挡）
        mode         = "random",# ← 可调："random" | "box" | "both"
        noise_sigma  = 0.05,
    ),
    "motion_deblur": dict(
        kernel_size  = 61,      # ← 可调：论文/DPS 值 61
        intensity    = 0.5,     # ← 可调：论文/DPS motionblur intensity
        seed         = 0,       # ← 可调：固定 motion kernel，便于复现
        angle_deg    = None,    # ← 可调：None 使用随机方向核；设角度时为线性核
        noise_sigma  = 0.05,
    ),
    "super_resolution": dict(
        scale_factor = 4,       # ← 可调：论文值 4
        noise_sigma  = 0.05,
    ),
}


# ═══════════════════════════════════════════════════════
# 评估指标配置
# ═══════════════════════════════════════════════════════
METRICS_CONFIG = dict(
    lpips_net  = "alex",    # ← 可调："alex" | "vgg" | "squeeze"
    ssim_win   = 11,        # ← 可调：SSIM 窗口尺寸
)


# ═══════════════════════════════════════════════════════
# 实验数据集选择
# ═══════════════════════════════════════════════════════
# "ffhq"：使用 ffhq_10m.pt，适用于人脸图像
# "imagenet"：使用 256x256_diffusion_uncond.pt，适用于自然图像
DATASET_MODEL_MAP = {
    "ffhq":     FFHQ_CKPT,
    "imagenet": IMAGENET_CKPT,
}
DEFAULT_DATASET = "ffhq"   # ← 可调
