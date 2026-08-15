"""
ZAPS 单图完整流程入口
用法：
    python modules/main_single.py --image path/to/img.png --task gaussian_deblur
    python modules/main_single.py --image path/to/img.png --task inpainting --dataset ffhq
"""

import os
import sys
import argparse
import torch
from PIL import Image

# 把 projects 根目录加入路径，保证各模块可找到
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from configs.config         import (ZAPS_CONFIG, ZETA_INIT_BY_TASK, TASK_CONFIGS,
                                    METRICS_CONFIG, DATASET_MODEL_MAP, DEFAULT_DATASET,
                                    RESULTS_DIR, EXPERIMENTS_DIR, IMG_SIZE)
from modules.diffusion_model import load_ffhq_model, load_imagenet_model
from modules.degradations    import get_operator
from modules.dataset_loader  import image_to_tensor, tensor_to_image
from modules.zaps_algorithm  import ZAPS
from utils.metrics           import compute_all_metrics
from utils.experiment_logger import ExperimentLogger


# ───────────────────────────────────────────────────────
# 辅助
# ───────────────────────────────────────────────────────

def load_image_as_tensor(image_path: str, img_size=IMG_SIZE) -> torch.Tensor:
    """加载图像 → [1, 3, H, W]，值域 [-1, 1]"""
    img = Image.open(image_path).convert("RGB").resize(img_size, Image.LANCZOS)
    return image_to_tensor(img, normalize=True).unsqueeze(0)


def save_tensor_as_image(tensor: torch.Tensor, path: str):
    """[1, 3, H, W] 张量 → PNG 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = tensor_to_image(tensor.squeeze(0), denormalize=True)
    img.save(path)
    print(f"  保存图像 → {path}")


def load_diffusion_model(dataset: str, device: str):
    """按数据集选择加载 FFHQ 或 ImageNet 模型"""
    model_dir = os.path.join(_ROOT, "modules", "models")
    if dataset == "ffhq":
        return load_ffhq_model(model_dir=model_dir, device=device)
    elif dataset == "imagenet":
        return load_imagenet_model(model_dir=model_dir, device=device)
    else:
        raise ValueError(f"未知 dataset: {dataset}，可选 'ffhq' | 'imagenet'")


# ───────────────────────────────────────────────────────
# 主流程
# ───────────────────────────────────────────────────────

def run_zaps_single(
    image_path: str,
    task:       str,
    dataset:    str = DEFAULT_DATASET,
    device:     str = None,
    save_dir:   str = RESULTS_DIR,
    verbose:    bool = True,
    final_mode: str = "sample",
    sample_eta: float = None,
    sample_init: str = "random",
    timestep_spacing: str = None,
    schedule_power: float = None,
    num_epochs: int = None,
    num_steps: int = None,
    schedule: tuple = None,
    purpose: str = "",
) -> dict:
    """
    单张图像 ZAPS 完整流程

    参数:
        image_path : 原始干净图像路径
        task       : 退化任务名称（见 TASK_CONFIGS）
        dataset    : 使用的模型数据集 "ffhq" | "imagenet"
        device     : 运行设备；None 自动选择
        save_dir   : 结果保存目录
        verbose    : 是否打印过程信息
        final_mode : 最终输出策略，sample 或 last_opt
        sample_eta : 最终采样 eta，None 表示使用配置 eta
        sample_init: random 或 opt_noise，用于最终采样初始噪声对照
        timestep_spacing : 时间步取点方式，None 时使用配置值
        schedule_power   : 非线性取点指数，None 时使用配置值
        num_epochs : 零样本优化轮数，None 时使用配置值
        num_steps  : 每轮采样步数 S，None 时使用配置值（NFE_opt = num_epochs × num_steps）
        schedule   : 低/中/高噪声区步数分配 (n_low,n_mid,n_high)，None 时使用配置值
    返回:
        dict: {"psnr", "ssim", "lpips", "x0_gt", "y_obs", "x0_recon"}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  任务: {task}   数据集: {dataset}   设备: {device}")
    print(f"{'='*55}")

    # ── 1. 加载原始图像 ──
    print("\n[1/5] 加载图像...")
    x0_gt = load_image_as_tensor(image_path).to(device)   # [1,3,256,256]

    # ── 2. 构建退化算子 ──
    print("[2/5] 构建退化算子...")
    task_cfg = TASK_CONFIGS[task]
    operator = get_operator(task, device=device, **task_cfg)

    # ── 3. 生成观测 y = H(x_0) + noise ──
    print("[3/5] 生成观测图像...")
    with torch.no_grad():
        y_obs = operator(x0_gt)                            # [1,3,H',W']

    # ── 4. 加载扩散模型 ──
    print("[4/5] 加载扩散模型...")
    diffusion_model = load_diffusion_model(dataset, device)

    # ── 5. ZAPS 重建 ──
    print("[5/5] 执行 ZAPS 重建...")
    zaps_cfg = {**ZAPS_CONFIG,
                "zeta_init": ZETA_INIT_BY_TASK.get(task, ZAPS_CONFIG["zeta_init"])}
    if timestep_spacing is not None:
        zaps_cfg["timestep_spacing"] = timestep_spacing
    if schedule_power is not None:
        zaps_cfg["schedule_power"] = schedule_power
    if num_epochs is not None:
        zaps_cfg["num_epochs"] = num_epochs
    if num_steps is not None:
        zaps_cfg["num_steps"] = num_steps
    if schedule is not None:
        zaps_cfg["schedule"] = schedule
    zaps = ZAPS(
        diffusion_model=diffusion_model,
        forward_operator=operator,
        img_size=IMG_SIZE[0],
        **zaps_cfg,
    )
    result = zaps.run(
        y_obs, verbose=verbose, x0_gt=x0_gt,
        final_mode=final_mode, sample_eta=sample_eta, sample_init=sample_init,
    )
    x0_recon  = result["x0_final"]
    loss_hist = result["loss_history"]

    # ── NFE & 运行时间汇报 ──
    print("\n[NFE & 时间]")
    print(f"  优化阶段  NFE = {result['nfe_opt']:4d}   耗时 = {result['time_opt_s']:.1f}s")
    print(f"  最终采样  NFE = {result['nfe_sample']:4d}   耗时 = {result['time_sample_s']:.1f}s")
    print(f"  全程合计  NFE = {result['nfe_total']:4d}   耗时 = {result['time_total_s']:.1f}s"
          f"  ({result['time_total_s']/60:.1f} min)")

    # ── 评估指标 ──
    print("\n[评估]")
    # 观测图像需插值回 256×256 才能与 GT 对比
    y_for_metric = torch.nn.functional.interpolate(
        y_obs, size=(256, 256), mode="bilinear", align_corners=False
    ) if y_obs.shape[-1] != 256 else y_obs
    obs_mse  = ((y_for_metric.clamp(-1,1) - x0_gt) ** 2).mean()
    obs_psnr = (-10 * torch.log10(obs_mse + 1e-8) + 20 * torch.log10(torch.tensor(2.0))).item()
    print(f"  观测PSNR = {obs_psnr:.2f} dB  (退化基线)")
    metrics = compute_all_metrics(
        x0_recon, x0_gt,
        lpips_net=METRICS_CONFIG["lpips_net"],
    )
    print(f"  PSNR  = {metrics['psnr']:.2f} dB")
    print(f"  SSIM  = {metrics['ssim']:.4f}")
    print(f"  LPIPS = {metrics['lpips']:.4f}")

    # ── 归档实验（CSV 索引 + 独立目录：图像 + run.json + conclusion.md）──
    log_config = {
        "num_steps": zaps_cfg["num_steps"], "schedule": zaps_cfg["schedule"],
        "timestep_spacing": zaps_cfg.get("timestep_spacing"),
        "schedule_power": zaps_cfg.get("schedule_power"),
        "num_epochs": zaps_cfg["num_epochs"], "lr": zaps_cfg["lr"],
        "zeta_init": zaps_cfg["zeta_init"], "d_init": zaps_cfg["d_init"],
        "wave": zaps_cfg["wave"], "level": zaps_cfg["level"], "eta": zaps_cfg["eta"],
        "final_mode": final_mode,
        "sample_eta": zaps_cfg["eta"] if sample_eta is None else sample_eta,
        "sample_init": sample_init,
        "noise_sigma": task_cfg.get("noise_sigma"),
    }
    exp_id = ExperimentLogger(EXPERIMENTS_DIR).log(
        task=task, dataset=dataset, image=image_path,
        config=log_config, metrics=metrics, obs_psnr=obs_psnr,
        nfe={"opt": result["nfe_opt"], "sample": result["nfe_sample"],
             "total": result["nfe_total"]},
        times={"opt_s": result["time_opt_s"], "sample_s": result["time_sample_s"],
               "total_s": result["time_total_s"]},
        images={"gt": x0_gt, "observed": y_obs, "recon": x0_recon},
        purpose=purpose,
    )
    print(f"\n  实验归档 → {os.path.join(EXPERIMENTS_DIR, exp_id)}  (exp_id={exp_id})")

    print(f"\n损失曲线（最后5轮）: {[f'{v:.4f}' for v in loss_hist[-5:]]}")
    print(f"{'='*55}\n")

    return {
        "exp_id":       exp_id,
        "psnr":         metrics["psnr"],
        "ssim":         metrics["ssim"],
        "lpips":        metrics["lpips"],
        "nfe_total":    result["nfe_total"],
        "time_total_s": result["time_total_s"],
        "x0_gt":        x0_gt,
        "y_obs":        y_obs,
        "x0_recon":     x0_recon,
    }


# ───────────────────────────────────────────────────────
# 命令行入口
# ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZAPS 单图推断")
    parser.add_argument("--image",   type=str, required=True,
                        help="原始图像路径")
    parser.add_argument("--task",    type=str, required=True,
                        choices=list(TASK_CONFIGS.keys()),
                        help="退化任务类型")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=["ffhq", "imagenet"],
                        help="使用的扩散模型 (default: ffhq)")
    parser.add_argument("--device",  type=str, default=None,
                        help="设备 cuda/cpu (default: 自动)")
    parser.add_argument("--save_dir", type=str, default=RESULTS_DIR,
                        help="结果保存目录")
    parser.add_argument("--quiet",   action="store_true",
                        help="不打印优化过程")
    parser.add_argument("--final_mode", type=str, default="sample",
                        choices=["sample", "last_opt"],
                        help="最终输出策略：sample 为默认采样，last_opt 为最后一轮优化轨迹")
    parser.add_argument("--sample_eta", type=float, default=None,
                        help="最终采样 eta；默认使用配置 eta")
    parser.add_argument("--sample_init", type=str, default="random",
                        choices=["random", "opt_noise"],
                        help="最终采样初始噪声：random 或复用优化噪声 opt_noise")
    parser.add_argument("--timestep_spacing", type=str, default=None,
                        choices=["linear", "quadratic", "power"],
                        help="时间步取点方式；默认使用配置值")
    parser.add_argument("--schedule_power", type=float, default=None,
                        help="非线性时间步取点指数；默认使用配置值")
    parser.add_argument("--purpose", type=str, default="",
                        help="本次实验目的，写入 conclusion.md")
    args = parser.parse_args()

    run_zaps_single(
        image_path=args.image,
        task=args.task,
        dataset=args.dataset,
        device=args.device,
        save_dir=args.save_dir,
        verbose=not args.quiet,
        final_mode=args.final_mode,
        sample_eta=args.sample_eta,
        sample_init=args.sample_init,
        timestep_spacing=args.timestep_spacing,
        schedule_power=args.schedule_power,
        purpose=args.purpose,
    )
