"""
FFHQ 批量复现实验入口。

默认只做 dry-run 预览，确认图像和任务列表后再加 --run 执行。
单图耗时较长，本脚本按 image × task 顺序串行调用 main_single.run_zaps_single。
"""

import argparse
import csv
import os
import sys
import time
from typing import Iterable, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from configs.config import EXPERIMENTS_DIR, TASK_CONFIGS

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DEFAULT_TASKS = ("gaussian_deblur", "inpainting", "motion_deblur", "super_resolution")


def default_data_dir() -> str:
    """优先使用当前工作目录下的数据集，适配 Codex 沙箱和本地直接运行。"""
    cwd_data = os.path.join(os.getcwd(), "datasets")
    return cwd_data if os.path.isdir(cwd_data) else os.path.join(_REPO_ROOT, "datasets")


def dataset_dir(data_dir: str, dataset: str) -> str:
    """返回当前复现使用的数据集目录，兼容 datasets/FFHQ 和 datasets/ffhq/256x256。"""
    if dataset != "ffhq":
        raise ValueError(f"当前批量入口只聚焦 FFHQ，收到: {dataset}")
    candidates = [
        os.path.join(data_dir, "ffhq", "256x256"),
        os.path.join(data_dir, "ffhq"),
        os.path.join(data_dir, "FFHQ"),
    ]
    for root in candidates:
        if os.path.isdir(root):
            return root
    return candidates[0]


def collect_images(data_dir: str, dataset: str, max_images: int = None,
                   start_index: int = 0) -> List[str]:
    """收集数据集图像路径，保持排序以便复现实验。"""
    root = dataset_dir(data_dir, dataset)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"数据集目录不存在: {root}")

    images = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(IMAGE_EXTS):
                images.append(os.path.join(dirpath, name))
    images.sort()

    if start_index:
        images = images[start_index:]
    if max_images is not None:
        images = images[:max_images]
    return images


def parse_tasks(tasks: Iterable[str]) -> List[str]:
    """解析任务列表，支持 all 或显式任务名。"""
    if not tasks or "all" in tasks:
        return list(DEFAULT_TASKS)
    bad = [task for task in tasks if task not in TASK_CONFIGS]
    if bad:
        raise ValueError(f"未知任务: {bad}，可选 {list(TASK_CONFIGS)} 或 all")
    return list(tasks)


def write_batch_summary(rows: list, batch_id: str) -> str:
    """写入批量运行摘要，单次实验详情仍以 experiments.csv 为准。"""
    batch_dir = os.path.join(EXPERIMENTS_DIR, "batch_runs")
    os.makedirs(batch_dir, exist_ok=True)
    path = os.path.join(batch_dir, f"{batch_id}.csv")
    fieldnames = [
        "batch_id", "dataset", "task", "image", "exp_id",
        "psnr", "ssim", "lpips", "nfe_total", "time_total_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_batch(args) -> None:
    tasks = parse_tasks(args.tasks)
    images = collect_images(
        data_dir=args.data_dir,
        dataset=args.dataset,
        max_images=args.max_images,
        start_index=args.start_index,
    )

    print(f"数据集: {args.dataset}")
    print(f"数据目录: {dataset_dir(args.data_dir, args.dataset)}")
    print(f"图像数: {len(images)}")
    print(f"任务: {', '.join(tasks)}")
    print(f"组合数: {len(images) * len(tasks)}")

    for image in images:
        print(f"  image: {image}")
    if not args.run:
        print("\n当前为 dry-run；确认无误后添加 --run 执行。")
        return

    from modules.main_single import run_zaps_single

    batch_id = time.strftime("%Y%m%d_%H%M%S_ffhq_batch")
    rows = []
    for image_path in images:
        for task in tasks:
            result = run_zaps_single(
                image_path=image_path,
                task=task,
                dataset=args.dataset,
                device=args.device,
                verbose=not args.quiet,
                final_mode=args.final_mode,
                sample_eta=args.sample_eta,
                sample_init=args.sample_init,
                timestep_spacing=args.timestep_spacing,
                schedule_power=args.schedule_power,
                num_epochs=args.num_epochs,
                num_steps=args.num_steps,
                schedule=args.schedule,
                purpose=args.purpose,
            )
            rows.append({
                "batch_id": batch_id,
                "dataset": args.dataset,
                "task": task,
                "image": os.path.basename(image_path),
                "exp_id": result["exp_id"],
                "psnr": result["psnr"],
                "ssim": result["ssim"],
                "lpips": result["lpips"],
                "nfe_total": result["nfe_total"],
                "time_total_s": result["time_total_s"],
            })

    summary_path = write_batch_summary(rows, batch_id)
    print(f"\n批量摘要已保存: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FFHQ ZAPS 批量复现实验")
    parser.add_argument("--dataset", default="ffhq", choices=["ffhq"],
                        help="当前聚焦 FFHQ 复现")
    parser.add_argument("--data_dir", default=default_data_dir(),
                        help="数据集根目录，默认使用仓库根目录下 datasets")
    parser.add_argument("--tasks", nargs="+", default=["all"],
                        help="任务列表，例如 all 或 gaussian_deblur inpainting")
    parser.add_argument("--max_images", type=int, default=None,
                        help="最多运行多少张图，调参时建议从 1 开始")
    parser.add_argument("--start_index", type=int, default=0,
                        help="从排序后的第几张图开始")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 cuda/cpu，默认自动选择")
    parser.add_argument("--quiet", action="store_true",
                        help="减少单图优化日志")
    parser.add_argument("--final_mode", type=str, default="sample",
                        choices=["sample", "last_opt"],
                        help="最终输出策略：sample 或 last_opt")
    parser.add_argument("--sample_eta", type=float, default=None,
                        help="最终采样 eta；默认使用配置 eta")
    parser.add_argument("--sample_init", type=str, default="random",
                        choices=["random", "opt_noise"],
                        help="最终采样初始噪声：random 或 opt_noise")
    parser.add_argument("--timestep_spacing", type=str, default=None,
                        choices=["linear", "quadratic", "power"],
                        help="时间步取点方式；默认使用配置值")
    parser.add_argument("--schedule_power", type=float, default=None,
                        help="非线性时间步取点指数；默认使用配置值")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="零样本优化轮数；默认使用配置值。NFE_opt = num_epochs × num_steps")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="每轮采样步数 S；默认使用配置值")
    parser.add_argument("--schedule", type=int, nargs=3, default=None,
                        metavar=("N_LOW", "N_MID", "N_HIGH"),
                        help="低/中/高噪声区步数分配，例如 --schedule 10 7 3；三者之和应等于 num_steps")
    parser.add_argument("--purpose", type=str, default="",
                        help="本次实验目的，写入每个实验的 conclusion.md，例如 --purpose 'ZAPS Table8 epochs-steps 权衡复现'")
    parser.add_argument("--run", action="store_true",
                        help="真正执行；不加此参数只预览任务")
    return parser


if __name__ == "__main__":
    run_batch(build_parser().parse_args())
