"""
实验追踪：每次运行落盘一行 CSV + 独立归档目录（重建图 + run.json + conclusion.md）

设计要点：
  - 用 git commit 短 hash 把每次实验结果绑定到确切代码版本（git_dirty 标记是否有未提交改动）
  - 重建图归档到 experiments/<exp_id>/，CSV 为主索引便于横向对比
  - 存图复用 dataset_loader.tensor_to_image，不重复造轮子
"""
import os
import csv
import json
import time
import subprocess

from modules.dataset_loader import tensor_to_image

# CSV 固定列顺序（schedule 元组存成 "15-10-5"）
CSV_FIELDS = [
    "exp_id", "datetime", "git_commit", "git_dirty",
    "dataset", "task", "image",
    "num_steps", "schedule", "num_epochs", "lr", "zeta_init", "d_init",
    "wave", "level", "eta", "noise_sigma",
    "obs_psnr", "psnr", "ssim", "lpips",
    "nfe_opt", "nfe_sample", "nfe_total",
    "time_opt_s", "time_sample_s", "time_total_s",
]


class ExperimentLogger:
    """实验记录器：log() 一次完成 归档图像 + 追加CSV + 写 run.json/conclusion.md"""

    def __init__(self, base_dir: str, repo_dir: str = None):
        """
        参数:
            base_dir : experiments 根目录
            repo_dir : git 仓库目录（取 commit 用）；None 时用 base_dir 上一级
        """
        self.base_dir = base_dir
        self.repo_dir = repo_dir or os.path.dirname(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.csv_path = os.path.join(self.base_dir, "experiments.csv")

    # ── git 信息（非 repo 时优雅降级）──────────────────────
    def _git_info(self) -> dict:
        def _run(args):
            return subprocess.check_output(
                args, cwd=self.repo_dir, stderr=subprocess.DEVNULL
            ).decode().strip()
        try:
            commit = _run(["git", "rev-parse", "--short", "HEAD"])
            branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            # dirty 只反映“代码”是否干净，排除 experiments/ 自身产物，避免历次实验互相污染
            porcelain = _run(["git", "status", "--porcelain"])
            code_dirty = any(
                line and "experiments/" not in line.replace("\\", "/")
                for line in porcelain.splitlines()
            )
            return {"commit": commit, "branch": branch, "dirty": code_dirty}
        except Exception:
            return {"commit": "nogit", "branch": "nogit", "dirty": False}

    # ── 主接口 ────────────────────────────────────────────
    def log(self, task: str, dataset: str, image: str,
            config: dict, metrics: dict, obs_psnr: float,
            nfe: dict, times: dict, images: dict, purpose: str = "",
            indicators: list = None) -> str:
        """
        参数:
            task/dataset/image : 实验标识
            config  : 扁平参数字典（num_steps/schedule/num_epochs/lr/zeta_init/d_init/wave/level/eta/noise_sigma）
            metrics : {"psnr","ssim","lpips"}
            obs_psnr: 退化观测基线 PSNR
            nfe     : {"opt","sample","total"}
            times   : {"opt_s","sample_s","total_s"}
            images  : {"gt","observed","recon"} 张量 [1,C,H,W]，值域[-1,1]
            purpose : 本次实验目的（人工标注，写入 conclusion.md）
        返回:
            exp_id
        """
        ts      = time.strftime("%Y%m%d_%H%M%S")
        stem    = os.path.splitext(os.path.basename(image))[0]
        exp_id  = f"{ts}_{task}_{dataset}_{stem}"
        exp_dir = os.path.join(self.base_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        git = self._git_info()

        # 归档三张图（复用 tensor_to_image）
        for name, tensor in images.items():
            tensor_to_image(tensor.detach().squeeze(0), denormalize=True).save(
                os.path.join(exp_dir, f"{name}.png"))

        sched = config["schedule"]
        sched_str = "-".join(map(str, sched)) if isinstance(sched, (tuple, list)) else str(sched)

        row = {
            "exp_id": exp_id, "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "dataset": dataset, "task": task, "image": os.path.basename(image),
            "num_steps": config["num_steps"], "schedule": sched_str,
            "num_epochs": config["num_epochs"], "lr": config["lr"],
            "zeta_init": config["zeta_init"], "d_init": config["d_init"],
            "wave": config["wave"], "level": config["level"],
            "eta": config["eta"], "noise_sigma": config.get("noise_sigma", ""),
            "obs_psnr": round(obs_psnr, 4),
            "psnr": round(metrics["psnr"], 4), "ssim": round(metrics["ssim"], 4),
            "lpips": round(metrics["lpips"], 4),
            "nfe_opt": nfe["opt"], "nfe_sample": nfe["sample"], "nfe_total": nfe["total"],
            "time_opt_s": round(times["opt_s"], 1),
            "time_sample_s": round(times["sample_s"], 1),
            "time_total_s": round(times["total_s"], 1),
        }

        # 追加 CSV（首次写表头）
        new_file = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

        # run.json（完整机器可读）
        with open(os.path.join(exp_dir, "run.json"), "w", encoding="utf-8") as f:
            json.dump({"exp_id": exp_id, "purpose": purpose, "git": git, "config": config,
                       "metrics": metrics, "obs_psnr": obs_psnr,
                       "nfe": nfe, "times": times,
                       "indicators": indicators or []}, f, ensure_ascii=False, indent=2)

        # conclusion.md（预填指标，留空结论）
        self._write_conclusion(exp_dir, exp_id, git, task, dataset, image,
                               config, metrics, obs_psnr, nfe, times, purpose)
        return exp_id

    def _write_conclusion(self, exp_dir, exp_id, git, task, dataset, image,
                          config, metrics, obs_psnr, nfe, times, purpose=""):
        purpose_line = purpose.strip() if purpose and purpose.strip() else "（待填写）"
        md = f"""# 实验 {exp_id}

- 代码版本: `{git['commit']}` (branch={git['branch']}, dirty={git['dirty']})
- 任务 / 数据集: **{task}** / {dataset}，图像 `{os.path.basename(image)}`
- 关键参数: steps={config['num_steps']} schedule={config['schedule']} epochs={config['num_epochs']} \
lr={config['lr']} zeta_init={config['zeta_init']} d_init={config['d_init']} wave={config['wave']} level={config['level']}
- 指标: **PSNR={metrics['psnr']:.2f}** / SSIM={metrics['ssim']:.4f} / LPIPS={metrics['lpips']:.4f}（观测基线 PSNR={obs_psnr:.2f}）
- NFE={nfe['total']}（优化{nfe['opt']}+采样{nfe['sample']}）
- 耗时: 优化={times['opt_s']:.2f}s + 采样={times['sample_s']:.2f}s = 合计 **{times['total_s']:.2f}s**

![recon](recon.png)

## 本次实验任务与目的

- 任务: **{task}**（{dataset}）
- 目的: {purpose_line}

## 结论 / 观察
（待填写）
"""
        with open(os.path.join(exp_dir, "conclusion.md"), "w", encoding="utf-8") as f:
            f.write(md)
