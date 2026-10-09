"""One-change ImageNet blur ablation: clamp VJP mask in likelihood guidance.

Reuse the completed guidance audit and archived raw baseline. Compare frozen
raw->mask with matched mask->mask optimization (same lr/grid/noise/zeta+D).
No core/default edits, GT selection, noise attenuation or parameter sweep.
"""

import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

PROJECTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECTS_ROOT.parent
sys.path.insert(0, str(PROJECTS_ROOT))

from utils.diag_dps_same_observation import file_sha256, set_seed
from utils.diag_zaps_guidance_clipping import (
    archive_fingerprint_check, restore_sampling_rng, saved_cuda_selection,
)
from utils.diag_zaps_sampler_parity import rng_state, same_rng


def validate_source(guidance, saved):
    if guidance.get("status") != "complete":
        raise ValueError("Use a completed guidance/clipping audit")
    if set(guidance.get("results", {})) != {"irregular_15_10_5", "uniform_30"}:
        raise ValueError("Need both audited schedule arms")
    for name, audit in guidance["results"].items():
        if not audit["replay_gate"]["passed"] or not audit["archive_fingerprint_gate"]["passed"]:
            raise ValueError("Source passive/archive gate did not pass")
        if not audit.get("probe_gates") or len(audit["probe_gates"]) != len(audit.get("rows", [])):
            raise ValueError("Source gradient probes are missing or incomplete")
        for probe_gate in audit["probe_gates"]:
            for key in ("epsilon_vs_captured", "independent_graph_epsilon", "frozen_epsilon_clamp_chain"):
                if not probe_gate[key]["passed"]:
                    raise ValueError("Source VJP gate did not pass")
        config = saved["results"][name]["config"]
        if (config["num_steps"] != 30 or config["num_epochs"] != 10 or config["lr"] != .001
                or config["eta"] != 1.0 or config["use_learned_var"]
                or config["sampler_mode"] != "ddpm" or config.get("surrogate_score_jacobian", False)):
            raise ValueError("Keep the archived 30x10/lr=.001/fixed variance/DDPM setting")


@contextmanager
def clamp_guidance_scope(zaps, enabled, rows):
    """Mask H^T r before the core's B multiplication; never alter H or posterior.

    A detached piecewise-constant mask has zero derivative almost everywhere.
    Multiplying it by the LIVE adjoint residual retains zeta/D/unroll gradients.
    Each mask tensor is replaced, not mutated; earlier autograd graphs stay valid.
    """
    import torch
    original_tweedie, original_transpose = zaps._tweedie_estimate, zaps.A.transpose
    holder = {"mask": None, "used": True}

    def tweedie(x, epsilon, ab, index):
        if not holder["used"]:
            raise RuntimeError("Previous guidance mask was not consumed")
        value = original_tweedie(x, epsilon, ab, index)
        raw = (x - (1-ab).sqrt() * epsilon.detach()) / ab.sqrt().clamp(min=1e-8)
        holder["mask"] = (raw.detach().abs() <= 1).to(raw.dtype)
        holder["used"] = False
        rows.append({"t": int(zaps.tau[index]), "input_x_rms": float(x.detach().square().mean().sqrt()),
                     "residual_norm": float((zaps._clamp_audit_y-zaps.A.H(value.detach())).norm()),
                     "raw_x0_clip_fraction": float((holder["mask"] == 0).float().mean())})
        return value

    def transpose(residual):
        if holder["used"] or holder["mask"] is None:
            raise RuntimeError("Likelihood transpose called without its current Tweedie mask")
        vector = original_transpose(residual)
        if vector.shape != holder["mask"].shape:
            raise RuntimeError("Clamp mask/adjoint vector shapes differ")
        holder["used"] = True
        return holder["mask"] * vector if enabled else vector

    with patch.object(zaps, "_tweedie_estimate", side_effect=tweedie), patch.object(zaps.A, "transpose", side_effect=transpose):
        yield
    if len(rows) != len(zaps.tau) or not holder["used"]:
        raise RuntimeError("Unexpected number/order of likelihood updates")


def make_clamp_zaps(snapshot_class):
    class ClampZAPS(snapshot_class):
        def __init__(self, *args, use_clamp_mask=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.use_clamp_mask = use_clamp_mask
            self.clamp_rows = []

        def _reverse_diffusion(self, y, *args, **kwargs):
            self.clamp_rows = []
            self._clamp_audit_y = y.detach()
            with clamp_guidance_scope(self, self.use_clamp_mask, self.clamp_rows):
                return super()._reverse_diffusion(y, *args, **kwargs)
    return ClampZAPS


def training_pairing(old_state, new_state, old_index, new_index):
    import torch
    return {
        "x_T_equal": bool(torch.equal(old_state["init_noise"], new_state["init_noise"])),
        "epoch10_cpu_rng_equal": bool(torch.equal(old_state["cpu_rng"], new_state["cpu_rng"])),
        "epoch10_sampling_cuda_rng_equal": bool(torch.equal(old_state["cuda_rng"][old_index],
                                                             new_state["cuda_rng"][new_index])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guidance-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--saved-cuda-index", type=int, default=None)
    args = parser.parse_args()
    guidance_dir = Path(args.guidance_dir).resolve()
    guidance = json.loads((guidance_dir / "audit.json").read_text(encoding="utf-8"))
    trace_dir = Path(guidance["source_trace"])
    saved = json.loads((trace_dir / "audit.json").read_text(encoding="utf-8"))
    source = json.loads((Path(saved["source"]) / "run.json").read_text(encoding="utf-8"))
    validate_source(guidance, saved)
    if (not str(args.device).startswith("cuda") or guidance["task"] != "gaussian_deblur"
            or source["arguments"]["dataset"] != "imagenet"):
        parser.error("Only the audited ImageNet CUDA Gaussian-deblurring path is in scope")

    import torch
    from configs.config import IMG_SIZE, METRICS_CONFIG
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from modules.degradations import get_operator
    from modules.zaps_algorithm import ZAPS
    from modules.dataset_loader import tensor_to_image
    from utils.diag_zaps_optimized_trace import LastUnrollSnapshotZAPS
    from utils.diag_zaps_trace_audit import relative_error, classify_parity
    from utils.metrics import compute_all_metrics

    model = load_diffusion_model("imagenet", args.device)
    model.model.eval()
    operator = get_operator("gaussian_deblur", device=args.device, **source["task_configs"]["gaussian_deblur"])
    measurement_path = Path(saved["source"]) / "gaussian_deblur/measurement.pt"
    print("Checking audited checkpoint/H/measurement/core identities...", flush=True)
    for key, path in (("checkpoint_sha256", model.ckpt_path), ("measurement_sha256", measurement_path),
                      ("operator_source_sha256", PROJECTS_ROOT / "modules/degradations.py"),
                      ("core_sha256", PROJECTS_ROOT / "modules/zaps_algorithm.py"),
                      ("image_sha256", source["arguments"]["image"])):
        if file_sha256(path) != guidance[key]:
            raise RuntimeError(f"Identity changed since the gradient audit: {key}")
    gt = load_image_as_tensor(source["arguments"]["image"]).to(args.device)
    y = torch.load(measurement_path, map_location=args.device, weights_only=True)
    seed = int(source["arguments"]["seed"])
    target = torch.device(args.device)
    target_index = target.index if target.index is not None else torch.cuda.current_device()
    ClampZAPS = make_clamp_zaps(LastUnrollSnapshotZAPS)
    output_dir = trace_dir / ("clamp_ablation_" + time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "experiment": "clamp_VJP_mask_one_change", "status": "running", "arguments": vars(args),
        "source_guidance": str(guidance_dir), "source_trace": str(trace_dir), "seed": seed,
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True, capture_output=True, check=True).stdout.strip(),
        "runtime": {"torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "visible_cuda_count": torch.cuda.device_count(),
                    "sampling_cuda_index": target_index},
        "settings": "same y/H/x_T/grid/lr=.001/eta=1/fixed variance/joint zeta+D/last_opt",
        "formula": "current B(v) versus B(Mv), v=H^T(y-H(clamp(x0_raw)))",
        "limits": "frozen raw->mask is sensitivity only; matched mask->mask is a diagnostic variant, "
                  "not state-aware gain or proof of the authors' implementation",
        "ground_truth_usage": "metrics only, no schedule/output/epoch selection",
        "new_nfe": 0, "results": [], "gates": {}, "loss_histories": {},
        "configs": {name: arm["config"] for name, arm in saved["results"].items()},
        "identity": {key: guidance[key] for key in ("checkpoint_sha256", "measurement_sha256",
                                                    "operator_source_sha256", "core_sha256", "image_sha256")},
    }

    def save_record():
        (output_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                                        allow_nan=False), encoding="utf-8")

    def load_state(zaps, state):
        zaps.tau = state["tau"].to(args.device)
        with torch.no_grad():
            zaps.zeta.copy_(state["zeta"].to(args.device))
            zaps.D.copy_(state["D"].to(args.device))
        return state["init_noise"].to(args.device)

    def report(name, policy, output, baseline_metrics, nfe, elapsed):
        metrics = compute_all_metrics(output, gt, lpips_net=METRICS_CONFIG["lpips_net"])
        if baseline_metrics is None:
            baseline_metrics = metrics
        row = {"schedule": name, "policy": policy, **metrics,
               "dPSNR": metrics["psnr"] - baseline_metrics["psnr"],
               "dSSIM": metrics["ssim"] - baseline_metrics["ssim"],
               "dLPIPS": metrics["lpips"] - baseline_metrics["lpips"],
               "residual_norm": float((y - operator.H(output)).detach().norm()),
               "new_policy_nfe": nfe,
               "method_budget_nfe": 330 if policy == "frozen_raw_to_mask" else 300,
               "seconds": elapsed}
        record["results"].append(row)
        torch.save(output.detach().cpu(), output_dir / f"{name}_{policy}.pt")
        tensor_to_image(output.detach().cpu().squeeze(0), denormalize=True).save(output_dir / f"{name}_{policy}.png")
        save_record()
        print(f"{name} {policy}: PSNR={metrics['psnr']:.4f} dPSNR={row['dPSNR']:+.4f} "
              f"SSIM={metrics['ssim']:.4f} LPIPS={metrics['lpips']:.4f}", flush=True)
        return metrics

    save_record()
    print("\n=== Clamp guidance ablation: no LR/grid/noise/default changes ===", flush=True)
    print(f"Records: {output_dir}", flush=True)
    try:
        for name, arm in saved["results"].items():
            old_state = torch.load(trace_dir / f"{name}_last_unroll_state.pt", map_location="cpu", weights_only=True)
            mapping = saved_cuda_selection(old_state, saved["arguments"].get("device"), args.saved_cuda_index)
            old_index = mapping["saved_sampling_index"]
            print(f"\n--- {name}; saved RNG {old_index} -> current cuda:{target_index} ---", flush=True)
            config = arm["config"]
            core = ZAPS(model, operator, img_size=IMG_SIZE[0], **config)
            initial = load_state(core, old_state)
            restore_sampling_rng(old_state, args.device, old_index)
            baseline_output, count_a, _ = core.sample(y, init_noise=initial)
            baseline_end = rng_state(args.device)
            restore_sampling_rng(old_state, args.device, old_index)
            repeat, count_b, _ = core.sample(y, init_noise=initial)
            repeat_error = relative_error(baseline_output, repeat)
            repeat_end = rng_state(args.device)
            record["new_nfe"] += count_a + count_b
            del core, repeat

            zaps = ClampZAPS(model, operator, img_size=IMG_SIZE[0], use_clamp_mask=False, **config)
            initial = load_state(zaps, old_state)
            restore_sampling_rng(old_state, args.device, old_index)
            null, count_c, _ = zaps.sample(y, init_noise=initial)
            record["new_nfe"] += count_c
            null_gate = classify_parity(repeat_error, relative_error(baseline_output, null),
                                        count_a == count_b == count_c == 30
                                        and same_rng(baseline_end, repeat_end)
                                        and same_rng(baseline_end, rng_state(args.device)))
            archive_gate = archive_fingerprint_check(zaps.clamp_rows, arm["rows"])
            record["gates"][name] = {"null_wrapper": null_gate, "archive_fingerprint": archive_gate,
                                    "rng_mapping": mapping}
            save_record()
            if not null_gate["passed"] or not archive_gate["passed"]:
                raise RuntimeError("Null wrapper or archived baseline gate failed; do not interpret the mask ablation")
            del null
            baseline_metrics = report(name, "raw_baseline", baseline_output, None, 0, 0.)
            record["gates"][name]["archived_psnr_delta"] = baseline_metrics["psnr"] - arm["metrics"]["psnr"]
            del baseline_output

            zaps.use_clamp_mask = True
            restore_sampling_rng(old_state, args.device, old_index)
            started = time.time()
            frozen, count_d, _ = zaps.sample(y, init_noise=initial)
            elapsed = time.time() - started
            record["new_nfe"] += count_d
            if count_d != 30 or not same_rng(baseline_end, rng_state(args.device)):
                raise RuntimeError("Frozen mask ablation changed random-number consumption")
            report(name, "frozen_raw_to_mask", frozen, baseline_metrics, count_d, elapsed)
            del zaps, frozen, initial

            print(f"{name}: train matched clamp mask, original lr={config['lr']}, joint zeta+D...", flush=True)
            set_seed(seed)
            zaps = ClampZAPS(model, operator, img_size=IMG_SIZE[0], use_clamp_mask=True, **config)
            zaps.tau = old_state["tau"].to(args.device)
            started = time.time()
            losses = zaps.optimize(y, verbose=True, x0_gt=gt)
            elapsed = time.time() - started
            training_nfe = zaps._last_nfe
            record["new_nfe"] += training_nfe
            last_opt = zaps._last_opt_x0.detach()
            new_state = zaps.audit_state
            if new_state is None or training_nfe != 300:
                raise RuntimeError("Masked training did not complete the original 30x10 budget")
            paired = training_pairing(old_state, new_state, old_index, target_index)
            record["gates"][name]["training_randomness"] = paired
            record["loss_histories"][name] = losses
            save_record()
            if not all(paired.values()):
                raise RuntimeError(f"Masked training is not paired to the archived baseline: {paired}")
            torch.save(new_state, output_dir / f"{name}_mask_last_unroll_state.pt")
            initial = load_state(zaps, new_state)
            restore_sampling_rng(new_state, args.device, target_index)
            replay, count_e, _ = zaps.sample(y, init_noise=initial)
            replay_end = rng_state(args.device)
            restore_sampling_rng(new_state, args.device, target_index)
            replay_repeat, count_f, _ = zaps.sample(y, init_noise=initial)
            record["new_nfe"] += count_e + count_f
            train_gate = classify_parity(relative_error(replay, replay_repeat), relative_error(replay, last_opt),
                                         count_e == count_f == 30 and same_rng(replay_end, rng_state(args.device)))
            record["gates"][name]["masked_last_opt_replay"] = train_gate
            save_record()
            if not train_gate["passed"]:
                raise RuntimeError("Matched-mask last_opt replay gate failed")
            report(name, "matched_mask_training", last_opt, baseline_metrics, training_nfe, elapsed)
            del zaps, old_state, new_state, initial, replay, replay_repeat, last_opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        record["status"] = "complete"
        save_record()
    except Exception as error:
        record["status"], record["error"] = "failed", repr(error)
        save_record()
        raise
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record["results"][0]))
        writer.writeheader()
        writer.writerows(record["results"])
    print("\n=== Paired clamp-guidance reconstruction summary ===", flush=True)
    print(f"{'schedule':>20} {'policy':>23} {'PSNR':>9} {'delta':>9} {'SSIM':>8} {'LPIPS':>8} {'dLPIPS':>9}", flush=True)
    for row in record["results"]:
        print(f"{row['schedule']:>20} {row['policy']:>23} {row['psnr']:9.4f} {row['dPSNR']:+9.4f} "
              f"{row['ssim']:8.4f} {row['lpips']:8.4f} {row['dLPIPS']:+9.4f}", flush=True)
    print("Frozen raw->mask is sensitivity only; matched mask training tests this one implementation choice.", flush=True)
    print("No original-paper recovery/state-aware gain claim from one image. Formal defaults and FFHQ unchanged.", flush=True)
    print(f"New NFE={record['new_nfe']} (default 960); records: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
