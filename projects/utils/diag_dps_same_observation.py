"""Official DPS sampler/PS conditioning on the existing ZAPS observation.

ImageNet Gaussian deblurring only. Keep the archived H, y, image and x_T;
run unmodified DPS p_sample_loop (1000 steps, learned_range, PS scale=0.3).
This is not the stock DPS blur operator: sharing H is intentional. Different
step counts/conditioning/variance policies mean this is NOT a one-factor ablation.
"""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import types

PROJECTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECTS_ROOT.parent
sys.path.insert(0, str(PROJECTS_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from utils.diag_zaps_sampler_parity import load_dps_reference


class PureOperatorAdapter:
    """DPS's forward() must compute H(x), not this project's noisy forward()."""

    def __init__(self, operator):
        self.operator = operator

    def forward(self, data, **kwargs):
        if kwargs:
            raise ValueError("Gaussian H does not accept extra conditioning arguments")
        return self.operator.H(data)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dps_config(diffusion, conditioning):
    required = {
        "sampler": "ddpm", "steps": 1000, "noise_schedule": "linear",
        "model_mean_type": "epsilon", "model_var_type": "learned_range",
        "dynamic_threshold": False, "clip_denoised": True,
        "rescale_timesteps": False,
    }
    for key, value in required.items():
        if diffusion.get(key) != value:
            raise ValueError(f"DPS baseline requires {key}={value!r}, got {diffusion.get(key)!r}")
    if str(diffusion.get("timestep_respacing")) != "1000":
        raise ValueError("DPS baseline requires all 1000 original timesteps")
    if conditioning["method"] != "ps":
        raise ValueError("DPS baseline requires PS conditioning")
    scale = conditioning["params"]["scale"]
    if not math.isfinite(scale) or scale != 0.3:
        raise ValueError("This diagnostic fixes official Gaussian PS scale=0.3; no scale search")


def set_seed(seed):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dps-root", default=str(REPO_ROOT / "DPS"))
    args = parser.parse_args()
    trace_dir = Path(args.trace_dir).resolve()
    saved = json.loads((trace_dir / "audit.json").read_text(encoding="utf-8"))
    source = Path(saved["source"])
    original = json.loads((source / "run.json").read_text(encoding="utf-8"))
    run_args = original["arguments"]
    task = saved["arguments"]["task"]
    if run_args["dataset"] != "imagenet" or task != "gaussian_deblur":
        parser.error("Use the saved ImageNet Gaussian-deblurring optimized trace")
    expected_arms = {"irregular_15_10_5", "uniform_30"}
    if set(saved["results"]) != expected_arms:
        raise RuntimeError("Source must contain both original ZAPS schedule arms")
    for arm in saved["results"].values():
        if not arm["gates"]["passed"] or arm["config"]["eta"] != 1.0:
            raise RuntimeError("Use the original eta=1 last-opt trace with a passed replay gate")

    import torch
    import yaml
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from modules.degradations import get_operator
    from modules.dataset_loader import tensor_to_image
    from utils.metrics import compute_all_metrics, compute_psnr
    from utils.diag_ffhq_regression import tensor_psnr

    dps_root = Path(args.dps_root).resolve()
    diffusion_path = dps_root / "configs/diffusion_config.yaml"
    task_path = dps_root / "configs/gaussian_deblur_config.yaml"
    diffusion = yaml.safe_load(diffusion_path.read_text())
    official_task = yaml.safe_load(task_path.read_text())
    conditioning_config = official_task["conditioning"]
    validate_dps_config(diffusion, conditioning_config)
    # The source H and observation replace the stock DPS measurement operator.
    # Its sampler/gradient guidance/settings are imported verbatim below.
    model = load_diffusion_model("imagenet", args.device)
    model.model.eval()
    if not all(parameter.requires_grad for parameter in model.model.parameters()):
        raise RuntimeError("Keep model parameter requires_grad=True for guided-diffusion's custom "
                           "checkpoint backward. No optimizer is used and no weights are updated.")
    dps, reference_metadata = load_dps_reference(dps_root)
    condition_module = importlib.import_module(dps.__package__ + ".condition_methods")
    reference_metadata["source_sha256"][condition_module.__file__] = file_sha256(condition_module.__file__)
    sampler = dps.create_sampler(**diffusion)
    if sampler.timestep_map != list(range(1000)):
        raise RuntimeError("DPS sampler does not visit the full original 1000-step grid")
    operator = get_operator(task, device=args.device, **original["task_configs"][task])
    adapter = PureOperatorAdapter(operator)
    # Official PS uses only noiser.__name__ to select the Gaussian L2 gradient.
    # There is no new noiser call: y already contains the archived noise draw.
    gaussian_tag = types.SimpleNamespace(__name__="gaussian", sigma=operator.noise.sigma)
    condition = condition_module.get_conditioning_method(
        name=conditioning_config["method"], operator=adapter, noiser=gaussian_tag,
        **conditioning_config["params"],
    )
    measurement_path = source / task / "measurement.pt"
    y = torch.load(measurement_path, map_location=args.device, weights_only=True)
    gt = load_image_as_tensor(run_args["image"]).to(args.device)
    first_state = torch.load(trace_dir / "irregular_15_10_5_last_unroll_state.pt",
                             map_location="cpu", weights_only=True)
    other_state = torch.load(trace_dir / "uniform_30_last_unroll_state.pt",
                             map_location="cpu", weights_only=True)
    if not torch.equal(first_state["init_noise"], other_state["init_noise"]):
        raise RuntimeError("Source arms have different x_T")
    init_noise = first_state["init_noise"].to(args.device)
    if tuple(init_noise.shape) != tuple(gt.shape) or tuple(y.shape) != tuple(gt.shape):
        raise RuntimeError("Unexpected ImageNet Gaussian task tensor shapes")
    if not torch.isfinite(y).all() or not torch.isfinite(gt).all():
        raise RuntimeError("Non-finite archived observation or ground truth")
    seed = int(run_args["seed"])
    # Reconstruct original seed initialization, then retain its post-x_T RNG.
    # 1000 DPS steps cannot be draw-for-draw paired with ZAPS's 30x10 unrolls.
    set_seed(seed)
    seed_noise = torch.randn(tuple(init_noise.shape), device=args.device, dtype=init_noise.dtype)
    if not torch.equal(seed_noise, init_noise):
        raise RuntimeError("Source seed does not reproduce archived x_T; refusing ambiguous RNG pairing")
    del seed_noise, other_state
    sampling_rng_cpu = torch.get_rng_state().clone()
    sampling_rng_cuda = torch.cuda.get_rng_state_all() if str(args.device).startswith("cuda") else None
    output_dir = trace_dir / ("dps_same_observation_" + time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    with torch.no_grad():
        physical_noise = y - operator.H(gt)
        clean_operator_check = torch.equal(adapter.forward(gt), operator.H(gt))
    if not clean_operator_check:
        raise RuntimeError("Adapter forward does not equal the saved task's pure H")
    record = {
        "experiment": "official_DPS_same_observation", "status": "preflight",
        "arguments": vars(args), "git_revision": revision, "source_trace": str(trace_dir),
        "source_git": original.get("git"), "source_trace_git": saved.get("git"),
        "reference": reference_metadata, "diffusion_config": diffusion,
        "conditioning_config": conditioning_config, "task": task, "seed": seed,
        "operator_config": original["task_configs"][task],
        "operator_source_sha256": file_sha256(PROJECTS_ROOT / "modules/degradations.py"),
        "operator_semantics": "source ZAPS H (including its zero-padding), NOT stock DPS blur H",
        "paired_scope": "same H, saved y, source image, model loader and x_T; not same full noise sequence",
        "measurement_path": str(measurement_path), "measurement_sha256": file_sha256(measurement_path),
        "image_path": run_args["image"], "image_sha256": file_sha256(run_args["image"]),
        "checkpoint_path": model.ckpt_path,
        "checkpoint_identity_limit": "source archive did not store a checkpoint hash; current hash recorded",
        "observed_psnr": compute_psnr(y, gt),
        "inferred_measurement_noise_mean": float(physical_noise.mean()),
        "inferred_measurement_noise_std": float(physical_noise.std()),
        "metric_convention": "same compute_all_metrics as ZAPS; clamp/unit-range conversion, no min-max stretch",
        "gates": {"source_xT_equal": True, "seed_xT_equal": True, "pure_H_equal": True},
        "zaps_archived": {name: {"metrics": arm["metrics"], "config": arm["config"],
                                  "nfe": arm["optimization_nfe"],
                                  "float_psnr": arm["summary"]["final_float_psnr"]}
                          for name, arm in saved["results"].items()},
    }

    def save_record():
        (output_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False,
                                                        indent=2, allow_nan=False), encoding="utf-8")

    print("\n=== Official DPS, same H/y/image/x_T as archived ZAPS ===", flush=True)
    print(f"1000 DDPM steps; learned_range; PS scale=0.3; seed={seed}", flush=True)
    print("No ZAPS optimization, no new measurement noise, no late-noise scaling.", flush=True)
    print(f"Observed PSNR: {record['observed_psnr']:.4f}; records: {output_dir}", flush=True)
    print("Recording current checkpoint SHA256 (local file read, no download)...", flush=True)
    record["checkpoint_sha256"] = file_sha256(model.ckpt_path)
    save_record()
    # A single real input-gradient check catches the earlier frozen-parameter
    # checkpoint error before spending minutes on the 1000-step run.
    print("Checking official PS input gradient at t=999...", flush=True)
    try:
        with torch.enable_grad():
            probe = init_noise.detach().clone().requires_grad_(True)
            t = torch.full((probe.shape[0],), 999, device=probe.device, dtype=torch.long)
            probe_out = sampler.p_mean_variance(model.model, probe, t)
            grad, norm = condition.grad_and_value(probe, probe_out["pred_xstart"], y)
            if not torch.isfinite(grad).all() or not torch.isfinite(norm):
                raise RuntimeError("Official DPS input gradient is not finite")
            record["preflight_gradient_norm"] = float(grad.detach().norm())
            record["preflight_residual_norm"] = float(norm.detach())
        del probe, probe_out, grad, norm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Input-gradient check passed; starting official 1000-step loop...", flush=True)
        save_record()
        calls = []

        def counted_model(x, t):
            expected_t = 999 - len(calls)
            if not torch.equal(t, torch.full_like(t, expected_t)):
                raise RuntimeError("Official DPS loop supplied an unexpected timestep")
            if not calls and not torch.equal(x.detach(), init_noise):
                raise RuntimeError("Official DPS loop changed x_T before its first model call")
            calls.append(expected_t)
            return model.model(x, t)

        torch.set_rng_state(sampling_rng_cpu)
        if sampling_rng_cuda is not None:
            torch.cuda.set_rng_state_all(sampling_rng_cuda)
        started = time.time()
        with torch.enable_grad():
            reconstruction = sampler.p_sample_loop(
                model=counted_model, x_start=init_noise.detach().clone(), measurement=y,
                measurement_cond_fn=condition.conditioning, record=False, save_root=str(output_dir),
            ).detach()
        elapsed = time.time() - started
        if calls != list(range(999, -1, -1)):
            raise RuntimeError("Official DPS did not finish exactly 1000 model calls")
        if not torch.isfinite(reconstruction).all():
            raise RuntimeError("DPS reconstruction is not finite")
        if any(parameter.grad is not None for parameter in model.model.parameters()):
            raise RuntimeError("Unexpected accumulated model parameter gradients")
        torch.save(reconstruction.cpu(), output_dir / "dps_reconstruction.pt")
        tensor_to_image(reconstruction.cpu().squeeze(0), denormalize=True).save(output_dir / "dps_reconstruction.png")
        metrics = compute_all_metrics(reconstruction, gt)
        record["dps"] = {**metrics, "float_psnr": tensor_psnr(gt, reconstruction),
                         "nfe": len(calls), "preflight_nfe": 1, "total_nfe": len(calls) + 1,
                         "seconds": elapsed,
                         "residual_norm": float((y - operator.H(reconstruction)).norm())}
        record["gates"].update(full_timestep_grid=True, model_grad_not_accumulated=True)
        record["status"] = "complete"
        save_record()
    except Exception as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        save_record()
        raise
    print("\n=== Same-observation baseline summary ===", flush=True)
    print(f"{'method':>22} {'PSNR':>9} {'SSIM':>8} {'LPIPS':>8} {'NFE':>6}", flush=True)
    dps_result = record["dps"]
    print(f"{'DPS_official_1000':>22} {dps_result['psnr']:9.4f} {dps_result['ssim']:8.4f} "
          f"{dps_result['lpips']:8.4f} {dps_result['nfe']:6d}", flush=True)
    for name, arm in record["zaps_archived"].items():
        m = arm["metrics"]
        print(f"{name:>22} {m['psnr']:9.4f} {m['ssim']:8.4f} {m['lpips']:8.4f} {arm['nfe']:6d}", flush=True)
    print("Archived ZAPS rows are not retrained. Different NFE/variance/guidance: not a one-factor ablation.", flush=True)
    print("A single-image comparison does not establish reproduction of paper dataset means.", flush=True)
    print(f"Records saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
