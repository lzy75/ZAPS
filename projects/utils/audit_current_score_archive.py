"""Read-only A49/A50r2 metadata audit; no Torch, model, or tensor replay.

Accepts extracted lightweight archives. Missing PNGs may be reported, but missing
JSON/sidecars are errors. Historical files and scientific decisions are untouched.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


TASKS = ("super_resolution", "gaussian_deblur", "inpainting", "motion_deblur")
ARMS = ("PROX_Q", "DIRECT_Q", "ALIGN_Q")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b):
    require(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12), f"Mismatch: {a} vs {b}")


def verify_archive(root):
    checked, missing = [], []
    manifest = read(root / "manifest.json")
    for entry in manifest["files"]:
        path = root / entry["path"]
        require(path.resolve().is_relative_to(root.resolve()), "Unsafe manifest path")
        if not path.exists():
            require(path.suffix == ".png", f"Missing required metadata: {path}")
            missing.append(entry["path"])
            continue
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
                f"Archive byte/hash mismatch: {path}")
        checked.append(entry["path"])
    for sidecar in root.rglob("*.sha256"):
        source = sidecar.with_suffix(".json")
        require(source.is_file() and digest(source) == sidecar.read_text().strip(),
                f"JSON sidecar mismatch: {sidecar}")
    return dict(manifest_sha256=digest(root / "manifest.json"), verified_files=len(checked),
                unextracted_pngs=missing, scope="Available archived bytes, not excluded PT/UNet replay")


def audit49(root):
    verification = verify_archive(root)
    report = read(root / "report.json")
    require(report["completed_cases"] == len(report["rows"]) == 20, "Need all A49 cases")
    for group in report["groups"]:
        rows = [r for r in report["rows"] if r["case"]["task"] == group["task"]]
        require(len(rows) == group["cases"] == 5, "A49 group incomplete")
        for field, source in (("local_changed_from_legacy", "changed_local_vs_legacy"),
                              ("trust_changed_from_local", "changed_trust_vs_local"),
                              ("local_J_improved_final_PSNR_worse", "local_J_improved_final_PSNR_worse")):
            require(sum(bool(r[source]) for r in rows) == group[field], "A49 count mismatch")
        for metric, value in group["mean_LOCAL_minus_LEGACY"].items():
            near(mean(r["LOCAL_minus_LEGACY"][metric] for r in rows), value)
    return dict(verification=verification, groups=report["groups"], method_nfe=0, diagnostic_nfe=0,
                source_report_sha256=digest(root / "report.json"),
                scope="Recomputed archived per-case contrasts; source A48 tensor metrics not re-evaluated")


def audit50(root):
    verification = verify_archive(root)
    p, summary, evaluation = (read(root / name) for name in ("plan.json", "summary.json", "evaluation.json"))
    require(p["experiment"] == "A50r2_current_score_geometry", "Wrong experiment")
    require(len(p["cases"]) == len(evaluation["rows"]) == 20, "Need all 20 cases")
    require(len({c["key"] for c in p["cases"]}) == 20, "Duplicate case key")
    original = Path(verification_source := read(root / "manifest.json")["source"])
    records, forward, backward = {}, 0, 0
    for case in p["cases"]:
        records[case["key"]] = {}
        for phase in ("generate", "control"):
            paths = sorted((root / case["key"]).glob(phase + "_*/run.json"))
            require(len(paths) == 1, "Ambiguous/incomplete phase")
            path, = paths
            r = read(path)
            require(r["status"] == "complete" and r["case"] == case, "Phase identity/status mismatch")
            require(r["plan_sha256"] == digest(root / "plan.json"), "Phase plan differs")
            require(r["gates"] and all(g["passed"] for g in r["gates"].values()), "Failed original gate")
            logical = str(original / path.relative_to(root))
            require(evaluation["inputs"][logical] == digest(path), "Evaluation not bound to this phase")
            require(r["nfe"] == len(r["calls"]), "Forward accounting mismatch")
            require(r["backward_attempts"] == r["backward_completed"], "Partial backward attempt")
            forward += r["nfe"]
            backward += r["backward_attempts"]
            records[case["key"]][phase] = r
    require(forward == summary["diagnostic_nfe"] == 4872 and backward == summary["backward_attempts"] == 775,
            "Total cost mismatch")
    groups, recomputed_contrasts = [], 0
    max_formula_error = 0.
    for task in TASKS:
        group = next(g for g in summary["groups"] if g["task"] == task)
        rows = [r for r in evaluation["rows"] if r["case"]["task"] == task]
        require(len(rows) == 5 and len({r["case"]["image_id"] for r in rows}) == 5, "Task cohort differs")
        metrics = {r["case"]["image_id"]: r["metrics"] for r in rows}
        require(metrics == {r["image_id"]: r["metrics"] for r in group["per_image"]}, "Summary per-image values differ")
        for arm, comparisons in group["mean_deltas"].items():
            for reference, deltas in comparisons.items():
                for metric, value in deltas.items():
                    near(mean(m[arm][metric] - m[reference][metric] for m in metrics.values()), value)
                    recomputed_contrasts += 1
        counts, geometry = {}, []
        for arm in ARMS:
            trace = [v for r in rows for v in records[r["case"]["key"]]["generate"]["traces"][arm]]
            require(len(trace) == 150, "Missing full-unroll trace")
            resolved = sum(v["readout"].get("resolved", False) for v in trace)
            require(resolved == group["resolved_states"][arm], "Coverage differs")
            cg_failures = Counter(v["readout"]["cg"]["reason"] for v in trace
                                  if not v["readout"]["cg"]["passed"])
            counts[arm] = dict(resolved=resolved, total=len(trace), cg_failures=dict(cg_failures),
                               gamma_min=min(v["gamma"] for v in trace), gamma_median=median(v["gamma"] for v in trace),
                               gamma_max=max(v["gamma"] for v in trace))
            if arm == "ALIGN_Q":
                geometry = trace
                for v in trace:
                    z = v["readout"]
                    if z.get("resolved"):
                        expected = max(0., z["coefficient"] * z["v_RMS"] / z["q_RMS"] *
                                       z["cosine_v_u"] * z["cosine_q_u"])
                        max_formula_error = max(max_formula_error, abs(expected-v["gamma"]) / max(1., abs(v["gamma"])))
                        near(expected, v["gamma"])
        stages = []
        for lo, hi in ((0, 10), (10, 20), (20, 30)):
            selected = [v for v in geometry if lo <= v["step"] < hi]
            resolved = [v["readout"] for v in selected if v["readout"].get("resolved")]
            ratios = [z["gamma"]["ALIGN_Q"] / z["gamma"]["PROX_Q"] for z in resolved
                      if z["gamma"]["PROX_Q"] > 0.]
            stages.append(dict(steps_inclusive=[lo, hi-1], n=len(selected), resolved=len(resolved),
                               gamma_median=median(v["gamma"] for v in selected),
                               R_clean_median=median(v["readout"]["R_clean"] for v in selected),
                               same_state_ALIGN_over_PROX_median=median(ratios) if ratios else None))
        groups.append(dict(task=task, quality_candidate=group["quality_candidate"],
                           matching_candidate=group["matching_candidate"],
                           numerical_coverage_passed=group["numerical_coverage_passed"],
                           ALIGN_deltas=group["mean_deltas"]["ALIGN_Q"], positive_images=group["positive_images"],
                           traces=counts, ALIGN_stages=stages,
                           per_image_ALIGN_minus_TIME={i: m["ALIGN_Q"]["psnr"]-m["TIME"]["psnr"] for i,m in metrics.items()}))
    return dict(verification=verification, source_server_root=verification_source,
                source_summary_sha256=digest(root / "summary.json"), recomputed_mean_contrasts=recomputed_contrasts,
                max_relative_alignment_formula_error=max_formula_error, original_recorded_gates=40,
                source_forward_calls=forward, source_backward_attempts=backward,
                method_nfe=0, diagnostic_nfe=0, groups=groups,
                limits="Cached metadata audit only. Does not repeat gates, CG, VJP, image metrics or GPU sampling. "
                       "Five reused image clusters; no independent transfer, novel theorem or full posterior/curvature claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a49-archive", type=Path, required=True)
    parser.add_argument("--a50-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refuse overwriting previous audit")
    report = dict(experiment="A49_A50r2_cached_evidence_audit", method_nfe=0, diagnostic_nfe=0,
                  A49=audit49(args.a49_archive), A50r2=audit50(args.a50_archive))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("CACHED_AUDIT_PASS: A49 20 cases; A50r2 40 phases;",
          report["A50r2"]["recomputed_mean_contrasts"], "mean contrasts; UNet NFE=0.")
    for g in report["A50r2"]["groups"]:
        d = g["ALIGN_deltas"]
        print(g["task"], json.dumps(dict(PSNR_vs_C=d["C_NATIVE"]["psnr"], PSNR_vs_TIME=d["TIME"]["psnr"],
                                           coverage=g["traces"]["ALIGN_Q"]["resolved"],
                                           quality=g["quality_candidate"], matching=g["matching_candidate"])))


if __name__ == "__main__":
    main()
