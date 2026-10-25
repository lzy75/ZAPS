"""Zero-UNet check of resolved-branch bounds for the existing A50 primitive."""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.current_score_geometry import score_targets


def check():
    rng = np.random.default_rng(9001)
    excess = 0.0
    identity_error = 0.0
    trials = 2000
    for _ in range(trials):
        q, u, v = (rng.normal(size=23) * 10.0 ** rng.uniform(-8, 8)
                   for _ in range(3))
        k = 10.0 ** rng.uniform(-4, 4)
        record = score_targets(q, v, u, k)
        gamma = record["decisions"]["ALIGN_Q"]["gamma"]
        nq, nv, nu = (np.linalg.norm(w) for w in (q, v, u))
        cosine_product = (v @ u / nv / nu) * (q @ u / nq / nu)
        expected = k * nv / nq * max(0.0, float(cosine_product))
        identity_error = max(identity_error, abs(gamma - expected) / max(1.0, abs(expected)))
        excess = max(excess, np.linalg.norm(gamma * q) / (k * nv) - 1.0)
    # Explicit alignment/anti-alignment cases meet the two norm boundaries.
    q = np.array([1.0, 0.0])
    aligned = score_targets(q, q, q, 3.0)["decisions"]["ALIGN_Q"]["gamma"]
    blocked = score_targets(-q, q, q, 3.0)["decisions"]["ALIGN_Q"]["gamma"]
    assert aligned == 3.0 and blocked == 0.0
    assert excess <= 1e-12 and identity_error <= 1e-12
    return {"status": "RESOLVED_BRANCH_ALGEBRA_ONLY", "trials": trials, "seed": 9001,
            "maximum_relative_norm_bound_excess": float(excess),
            "maximum_scaled_cosine_identity_error": float(identity_error),
            "aligned_gamma": aligned, "anti_aligned_gamma": blocked,
            "method_nfe": 0, "diagnostic_nfe": 0,
            "limits": "Float64 vectors only;not GPU,quality,safe-step or posterior proof;fallback excluded."}


if __name__ == "__main__":
    result = check()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
