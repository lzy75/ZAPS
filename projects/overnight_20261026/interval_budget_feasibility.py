"""Exact combinatorial cost bound for the proposed B contract; no model run.

Thirty original anchor evaluations define 29 intervals and one terminal read.
A minimum block uses one evaluation. Every larger coarse/two-half attempt is
charged two evaluations; rejected attempts are charged separately, without
assuming cached-left reuse. This proves a budget bound, not accuracy or speed.
"""
from __future__ import annotations

import json
from pathlib import Path


def best_partition(intervals: int, maximum: int) -> tuple[int, list[int]]:
    widths = [w for w in (1, 2, 4, 8) if w <= maximum]
    costs = {w: 1 if w == 1 else 2 for w in widths}
    dp: list[tuple[int, list[int]]] = [(0, [])]
    for n in range(1, intervals + 1):
        options = [(dp[n - w][0] + costs[w], dp[n - w][1] + [w])
                   for w in widths if w <= n]
        dp.append(min(options, key=lambda v: (v[0], len(v[1]))))
    return dp[intervals]


def summarize() -> dict:
    original_anchors, frozen_prefix = 30, 270
    intervals = original_anchors - 1
    rows = []
    for maximum in (1, 2, 4, 8):
        internal_calls, partition = best_partition(intervals, maximum)
        suffix_calls = internal_calls + 1  # terminal t0 network read retained
        total_calls = frozen_prefix + suffix_calls
        saving = original_anchors - suffix_calls
        rows.append({
            "maximum_intervals_per_block": maximum,
            "one_minimum_cost_partition": partition,
            "accepted_blocks": len(partition),
            "all_accepted_suffix_forward_lower_bound": suffix_calls,
            "full_method_forward_lower_bound": total_calls,
            "maximum_forward_saving": saving,
            "maximum_whole_method_saving_fraction": saving / 300,
            "two_call_extra_attempts_to_erase_best_possible_saving": saving // 2,
        })
    # Independent closed forms for these four maxima; no stochastic simulation.
    assert [r["all_accepted_suffix_forward_lower_bound"] for r in rows] == [30, 30, 16, 10]
    assert all(sum(r["one_minimum_cost_partition"]) == intervals for r in rows)
    return {
        "status": "EXACT_DISCRETE_COST_BOUND_ONLY",
        "original_anchors": original_anchors,
        "original_intervals": intervals,
        "frozen_prefix_calls": frozen_prefix,
        "rows": rows,
        "method_nfe_executed": 0,
        "diagnostic_nfe_executed": 0,
        "assumptions": [
            "B uses widths1,2,4,8 original intervals and retains t999/t0 reads",
            "One forward per width1;two per accepted nonminimum block",
            "No rejection,CG,extra sensor or warmup is included in lower bound",
            "Each rejected nonminimum attempt adds two forward under this contract",
            "No cross-attempt left-read reuse assumed;different reuse needs new accounting",
        ],
        "limits": [
            "No actual model,quality,error-acceptance probability or walltime result",
            "Best partition is cost-only and not an online sampling policy",
            "Savings apply only to this last-unroll contract,not arbitrary adaptive solvers",
            "Total prefix cost changes and jointgamma/backward costs need independent protocol",
        ],
    }


if __name__ == "__main__":
    destination = Path(__file__).with_suffix(".json")
    result = summarize()
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "rows": result["rows"],
                      "method_nfe": 0, "diagnostic_nfe": 0}, ensure_ascii=False))
