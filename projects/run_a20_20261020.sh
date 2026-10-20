#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a20_parent=/data0/lzy/ZAPS-results/diag_cached_prior_ablation
mkdir -p "$zaps_a20_parent"
exec 9>"$zaps_a20_parent/launcher.lock"
if ! flock -n 9; then echo 'A20 already running; duplicate refused.' >&2; exit 1; fi
zaps_a20_log="$zaps_a20_parent/launcher_20261020.log"
echo 'A20 cached fixed-q prior-displacement ablation: CPU only, UNet NFE=0. Waiting for combined summary and comparison figures.'
python -m unittest utils.test_diag_cached_prior_ablation -v
if ! env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u utils/diag_cached_prior_ablation.py > "$zaps_a20_log" 2>&1; then
  echo 'Offline audit failed; source experiments unchanged.'
  tail -n 45 "$zaps_a20_log"; exit 1
fi
python -u utils/diag_cached_prior_ablation.py --summary
echo 'Figures: /data0/lzy/ZAPS-results/diag_cached_prior_ablation/A20_20261020/comparison_00.png through comparison_03.png'
if git push origin results/refinement-confirmation-20261020 >> "$zaps_a20_log" 2>&1; then
  echo 'RESULTS_PUSH_PASS: A20 cached p-removal ablation and comparison figures appended; UNet NFE=0.'
else
  echo 'RESULTS_PUSH_PENDING: offline records retained; no experiment rerun needed.'
  tail -n 12 "$zaps_a20_log"
fi
