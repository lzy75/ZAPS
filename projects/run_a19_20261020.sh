#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a19_parent=/data0/lzy/ZAPS-results/diag_terminal_failure_attribution
mkdir -p "$zaps_a19_parent"
exec 9>"$zaps_a19_parent/launcher.lock"
if ! flock -n 9; then echo 'A19 already running; duplicate refused.' >&2; exit 1; fi
zaps_a19_log="$zaps_a19_parent/launcher_20261020.log"
echo 'A19 cached failure attribution: CPU only, UNet NFE=0. Waiting for combined summary and comparison figures.'
python -m unittest utils.test_diag_terminal_failure_attribution -v
if ! env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u utils/diag_terminal_failure_attribution.py > "$zaps_a19_log" 2>&1; then
  echo 'Offline audit failed; source experiments unchanged.'
  tail -n 45 "$zaps_a19_log"; exit 1
fi
python -u utils/diag_terminal_failure_attribution.py --summary
echo 'Figures: /data0/lzy/ZAPS-results/diag_terminal_failure_attribution/A19_20261020/comparison_00.png through comparison_03.png'
if git push origin results/refinement-confirmation-20261020 >> "$zaps_a19_log" 2>&1; then
  echo 'RESULTS_PUSH_PASS: A19 offline attribution and comparison figures appended; UNet NFE=0.'
else
  echo 'RESULTS_PUSH_PENDING: offline records retained; no experiment rerun needed.'
  tail -n 12 "$zaps_a19_log"
fi
