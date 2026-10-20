#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a21_parent=/data0/lzy/ZAPS-results/diag_equal_nfe_state_guidance
mkdir -p "$zaps_a21_parent"
exec 9>"$zaps_a21_parent/launcher.lock"
if ! flock -n 9; then echo 'A21 already running; duplicate refused.' >&2; exit 1; fi
zaps_a21_log="$zaps_a21_parent/launcher_20261020.log"
echo 'A21 state guidance: every scientific arm=300 NFE; cached original-step experiment, new UNet calls=0. Waiting for summary.'
python -m unittest utils.test_diag_equal_nfe_state_guidance -v
if ! env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u utils/diag_equal_nfe_state_guidance.py > "$zaps_a21_log" 2>&1; then
  echo 'A21 failed; original experiments unchanged.'
  tail -n 45 "$zaps_a21_log"; exit 1
fi
python -u utils/diag_equal_nfe_state_guidance.py --summary
if git push origin results/refinement-confirmation-20261020 >> "$zaps_a21_log" 2>&1; then
  echo 'RESULTS_PUSH_PASS: A21 equal-300-NFE state guidance diagnosis appended; no new UNet calls.'
else
  echo 'RESULTS_PUSH_PENDING: records retained; no GPU rerun needed.'
  tail -n 12 "$zaps_a21_log"
fi
