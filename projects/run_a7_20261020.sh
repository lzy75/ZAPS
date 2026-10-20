#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a7_root=/data0/lzy/ZAPS-results/diag_direction_value
mkdir -p "$zaps_a7_root"
exec 9>"$zaps_a7_root/A7_20261020.launcher.lock"
if ! flock -n 9; then echo 'A7 already running; duplicate submission refused.' >&2; exit 1; fi
zaps_a7_log="$zaps_a7_root/A7_20261020.log"
echo 'A7 started: auditing completed data and testing incremental direction information; NO new UNet calls. Waiting for summary.'
if (
  date -Is
  python -m unittest utils.test_diag_direction_value -v
  zaps_a7_checks=$?
  if [ "$zaps_a7_checks" -ne 0 ]; then exit "$zaps_a7_checks"; fi
  python -u utils/diag_direction_value.py
) >> "$zaps_a7_log" 2>&1; then
  python - <<'PY'
import json
from pathlib import Path
p=Path('/data0/lzy/ZAPS-results/diag_direction_value/A7_20261020')
print('A7 Summary:',json.dumps(json.loads((p/'summary.json').read_text()),ensure_ascii=False))
print('Records:',p)
PY
  if git push origin results/refinement-confirmation-20261020 >> "$zaps_a7_log" 2>&1; then
    echo 'RESULTS_PUSH_PASS: results/refinement-confirmation-20261020 (A7 appended; source experiments preserved).'
  else
    echo 'A7 analysis complete. RESULTS_PUSH_PENDING; local results commit retained; no GPU rerun needed.'
    tail -n 10 "$zaps_a7_log"
  fi
else
  echo "A7 stopped; full log: $zaps_a7_log" >&2
  tail -n 50 "$zaps_a7_log"
  exit 1
fi
