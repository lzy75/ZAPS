#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")" || exit 1
python_bin=${PYTHON_BIN:-/home/lzy/anaconda3/bin/python}
root=${A147_ROOT:-/data0/lzy/ZAPS-results/diag_a147_temporal_confirmation}
first=${1:-0}
second=${2:-1}
if [[ "$first" == "$second" ]]; then
  echo 'Pass two different GPU indices.' >&2
  exit 2
fi
mkdir -p "$root" || exit 1
exec 9>"$root/launcher.lock"
flock -n 9 || { echo 'An A147 queue already holds the launcher lock.' >&2; exit 1; }
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

free() {
  local device=$1 mem apps
  mem=$(nvidia-smi -i "$device" --query-gpu=memory.free --format=csv,noheader,nounits) || return 2
  apps=$(nvidia-smi -i "$device" --query-compute-apps=pid --format=csv,noheader,nounits) || return 2
  [[ "$mem" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || return 2
  [[ "$mem" -ge 6144 ]] && ! grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$' <<<"$apps"
}

wait_gpu() {
  local device=$1 quiet=0 status
  while :; do
    free "$device"
    status=$?
    case "$status" in
      0) quiet=$((quiet+1)); if ((quiet>=2)); then return 0; fi ;;
      1) quiet=0; printf 'A147 WAIT gpu=%s still occupied\n' "$device" ;;
      *) printf 'A147 invalid GPU query for %s\n' "$device" >&2; return 2 ;;
    esac
    sleep 10
  done
}

lane() {
  local device=$1 job host task status=0
  shift
  for job in "$@"; do
    host=${job%%:*}
    task=${job#*:}
    wait_gpu "$device" || return
    printf 'A147 START gpu=%s %s %s\n' "$device" "$host" "$task"
    CUDA_VISIBLE_DEVICES="$device" "$python_bin" -u -m utils.diag_a147_temporal_confirmation "$host" "$task" --root "$root" \
      2>&1 | tee -a "$root/${host}_${task}.log" || status=1
  done
  return "$status"
}

lane "$first" dps:inpainting &
lane_a=$!
lane "$second" pigdm:inpainting dps:super_resolution &
lane_b=$!
failed=0
wait "$lane_a" || failed=1
wait "$lane_b" || failed=1
CUDA_VISIBLE_DEVICES="" "$python_bin" -m utils.diag_a147_temporal_confirmation matrix --root "$root" || failed=1
exit "$failed"
