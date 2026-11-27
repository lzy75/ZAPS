#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")" || exit 1
python_bin=${PYTHON_BIN:-/home/lzy/anaconda3/bin/python}
root=${A150_ROOT:-/data0/lzy/ZAPS-results/diag_a150_guarded_temporal_confirmation}
first=${1:-0}
second=${2:-1}
if [[ "$first" == "$second" ]]; then
  echo 'A150 needs two different GPU indices.' >&2
  exit 2
fi
mkdir -p "$root" || exit 1
exec 9>"$root/launcher.lock"
flock -n 9 || { echo 'An A150 queue already holds the launcher lock.' >&2; exit 1; }
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL
CUDA_VISIBLE_DEVICES="" "$python_bin" -m unittest -v utils.test_a146_guarded_panel utils.test_a145_temporal_panel \
  2>&1 | tee -a "$root/tests.log" || exit 1

wait_gpu() {
  local device=$1 free apps quiet=0
  while (( quiet < 2 )); do
    free=$(nvidia-smi -i "$device" --query-gpu=memory.free --format=csv,noheader,nounits) || return 2
    apps=$(nvidia-smi -i "$device" --query-compute-apps=pid --format=csv,noheader,nounits) || return 2
    [[ "$free" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || return 2
    if [[ "$free" -ge 6144 ]] && ! grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$' <<<"$apps"; then
      quiet=$((quiet+1))
    else
      quiet=0
      printf 'A150 WAIT gpu=%s occupied or under 6 GiB free\n' "$device"
    fi
    (( quiet >= 2 )) || sleep 10
  done
}

lane() {
  local device=$1 job host task failed=0
  shift
  for job in "$@"; do
    host=${job%%:*}
    task=${job#*:}
    wait_gpu "$device" || return
    printf 'A150 START gpu=%s %s %s\n' "$device" "$host" "$task"
    CUDA_VISIBLE_DEVICES="$device" "$python_bin" -u -m utils.diag_a150_guarded_temporal_confirmation "$host" "$task" --root "$root" \
      2>&1 | tee -a "$root/${host}_${task}.log" || failed=1
  done
  return "$failed"
}

lane "$first" dps:super_resolution zaps:super_resolution pigdm:super_resolution \
  dps:gaussian_deblur zaps:gaussian_deblur pigdm:gaussian_deblur &
lane_a=$!
lane "$second" dps:inpainting zaps:inpainting pigdm:inpainting \
  dps:motion_deblur zaps:motion_deblur pigdm:motion_deblur &
lane_b=$!
failed=0
wait "$lane_a" || failed=1
wait "$lane_b" || failed=1
CUDA_VISIBLE_DEVICES="" "$python_bin" -m utils.diag_a150_guarded_temporal_confirmation matrix --root "$root" || failed=1
exit "$failed"
