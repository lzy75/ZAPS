#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a36_selection_file=$(mktemp)
trap 'rm -f -- "$zaps_a36_selection_file"' EXIT
python -m utils.experiment_gpu_selection > "$zaps_a36_selection_file"
mapfile -t zaps_a36_selected < "$zaps_a36_selection_file"
if [ "${#zaps_a36_selected[@]}" -ne 2 ];then echo 'Invalid GPU selection' >&2;exit 1;fi
# Keep the hash-bound scientific launcher, plan, image list and algorithm files byte-identical.
ZAPS_A36_GPU0="${zaps_a36_selected[0]}" ZAPS_A36_GPU1="${zaps_a36_selected[1]}" \
  bash run_a36_20261023.sh
