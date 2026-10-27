#!/usr/bin/env bash
set -euo pipefail
zaps_night_pkg=${1:-}
case "$zaps_night_pkg" in
  A56) zaps_night_min=648030e;zaps_night_entry=a56_dataset_queue_session.py;;
  A57) zaps_night_min=cb449fd;zaps_night_entry=a57_operator_repair_session.py;;
  A58) zaps_night_min=309e291;zaps_night_entry=a58_session.py;;
  A59) zaps_night_min=fa2d7fa;zaps_night_entry=a59_session.py;;
  A60) zaps_night_min=3e855f2;zaps_night_entry=a60_session.py;;
  *) echo 'Usage: bash /home/lzy/ZAPS/projects/overnight_20261027/run_commands.sh A56|A57|A58|A59|A60' >&2;exit 2;;
esac
source /home/lzy/anaconda3/etc/profile.d/conda.sh
conda activate base
cd /home/lzy/ZAPS
env -u NO_PROXY -u no_proxy git -c http.proxy=http://127.0.0.1:10809 \
  fetch https://github.com/lzy75/ZAPS.git refs/heads/master:refs/remotes/origin/master
git merge --ff-only origin/master
git merge-base --is-ancestor "$zaps_night_min" HEAD
cd projects
case "$zaps_night_pkg" in
  A56|A57) python -u "utils/$zaps_night_entry" --status;;
esac
exec python -u "utils/$zaps_night_entry"
