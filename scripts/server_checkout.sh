#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "用法: $0 <commit-or-tag>" >&2
  exit 2
fi

target="$1"
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "拒绝切换：服务器存在已跟踪文件的本地修改。" >&2
  git status --short >&2
  exit 1
fi

git fetch --prune origin
git rev-parse --verify "${target}^{commit}" >/dev/null
git checkout --detach "$target"

echo "服务器代码已切换："
git show -s --format='commit=%H%ncommit_time=%cI%nsubject=%s' HEAD
echo "tracked_status=$(git status --porcelain --untracked-files=no | wc -l)"
