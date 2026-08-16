#!/usr/bin/env bash
# 一键上传实验结果到 GitHub。
# 用法(在服务器仓库根目录):
#   bash scripts/upload_results.sh "自适应 w_cos=0.3 vs 基线"
# 只提交文本结果(CSV/run.json/conclusion.md);重建图 *.png 由 .gitignore 排除。
set -euo pipefail

MSG="${1:-exp: 实验结果}"
cd "$(git rev-parse --show-toplevel)"

echo "[1/4] 同步远程(避免推送冲突)..."
git pull --no-edit origin master

echo "[2/4] 暂存实验结果..."
git add projects/experiments/

echo "[3/4] 待提交内容(确认无大文件/权重):"
git status --short projects/experiments/ | head -40
# 安全检查:拦截意外的大文件(权重/数据)
if git diff --cached --name-only | grep -qiE '\.(pt|pth|npz|ckpt)$'; then
  echo "❌ 检测到权重/大文件被暂存,已中止。请检查 .gitignore。" >&2
  git reset >/dev/null
  exit 1
fi

if git diff --cached --quiet; then
  echo "(无新结果可提交)"; exit 0
fi

echo "[4/4] 提交并推送..."
git commit -m "exp: ${MSG}"
git push origin master
echo "✅ 完成。最新提交:"
git log --oneline -1
