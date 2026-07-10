param(
    [string]$Remote = "origin",
    [string]$Branch = "master"
)

$ErrorActionPreference = "Stop"
$repoRoot = git rev-parse --show-toplevel
Set-Location $repoRoot

$trackedChanges = git status --porcelain --untracked-files=no
if ($trackedChanges) {
    throw "Tracked files have uncommitted changes. Commit or restore them first."
}

$commit = git rev-parse HEAD
git push $Remote "HEAD:refs/heads/$Branch"

Write-Host "Published commit: $commit"
Write-Host "Server command: bash scripts/server_checkout.sh $commit"
