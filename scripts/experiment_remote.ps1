[CmdletBinding()]
param(
    [ValidateSet('Sync', 'Check', 'Submit', 'Status', 'Logs', 'Collect', 'AuditExisting')]
    [string]$Action = 'Status',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$')]
    [string]$RunId,
    [string]$Spec,
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$Commit,
    [string]$OutputDirectory,
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion -lt [version]'7.4') { throw 'PowerShell 7.4+ required for binary SSH output.' }
$sshArgs = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ClearAllForwardings=yes', 'server')
$sourceFile = Join-Path $PSScriptRoot 'server_job.py'
if ($Action -eq 'AuditExisting') { $sourceFile = Join-Path $PSScriptRoot 'audit_existing_results.py' }
$code = [Convert]::ToBase64String([IO.File]::ReadAllBytes($sourceFile))
$bootstrap = "import base64;CONTROLLER_SOURCE=base64.b64decode('$code').decode('utf-8');exec(compile(CONTROLLER_SOURCE,'server_job.py','exec'))"
$remoteCommand = "/home/lzy/anaconda3/bin/python -B -c `"$bootstrap`""
if ($Action -eq 'Sync') {
    if (-not $Commit) { throw '-Commit is required.' }
    $remoteCommand += " sync --commit $Commit"
} elseif ($Action -in @('Check', 'Submit')) {
    if (-not $Spec) { throw '-Spec is required.' }
    if ($Action -eq 'Submit' -and -not $Execute) { throw 'Submit requires explicit -Execute and an approved new experiment card.' }
    $specData = [Convert]::ToBase64String([IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $Spec)))
    $remoteCommand += " $($Action.ToLowerInvariant()) --spec-base64 $specData"
} elseif ($Action -ne 'AuditExisting') {
    if (-not $RunId) { throw '-RunId is required.' }
    $remoteCommand += " $($Action.ToLowerInvariant()) --run-id $RunId"
}
if ($Action -in @('Collect', 'AuditExisting')) {
    if (-not $OutputDirectory) { throw '-OutputDirectory is required and must be a new directory.' }
    if (Test-Path -LiteralPath $OutputDirectory) { throw 'Refusing to overwrite an existing collection.' }
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
    $partial = Join-Path $OutputDirectory 'results.tar.gz.partial'
    $archive = Join-Path $OutputDirectory 'results.tar.gz'
    & ssh @sshArgs $remoteCommand > $partial
    if ($LASTEXITCODE -ne 0) { throw 'SSH failed; partial output retained. No task resubmitted.' }
    Move-Item -LiteralPath $partial -Destination $archive
    & python -B (Join-Path $PSScriptRoot 'verify_light_archive.py') $archive (Join-Path $OutputDirectory 'verified')
    if ($LASTEXITCODE -ne 0) { throw 'Archive verification failed.' }
    Get-FileHash -LiteralPath $archive -Algorithm SHA256
} else {
    & ssh @sshArgs $remoteCommand
    if ($LASTEXITCODE -ne 0) { throw 'SSH or remote action failed; query the same run ID before any retry.' }
}
