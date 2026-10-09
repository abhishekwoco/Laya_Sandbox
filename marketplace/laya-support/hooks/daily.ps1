# SessionStart (startup) hook: once per calendar day, on the first session this
# machine opens, nudge Claude to OFFER the daily triage digest. The stamp is
# written when the nudge fires, so a declined offer stays declined for the day.
# Always exits 0.
$ErrorActionPreference = 'Stop'
try {
    $dir = Join-Path $env:USERPROFILE '.laya-support'
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    $stamp = Join-Path $dir ('nudge-' + (Get-Date -Format 'yyyy-MM-dd') + '.txt')
    if (Test-Path $stamp) { exit 0 }
    New-Item -ItemType File -Path $stamp | Out-Null

    $pluginRoot = Split-Path -Parent $PSScriptRoot
    $recipePath = Join-Path (Join-Path $pluginRoot 'recipes') 'triage.md'
    $msg = "laya-support: today's support triage digest has not been run from this machine yet. Early in this session, at a natural moment, ask the user ONCE in one short line whether to run the daily triage now. If they say yes, read and follow the recipe file at $recipePath. If they decline or ignore the offer, drop the subject for the rest of the session."
    $out = @{ hookSpecificOutput = @{ hookEventName = 'SessionStart'; additionalContext = $msg } }
    $out | ConvertTo-Json -Depth 4 -Compress
    exit 0
} catch {
    exit 0
}
