# UserPromptSubmit hook: expands the support team's one-word commands into full
# recipes so members never write long prompts. Reads the hook event JSON from
# stdin; when the first word of the prompt is a known trigger, emits the matching
# recipe file as additionalContext. Always exits 0 - a hook failure must never
# break the member's prompt.
$ErrorActionPreference = 'Stop'
try {
    $raw = [Console]::In.ReadToEnd()
    if ([string]::IsNullOrWhiteSpace($raw)) { exit 0 }
    $evt = $raw | ConvertFrom-Json
    $prompt = [string]$evt.prompt
    if ([string]::IsNullOrWhiteSpace($prompt)) { exit 0 }

    $firstWord = ($prompt.Trim() -replace '^[/!]', '').Split(" `t`r`n")[0].ToLowerInvariant()
    $map = @{
        'triage' = 'triage.md'
        'chase'  = 'chase.md'
        'labels' = 'labels.md'
    }
    if (-not $map.ContainsKey($firstWord)) { exit 0 }

    $pluginRoot = Split-Path -Parent $PSScriptRoot
    $recipePath = Join-Path (Join-Path $pluginRoot 'recipes') $map[$firstWord]
    if (-not (Test-Path $recipePath)) { exit 0 }
    $recipe = Get-Content -Raw -Encoding UTF8 $recipePath

    $context = "The user typed the laya-support shortcut '$firstWord'. Follow the recipe below now; the rest of their message (if any) carries options such as a channel name. If their full message is clearly about something unrelated, ignore the recipe and answer normally.`n`n" + $recipe
    $out = @{ hookSpecificOutput = @{ hookEventName = 'UserPromptSubmit'; additionalContext = $context } }
    $out | ConvertTo-Json -Depth 4 -Compress
    exit 0
} catch {
    exit 0
}
