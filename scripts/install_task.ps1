#Requires -Version 5.1
<#
.SYNOPSIS
    Installs (or removes) the "LayaMCP" Windows Scheduled Task that runs the
    laya-mcp server at boot, plus the matching inbound firewall rule.

.DESCRIPTION
    Registers a scheduled task named "LayaMCP" that:
      - Triggers "at startup" (boot), independent of any interactive logon.
      - Runs "<ProjectRoot>\venv\Scripts\laya-mcp.exe" (falling back to
        "<ProjectRoot>\venv\Scripts\python.exe -m laya_mcp.app" if the console
        script isn't installed) with working directory <ProjectRoot>.
      - Restarts automatically on failure: 3 attempts, 1 minute apart.
      - Has no execution time limit (the default scheduled-task settings cap
        a run at a few days; a long-running server must be exempt from that).
      - Does NOT request "Run with highest privileges" (-RunLevel Limited) --
        the server binds one unprivileged TCP port and writes only under its
        own project directory, so it should run at normal user rights.

    It also opens an inbound firewall rule "LayaMCP (TCP <Port>)" scoped to
    -RemoteAddress. The default, 'LocalSubnet', only allows this host's own
    subnet. Pass the real WoCo intranet range (e.g. 10.10.0.0/16) once you've
    confirmed it, to let other intranet machines reach the server.

    RUN THIS SCRIPT AS ADMINISTRATOR. It only registers OS-level configuration
    (a scheduled task + a firewall rule) -- it does not install the project,
    touch the venv, or modify laya-mcp's own files.

.PARAMETER Port
    TCP port laya-mcp listens on. Must match LAYA_MCP_PORT in .env. Default 8765.

.PARAMETER RemoteAddress
    Firewall rule scope: who may reach this port. Default 'LocalSubnet' (this
    host's own subnet only). Pass a CIDR (e.g. 10.10.0.0/16) or a
    comma-separated list of ranges/addresses to open it to the wider WoCo
    intranet.

.PARAMETER ProjectRoot
    Root of the laya-mcp checkout. Default D:\Laya_Sandbox.

.PARAMETER UserAccount
    Windows account the task runs as, as DOMAIN\User or COMPUTER\User.
    Default: the account running this script. This MUST be the "Woco"
    account (or whichever account's profile holds the model cache) --
    HF_HOME resolves to C:\Users\<that account>\.cache\huggingface, so
    running as a different account (or as SYSTEM) means the checkpoints
    look missing: with HF_HUB_OFFLINE=1 the server then fails to start
    instead of silently re-downloading.

    You will be prompted for that account's password via Get-Credential, so
    Task Scheduler can start the task "whether or not a user is logged on"
    (LogonType Password). See the S4U trade-off below for a passwordless
    alternative.

.PARAMETER UseS4U
    Register the task with LogonType S4U instead of a stored password: no
    password is saved, but on some Windows builds an S4U task only starts
    once some user has interactively logged on to the machine at least once
    since boot -- which can defeat "starts at boot with nobody logged on".
    It can also behave differently around loading the user's profile (where
    HF_HOME resolves from). Verify actual boot behavior on this machine
    before relying on it; Password logon (the default) is the safer choice
    for an unattended host and is what this script uses unless you pass
    -UseS4U.

.PARAMETER Uninstall
    Removes the "LayaMCP" scheduled task and its firewall rule, then exits.
    Safe to run even if neither exists.

.EXAMPLE
    .\scripts\install_task.ps1
    Installs with defaults: port 8765, LocalSubnet firewall scope, project
    root D:\Laya_Sandbox. Prompts for the running account's password.

.EXAMPLE
    .\scripts\install_task.ps1 -RemoteAddress 10.10.0.0/16
    Installs and opens the firewall to the wider WoCo intranet range instead
    of just this host's local subnet.

.EXAMPLE
    .\scripts\install_task.ps1 -Uninstall
    Removes the "LayaMCP" scheduled task and its firewall rule.

.NOTES
    Idempotent: re-running with the same (or different) parameters
    unregisters and re-registers the task/rule rather than erroring because
    they already exist, so it's safe to re-run after editing .env or moving
    the project.

    This script only prepares configuration for the user to apply -- per
    project policy it is not executed as part of authoring it. Review it,
    then run it yourself from an elevated PowerShell prompt.
#>
[CmdletBinding()]
param(
    [int]$Port = 8765,
    [string]$RemoteAddress = 'LocalSubnet',
    [string]$ProjectRoot = 'D:\Laya_Sandbox',
    [string]$UserAccount = "$env:COMPUTERNAME\$env:USERNAME",
    [switch]$UseS4U,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'
$TaskName = 'LayaMCP'
$FirewallRuleName = "LayaMCP (TCP $Port)"

function Assert-Admin {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Run this script from an elevated (Administrator) PowerShell prompt."
    }
}

function Remove-LayaTaskAndRule {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Write-Host "Removing scheduled task '$TaskName'..."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
    else {
        Write-Host "Scheduled task '$TaskName' not present; nothing to remove."
    }

    $rule = Get-NetFirewallRule -DisplayName $FirewallRuleName -ErrorAction SilentlyContinue
    if ($rule) {
        Write-Host "Removing firewall rule '$FirewallRuleName'..."
        $rule | Remove-NetFirewallRule
    }
    else {
        # Catch a rule created earlier with a different -Port value.
        $legacy = Get-NetFirewallRule -DisplayName 'LayaMCP (TCP *)' -ErrorAction SilentlyContinue
        if ($legacy) {
            Write-Host "Removing firewall rule(s) matching 'LayaMCP (TCP *)'..."
            $legacy | Remove-NetFirewallRule
        }
        else {
            Write-Host "No LayaMCP firewall rule present; nothing to remove."
        }
    }
}

Assert-Admin

if ($Uninstall) {
    Remove-LayaTaskAndRule
    Write-Host "LayaMCP scheduled task and firewall rule removed."
    return
}

if (-not (Test-Path $ProjectRoot)) {
    throw "ProjectRoot '$ProjectRoot' does not exist."
}

$exePath = Join-Path $ProjectRoot 'venv\Scripts\laya-mcp.exe'
$pythonPath = Join-Path $ProjectRoot 'venv\Scripts\python.exe'

if (Test-Path $exePath) {
    $action = New-ScheduledTaskAction -Execute $exePath -WorkingDirectory $ProjectRoot
    Write-Host "Task will run: $exePath"
}
elseif (Test-Path $pythonPath) {
    $action = New-ScheduledTaskAction -Execute $pythonPath -Argument '-m laya_mcp.app' -WorkingDirectory $ProjectRoot
    Write-Host "Task will run: $pythonPath -m laya_mcp.app"
}
else {
    throw "Neither $exePath nor $pythonPath was found. Install the project first, e.g.:`n" +
    "  $ProjectRoot\venv\Scripts\python.exe -m pip install -e `"$ProjectRoot`""
}

# Start at boot, independent of any interactive logon.
$trigger = New-ScheduledTaskTrigger -AtStartup

# Restart on failure: 3 attempts, 1 minute apart. No execution time limit.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Task '$TaskName' already exists; unregistering before re-creating (idempotent re-install)."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

if ($UseS4U) {
    # No stored password -- see the -UseS4U trade-off in the help text above.
    $principal = New-ScheduledTaskPrincipal -UserId $UserAccount -LogonType S4U -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal `
        -Description 'laya-mcp MCP server (S4U logon, no stored password).' | Out-Null
}
else {
    Write-Host "Task will run as '$UserAccount'."
    $cred = Get-Credential -UserName $UserAccount -Message "Password for $UserAccount (used to run LayaMCP at boot, whether or not anyone is logged on)"
    # -RunLevel Limited: highest privileges are deliberately NOT requested.
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -User $cred.UserName -Password $cred.GetNetworkCredential().Password `
        -RunLevel Limited -Description 'laya-mcp MCP server (starts at boot, restarts on failure).' | Out-Null
}

Write-Host "Scheduled task '$TaskName' registered."

# Firewall: inbound TCP $Port, scoped to $RemoteAddress (default LocalSubnet).
$existingRule = Get-NetFirewallRule -DisplayName $FirewallRuleName -ErrorAction SilentlyContinue
if ($existingRule) {
    Write-Host "Firewall rule '$FirewallRuleName' already exists; removing before re-creating."
    $existingRule | Remove-NetFirewallRule
}

New-NetFirewallRule -DisplayName $FirewallRuleName `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort $Port `
    -RemoteAddress $RemoteAddress `
    -Action Allow `
    -Profile Any | Out-Null

Write-Host "Firewall rule '$FirewallRuleName' created, scoped to -RemoteAddress $RemoteAddress."
Write-Host ""
Write-Host "Done. Verify with:"
Write-Host "  Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "  Get-NetFirewallRule -DisplayName '$FirewallRuleName' | Get-NetFirewallAddressFilter"
Write-Host ""
Write-Host "Start it immediately without rebooting:"
Write-Host "  Start-ScheduledTask -TaskName $TaskName"
Write-Host ""
Write-Host "Uninstall (removes the task and firewall rule):"
Write-Host "  .\scripts\install_task.ps1 -Uninstall"
