<#
.SYNOPSIS
    Start/stop a local, self-hosted Arize Phoenix instance for reviewing eval-run
    traces and attaching human feedback (annotations). Windows/PowerShell port of
    phoenix.sh -- same workflow (eval -> traces -> annotate), see README.md.

    Unlike deployment/langfuse/, this needs NO container runtime -- Phoenix runs
    as a single Python process (`phoenix serve`). Unlike phoenix.sh, `up` will
    auto-install arize-phoenix into this repo's .venv (via uv) if it isn't
    present yet, so there's no separate manual setup step on a fresh machine.

.USAGE
    .\phoenix.ps1 up               # install (if needed) + start the server in the background
    .\phoenix.ps1 status           # show whether it's running + the data dir
    .\phoenix.ps1 logs             # show recent server logs
    .\phoenix.ps1 logs -Follow     # follow server logs
    .\phoenix.ps1 down             # stop, keep local trace data (.data\)
    .\phoenix.ps1 down -Purge      # stop and delete .data\ (all local trace history)

.ENV OVERRIDES
    PHOENIX_BIN           path to the `phoenix` executable (default: resolved via PATH,
                          then this repo's .venv, then ~\.venvs\phoenix)
    PHOENIX_WORKING_DIR   where Phoenix stores its local SQLite trace data
                          (default: .\.data next to this script)
    PHOENIX_PORT          UI + OTLP/HTTP port (default: Phoenix's own default, 6006)
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet('up', 'status', 'logs', 'down', 'help')]
    [string]$Command = 'help',

    [switch]$Follow,
    [switch]$Purge
)

$ErrorActionPreference = 'Stop'

$ScriptPath = $MyInvocation.MyCommand.Path
$ScriptDir = Split-Path -Parent $ScriptPath
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir '..\..')
$RepoVenv  = Join-Path $RepoRoot '.venv'
$RunDir    = Join-Path $ScriptDir '.run'
$DataDir   = if ($env:PHOENIX_WORKING_DIR) { $env:PHOENIX_WORKING_DIR } else { Join-Path $ScriptDir '.data' }
$PidFile   = Join-Path $RunDir 'phoenix.pid'
$OutLog    = Join-Path $RunDir 'phoenix.out.log'
$ErrLog    = Join-Path $RunDir 'phoenix.err.log'

function Write-Log([string]$Message) {
    Write-Host "[phoenix] $Message"
}

function Die([string]$Message) {
    Write-Host "[phoenix] ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Get-RunningPid {
    if (-not (Test-Path $PidFile)) { return $null }
    $procId = Get-Content $PidFile -ErrorAction SilentlyContinue
    if (-not $procId) { return $null }
    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($proc) { return $procId }
    return $null
}

function Resolve-PhoenixBin {
    if ($env:PHOENIX_BIN -and (Test-Path $env:PHOENIX_BIN)) { return $env:PHOENIX_BIN }

    $onPath = Get-Command phoenix -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    $repoVenvBin = Join-Path $RepoVenv 'Scripts\phoenix.exe'
    if (Test-Path $repoVenvBin) { return $repoVenvBin }

    $userVenvBin = Join-Path $HOME '.venvs\phoenix\Scripts\phoenix.exe'
    if (Test-Path $userVenvBin) { return $userVenvBin }

    return $null
}

function Install-PhoenixIfMissing {
    $bin = Resolve-PhoenixBin
    if ($bin) { return $bin }

    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        $msg = "'phoenix' executable not found and 'uv' isn't on PATH to auto-install it. " + `
            "Install manually: uv venv `"$RepoVenv`"; uv pip install --python `"$RepoVenv\Scripts\python.exe`" arize-phoenix"
        Die $msg
    }
    if (-not (Test-Path $RepoVenv)) {
        $msg = "'phoenix' executable not found and no .venv at '$RepoVenv' to install into. " + `
            "Run 'uv venv' in the repo root first, or set `$env:PHOENIX_BIN to an existing install."
        Die $msg
    }

    Write-Log "arize-phoenix not found -- installing into $RepoVenv (one-time)..."
    & uv pip install --python (Join-Path $RepoVenv 'Scripts\python.exe') arize-phoenix
    if ($LASTEXITCODE -ne 0) { Die "uv pip install arize-phoenix failed (exit $LASTEXITCODE)." }

    $bin = Resolve-PhoenixBin
    if (-not $bin) { Die "Installed arize-phoenix but still can't find phoenix.exe under $RepoVenv\Scripts." }
    Write-Log "Installed. Using $bin"
    return $bin
}

function Cmd-Up {
    $procId = Get-RunningPid
    if ($procId) {
        Write-Log "Already running (pid $procId). Use '.\phoenix.ps1 status' or '.\phoenix.ps1 down' first."
        return
    }

    $phoenixBin = Install-PhoenixIfMissing

    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

    Write-Log "Starting Phoenix (data dir: $DataDir)..."
    $env:PHOENIX_WORKING_DIR = $DataDir

    $startArgs = @{
        FilePath               = $phoenixBin
        ArgumentList           = @('serve')
        RedirectStandardOutput = $OutLog
        RedirectStandardError  = $ErrLog
        WindowStyle            = 'Hidden'
        PassThru                = $true
    }
    $proc = Start-Process @startArgs
    $proc.Id | Out-File -FilePath $PidFile -Encoding ascii

    Start-Sleep -Seconds 1
    if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) {
        $port = if ($env:PHOENIX_PORT) { $env:PHOENIX_PORT } else { 6006 }
        Write-Log "Started (pid $($proc.Id)). UI: http://localhost:$port"
        Write-Log "Logs: .\phoenix.ps1 logs -Follow    Data: $DataDir"
    } else {
        Remove-Item -Force $PidFile -ErrorAction SilentlyContinue
        Die "Phoenix exited immediately -- check $ErrLog for details."
    }
}

function Cmd-Status {
    $procId = Get-RunningPid
    if ($procId) {
        Write-Log "Running (pid $procId). Data dir: $DataDir"
    } else {
        Write-Log "Not running."
    }
}

function Cmd-Logs {
    if (-not (Test-Path $OutLog)) { Die "No log file yet -- run '.\phoenix.ps1 up' first." }
    if ($Follow) {
        Write-Log "Following $OutLog (Ctrl+C to stop). Errors: $ErrLog"
        Get-Content -Path $OutLog -Wait -Tail 20
    } else {
        Get-Content -Path $OutLog -Tail 200
        if ((Test-Path $ErrLog) -and (Get-Item $ErrLog).Length -gt 0) {
            Write-Log "--- stderr ($ErrLog) ---"
            Get-Content -Path $ErrLog -Tail 50
        }
    }
}

function Cmd-Down {
    $procId = Get-RunningPid
    if ($procId) {
        Write-Log "Stopping Phoenix (pid $procId)..."
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    } else {
        Write-Log "Not running."
    }
    Remove-Item -Force $PidFile -ErrorAction SilentlyContinue

    if ($Purge) {
        Write-Log "Deleting local trace data at $DataDir..."
        Remove-Item -Recurse -Force $DataDir -ErrorAction SilentlyContinue
    }
}

function Show-Usage {
    Get-Content $ScriptPath | Select-Object -First 26 | Select-Object -Skip 1 | ForEach-Object {
        $_ -replace '^\.', ''
    }
}

switch ($Command) {
    'up'     { Cmd-Up }
    'status' { Cmd-Status }
    'logs'   { Cmd-Logs }
    'down'   { Cmd-Down }
    default  { Show-Usage }
}
