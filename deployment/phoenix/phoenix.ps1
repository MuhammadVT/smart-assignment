<#
.SYNOPSIS
    Start/stop a local, self-hosted Arize Phoenix instance for reviewing eval-run
    traces and attaching human feedback (annotations). Windows/PowerShell port of
    phoenix.sh -- same workflow (eval -> traces -> annotate), see README.md.

    Unlike deployment/langfuse/, this needs NO container runtime -- Phoenix runs
    as a single Python process (`phoenix serve`). Unlike phoenix.sh, `up` will
    auto-install arize-phoenix (via uv) if it isn't present yet, so there's no
    separate manual setup step on a fresh machine.

    It installs into a DEDICATED Phoenix venv (default ~\.venvs\phoenix), NOT the
    project's .venv, and pins that venv to Python 3.13 by default. This is on
    purpose: Phoenix needs `sqlean-py`, which has no Windows wheel for Python
    3.14, so installing Phoenix into a 3.14 project venv fails at `import
    sqlean`. Phoenix is an external backend that talks to the agent over OTLP
    HTTP -- its interpreter is independent of the project's, so running it on
    3.13 keeps the project on 3.14 untouched. uv downloads 3.13 automatically.

.USAGE
    .\phoenix.ps1 up               # install (if needed) + start the server in the background
    .\phoenix.ps1 status           # show whether it's running + the data dir
    .\phoenix.ps1 logs             # show recent server logs
    .\phoenix.ps1 logs -Follow     # follow server logs
    .\phoenix.ps1 down             # stop, keep local trace data (.data\)
    .\phoenix.ps1 down -Purge      # stop and delete .data\ (all local trace history)

.ENV OVERRIDES
    PHOENIX_BIN           path to the `phoenix` executable (default: resolved via PATH,
                          then the dedicated Phoenix venv, then this repo's .venv)
    PHOENIX_VENV          dedicated venv dir to install/run Phoenix from
                          (default: ~\.venvs\phoenix)
    PHOENIX_PYTHON        Python version for the dedicated venv when auto-installing
                          (default: 3.13 -- has prebuilt sqlean-py Windows wheels)
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
$PhoenixVenv   = if ($env:PHOENIX_VENV)   { $env:PHOENIX_VENV }   else { Join-Path $HOME '.venvs\phoenix' }
$PhoenixPython = if ($env:PHOENIX_PYTHON) { $env:PHOENIX_PYTHON } else { '3.13' }
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

    # Dedicated Phoenix venv first: it's pinned to a Python that has prebuilt
    # sqlean-py wheels, so it's preferred over the project .venv, which may be on
    # a newer Python (e.g. 3.14) that has none on Windows and would fail to run.
    $dedicatedBin = Join-Path $PhoenixVenv 'Scripts\phoenix.exe'
    if (Test-Path $dedicatedBin) { return $dedicatedBin }

    $onPath = Get-Command phoenix -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    # Last resort: a phoenix already installed into the project .venv (fine when
    # that venv's Python has sqlean-py wheels; auto-install never puts it here).
    $repoVenvBin = Join-Path $RepoVenv 'Scripts\phoenix.exe'
    if (Test-Path $repoVenvBin) { return $repoVenvBin }

    return $null
}

function Install-PhoenixIfMissing {
    # Honor an explicit override, an already-installed dedicated venv, or a
    # phoenix on PATH. Deliberately do NOT accept a project-.venv install here:
    # that venv may be on a Python (e.g. 3.14) with no sqlean-py Windows wheel,
    # so a phoenix.exe sitting there can still be broken -- we'd rather build the
    # dedicated venv than short-circuit to it.
    if ($env:PHOENIX_BIN -and (Test-Path $env:PHOENIX_BIN)) { return $env:PHOENIX_BIN }
    $dedicatedBin = Join-Path $PhoenixVenv 'Scripts\phoenix.exe'
    if (Test-Path $dedicatedBin) { return $dedicatedBin }
    $onPath = Get-Command phoenix -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        $py = Join-Path $PhoenixVenv 'Scripts\python.exe'
        $msg = "'phoenix' executable not found and 'uv' isn't on PATH to auto-install it. " + `
            "Install manually: uv venv --python $PhoenixPython `"$PhoenixVenv`"; uv pip install --python `"$py`" arize-phoenix"
        Die $msg
    }

    # Create a dedicated venv pinned to a Python that has sqlean-py wheels (uv
    # downloads it if absent), keeping the project's interpreter untouched.
    Write-Log "arize-phoenix not found -- creating a dedicated Phoenix venv at $PhoenixVenv (Python $PhoenixPython) and installing (one-time)..."
    & uv venv --python $PhoenixPython $PhoenixVenv
    if ($LASTEXITCODE -ne 0) { Die "uv venv --python $PhoenixPython '$PhoenixVenv' failed (exit $LASTEXITCODE)." }
    & uv pip install --python (Join-Path $PhoenixVenv 'Scripts\python.exe') arize-phoenix
    if ($LASTEXITCODE -ne 0) { Die "uv pip install arize-phoenix failed (exit $LASTEXITCODE)." }

    if (-not (Test-Path $dedicatedBin)) { Die "Installed arize-phoenix but phoenix.exe not found under $PhoenixVenv\Scripts." }
    Write-Log "Installed. Using $dedicatedBin"
    return $dedicatedBin
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
    # Print the leading <# ... #> comment block (minus its delimiters).
    $inBlock = $false
    foreach ($line in Get-Content $ScriptPath) {
        if ($line -match '^\s*<#') { $inBlock = $true; continue }
        if ($line -match '^\s*#>') { break }
        if ($inBlock) { $line -replace '^\.', '' }
    }
}

switch ($Command) {
    'up'     { Cmd-Up }
    'status' { Cmd-Status }
    'logs'   { Cmd-Logs }
    'down'   { Cmd-Down }
    default  { Show-Usage }
}
