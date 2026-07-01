# RAG Q&A launcher.
# Double-click start.bat, or right-click this file -> "Run with PowerShell".
# Starts Ollama (if needed) + the Streamlit UI. Closing this window shuts everything
# down: child processes are tied to a Job Object that kills them when the window dies.

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Default install locations; edit if your conda env / Ollama live elsewhere.
$Python    = "$env:USERPROFILE\miniconda3\envs\rag\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }            # fall back to PATH
$OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $OllamaExe)) { $OllamaExe = "ollama" }      # fall back to PATH
$Port      = 8501
$Model     = "mistral"

Set-Location $ProjectDir
Write-Host "=== RAG Q&A launcher ===" -ForegroundColor Cyan

# Direct TCP probe to 127.0.0.1 — fast and reliable, unlike Invoke-WebRequest which
# can false-negative on a cold shell (proxy autodetect, localhost->IPv6 ambiguity).
function Test-Port($portNum) {
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $portNum)
        $ok = $c.Connected
        $c.Close()
        return $ok
    } catch { return $false }
}

# If the UI is already running, just open it and stop.
if (Test-Port $Port) {
    Write-Host "UI already running - opening browser." -ForegroundColor Green
    Start-Process "http://localhost:$Port"
    return
}

if (-not (Test-Path $Python)) {
    Write-Host "Python env not found at:`n  $Python" -ForegroundColor Red
    Read-Host "Press Enter to exit"; return
}

# --- Job Object: anything we Add-ToJob dies when this window/process closes ---
$JobReady = $false
try {
    Add-Type -Name Win32Job -Namespace RagLauncher -MemberDefinition @"
[DllImport("kernel32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr CreateJobObject(IntPtr a, string n);
[DllImport("kernel32.dll")] public static extern bool SetInformationJobObject(IntPtr j, int c, IntPtr i, uint l);
[DllImport("kernel32.dll")] public static extern bool AssignProcessToJobObject(IntPtr j, IntPtr p);
"@
    $script:Job = [RagLauncher.Win32Job]::CreateJobObject([IntPtr]::Zero, $null)
    # JOBOBJECT_EXTENDED_LIMIT_INFORMATION: LimitFlags at byte offset 16; set
    # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (0x2000). Struct is 144 bytes on x64.
    $size = 144
    $buf  = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
    for ($o = 0; $o -lt $size; $o++) { [System.Runtime.InteropServices.Marshal]::WriteByte($buf, $o, 0) }
    [System.Runtime.InteropServices.Marshal]::WriteInt32($buf, 16, 0x2000)
    [void][RagLauncher.Win32Job]::SetInformationJobObject($script:Job, 9, $buf, $size)
    [System.Runtime.InteropServices.Marshal]::FreeHGlobal($buf)
    $JobReady = $true
} catch {
    Write-Host "(auto-shutdown guard unavailable - closing the window still stops the UI)" -ForegroundColor DarkYellow
}

function Add-ToJob($procId) {
    if ($JobReady) {
        try {
            $h = (Get-Process -Id $procId).Handle
            [void][RagLauncher.Win32Job]::AssignProcessToJobObject($script:Job, $h)
        } catch {}
    }
}

# Ensure the Ollama server is up (start it if not; only ours is tied to the window).
if (-not (Test-Port 11434)) {
    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    if (Test-Path $OllamaExe) {
        $oll = Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden -PassThru
        Add-ToJob $oll.Id
        for ($i = 0; $i -lt 20; $i++) {
            if (Test-Port 11434) { break }
            Start-Sleep -Seconds 1
        }
    }
    if (-not (Test-Port 11434)) {
        Write-Host "Could not reach Ollama. Open the Ollama app, then retry." -ForegroundColor Red
        Read-Host "Press Enter to exit"; return
    }
}
Write-Host "Ollama is up." -ForegroundColor Green

# Ensure the model is present (one-time ~4GB pull if missing).
if ((& $OllamaExe list | Out-String) -notmatch $Model) {
    Write-Host "Model '$Model' not found - pulling (~4GB, one time)..." -ForegroundColor Yellow
    & $OllamaExe pull $Model
}

# Launch the UI in this console and wait on it. Close the window (or Ctrl+C) to stop.
Write-Host "Starting UI at http://localhost:$Port  (close this window to shut everything down)" -ForegroundColor Cyan
$env:PYTHONIOENCODING = "utf-8"
$uiLog = Join-Path $env:TEMP "rag_ui.log"
try {
    # Hidden + redirected (NOT -NoNewWindow, which needs a console the detached
    # launcher doesn't have and would prevent Streamlit from starting).
    $ui = Start-Process -FilePath $Python `
            -ArgumentList "-m","streamlit","run","src/app.py","--server.port",$Port,"--server.address","127.0.0.1","--server.headless","true" `
            -WorkingDirectory $ProjectDir -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $uiLog -RedirectStandardError "$uiLog.err"
    Add-ToJob $ui.Id
    # Wait for it to bind, then open the browser
    for ($i = 0; $i -lt 30; $i++) { if (Test-Port $Port) { break }; Start-Sleep -Seconds 1 }
    if (Test-Port $Port) {
        Write-Host "UI is up. Opening browser." -ForegroundColor Green
        Start-Process "http://localhost:$Port"
    }
    Wait-Process -Id $ui.Id
} finally {
    # Graceful path (Ctrl+C / UI exit): unload the model from VRAM. On a hard window
    # close the Job Object handles the kills instead.
    Write-Host "`nShutting down..." -ForegroundColor Cyan
    try { & $OllamaExe stop $Model 2>$null | Out-Null } catch {}
    try { if (-not $ui.HasExited) { Stop-Process -Id $ui.Id -Force -ErrorAction SilentlyContinue } } catch {}
}
