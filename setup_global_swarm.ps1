$ProfilePath = $PROFILE
if (!(Test-Path -Path $ProfilePath)) {
    New-Item -ItemType File -Path $ProfilePath -Force | Out-Null
}

$SwarmBlock = @"

# ==========================================
# 🌌 FRACTAL SWARM: GLOBAL OS INTEGRATION
# ==========================================
`$GLOBAL_SWARM_ROOT = "D:\Temp\Fractal_Swarm"
`$VIBE_BACKEND = "D:\Temp\vibe_backend"
`$KNOWLEDGE_BROKER = "D:\Temp\knowledge_broker"

# Initialize a new Swarm Workspace anywhere
function Initialize-SwarmWorkspace {
    param([string]`$TargetDir = ".")
    Write-Host "🚀 Terraforming `$TargetDir into a Swarm Workspace..." -ForegroundColor Cyan
    python "`$GLOBAL_SWARM_ROOT\tools\orchestrator\swarm_init.py" `$TargetDir
    # Autonomously inject the master protocol into the new workspace
    Copy-Item -Path "`$GLOBAL_SWARM_ROOT\.antigravityrules" -Destination "`$TargetDir\.antigravityrules" -Force
    Write-Host "✅ .antigravityrules injected. Agent is now in God-Mode." -ForegroundColor Green
}

# Interrogate the Oracle from any directory
function Query-SovereignOracle {
    param([Parameter(Mandatory=`$true)][string]`$Query)
    # Using myBrAIn venv to run the query script
    & "D:\Temp\myBrAIn\venv\Scripts\python.exe" "`$KNOWLEDGE_BROKER\query_brain.py" "`$Query"
}

# Boot the background Oracle daemon
function Start-SwarmOracle {
    Write-Host "🧠 Booting Sovereign Oracle Daemon on port 8000..." -ForegroundColor Magenta
    Start-Process -WindowStyle Hidden -FilePath "C:\Users\Admin\AppData\Local\Programs\Jan\uv.exe" -ArgumentList "run uvicorn main:app --host 127.0.0.1 --port 8000" -WorkingDirectory "`$VIBE_BACKEND"
    
    Write-Host "🌙 Waking Nightcrawler continuous ingestion daemon..." -ForegroundColor DarkMagenta
    Start-Process -WindowStyle Hidden -FilePath "C:\Users\Admin\AppData\Local\Programs\Jan\uv.exe" -ArgumentList "run python nightcrawler.py" -WorkingDirectory "`$VIBE_BACKEND"
}

# Aliases for lightning-fast execution
Set-Alias swarm-init Initialize-SwarmWorkspace
Set-Alias swarm-query Query-SovereignOracle
Set-Alias swarm-boot Start-SwarmOracle
"@

# Append strictly if not already present
$CurrentProfile = ""
if (Test-Path $ProfilePath) {
    $CurrentProfile = Get-Content $ProfilePath -Raw
    if ($null -eq $CurrentProfile) { $CurrentProfile = "" }
}

if (-not $CurrentProfile.Contains("FRACTAL SWARM: GLOBAL OS INTEGRATION")) {
    Add-Content -Path $ProfilePath -Value $SwarmBlock
    Write-Host "✅ Swarm aliases injected into PowerShell profile: $ProfilePath" -ForegroundColor Green
} else {
    Write-Host "⚠️ Swarm aliases already exist in PowerShell profile." -ForegroundColor Yellow
}

# Register the Scheduled Task to run silently at logon
$TaskName = "FractalSwarmDaemon"
$TaskExists = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if (-not $TaskExists) {
    Write-Host "⚙️ Registering Windows Scheduled Task '$TaskName'..." -ForegroundColor Cyan
    $Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-WindowStyle Hidden -Command `". `$PROFILE; Start-SwarmOracle`""
    $Trigger = New-ScheduledTaskTrigger -AtLogon
    $Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
    $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description "Autonomously boots the Fractal Swarm Oracle and Nightcrawler daemons at logon." | Out-Null
    Write-Host "✅ Task Scheduler configured successfully. The Swarm is now natively embedded in the OS startup sequence." -ForegroundColor Green
} else {
    Write-Host "⚠️ Scheduled Task '$TaskName' already exists." -ForegroundColor Yellow
}
