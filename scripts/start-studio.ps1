param([int]$Port=8766,[int]$AliasPort=8765,[string]$Device='0',[string]$PoseDevice='auto',[switch]$NoBrowser,[switch]$Restart)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$python=Join-Path $project '.venv\Scripts\python.exe'
$ports=@($Port,$AliasPort) | Where-Object { $_ -gt 0 } | Select-Object -Unique
$pattern=($ports | ForEach-Object { [regex]::Escape("$_") }) -join '|'
$servers=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "-m selfwatch_v2\.studio --port ($pattern)(?: |$)" })
$ownedRoots=@($servers | Where-Object { $_.CommandLine.Contains($project) })
$ownedIds=@($ownedRoots.ProcessId)
$children=@($servers | Where-Object { $_.ParentProcessId -in $ownedIds })
$ownedIds+=@($children.ProcessId)
$listeners=@(Get-NetTCPConnection -LocalPort $ports -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -gt 0 })
if (@($listeners | Where-Object { $_.OwningProcess -notin $ownedIds }).Count) { throw 'A requested port belongs to another application. Choose another -Port / -AliasPort.' }
if ($ownedRoots.Count) {
    $canonical=@($ownedRoots | Where-Object { $_.CommandLine -match "--port $Port(?: |$)" })
    if (-not $Restart -and $canonical.Count) {
        if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$Port/" }
        Write-Output "SELFWATCH is already running at http://127.0.0.1:$Port/"
        return
    }
    if (-not $Restart) { throw 'An earlier SELFWATCH portal is running. Use -Restart to migrate the idle workspace.' }
    foreach ($server in $ownedRoots) {
        $null=$server.CommandLine -match '--port (\d+)';$previousPort=[int]$Matches[1]
        $state=Invoke-RestMethod -Uri "http://127.0.0.1:$previousPort/api/state" -TimeoutSec 3
        if ($state.status -in @('loading','running','paused','stopping')) { throw 'Stop the active analysis in the portal before restarting.' }
    }
    foreach ($server in @($children)+@($ownedRoots)) { Stop-Process -Id $server.ProcessId -ErrorAction SilentlyContinue }
}
$log=Join-Path $project 'runs\studio-server.log'
$errorLog=Join-Path $project 'runs\studio-server-error.log'
$arguments=@('-u','-m','selfwatch_v2.studio','--port',"$Port",'--alias-port',"$AliasPort",'--device',$Device,'--pose-device',$PoseDevice)
Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $errorLog -PassThru | Select-Object Id,ProcessName
$ready=$false
for ($attempt=0;$attempt -lt 60;$attempt++) {
    try { $response=Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/bootstrap" -TimeoutSec 1; if ($response.token) { $ready=$true; break } } catch { Start-Sleep -Milliseconds 150 }
}
if (-not $ready) { throw "Portal did not start. Inspect $errorLog" }
if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$Port/" }
