param([string]$ReviewDirectory = 'runs\combined-cognitive-v2-000398', [int]$Port = 8766, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$review = [System.IO.Path]::GetFullPath((Join-Path $project $ReviewDirectory))
$hasRecording = Test-Path -LiteralPath (Join-Path $review 'data.json')
$hasStudy = (Test-Path -LiteralPath (Join-Path $review 'study.json')) -and (Test-Path -LiteralPath (Join-Path $review 'index.html'))
if (-not ($hasRecording -or $hasStudy)) { throw "Review data missing: $review" }
$python = Join-Path $project '.venv\Scripts\python.exe'
# Replace only a verified SELFWATCH review server on this port. Otherwise a
# failed second server would silently leave the browser on the previous run.
$servers = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match "-m http\.server $Port(?: |$)" })
$ownedRoots = @($servers | Where-Object {
    $_.CommandLine.Contains($project) -and $_.CommandLine.Contains('--directory')
})
$ownedIds = @($ownedRoots.ProcessId)
# A Windows venv launcher creates a base-Python child. The child can use a
# relative directory, so verify it through its already-verified parent.
$ownedChildren = @($servers | Where-Object { $_.ParentProcessId -in $ownedIds })
$ownedIds += @($ownedChildren.ProcessId)
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.OwningProcess -gt 0 })
if (@($listeners | Where-Object { $_.OwningProcess -notin $ownedIds }).Count) {
    throw "Port $Port belongs to another application. Choose a different -Port."
}
foreach ($server in @($ownedChildren)+@($ownedRoots)) {
    Stop-Process -Id $server.ProcessId -ErrorAction SilentlyContinue
}
$serverArgs = @('-m','http.server', "$Port", '--bind','127.0.0.1','--directory', ('"' + $review + '"'))
Start-Process -FilePath $python -ArgumentList $serverArgs -WindowStyle Hidden -PassThru | Select-Object Id,ProcessName
if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$Port/" }
