Push-Location
try {
$ErrorActionPreference = "Stop"

$corePath = "C:\Users\Ivan\twistor-render-core"
$shieldPath = "C:\Users\Ivan\ai-vector-shield"

function Get-PythonExe($repo) {
    if (Test-Path "$repo\venv\Scripts\python.exe") {
        return "$repo\venv\Scripts\python.exe"
    } elseif (Test-Path "C:\Users\Ivan\AppData\Local\Programs\Python\Python313\python.exe") {
        return "C:\Users\Ivan\AppData\Local\Programs\Python\Python313\python.exe"
    } else {
        return "python"
    }
}

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host " [1/2] RUNNING TWISTOR-RENDER-CORE UNIT TEST SUITE" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

Set-Location $corePath
$pyCore = Get-PythonExe $corePath
$env:PYTHONPATH = $corePath
& $pyCore -m unittest discover -s "$corePath\tests" -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[FAILURE] Twistor-Render-Core unit tests failed!" -ForegroundColor Red
    exit 1
}
Write-Host "`n[SUCCESS] Twistor-Render-Core unit tests passed!" -ForegroundColor Green

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host " [2/2] RUNNING AI-VECTOR-SHIELD BENCHMARK & UNIT TESTS" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

Set-Location $shieldPath
$pyShield = Get-PythonExe $shieldPath
$env:PYTHONPATH = $shieldPath

& $pyShield "$shieldPath\topological_poison_shield.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[FAILURE] AI-Vector-Shield benchmark failed!" -ForegroundColor Red
    exit 1
}

& $pyShield -m unittest discover -s "$shieldPath\tests" -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[FAILURE] AI-Vector-Shield unit tests failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[SUCCESS] AI-Vector-Shield benchmark and unit tests passed!" -ForegroundColor Green
Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host " ALL REPOSITORIES VERIFIED & GREEN AT $(Get-Date) " -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Cyan
} finally {
    Pop-Location
}

