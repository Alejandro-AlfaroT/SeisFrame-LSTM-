param([string]$RepoRoot = (Get-Location).Path, [string]$Python = 'python')
. (Join-Path $PSScriptRoot 'Common.ps1')
$context = Get-CalibrationContext $RepoRoot $Python
Invoke-CalibrationPython $context @('pilots/v2WeightedCalibrationV3/verify_setup.py')
