param(
    [Parameter(Mandatory = $true)][ValidateRange(1,4)][int]$Device,
    [string]$RepoRoot = (Get-Location).Path,
    [string]$Python = 'python',
    [string]$Plan = 'pilots/v2WeightedCalibration100/prepared/wcal100_CD_Fc6_r1/plan.json',
    [switch]$PreflightOnly,
    [switch]$FirstJobOnly,
    [switch]$Resume,
    [switch]$SummarizeOnly
)
. (Join-Path $PSScriptRoot 'Common.ps1')
$context = Get-CalibrationContext $RepoRoot $Python
if ((@($PreflightOnly, $FirstJobOnly, $SummarizeOnly) | Where-Object { $_ }).Count -gt 1) { throw 'Choose only one of PreflightOnly, FirstJobOnly or SummarizeOnly.' }
$command = @('Data_Generation/Run_Weighted_Seismic_Calibration.py', '--plan', $Plan, '--shard', "$Device")
if ($SummarizeOnly) { Invoke-CalibrationPython $context ($command + @('--summarize-only')); return }
Invoke-CalibrationPython $context @('pilots/v2WeightedCalibration100/verify_setup.py')
Invoke-CalibrationPython $context ($command + @('--preflight-only'))
if ($PreflightOnly) { return }
if ($FirstJobOnly) {
    $extra = @('--first-job-only')
    if ($Resume) { $extra += '--resume' }
    Invoke-CalibrationPython $context ($command + $extra)
    return
}
if (-not $Resume) {
    Write-Host 'First assigned full-record search runs before the remainder. A failed or unmet canary target stops this launcher for review.'
    Invoke-CalibrationPython $context ($command + @('--first-job-only'))
}
Invoke-CalibrationPython $context ($command + @('--resume'))
