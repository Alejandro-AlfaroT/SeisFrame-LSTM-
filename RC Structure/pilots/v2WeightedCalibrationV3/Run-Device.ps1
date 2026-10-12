param(
    [Parameter(Mandatory = $true)][ValidateRange(1,4)][int]$Device,
    [string]$RepoRoot = (Get-Location).Path,
    [string]$Python = 'python',
    [string]$Plan = 'pilots/v2WeightedCalibrationV3/prepared/wcalV3_r1/plan.json',
    [switch]$PreflightOnly,
    [switch]$FirstJobOnly,
    [switch]$Resume,
    [switch]$SummarizeOnly
)
. (Join-Path $PSScriptRoot 'Common.ps1')
$context = Get-CalibrationContext $RepoRoot $Python
if ((@($PreflightOnly, $FirstJobOnly, $SummarizeOnly) | Where-Object { $_ }).Count -gt 1) { throw 'Choose only one of PreflightOnly, FirstJobOnly or SummarizeOnly.' }
$command = @('pilots/v2WeightedCalibrationV3/school.py', 'run', '--plan', $Plan, '--device', "$Device")
if ($SummarizeOnly) { $command += '--summarize-only' }
if ($PreflightOnly) { $command += '--preflight-only' }
if ($FirstJobOnly) { $command += '--first-job-only' }
if ($Resume) { $command += '--resume' }
Invoke-CalibrationPython $context $command
