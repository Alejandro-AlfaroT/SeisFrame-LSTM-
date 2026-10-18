param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$Python = 'python',
    [string]$Plan = 'pilots/v2WeightedCalibrationV3/prepared/wcalV3_r1/plan.json',
    [string[]]$DeviceRoots = @(),
    [string]$Output = '',
    [switch]$AllowPartial
)
. (Join-Path $PSScriptRoot 'Common.ps1')
$context = Get-CalibrationContext $RepoRoot $Python
$command = @('pilots/v2WeightedCalibrationV3/school.py', 'collect', '--plan', $Plan)
if ($Output) { $command += @('--output', $Output) }
foreach ($folder in $DeviceRoots) { $command += @('--device-root', $folder) }
if ($AllowPartial) { $command += '--allow-partial' }
Invoke-CalibrationPython $context $command
