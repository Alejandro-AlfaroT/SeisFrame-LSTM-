param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$Python = 'python',
    [string]$Plan = 'pilots/v2WeightedCalibration100/prepared/wcal100_CD_Fc6_r1/plan.json',
    [string[]]$DeviceRoots = @(),
    [string]$Output = 'outputs/wcal100_CD_Fc6_r1_collection',
    [switch]$AllowPartial
)
. (Join-Path $PSScriptRoot 'Common.ps1')
$context = Get-CalibrationContext $RepoRoot $Python
$planFile = if ([IO.Path]::IsPathRooted($Plan)) { $Plan } else { Join-Path $context.Rc $Plan }
$data = Get-Content -LiteralPath $planFile -Raw | ConvertFrom-Json
if ($DeviceRoots.Count -eq 0) {
    $DeviceRoots = @(1..$data.shards | ForEach-Object { Join-Path $data.output_root ('device_{0:D2}' -f $_) })
}
$command = @('Data_Generation/Collect_Weighted_Seismic_Calibration.py', '--plan', $Plan, '--output', $Output)
foreach ($folder in $DeviceRoots) { $command += @('--device-root', $folder) }
if ($AllowPartial) { $command += '--allow-partial' }
Invoke-CalibrationPython $context $command
