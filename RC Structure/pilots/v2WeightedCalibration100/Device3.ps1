param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$Python = 'python',
    [string]$Plan = 'pilots/v2WeightedCalibration100/prepared/wcal100_CD_Fc6_r1/plan.json',
    [switch]$PreflightOnly, [switch]$FirstJobOnly, [switch]$Resume, [switch]$SummarizeOnly
)
& (Join-Path $PSScriptRoot 'Run-Device.ps1') -Device 3 @PSBoundParameters