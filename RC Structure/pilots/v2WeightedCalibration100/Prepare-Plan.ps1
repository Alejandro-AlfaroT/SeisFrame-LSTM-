param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$Python = 'python',
    [string[]]$DesignRoots = @('outputs/v2ResearchPilot100_CD_shard_01', 'outputs/v2ResearchPilot100_CD_shard_02', 'outputs/v2ResearchPilot100_CD_shard_03', 'outputs/v2ResearchPilot100_CD_shard_04'),
    [string]$CaseManifest = 'pilots/v2ResearchPilot100/screening_plan.json',
    [string]$RunName = 'wcal100_CD_Fc6_r1',
    [ValidateRange(1,4)][int]$Workers = 1,
    [ValidateRange(1,4)][int]$Devices = 4,
    [double]$MinimumScale = 0.125,
    [double]$InitialScale = 1.0,
    [double]$MaximumScale = 8.0,
    [ValidateRange(1,30)][int]$MaximumTrials = 7,
    [ValidateRange(60,86400)][int]$TimeoutSeconds = 3600,
    [switch]$RequireQualified,
    [switch]$AllowExcludedCases,
    [switch]$InventoryOnly
)
. (Join-Path $PSScriptRoot 'Common.ps1')
$context = Get-CalibrationContext $RepoRoot $Python
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunName must use letters, digits, underscores and hyphens.' }
Invoke-CalibrationPython $context @('pilots/v2WeightedCalibration100/verify_setup.py')
$prepared = "pilots/v2WeightedCalibration100/prepared/$RunName"
$plan = "$prepared/plan.json"
if ((-not $InventoryOnly) -and (Test-Path -LiteralPath (Join-Path $context.Rc $plan))) { throw "Frozen plan already exists: $plan. Use it unchanged or choose a new RunName." }
$rootArgs = @()
foreach ($root in $DesignRoots) { $rootArgs += @('--design-root', $root) }
$mode = if ($RequireQualified) { @() } else { @('--allow-open-m1-diagnostic') }
if (-not $RequireQualified) { Write-Host 'Declared intensity diagnostic: only an open M1 item may remain; no M1 assertion or training release is made.' }
$inspect = @('Data_Generation/Inspect_Calibration_Designs.py') + $rootArgs + @('--case-manifest', $CaseManifest, '--output', $prepared) + $mode
Invoke-CalibrationPython $context $inspect @(0, 2)
$inventory = Get-Content -LiteralPath (Join-Path $context.Rc "$prepared/design_inventory.json") -Raw | ConvertFrom-Json
if ($InventoryOnly) { return }
if ($inventory.eligible_count -lt $Devices) { throw 'Too few eligible cases for the requested number of devices. Review the inventory and reduce -Devices if appropriate.' }
if (($inventory.excluded_count -gt 0) -and (-not $AllowExcludedCases)) { throw "Review $prepared/design_inventory.json. If proceeding with the explicitly listed eligible subset, rerun with -AllowExcludedCases." }
$build = @('Data_Generation/Build_Weighted_Seismic_Calibration_Plan.py') + $rootArgs + $mode + @(
    '--case-manifest', "$prepared/eligible_cases.json",
    '--set-name', 'peer_mle_all', '--all-pairs-in-set', '--max-motion-points', '15000',
    '--assignment', 'one-per-case', '--seed', '20261006', '--shards', "$Devices", '--workers', "$Workers",
    '--minimum-scale', "$MinimumScale", '--initial-scale', "$InitialScale", '--maximum-scale', "$MaximumScale",
    '--max-trials', "$MaximumTrials", '--timeout-seconds', "$TimeoutSeconds",
    '--output-root', "outputs/$RunName", '--output', $plan)
Invoke-CalibrationPython $context $build
Write-Host "Copy the same prepared folder and the complete design roots to each device, preserving paths inside RC Structure. Plan: $plan"
