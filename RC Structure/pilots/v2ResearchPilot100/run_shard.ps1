# v2ResearchPilot100: one shard of the Latin hypercube design pilot on this machine.
# Usage, from anywhere:  .\run_shard.ps1 -Shard 1 -ProbeDate 2026-10-05 [-Workers 4] [-Python python]
# The initial case runs alone, then the rest of the shard on the workers. Output: RC Structure\outputs\v2ResearchPilot100_shard_NN
param([Parameter(Mandatory = $true)][int]$Shard, [Parameter(Mandatory = $true)][string]$ProbeDate,
      [int]$Workers = 4, [string]$Python = "python")
$rc = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$label = "shard_{0:D2}" -f $Shard
$manifest = Join-Path $PSScriptRoot "${label}_screening_plan.json"
$root = Join-Path $rc "outputs\v2ResearchPilot100_$label"
$initial = (Get-Content $manifest -Raw | ConvertFrom-Json).execution.initial_case_id
Set-Location $rc
& $Python -X utf8 -B "Design\Verify_Designs.py" --manifest $manifest --case-ids $initial --probe-assertions --probe-date $ProbeDate --stages design gravity_modal figures --output-root $root --workers 1
& $Python -X utf8 -B "Design\Verify_Designs.py" --manifest $manifest --remaining --probe-assertions --probe-date $ProbeDate --stages design gravity_modal figures --output-root $root --workers $Workers
