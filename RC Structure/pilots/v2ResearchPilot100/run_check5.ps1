# v2ResearchCheck5: five structures end to end (design, then one full-length response-history run each).
# Usage, from anywhere:  .\run_check5.ps1 -ProbeDate 2026-10-05 [-Workers 4] [-Python python]
# Designs: RC Structure\outputs\v2ResearchCheck5_designs   Runs: RC Structure\outputs\v2ResearchCheck5_batch
# Running it again continues: finished designs and finished runs are skipped.
param([Parameter(Mandatory = $true)][string]$ProbeDate, [int]$Workers = 4, [string]$Python = "python")
$rc = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$manifest = Join-Path $PSScriptRoot "check5_screening_plan.json"
$designs = Join-Path $rc "outputs\v2ResearchCheck5_designs"
$batch = Join-Path $rc "outputs\v2ResearchCheck5_batch"
$initial = (Get-Content $manifest -Raw | ConvertFrom-Json).execution.initial_case_id
Set-Location $rc
& $Python -X utf8 -B "Design\Verify_Designs.py" --manifest $manifest --case-ids $initial --probe-assertions --probe-date $ProbeDate --stages design gravity_modal --output-root $designs --workers 1
& $Python -X utf8 -B "Design\Verify_Designs.py" --manifest $manifest --remaining --probe-assertions --probe-date $ProbeDate --stages design gravity_modal --output-root $designs --workers $Workers
& $Python -X utf8 -B "Data_Generation\Run_Research_Batch.py" --manifest $manifest --design-roots $designs --output-root $batch --workers $Workers
