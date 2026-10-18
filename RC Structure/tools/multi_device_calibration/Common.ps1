$ErrorActionPreference = 'Stop'
function Get-CalibrationContext([string]$RepoRoot, [string]$Python) {
    $candidate = [IO.Path]::GetFullPath($RepoRoot)
    $rc = if (Test-Path -LiteralPath (Join-Path $candidate 'Data_Generation\Run_Weighted_Seismic_Calibration.py')) { $candidate } else { Join-Path $candidate 'RC Structure' }
    if (-not (Test-Path -LiteralPath (Join-Path $rc 'Data_Generation\Run_Weighted_Seismic_Calibration.py'))) { throw 'Use the repository root or RC Structure directory with -RepoRoot.' }
    return @{ Rc = $rc; Python = (Get-Command $Python -ErrorAction Stop).Source }
}
function Invoke-CalibrationPython($Context, [string[]]$Arguments, [int[]]$AllowedExitCodes = @(0)) {
    Push-Location $Context.Rc
    try {
        & $Context.Python -X utf8 -B @Arguments
        $code = $LASTEXITCODE
        if ($AllowedExitCodes -notcontains $code) { throw "Calibration command stopped with exit code $code. Preserve the saved logs and results; do not delete or relabel them." }
        return
    } finally { Pop-Location }
}
