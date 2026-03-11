param(
    [string]$Data = "dataset_home.csv",
    [string]$ExportBets = "output\\best_roi_bets.csv"
)

function Assert-LastExitCode([string]$StepName) {
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir

try {
    if (-not (Test-Path $Data)) {
        python .\make_dataset.py
        Assert-LastExitCode "make_dataset.py"
    }

    python .\train_model.py `
        --data $Data `
        --export-bets $ExportBets
    Assert-LastExitCode "train_model.py"
}
finally {
    Pop-Location
}
