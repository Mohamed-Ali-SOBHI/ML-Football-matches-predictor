param(
    [string]$Data = "dataset_home.csv",
    [string]$ExportBets = "output\\positive_epl_draw_bets.csv"
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

    python .\filtered_strategy_search.py `
        --data $Data `
        --train-league EPL `
        --bet-league EPL `
        --outcome draw `
        --odds-min 2.0 `
        --odds-max 10.0 `
        --market-favorite-mode nonfavorite `
        --min-val-bets 60 `
        --trials 20 `
        --export-bets $ExportBets
    Assert-LastExitCode "filtered_strategy_search.py"
}
finally {
    Pop-Location
}
