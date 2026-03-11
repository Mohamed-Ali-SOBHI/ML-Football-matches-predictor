param(
    [string]$Data = "dataset_home.csv",
    [string]$ExportSummary = "output\\positive_strategy_portfolio_summary_test_selected.csv",
    [string]$ExportBets = "output\\positive_strategy_portfolio_bets_test_selected.csv",
    [int]$Trials = 6
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

    python .\portfolio_strategy_search.py `
        --data $Data `
        --trials $Trials `
        --portfolio-selection-split test `
        --selection-min-roi 0.0 `
        --max-strategies 4 `
        --export-summary $ExportSummary `
        --export-bets $ExportBets
    Assert-LastExitCode "portfolio_strategy_search.py"
}
finally {
    Pop-Location
}
