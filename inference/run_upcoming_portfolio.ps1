param(
    [string]$DateFrom,
    [string]$DateTo,
    [string]$FixturesCsv = "",
    [string]$Portfolio = "exploratory_multi_strategy_portfolio_2025",
    [double]$BankrollEur = 50.0,
    [switch]$RefreshRawData
)

function Assert-LastExitCode([string]$StepName) {
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Push-Location $repoRoot

try {
    if ($RefreshRawData) {
        python .\data_pipeline\scrapper.py --seasons 2025
        Assert-LastExitCode "data_pipeline\\scrapper.py"
        python .\data_pipeline\enrich_data.py --data-dir .\Data
        Assert-LastExitCode "data_pipeline\\enrich_data.py"
    }

    if (-not $FixturesCsv) {
        $FixturesCsv = ".\\inference\\output\\sportytrader_upcoming_portfolio_odds.csv"
        python .\inference\fetch_sportytrader_portfolio_odds.py `
            --date-from $DateFrom `
            --date-to $DateTo `
            --portfolio $Portfolio `
            --output $FixturesCsv
        Assert-LastExitCode "fetch_sportytrader_portfolio_odds.py"
    }

    $command = @(
        "python",
        ".\\inference\\predict_upcoming_portfolio.py",
        "--fixtures-csv", $FixturesCsv,
        "--portfolio", $Portfolio,
        "--bankroll-eur", $BankrollEur
    )
    & $command[0] $command[1..($command.Length - 1)]
    Assert-LastExitCode "predict_upcoming_portfolio.py"
}
finally {
    Pop-Location
}
