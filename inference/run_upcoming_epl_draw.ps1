param(
    [string]$DateFrom,
    [string]$DateTo,
    [string]$FixturesCsv = "",
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
        $FixturesCsv = ".\\inference\\output\\sportytrader_upcoming_epl_odds.csv"
        python .\inference\fetch_sportytrader_epl_odds.py `
            --date-from $DateFrom `
            --date-to $DateTo `
            --output $FixturesCsv
        Assert-LastExitCode "fetch_sportytrader_epl_odds.py"
    }

    $command = @(
        "python",
        ".\\inference\\predict_upcoming_epl_draw.py",
        "--fixtures-csv", $FixturesCsv,
        "--bankroll-eur", $BankrollEur
    )
    & $command[0] $command[1..($command.Length - 1)]
    Assert-LastExitCode "predict_upcoming_epl_draw.py"
}
finally {
    Pop-Location
}
