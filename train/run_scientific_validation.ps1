param(
    [string]$Bets = "output\\positive_strategy_portfolio_bets.csv",
    [string]$Summary = "output\\positive_strategy_portfolio_summary.csv",
    [int]$BootstrapIterations = 10000,
    [string]$OutputMd = "",
    [string]$OutputJson = ""
)

function Assert-LastExitCode([string]$StepName) {
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir

try {
    $command = @(
        "python",
        ".\\scientific_validation_report.py",
        "--bets", $Bets,
        "--summary", $Summary,
        "--bootstrap-iterations", $BootstrapIterations
    )

    if ($OutputMd) {
        $command += @("--output-md", $OutputMd)
    }
    if ($OutputJson) {
        $command += @("--output-json", $OutputJson)
    }

    & $command[0] $command[1..($command.Length - 1)]
    Assert-LastExitCode "scientific_validation_report.py"
}
finally {
    Pop-Location
}
