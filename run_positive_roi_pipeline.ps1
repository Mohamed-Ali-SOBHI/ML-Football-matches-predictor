param(
    [switch]$RefreshRawData,
    [string]$DataDir = "Data",
    [string]$DatasetPath = "train\\dataset_home.csv",
    [string]$ExportBetsPath = "train\\output\\best_roi_bets.csv",
    [int]$Trials = 40,
    [switch]$SkipTune
)

function Assert-LastExitCode([string]$StepName) {
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

try {
    if ($RefreshRawData) {
        python .\data_pipeline\scrapper.py
        Assert-LastExitCode "data_pipeline\\scrapper.py"
    }

    python .\data_pipeline\enrich_data.py --data-dir $DataDir
    Assert-LastExitCode "data_pipeline\\enrich_data.py"

    python .\train\make_dataset.py --data-dir $DataDir --output $DatasetPath
    Assert-LastExitCode "train\\make_dataset.py"

    if (-not $SkipTune) {
        python .\train\tune_model.py --data $DatasetPath --trials $Trials
        Assert-LastExitCode "train\\tune_model.py"
    }

    python .\train\train_model.py --data $DatasetPath --export-bets $ExportBetsPath
    Assert-LastExitCode "train\\train_model.py"
}
finally {
    Pop-Location
}
