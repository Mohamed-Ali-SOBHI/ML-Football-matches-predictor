param(
    [string]$DateFrom = "",
    [string]$DateTo = "",
    [double]$BankrollEur = 50.0,
    [switch]$RefreshRawData
)

function Get-WeekendWindow {
    $today = (Get-Date).Date
    $day = [int]$today.DayOfWeek

    if ($day -in 5, 6, 0, 1) {
        $daysBackToFriday = ($day - 5 + 7) % 7
        $start = $today.AddDays(-$daysBackToFriday)
        $end = $start.AddDays(3)
    }
    else {
        $daysToFriday = (5 - $day + 7) % 7
        $start = $today.AddDays($daysToFriday)
        $end = $start.AddDays(3)
    }

    return @{
        DateFrom = $start.ToString("yyyy-MM-dd")
        DateTo = $end.ToString("yyyy-MM-dd")
    }
}

if (-not $DateFrom -or -not $DateTo) {
    $window = Get-WeekendWindow
    if (-not $DateFrom) { $DateFrom = $window.DateFrom }
    if (-not $DateTo) { $DateTo = $window.DateTo }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $scriptDir "run_upcoming_epl_draw.ps1"

$argsList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $runner,
    "-DateFrom", $DateFrom,
    "-DateTo", $DateTo,
    "-BankrollEur", $BankrollEur
)
if ($RefreshRawData) {
    $argsList += "-RefreshRawData"
}

powershell @argsList
