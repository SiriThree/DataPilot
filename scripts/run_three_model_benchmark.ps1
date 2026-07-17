param(
    [int]$Limit = 0,
    [switch]$Sequential
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$configs = @(
    @{ Name = "qwen"; Config = "configs/react_baseline.qwen.example.yaml" },
    @{ Name = "kimi"; Config = "configs/react_baseline.kimi.example.yaml" },
    @{ Name = "mimo"; Config = "configs/react_baseline.mimo.example.yaml" }
)

function New-BenchmarkArgs {
    param([string]$ConfigPath)
    $args = @("-m", "uv", "run", "dabench", "run-benchmark", "--config", $ConfigPath)
    if ($Limit -gt 0) {
        $args += @("--limit", "$Limit")
    }
    return $args
}

if ($Sequential) {
    foreach ($item in $configs) {
        Write-Host "== Running $($item.Name): $($item.Config) =="
        Push-Location $repoRoot
        try {
            & python @(New-BenchmarkArgs -ConfigPath $item.Config)
        }
        finally {
            Pop-Location
        }
    }
    exit 0
}

$jobs = foreach ($item in $configs) {
    $args = New-BenchmarkArgs -ConfigPath $item.Config
    Start-Job -Name "dabench-$($item.Name)" -ScriptBlock {
        param($Root, $PythonArgs)
        Set-Location $Root
        & python @PythonArgs
    } -ArgumentList $repoRoot.Path, $args
}

Write-Host "Started benchmark jobs:"
$jobs | Select-Object Id, Name, State

Wait-Job -Job $jobs
foreach ($job in $jobs) {
    Write-Host ""
    Write-Host "== Output: $($job.Name) =="
    Receive-Job -Job $job
}

$failed = $jobs | Where-Object { $_.State -ne "Completed" }
Remove-Job -Job $jobs
if ($failed) {
    exit 1
}
