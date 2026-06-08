<#
.SYNOPSIS
    Convenience wrapper that runs scripts/run_cache.py against the H1/H0
    dataset with sensible defaults, so you don't need to remember the flags.

.DESCRIPTION
    Builds the cache with MLCache.from_preset (via run_cache.py), calibrates
    it with cache.prefit_and_calibrate(...) against a target false-accept-rate
    budget, replays the query stream, and writes the experiment artifacts
    (summary_metrics.json, per_request_decisions.csv, runtime_config.json,
    schema_report.json, calibration_report.json) to the output directory.

.EXAMPLE
    .\scripts\run_cache.ps1
        Runs the ensemble scorer over the full dataset.

.EXAMPLE
    .\scripts\run_cache.ps1 -MaxRows 200 -Scorer cosine
        Fast trial run with the cosine scorer over the first 200 rows.

.EXAMPLE
    .\scripts\run_cache.ps1 -Npz "D:\datasets\h1h0_final.npz" -OutputDir "experiments\my_run"
        Point at a different dataset / output location.
#>

[CmdletBinding()]
param(
    [string]$Npz = "$PSScriptRoot\..\..\data\h1h0_final.npz",
    [string]$OutputDir = "$PSScriptRoot\..\experiments\h1h0_ensemble",
    [string]$Scorer = "ensemble",
    [string]$Scorers = "cosine,lda,pca_whitened_cosine,xgboost,mlp",
    [double]$TargetFpr = 0.05,
    [int]$TopK = 5,
    [Nullable[int]]$MaxRows = $null,
    [switch]$NoFilePersistence
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\.."
$Python = Join-Path $RepoRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$RunCache = Join-Path $RepoRoot "scripts\run_cache.py"

$arguments = @(
    $RunCache,
    "--npz", $Npz,
    "--output-dir", $OutputDir,
    "--label-field", "label",
    "--query-field", "text",
    "--anchor-field", "global_cluster",
    "--query-embedding-field", "emb",
    "--scorer", $Scorer,
    "--scorers", $Scorers,
    "--target-fpr", $TargetFpr,
    "--top-k", $TopK
)

if ($null -ne $MaxRows) {
    $arguments += @("--max-rows", $MaxRows)
}
if ($NoFilePersistence) {
    $arguments += "--no-file-persistence"
}

Write-Host "Running: $Python $($arguments -join ' ')"
& $Python @arguments
exit $LASTEXITCODE
