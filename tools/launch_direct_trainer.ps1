param(
    [string]$RepoRoot = "C:\Users\Laure\Desktop\AlphaXiang Transformer",
    [int]$MaxSteps = 70106
)

$ErrorActionPreference = "Stop"

$repoRootWin = (Resolve-Path $RepoRoot).Path
$repoRootWsl = "/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
$trainCommand = @(
    "source '/home/laure/.virtualenvs/AlphaXiang Transformer/bin/activate'",
    "python '$repoRootWsl/xiangqi_train.py'",
    "--foreground",
    "--human-data-dir '$repoRootWsl/human_bootstrap_data_elite_wdl'",
    "--selfplay-dirs '$repoRootWsl/selfplay_runs_bootstrap'",
    "--output-dir '$repoRootWsl/training_runs/run_001'",
    "--resume-path '$repoRootWsl/training_runs/run_001/latest.pt'",
    "--device cuda:0",
    "--max-steps $MaxSteps",
    "--log-interval-steps 20"
) -join " "

& wsl.exe -e bash -lc $trainCommand
