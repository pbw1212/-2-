# 场景一：开放式长句 + 句中停顿（dev_01，停顿 1000 ms）
#
# 同一段录音，三种端点：
#   fixed:600      -> 在停顿处截断，后半句"下周四下午"直接丢失
#   fixed:1500     -> 等到后半句说完，但每条短答都要多等 1.5 s
#   adaptive_text  -> 用"我想..."这类续说线索自动升到 1500 ms
#
# 用法：powershell -ExecutionPolicy Bypass -File b_module/demo/demo_open_pause.ps1 [-Fast]

param([switch]$Fast)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$Python = if ($env:B_PYTHON) { $env:B_PYTHON } else { "python" }
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $Root

$traceDir = "b_module/results/traces"
New-Item -ItemType Directory -Force -Path $traceDir | Out-Null

function Invoke-Replay($case, $strategy, $traceName) {
    $cli = @("-m", "b_module", "replay", "--case", $case, "--strategy", $strategy,
             "--trace", "$traceDir/$traceName")
    if (-not $Fast) { $cli += "--realtime" }
    & $Python @cli
}

function Show-Banner($text) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkGray
    Write-Host $text -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkGray
}

Show-Banner "场景一 / 1：固定 600 ms —— 在句中停顿处截断"
Invoke-Replay "dev_01" "fixed:600" "dev_01__fixed600.jsonl"

Show-Banner "场景一 / 2：固定 1500 ms —— 等到后半句，但每次都多等 1.5 s"
Invoke-Replay "dev_01" "fixed:1500" "dev_01__fixed1500.jsonl"

Show-Banner "场景一 / 3：自适应二 —— 靠续说线索把等待拉到上限 1500 ms"
Invoke-Replay "dev_01" "adaptive_text" "dev_01__adaptive_text.jsonl"

Show-Banner "结论：开放轮必须等到 1500 ms 才不截断；自适应的价值是按轮次类型分配耐心（短答轮只等 450 ms，见场景二）"
Write-Host "trace 已写入 $traceDir" -ForegroundColor Green
