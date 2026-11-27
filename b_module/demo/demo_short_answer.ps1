# 场景二：定向短答与确认轮（smoke_short 语音 160 ms；test_02 停顿 650 ms）
#
# 要展示：
#   1) 确认轮用户只说一个"对"，固定 900 ms 会白等到 1060 ms；
#      自适应二认出"这是短答"后 620 ms 就结束（快了 440 ms）。
#   2) 同一套参数下，650 ms 的句中停顿仍然不会被截断。
#
# 用法：powershell -ExecutionPolicy Bypass -File b_module/demo/demo_short_answer.ps1 [-Fast]

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

Show-Banner "场景二 / 1：确认轮短答 + 固定 900 ms —— 用户说完还要等 1 秒才反应"
Invoke-Replay "smoke_short" "fixed:900" "smoke_short__fixed900.jsonl"

Show-Banner "场景二 / 2：确认轮短答 + 自适应二 —— 认出短答，450 ms 就结束"
Invoke-Replay "smoke_short" "adaptive_text" "smoke_short__adaptive_text.jsonl"

Show-Banner "场景二 / 3：同一套自适应参数下，650 ms 句中停顿仍不截断（test_02）"
Invoke-Replay "test_02" "adaptive_text" "test_02__adaptive_text.jsonl"

Show-Banner "结论：短答更快，整句不截断——两个目标由同一条自适应策略同时满足"
Write-Host "trace 已写入 $traceDir" -ForegroundColor Green
