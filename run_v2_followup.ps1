# V2 Followup 一键串行脚本
# 在 E2 完成后启动，按序执行 A1 -> A2 -> E3 -> E4 -> E1 收尾
# 用法：在独立 PowerShell 窗口执行：
#   cd path\to\clinifs-benchmark
#   .\run_v2_followup.ps1
# 可选参数：
#   -SkipE2Check : 跳过 E2 完成度检查，立即开始
#   -OnlyAnalysis : 只跑 A1 + A2，不跑 E3/E4
#   -WorkersE3 N  : E3 dispatcher 的 workers（默认 2）
#   -WorkersE4 N  : E4 dispatcher 的 workers（默认 2）

param(
    [switch]$SkipE2Check,
    [switch]$OnlyAnalysis,
    [int]$WorkersE3 = 2,
    [int]$WorkersE4 = 2
)

$ErrorActionPreference = "Continue"
$ROOT = Split-Path -Parent $PSCommandPath
Set-Location $ROOT

$LogDir = Join-Path $ROOT "output\parallel_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "followup.log"

function Log($msg) {
    $t = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$t] $msg"
    $line | Tee-Object -FilePath $LogFile -Append
}

function Wait-ForE2Complete {
    # E2 total = 15 methods x 7 k-values x 11 datasets = 1155
    # Tolerance: accept >= 99% (1143) to handle SFE/MEL edge-case datasets that may not produce summary.json
    # Stability: if no python procs AND summary count unchanged for 3 consecutive polls, treat as complete
    Log "Waiting for E2 to complete (target 1155, min acceptable 1143)..."
    $target = 1155
    $min_acceptable = 1143
    $prev_c = -1
    $stable_count = 0
    while ($true) {
        $c = (Get-ChildItem "output\v2\E2_constrained_k" -Recurse -File -Filter summary.json -ErrorAction SilentlyContinue | Measure-Object).Count
        $py = (Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($c -eq $prev_c) { $stable_count++ } else { $stable_count = 0 }
        Log "  E2 summary.json = $c / $target; python procs = $py; stable_polls = $stable_count"

        # Primary exit: reached full target AND no workers running
        if ($c -ge $target -and $py -eq 0) {
            Log "E2 complete (hit target)."
            return
        }
        # Secondary exit: >= 99% AND no workers AND unchanged for 3 polls (6 min)
        if ($c -ge $min_acceptable -and $py -eq 0 -and $stable_count -ge 3) {
            Log "E2 effectively complete: $c/$target (>= $min_acceptable), workers exited, count stable for $stable_count polls."
            return
        }
        # Tertiary: target hit but workers alive - short wait
        if ($c -ge $target) {
            Log "E2 summary count reached target but python procs still alive; waiting 30s..."
            Start-Sleep -Seconds 30
            $prev_c = $c
            continue
        }
        $prev_c = $c
        Start-Sleep -Seconds 120
    }
}

function Run-Stage($name, $cmd) {
    Log "==================== Stage: $name START ===================="
    Log "CMD: $cmd"
    $start = Get-Date
    try {
        Invoke-Expression $cmd 2>&1 | Tee-Object -FilePath $LogFile -Append
        $ec = $LASTEXITCODE
    } catch {
        Log "EXCEPTION: $_"
        $ec = -1
    }
    $dur = ((Get-Date) - $start).TotalMinutes
    Log ("==================== Stage: $name DONE (exit={0}, wall={1:N1} min) ====================" -f $ec, $dur)
    return $ec
}

# ───── 起始检查 ─────
Log "V2 Followup starting (SkipE2Check=$SkipE2Check, OnlyAnalysis=$OnlyAnalysis, WorkersE3=$WorkersE3, WorkersE4=$WorkersE4)"
$mem = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($mem.FreePhysicalMemory / 1MB, 1)
Log "System free memory: $freeGB GB"

if (-not $SkipE2Check) {
    Wait-ForE2Complete
} else {
    Log "Skipping E2 completion check (user-requested)."
}

# ───── Stage 1: A1 overlap ─────
Run-Stage "A1_overlap" "python run_v2_a1_overlap.py"

# ───── Stage 2: A2 enrichment ─────
Run-Stage "A2_enrichment" "python run_v2_a2_enrichment.py"

if ($OnlyAnalysis) {
    Log "OnlyAnalysis mode: skipping E3/E4/E1. Followup finished."
    exit 0
}

# ───── Stage 3: E1 EA sparsity 收尾 ─────
Run-Stage "E1_ea_sparsity_tail" "python run_v2_dispatcher.py --experiment E1 --workers 1"

# ───── Stage 4: E3 pipelines ─────
Run-Stage "E3_pipelines" "python run_v2_dispatcher.py --experiment E3 --workers $WorkersE3"

# ───── Stage 5: E4 EA consensus ─────
Run-Stage "E4_consensus" "python run_v2_dispatcher.py --experiment E4 --workers $WorkersE4"

# ───── 最终覆盖度报告 ─────
Log "==================== Final coverage report ===================="
foreach ($exp in @('E1_ea_sparsity','E2_constrained_k','E3_pipelines','E4_consensus','E5_external_pairs')) {
    $path = "output\v2\$exp"
    if (Test-Path $path) {
        $c = (Get-ChildItem $path -Recurse -File -Filter summary.json | Measure-Object).Count
        Log ("  {0,-22} summary.json = {1}" -f $exp, $c)
    }
}
foreach ($ana in @('A1_overlap','A2_enrichment')) {
    $path = "output\v2\$ana"
    if (Test-Path $path) {
        $c = (Get-ChildItem $path -Recurse -File | Measure-Object).Count
        Log ("  {0,-22} files = {1}" -f $ana, $c)
    }
}
Log "Followup finished."
