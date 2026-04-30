#!/usr/bin/env bash
# ==============================================================
# V2 Followup — Mac / Linux 版本
# 与 run_v2_followup.ps1 功能等价：在 E2 完成后按序执行
#   A1 overlap -> A2 enrichment -> E1 收尾 -> E3 pipelines -> E4 consensus
# 所有阶段基于 summary.json 级 checkpoint，可随时 Ctrl+C 中断再续跑。
#
# 用法（推荐在独立终端窗口运行）：
#   chmod +x run_v2_followup.sh
#   ./run_v2_followup.sh                           # 全流程
#   ./run_v2_followup.sh --skip-e2-check           # 跳过 E2 完成度等待
#   ./run_v2_followup.sh --only-analysis           # 只跑 A1+A2
#   ./run_v2_followup.sh --workers-e3 2 --workers-e4 2
#   ./run_v2_followup.sh --a2-online                # A2 走在线 g:Profiler（需代理）
# ==============================================================
set -u
# 故意不 set -e：单阶段失败不应中断后续阶段

# ---- 参数 ----
SKIP_E2_CHECK=0
ONLY_ANALYSIS=0
WORKERS_E3=2
WORKERS_E4=2
A2_OFFLINE=1  # 默认离线（已有 KEGG+Hallmark GMT）
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-e2-check) SKIP_E2_CHECK=1; shift ;;
    --only-analysis) ONLY_ANALYSIS=1; shift ;;
    --workers-e3)    WORKERS_E3="$2"; shift 2 ;;
    --workers-e4)    WORKERS_E4="$2"; shift 2 ;;
    --a2-online)     A2_OFFLINE=0; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

# ---- 路径与日志 ----
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT"
LOG_DIR="$ROOT/output/parallel_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/followup.log"

log() {
  local msg="$1"
  local t
  t=$(date +"%Y-%m-%d %H:%M:%S")
  echo "[$t] $msg" | tee -a "$LOG_FILE"
}

# ---- Python 解释器解析（优先当前 env）----
PY_BIN="${PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN=python3
fi
log "Using Python: $($PY_BIN --version 2>&1) at $(command -v "$PY_BIN")"

# ---- 统计当前 python 进程 (排除 grep 自己) ----
count_py_procs() {
  # 匹配进程名 python 且 CWD 在本 ROOT 下的
  pgrep -af "python" 2>/dev/null \
    | grep -v "pgrep" | grep -v "followup.sh" \
    | wc -l | tr -d ' '
}

count_summary() {
  local subdir="$1"
  if [[ -d "output/v2/$subdir" ]]; then
    find "output/v2/$subdir" -type f -name summary.json 2>/dev/null | wc -l | tr -d ' '
  else
    echo 0
  fi
}

wait_for_e2_complete() {
  # E2 total = 15 methods * 7 k-values * 11 datasets = 1155
  # 容忍 >= 99% (1143)；无 python 进程且 3 轮 count 不变视为完成
  local target=1155
  local min_acceptable=1143
  local prev=-1
  local stable=0
  log "Waiting for E2 to complete (target $target, min $min_acceptable)..."
  while true; do
    local c py
    c=$(count_summary E2_constrained_k)
    py=$(count_py_procs)
    if [[ "$c" == "$prev" ]]; then stable=$((stable+1)); else stable=0; fi
    log "  E2 summary = $c/$target; python procs = $py; stable = $stable"

    if (( c >= target )) && (( py == 0 )); then
      log "E2 complete (hit target)."; return 0
    fi
    if (( c >= min_acceptable )) && (( py == 0 )) && (( stable >= 3 )); then
      log "E2 effectively complete: $c/$target, workers exited, stable."; return 0
    fi
    if (( c >= target )); then
      log "Count reached target but workers alive; 30s wait..."
      sleep 30
      prev=$c
      continue
    fi
    prev=$c
    sleep 120
  done
}

run_stage() {
  local name="$1"; shift
  local cmd="$*"
  log "==================== Stage: $name START ===================="
  log "CMD: $cmd"
  local t0
  t0=$(date +%s)
  # 用 bash -c 执行，允许 tee 抓取 stdout/stderr
  bash -c "$cmd" 2>&1 | tee -a "$LOG_FILE"
  local ec=${PIPESTATUS[0]}
  local t1
  t1=$(date +%s)
  local dur=$(( (t1 - t0) / 60 ))
  log "==================== Stage: $name DONE (exit=$ec, wall=${dur} min) ===================="
}

# ==================== 主流程 ====================
log "V2 Followup starting (skip_e2=$SKIP_E2_CHECK, only_analysis=$ONLY_ANALYSIS, e3=$WORKERS_E3, e4=$WORKERS_E4)"

# 报告内存情况（macOS 与 Linux 通用）
if command -v vm_stat >/dev/null 2>&1; then
  # macOS
  free_pages=$(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}')
  free_mb=$(( free_pages * 4096 / 1024 / 1024 ))
  log "System free memory (macOS): ${free_mb} MB"
elif command -v free >/dev/null 2>&1; then
  free -h | awk '/^Mem:/ {print "[Mem] total="$2" used="$3" free="$4" avail="$7}' | while read l; do log "$l"; done
fi

if [[ $SKIP_E2_CHECK -eq 0 ]]; then
  wait_for_e2_complete
else
  log "Skipping E2 completion check (user-requested)."
fi

run_stage "A1_overlap"    "$PY_BIN run_v2_a1_overlap.py"
if [[ $A2_OFFLINE -eq 1 ]]; then
  run_stage "A2_enrichment" "$PY_BIN run_v2_a2_enrichment.py --offline"
else
  run_stage "A2_enrichment" "$PY_BIN run_v2_a2_enrichment.py"
fi

if [[ $ONLY_ANALYSIS -eq 1 ]]; then
  log "OnlyAnalysis mode: skipping E3/E4/E1. Followup finished."
  exit 0
fi

run_stage "E1_ea_sparsity_tail" "$PY_BIN run_v2_dispatcher.py --experiment E1 --workers 1"
run_stage "E3_pipelines"        "$PY_BIN run_v2_dispatcher.py --experiment E3 --workers $WORKERS_E3"
run_stage "E4_consensus"        "$PY_BIN run_v2_dispatcher.py --experiment E4 --workers $WORKERS_E4"

log "==================== Final coverage report ===================="
for exp in E1_ea_sparsity E2_constrained_k E3_pipelines E4_consensus E5_external_pairs; do
  c=$(count_summary "$exp")
  printf "  %-22s summary.json = %s\n" "$exp" "$c" | tee -a "$LOG_FILE"
done
for ana in A1_overlap A2_enrichment; do
  path="output/v2/$ana"
  if [[ -d "$path" ]]; then
    n=$(find "$path" -type f | wc -l | tr -d ' ')
    printf "  %-22s files        = %s\n" "$ana" "$n" | tee -a "$LOG_FILE"
  fi
done
log "Followup finished."
