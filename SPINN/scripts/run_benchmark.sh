#!/bin/bash
# SPINN 自动化基准测试脚本
# 用法: ./scripts/run_benchmark.sh [--full]
#   --full: 同时跑 ORT 对比 (需要 onnxruntime)
#
# 输出: 标准输出打印汇总表, 可重定向到文件存档
# 退出码: 0=成功, 1=失败

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPINN_DIR="$(dirname "$SCRIPT_DIR")"
RUNTIME_DIR="$SPINN_DIR/run_time"
SPINN_EXE="$RUNTIME_DIR/spinn_run"
ONNX2ENNF="$SPINN_DIR/onnx2ennf"

OMP_THREADS="${OMP_NUM_THREADS:-4}"
NUM_RUNS=5
WARMUP=3

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

die() { echo -e "${RED}ERROR: $*${NC}" >&2; exit 1; }

# ---- 检查依赖 ----
[ -x "$SPINN_EXE" ] || die "spinn_run not found: $SPINN_EXE (run 'make' first?)"

# ---- 模型列表 ----
MODELS=()
if [ -f "$RUNTIME_DIR/yolov10n.ennf" ]; then
    MODELS+=("yolov10n")
fi
if [ -f "$RUNTIME_DIR/resnet101.ennf" ]; then
    MODELS+=("resnet101")
fi
[ ${#MODELS[@]} -gt 0 ] || die "No .ennf models found in $RUNTIME_DIR"

# ---- 可选: 重新转换模型 ----
RECONVERT=false
if [ "${1:-}" = "--reconvert" ]; then
    RECONVERT=true
    shift
fi

if $RECONVERT; then
    echo "=== Re-converting models with graph optimization ==="
    for m in "${MODELS[@]}"; do
        onnx_file="$RUNTIME_DIR/${m}.onnx"
        ennf_file="$RUNTIME_DIR/${m}.ennf"
        [ -f "$onnx_file" ] || { echo "WARNING: $onnx_file not found, skipping reconvert for $m"; continue; }
        echo "  Converting $m..."
        "$ONNX2ENNF" "$onnx_file" "$ennf_file" 2>&1 | tail -5
    done
    echo ""
fi

# ---- 编译检查 ----
echo "=== Build Check ==="
echo "  spinn_run: $(ls -lh "$SPINN_EXE" | awk '{print $5, $6, $7}')"
echo "  OMP_NUM_THREADS=$OMP_THREADS"
echo ""

# ---- SPINN Benchmark ----
echo "=== SPINN Benchmark (num_runs=$NUM_RUNS, warmup=$WARMUP) ==="
echo ""

declare -A SPINN_BEST SPINN_AVG SPINN_RSS

for m in "${MODELS[@]}"; do
    ennf_file="$RUNTIME_DIR/${m}.ennf"
    [ -f "$ennf_file" ] || continue

    echo "--- $m ---"
    output=$(OMP_NUM_THREADS=$OMP_THREADS SPINN_PROFILE=1 \
        /usr/bin/time -v "$SPINN_EXE" "$ennf_file" $NUM_RUNS 2>&1)

    # 解析 Best/Avg
    best=$(echo "$output" | grep -oP 'Best:\s+\K[\d.]+')
    avg=$(echo "$output" | grep -oP 'Avg:\s+\K[\d.]+')
    rss=$(echo "$output" | grep -oP 'Maximum resident set size \(kbytes\):\s+\K\d+')

    SPINN_BEST[$m]=$best
    SPINN_AVG[$m]=$avg
    SPINN_RSS[$m]=$rss

    echo "  Best: ${best}ms  Avg: ${avg}ms  RSS: $((rss/1024))MB"

    # 解析 per-op profile
    echo "  Per-op profile:"
    echo "$output" | sed -n '/===== SPINN per-op profile =====/,/Total tracked/p' | grep -E '^\s*[0-9]' | head -8 | while read line; do
        echo "    $line"
    done
    echo ""
done

# ---- ORT Benchmark (可选) ----
RUN_ORT=false
if [ "${1:-}" = "--full" ] || [ "${1:-}" = "--ort" ]; then
    RUN_ORT=true
fi

if $RUN_ORT && command -v python3 &>/dev/null; then
    echo "=== ORT Benchmark ==="
    echo ""

    declare -A ORT_BEST ORT_AVG ORT_RSS

    for m in "${MODELS[@]}"; do
        onnx_file="$RUNTIME_DIR/${m}.onnx"
        [ -f "$onnx_file" ] || continue

        echo "--- $m ---"
        ort_output=$(OMP_NUM_THREADS=$OMP_THREADS /usr/bin/time -v python3 -c "
import time, numpy as np, onnxruntime as ort
sess = ort.InferenceSession('$onnx_file', providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
shape = [d if isinstance(d, int) else 1 for d in inp.shape]
n = int(np.prod(shape))
data = (np.arange(n, dtype=np.float32) / 1000.0).reshape(shape)
for _ in range($WARMUP): sess.run(None, {inp.name: data})
best = 1e18; total = 0.0
for i in range($NUM_RUNS):
    t = time.perf_counter()
    sess.run(None, {inp.name: data})
    e = (time.perf_counter() - t) * 1000.0
    if e < best: best = e
    total += e
print(f'ORT Best: {best:.2f}ms, Avg: {total/$NUM_RUNS:.2f}ms')
" 2>&1)

        ort_best=$(echo "$ort_output" | grep -oP 'ORT Best:\s+\K[\d.]+')
        ort_avg=$(echo "$ort_output" | grep -oP 'Avg:\s+\K[\d.]+')
        ort_rss=$(echo "$ort_output" | grep -oP 'Maximum resident set size \(kbytes\):\s+\K\d+')

        ORT_BEST[$m]=${ort_best:-"n/a"}
        ORT_AVG[$m]=${ort_avg:-"n/a"}
        ORT_RSS[$m]=${ort_rss:-"0"}

        echo "  Best: ${ort_best:-n/a}ms  Avg: ${ort_avg:-n/a}ms  RSS: $((ort_rss/1024))MB"
        echo ""
    done
fi

# ---- 汇总表 ----
echo "================================================================"
echo "  SUMMARY"
echo "================================================================"
echo ""
printf "  %-12s %10s %10s %10s %10s %10s\n" "Model" "SPINN Best" "SPINN Avg" "ORT Best" "ORT Avg" "Ratio"
echo "  ----------------------------------------------------------------------"

for m in "${MODELS[@]}"; do
    sb=${SPINN_BEST[$m]:-"n/a"}
    sa=${SPINN_AVG[$m]:-"n/a"}
    ob=${ORT_BEST[$m]:-"n/a"}
    oa=${ORT_AVG[$m]:-"n/a"}

    if [ "$ob" != "n/a" ] && [ "$sb" != "n/a" ]; then
        ratio=$(python3 -c "print(f'{$sb/$ob:.2f}x')" 2>/dev/null || echo "n/a")
    else
        ratio="n/a"
    fi

    printf "  %-12s %10s %10s %10s %10s %10s\n" "$m" "${sb}ms" "${sa}ms" "${ob}ms" "${oa}ms" "$ratio"
done

echo ""
echo "  RSS Peak:"
printf "  %-12s %12s %12s %12s\n" "Model" "SPINN" "ORT" "Ratio"
echo "  --------------------------------------------------"
for m in "${MODELS[@]}"; do
    sr=$((SPINN_RSS[$m]/1024))
    or=$((ORT_RSS[$m]/1024))
    if [ "$or" -gt 0 ] 2>/dev/null; then
        rratio=$(python3 -c "print(f'{$sr/$or:.2f}x')" 2>/dev/null || echo "n/a")
    else
        rratio="n/a"
    fi
    printf "  %-12s %8d MB %8d MB %12s\n" "$m" "$sr" "$or" "$rratio"
done

echo ""
echo "  Test environment:"
echo "    CPU: $(lscpu | grep -i "model name" | head -1 | sed 's/Model name:\s*//' | sed 's/型号名称：\s*//')"
echo "    ORT: $(python3 -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null || echo 'n/a')"
echo "    Date: $(date -Iseconds)"
echo ""
