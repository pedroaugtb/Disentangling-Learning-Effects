#!/usr/bin/env bash
# Final queue: finishes every pending experiment and runs Addendum C (teacher / input), in one
# command meant for tmux.  Safe to re-run: every stage skips finished fits and draws.
#
#   scripts/run_final.sh [cpu_jobs] [gpu_workers]        defaults: 40 and 3
#
# Two lanes run in parallel, each with its own log; the collection runs when both have ended,
# whatever their status, and reports what is missing.
#   cpu  1. any unfinished primary-profile stage (core, pre-onset, sensitivity), SPX then NDX
#        2. har_ols decomposition (theory anchor)
#        3. Addendum C on real data: observed (all specs), placebo (primary, garch, h5td, h10td),
#           refit bootstrap (primary)
#        4. Addendum C Monte Carlo audit
#   gpu  nn2 profile on ONE card (GPU0, PCI 41:00.0, is defective): SPX core (finishes the tcn
#        refit bootstrap, multiplicity, collect), NDX core, then Addendum C observed + placebo
#   end  scripts/11 for every profile/asset, scripts/14 (claims, secondary, REPORT.md)
set -uo pipefail
cd "$(dirname "$0")/.."

CPU_JOBS="${1:-40}"
GPU_WORKERS="${2:-3}"
GPU_UUID="${IVBEHAV_GPU_UUID:-GPU-de89eed4-e1b8-5b8b-a676-0a99025cc078}"
PY=.venv/bin/python
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PWD/.tmp" JOBLIB_TEMP_FOLDER="$PWD/.tmp" IVBEHAV_SKIP_TESTS=1
mkdir -p logs "$TMPDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="logs/final_${STAMP}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1
echo "=== $(date '+%F %T') run_final cpu_jobs=$CPU_JOBS gpu_workers=$GPU_WORKERS log=$MAIN_LOG"

# Specs and families come from configs/teacher.yaml, so this script holds no design choice.
tcfg() { "$PY" -c "import yaml,sys; c=yaml.safe_load(open('configs/teacher.yaml')); print(' '.join(map(str, eval(sys.argv[1], {'c': c}))))" "$1"; }
OBS_SPECS=$(tcfg "c['observed']['specs']")
PLACEBO_SPECS=$(tcfg "c['placebo']['specs']")
REFIT_SPECS=$(tcfg "c['refit']['specs']")
CORE_FAMS=$(tcfg "c['confirmatory']['families']")
NN_FAMS=$(tcfg "c['neural']['families']")
NN_PLACEBO_SPECS=$(tcfg "c['neural']['placebo_specs']")
SENS_SPECS=$("$PY" -c "import sys; sys.path.insert(0,'src'); from ivbehav.config import load_config; print(' '.join(load_config('primary.yaml')['sensitivity_specs']))")

step() {   # step <description> <command...>: one attempt, timed, never aborts the lane
    echo; echo "=== $(date '+%F %T') $1"
    local start=$SECONDS
    shift
    if "$@"; then echo "=== ok in $(( SECONDS - start ))s: $*"; return 0; fi
    echo "!!! $(date '+%F %T') FAILED after $(( SECONDS - start ))s: $*"; return 1
}

try() {    # up to 3 attempts; stages resume, so a retry continues where the last one stopped
    local attempt
    for attempt in 1 2 3; do
        step "attempt $attempt: $*" "$@" && return 0
        sleep 60
    done
    return 1
}

# ------------------------------------------------------------------------------------
cpu_lane() {
    local A S
    for A in SPX NDX; do
        local T="results/primary/$A/primary/tables"
        # 1. unfinished stages of the primary profile (all done as of 2026-11-17; checked anyway)
        [[ -f "$T/hierarchy.csv" ]] || try scripts/run_all.sh core "$A" "$CPU_JOBS"
        [[ -f "$T/preonset_paths.csv" || -n "$(ls "$T"/preonset_* 2>/dev/null)" ]] || try scripts/run_all.sh preonset "$A" "$CPU_JOBS"
        local missing=0
        for S in $SENS_SPECS; do [[ -f "results/primary/$A/$S/tables/pairs_boot_xgb.csv" ]] || missing=1; done
        (( missing )) && try scripts/run_all.sh sensitivity "$A" "$CPU_JOBS"

        # 2. theory anchor: OLS HAR decomposition (fits exist from script 01)
        [[ -f "results/primary/$A/primary/behavior/har_ols/origin.parquet" ]] || \
            try "$PY" scripts/04_decomposition.py --asset "$A" --families har_ols --n-jobs "$CPU_JOBS"

        # 3. Addendum C on real data
        for S in $OBS_SPECS; do
            try "$PY" scripts/12_teacher_input.py --asset "$A" --spec "$S" --part observed --n-jobs "$CPU_JOBS"
        done
        for S in $PLACEBO_SPECS; do
            try "$PY" scripts/12_teacher_input.py --asset "$A" --spec "$S" --part placebo --n-jobs "$CPU_JOBS"
        done
        for S in $REFIT_SPECS; do
            # shellcheck disable=SC2086
            try "$PY" scripts/12_teacher_input.py --asset "$A" --spec "$S" --part refit --families $CORE_FAMS --n-jobs "$CPU_JOBS"
        done
    done
    # 4. Monte Carlo audit of the decomposition (pre-2017 SPX calibration only)
    try "$PY" scripts/13_teacher_simulation.py --n-jobs "$CPU_JOBS"
}

cuda_ok() {
    "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)" 2>/dev/null
}

gpu_try() {   # like try, but re-checks CUDA first: after a GPU fault, retrying only burns time
    local attempt
    for attempt in 1 2 3; do
        if ! cuda_ok; then
            echo "!!! $(date '+%F %T') CUDA unavailable (nvidia-smi / GPU reset needed); gpu lane stops here"
            return 2
        fi
        step "attempt $attempt: $*" "$@" && return 0
        sleep 60
    done
    return 1
}

gpu_lane() {
    export CUDA_VISIBLE_DEVICES="$GPU_UUID"
    if nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | grep -qi "41:00.0"; then
        echo "!!! GPU0 (PCI 41:00.0) is still on the bus. It is hidden from CUDA here, but when it faults it"
        echo "!!! has taken CUDA down on both cards.  To remove it first:"
        echo "!!!     echo 1 | sudo tee /sys/bus/pci/devices/0000:41:00.0/remove"
    fi
    cuda_ok || { echo "!!! CUDA unavailable with CUDA_VISIBLE_DEVICES=$GPU_UUID; gpu lane not started"; return 2; }
    local A S rc
    gpu_try scripts/run_all.sh core SPX "$GPU_WORKERS" nn.yaml; rc=$?; (( rc == 2 )) && return 2
    gpu_try scripts/run_all.sh core NDX "$GPU_WORKERS" nn.yaml; rc=$?; (( rc == 2 )) && return 2
    for A in SPX NDX; do
        # observed is CPU only; placebo refits the neural f1 on the GPU unless the draw already
        # carries the statistics (NDX draws made after 2026-11-17 do, and are copied)
        # shellcheck disable=SC2086
        gpu_try "$PY" scripts/12_teacher_input.py --config nn.yaml --asset "$A" --part observed --families $NN_FAMS --n-jobs "$GPU_WORKERS"
        for S in $NN_PLACEBO_SPECS; do
            # shellcheck disable=SC2086
            gpu_try "$PY" scripts/12_teacher_input.py --config nn.yaml --asset "$A" --spec "$S" --part placebo --families $NN_FAMS --n-jobs "$GPU_WORKERS"
            rc=$?; (( rc == 2 )) && return 2
        done
    done
    return 0
}

# ------------------------------------------------------------------------------------
CPU_LOG="logs/final_cpu_${STAMP}.log"
GPU_LOG="logs/final_gpu_${STAMP}.log"
echo "cpu lane -> $CPU_LOG"
echo "gpu lane -> $GPU_LOG"
cpu_lane > "$CPU_LOG" 2>&1 &
CPU_PID=$!
gpu_lane > "$GPU_LOG" 2>&1 &
GPU_PID=$!
echo "follow with:  tail -f $CPU_LOG   |   tail -f $GPU_LOG"
wait "$CPU_PID"; CPU_RC=$?
echo "=== $(date '+%F %T') cpu lane ended with status $CPU_RC"
wait "$GPU_PID"; GPU_RC=$?
echo "=== $(date '+%F %T') gpu lane ended with status $GPU_RC"

# ------------------------------------------------------------------------------------
step "hierarchy, primary SPX" "$PY" scripts/11_collect_primary.py --asset SPX
step "hierarchy, primary NDX" "$PY" scripts/11_collect_primary.py --asset NDX
for A in SPX NDX; do
    [[ -f "results/nn2/$A/primary/tables/refit_boot_tcn.csv" ]] && \
        step "hierarchy, nn2 $A" "$PY" scripts/11_collect_primary.py --config nn.yaml --asset "$A"
done
step "Addendum C collection" "$PY" scripts/14_collect_teacher.py

echo
echo "=== $(date '+%F %T') run_final finished: cpu=$CPU_RC gpu=$GPU_RC"
echo "Grep the lane logs for '!!!' to see every failure.  Results:"
echo "  results/primary/teacher/REPORT.md, claims.csv, secondary.csv, simulation_oc.csv"
echo "  results/primary/<asset>/<spec>/tables/teacher_*.csv;  results/nn2/<asset>/primary/tables/hierarchy.csv"
exit $(( CPU_RC != 0 || GPU_RC != 0 ))
