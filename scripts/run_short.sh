#!/usr/bin/env bash
# Addendum D: refit bootstrap of the short-horizon decompositions, then the post-hoc confirmation.
# CPU only; safe to re-run (finished refit draws are skipped).
#
#   scripts/run_short.sh [cpu_jobs]        default 40
set -uo pipefail
cd "$(dirname "$0")/.."

CPU_JOBS="${1:-40}"
PY=.venv/bin/python
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PWD/.tmp" JOBLIB_TEMP_FOLDER="$PWD/.tmp" IVBEHAV_SKIP_TESTS=1
mkdir -p logs "$TMPDIR"
LOG="logs/short_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') run_short cpu_jobs=$CPU_JOBS log=$LOG"

dcfg() { "$PY" -c "import yaml,sys; c=yaml.safe_load(open('configs/teacher_short.yaml')); print(' '.join(map(str, eval(sys.argv[1], {'c': c}))))" "$1"; }
ASSETS=$(dcfg "c['confirmatory']['assets']")
SPECS=$(dcfg "c['confirmatory']['specs']")
FAMS=$(dcfg "c['confirmatory']['families']")

step() {
    echo; echo "=== $(date '+%F %T') $1"
    local start=$SECONDS
    shift
    if "$@"; then echo "=== ok in $(( SECONDS - start ))s: $*"; return 0; fi
    echo "!!! $(date '+%F %T') FAILED after $(( SECONDS - start ))s: $*"; return 1
}
try() {
    local attempt
    for attempt in 1 2 3; do
        step "attempt $attempt: $*" "$@" && return 0
        sleep 60
    done
    return 1
}

RC=0
for A in $ASSETS; do
    for S in $SPECS; do
        # observed and placebo exist from Addendum C (fast: recomputed / copied); refit is the new work
        # shellcheck disable=SC2086
        try "$PY" scripts/12_teacher_input.py --asset "$A" --spec "$S" --part observed --families $FAMS --n-jobs "$CPU_JOBS" || RC=1
        # shellcheck disable=SC2086
        try "$PY" scripts/12_teacher_input.py --asset "$A" --spec "$S" --part placebo --families $FAMS --n-jobs "$CPU_JOBS" || RC=1
        # shellcheck disable=SC2086
        try "$PY" scripts/12_teacher_input.py --asset "$A" --spec "$S" --part refit --families $FAMS --n-jobs "$CPU_JOBS" || RC=1
    done
done
step "Addendum D collection" "$PY" scripts/15_collect_short.py || RC=1

echo
echo "=== $(date '+%F %T') run_short finished: status $RC"
echo "Results: results/primary/teacher_short/REPORT.md, claims.csv, headline.csv, term_structure.csv"
exit $RC
