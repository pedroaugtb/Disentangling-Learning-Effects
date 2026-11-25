#!/usr/bin/env bash
# Addendum E: raise the refit-bootstrap budget of the Addendum C and D cells to 2000 draws, then
# re-collect both addenda.  CPU only; safe to re-run (finished draws are skipped).
#
#   scripts/run_amend.sh [cpu_jobs]        default 56
set -uo pipefail
cd "$(dirname "$0")/.."

CPU_JOBS="${1:-56}"
PY=.venv/bin/python
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PWD/.tmp" JOBLIB_TEMP_FOLDER="$PWD/.tmp" IVBEHAV_SKIP_TESTS=1
mkdir -p logs "$TMPDIR"
LOG="logs/amend_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') run_amend cpu_jobs=$CPU_JOBS log=$LOG"

acfg() { "$PY" -c "import yaml,sys; c=yaml.safe_load(open('configs/teacher_amend.yaml')); v=c[sys.argv[1]]; print(' '.join(map(str, v)) if isinstance(v, list) else v)" "$1"; }
DRAWS=$(acfg refit_draws); ASSETS=$(acfg assets); SPECS=$(acfg specs); FAMS=$(acfg families)
SNAP="results/primary/$(acfg snapshot_dir)"

# 1. Keep the pre-amendment claims (once).  The old draws stay in place and are a subset of the new.
if [ ! -d "$SNAP" ]; then
    mkdir -p "$SNAP"
    cp -r results/primary/teacher "$SNAP/teacher"
    cp -r results/primary/teacher_short "$SNAP/teacher_short"
    for A in $ASSETS; do for S in $SPECS; do
        mkdir -p "$SNAP/tables/$A/$S"
        cp results/primary/"$A"/"$S"/tables/teacher_refit_*.csv "$SNAP/tables/$A/$S/" 2>/dev/null
    done; done
    echo "=== snapshot written to $SNAP"
fi

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

# 2. Extra refit draws (largest cells last so partial runs still finish whole cells).
RC=0
for F in $FAMS; do
    for A in $ASSETS; do
        for S in $SPECS; do
            try "$PY" scripts/12_teacher_input.py --asset "$A" --spec "$S" --part refit --families "$F" \
                --refit-draws "$DRAWS" --n-jobs "$CPU_JOBS" || RC=1
        done
    done
done

# 3. Re-collect both addenda with every draw.
step "Addendum C collection" "$PY" scripts/14_collect_teacher.py || RC=1
step "Addendum D collection" "$PY" scripts/15_collect_short.py || RC=1

echo
echo "=== $(date '+%F %T') run_amend finished: status $RC"
echo "New: results/primary/teacher/REPORT.md, results/primary/teacher_short/REPORT.md"
echo "Before Addendum E: $SNAP/"
exit $RC
