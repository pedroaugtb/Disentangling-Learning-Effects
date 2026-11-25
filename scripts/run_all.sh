#!/usr/bin/env bash
# Runs the study stage by stage.  Safe to re-run: finished folds and draws are skipped.
#
#   scripts/run_all.sh [stage] [asset] [n_jobs] [config]
#
# Stages, in order:
#   check        unit tests and the outcome-blind design check
#   simulation   Monte Carlo audit (Sec. 8); reads pre-2017 SPX rows only
#   forecasts    walk-forward fits of the primary spec (WP1 inputs)
#   behavior     decomposition, response functions, matching, accuracy checks (WP2, WP3, WP5)
#   placebo      negative controls (WP4); needs `behavior`
#   refit        refit bootstrap for intervals and repeated-fit variability (Sec. 9)
#   multiplicity near-equivalent fits, windows and learners (WP7)
#   preonset     exploratory event paths (WP6)
#   collect      assembles the prespecified hierarchy
#   sensitivity  non-confirmatory specs (lagged IV, rolling window, GARCH control, horizons, ...)
#   core         check + forecasts + behavior + placebo + refit + multiplicity + collect
#   all          simulation + core + preonset + sensitivity
#
# Replication: run SPX first, and NDX only after the SPX decisions are final.  There is no
# NDX-specific option on purpose: the same protocol runs with the asset swapped.
set -euo pipefail

STAGE="${1:-all}"
ASSET="${2:-SPX}"
JOBS="${3:-24}"
CONFIG="${4:-primary.yaml}"

cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# / is shared and can fill up: keep every temp file of the run on /scratch, inside the project.
export TMPDIR="$PWD/.tmp" JOBLIB_TEMP_FOLDER="$PWD/.tmp"
mkdir -p logs "$TMPDIR"
LOG="logs/${ASSET}_${STAGE}_$(date +%Y%m%d_%H%M%S).log"
echo "stage=$STAGE asset=$ASSET jobs=$JOBS config=$CONFIG log=$LOG"
exec > >(tee -a "$LOG") 2>&1

COMMON="--config $CONFIG --asset $ASSET"

run_step() {
    echo; echo "=== $(date '+%F %T') $*"
    local start=$SECONDS
    "$@"
    echo "=== finished in $(( SECONDS - start ))s"
}

stage_check() {
    # IVBEHAV_SKIP_TESTS=1 skips the unit tests (about 6 min) when they just passed; the design check still runs.
    [[ -n "${IVBEHAV_SKIP_TESTS:-}" ]] || run_step "$PY" -m pytest tests -q -p no:cacheprovider
    run_step "$PY" scripts/design_check.py $COMMON --n-jobs "$JOBS"
}
stage_simulation() { run_step "$PY" scripts/02_simulation.py --config "$CONFIG" --n-jobs "$JOBS"; }
stage_forecasts()  { run_step "$PY" scripts/01_forecasts.py $COMMON --n-jobs "$JOBS"; }
stage_behavior() {
    run_step "$PY" scripts/03_forecast_eval.py $COMMON
    run_step "$PY" scripts/04_decomposition.py $COMMON --n-jobs "$JOBS"
    run_step "$PY" scripts/05_response.py $COMMON --n-jobs "$JOBS"
    run_step "$PY" scripts/07_matching.py $COMMON --n-jobs "$JOBS"
}
stage_placebo()      { run_step "$PY" scripts/06_placebo.py $COMMON --n-jobs "$JOBS"; }
stage_refit()        { run_step "$PY" scripts/10_refit_bootstrap.py $COMMON --n-jobs "$JOBS"; }
stage_multiplicity() { run_step "$PY" scripts/09_multiplicity.py $COMMON --n-jobs "$JOBS"; }
stage_preonset()     { run_step "$PY" scripts/08_preonset.py $COMMON; }
stage_collect()      { run_step "$PY" scripts/11_collect_primary.py $COMMON; }
stage_sensitivity() {
    local specs
    specs=$("$PY" -c "import sys; sys.path.insert(0, 'src'); from ivbehav.config import load_config; print(' '.join(load_config('$CONFIG')['sensitivity_specs']))")
    for spec in $specs; do
        run_step "$PY" scripts/01_forecasts.py $COMMON --spec "$spec" --n-jobs "$JOBS"
        run_step "$PY" scripts/03_forecast_eval.py $COMMON --spec "$spec"
        run_step "$PY" scripts/04_decomposition.py $COMMON --spec "$spec" --n-jobs "$JOBS"
        run_step "$PY" scripts/05_response.py $COMMON --spec "$spec" --n-jobs "$JOBS" --no-shapley
        run_step "$PY" scripts/07_matching.py $COMMON --spec "$spec" --n-jobs "$JOBS"
    done
}
stage_core() {
    stage_check; stage_forecasts; stage_behavior; stage_placebo; stage_refit; stage_multiplicity; stage_collect
}
stage_all() {
    stage_simulation; stage_core; stage_preonset; stage_sensitivity
}

case "$STAGE" in
    check|simulation|forecasts|behavior|placebo|refit|multiplicity|preonset|collect|sensitivity|core|all)
        "stage_$STAGE" ;;
    *) echo "unknown stage: $STAGE"; sed -n '2,20p' "$0"; exit 2 ;;
esac
echo; echo "=== $(date '+%F %T') stage $STAGE done for $ASSET"
