"""Behavioral interpretability of information-set expansion in daily variance forecasts."""
import os

# One BLAS/OMP thread per process: parallelism is at the job level (joblib).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
