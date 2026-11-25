#!/usr/bin/env python3
"""Stage 2: Monte Carlo audit of the diagnostics (proposal, Sec. 8).

Reads only pre-out-of-sample SPX rows, to calibrate the DGP.
"""
import _bootstrap  # noqa: F401
import argparse
import dataclasses
import time

import joblib
import numpy as np
import pandas as pd

from ivbehav.config import RESULTS_DIR, check_config_stamp, load_config, write_csv, write_json, write_manifest, write_parquet
from ivbehav.data import load_spec_data
from ivbehav.simulation import calibrate, placebo_gate_replicate, replicate


def designs(cfg):
    s = cfg["simulation"]
    return [("redundant", 0.0)] + [(m, b) for m in ("constant", "state") for b in s["informative_b"]]


def summarize_main(reps: pd.DataFrame) -> pd.DataFrame:
    keys = ["mode", "b_bar", "learner", "law"]
    means = ["rms_D", "rms_HFS", "mean_HFS", "rms_fit_contrast", "rms_proj_err", "rms_ISD", "mean_ISD",
             "rmse_ISD_vs_true", "true_rms_ISD", "spearman_absu_absISD", "calib_slope_rel", "dqlike", "dmse",
             "pop_dqlike_fit", "ideal_dmse", "ideal_dmse_identity", "ideal_dqlike", "ideal_dqlike_identity", "ideal_rms_isd",
             "share_supported", "law_cov80", "share_floored_f1", "pairs_n", "pairs:conc0", "pairs:conc1",
             "pairs:conc_diff", "pairs:sep_slope_rel", "pairs:dqlike", "pairs:mean_dlogY"]
    grouped = reps.groupby(keys, sort=False)
    out = grouped[means].mean()
    for col in ("rms_HFS", "rms_ISD", "calib_slope_rel", "dqlike", "pairs:sep_slope_rel", "pairs:conc_diff"):
        out[f"{col}_mcsd"] = grouped[col].std()
    for col in ("calib_slope_rel", "pairs:sep_slope_rel"):
        out[f"{col}_median"] = grouped[col].median()     # a nearly unused V makes means erratic
    out["identity_residual_max"] = grouped["identity_residual"].max()
    out["dq_ci_coverage"] = grouped["dq_ci_covers_pop"].mean()
    # Detection in the direction a claim would need (docs/PREANALYSIS.md): false detection under
    # the redundant design, power under the others.  A slope undefined because V went unused
    # counts as no detection.
    def rate(mask: pd.Series) -> pd.Series:
        return mask.groupby([reps[k] for k in keys], sort=False).mean()

    out["detect_dqlike_neg"] = rate((reps["dq_p_zero"] < 0.05) & (reps["dqlike"] < 0))
    out["detect_calib_pos"] = rate((reps["calib_p_zero"] < 0.05) & (reps["calib_slope_rel"] > 0))
    out["detect_pairs_slope_pos"] = rate((reps["pairs:sep_slope_rel_p"] < 0.05) & (reps["pairs:sep_slope_rel"] > 0))
    out["detect_pairs_dqlike_neg"] = rate((reps["pairs:dqlike_p"] < 0.05) & (reps["pairs:dqlike"] < 0))
    out["detect_pairs_conc_pos"] = rate((reps["pairs:conc_diff_p"] < 0.05) & (reps["pairs:conc_diff"] > 0))
    out["n_reps"] = grouped.size()
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--part", choices=["main", "gate", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    check_config_stamp(cfg)
    s = cfg["simulation"]
    directory = RESULTS_DIR / cfg["profile"] / "simulation"
    directory.mkdir(parents=True, exist_ok=True)
    # Calibration needs the primary panel; the dev profile reads its pre-2017 copy of the same rows.
    data = load_spec_data(cfg, "SPX", "primary")
    cal = calibrate(data, cfg["sample"]["oos_start"])
    write_json(directory / "calibration.json", dataclasses.asdict(cal) | {k: np.asarray(v).tolist() for k, v in dataclasses.asdict(cal).items()})

    if args.part in ("main", "all"):
        frames = []
        for mode, b_bar in designs(cfg):
            path = directory / f"reps_{mode}_b{b_bar:g}.parquet"
            if args.force or not path.exists():
                start = time.time()
                rows = joblib.Parallel(n_jobs=args.n_jobs)(
                    joblib.delayed(replicate)(rep, mode, b_bar, cal, cfg, cfg["seed"]) for rep in range(s["n_reps"]))
                write_parquet(path, pd.DataFrame([r for chunk in rows for r in chunk]))
                print(f"main {mode} b={b_bar}: {time.time() - start:.0f}s", flush=True)
            frames.append(pd.read_parquet(path))
        write_csv(directory / "summary_main.csv", summarize_main(pd.concat(frames, ignore_index=True)))

    if args.part in ("gate", "all"):
        frames = []
        for mode, b_bar in designs(cfg):
            path = directory / f"gate_{mode}_b{b_bar:g}.parquet"
            if args.force or not path.exists():
                start = time.time()
                rows = joblib.Parallel(n_jobs=args.n_jobs)(
                    joblib.delayed(placebo_gate_replicate)(rep, mode, b_bar, cal, cfg, cfg["seed"])
                    for rep in range(s["placebo_reps"]))
                write_parquet(path, pd.DataFrame([r for chunk in rows for r in chunk]))
                print(f"gate {mode} b={b_bar}: {time.time() - start:.0f}s", flush=True)
            frames.append(pd.read_parquet(path))
        gate = pd.concat(frames, ignore_index=True)
        summary = gate.groupby(["mode", "b_bar", "learner", "statistic"], sort=False).agg(
            observed=("observed", "mean"), placebo_mean=("placebo_mean", "mean"),
            detect=("p_upper", lambda p: float((p <= 0.05).mean())), n_reps=("rep", "size")).reset_index()
        write_csv(directory / "summary_gate.csv", summary)
    write_manifest(cfg, directory, "02_simulation")


if __name__ == "__main__":
    main()
