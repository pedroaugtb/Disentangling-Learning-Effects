#!/usr/bin/env python3
"""Sec. 9: intervals that include forecaster and projection-law estimation.

Each draw block-resamples every training window, refits f0, f1 and law A, and evaluates at the
same OOS origins.  `fit_only` keeps the OOS dates fixed; `combined` also block-resamples them.
Distances of the refits from the observed forecasts measure repeated-fit variability for P1.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from ivbehav.config import load_config, out_dir, table_dir, write_csv, write_json, write_manifest
from ivbehav.inference import boot_p_two_sided, mbb_weights, percentile_ci
from ivbehav.pipeline import hp_for_folds, load_tables, run_refit, setup
from ivbehav.stats import run_stats


def draw_job(cfg, asset, spec, family, draw, path, block):
    if path.exists():
        return
    data, folds, oos_pos = setup(cfg, asset, spec)
    hp = hp_for_folds(folds, load_tables(cfg, asset, spec, family))
    seed = cfg["seed"] + 100_000 + draw
    _, results = run_refit(cfg, data, folds, oos_pos, family, hp, seed, block=block)
    stat = run_stats(results, oos_pos, cfg)
    rng = np.random.default_rng([cfg["seed"], draw])
    combined = stat.compute(mbb_weights(len(oos_pos), cfg["bootstrap"]["primary_block"], rng))

    observed = pd.read_parquet(out_dir(cfg, asset, spec, "forecasts") / f"{family}.parquet").set_index("pos")
    origin = pd.concat([r["origin"] for r in results], ignore_index=True).set_index("pos")
    f0_obs, f1_obs = observed.loc[origin.index, "f0"], observed.loc[origin.index, "f1"]
    repeat = {"rms_self_f0": float(np.sqrt(np.mean((origin["f0"] - f0_obs) ** 2))),
              "rms_self_f1": float(np.sqrt(np.mean((origin["f1"] - f1_obs) ** 2))),
              "rms_self_f0_rel": float(np.sqrt(np.mean(((origin["f0"] - f0_obs) / f0_obs) ** 2))),
              "rms_self_f1_rel": float(np.sqrt(np.mean(((origin["f1"] - f1_obs) / f1_obs) ** 2)))}
    record = {"draw": draw, "fit_only": stat.compute(), "combined": combined, "repeat": repeat}
    if family == "har":
        record["coef"] = [r["coef"] for r in results]
    write_json(path, record)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--asset", default="SPX")
    ap.add_argument("--spec", default="primary")
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--n-jobs", type=int, default=24)
    args = ap.parse_args()

    cfg = load_config(args.config)
    r = cfg["refit_bootstrap"]
    tdir = table_dir(cfg, args.asset, args.spec)
    for family in args.families or cfg["core_families"]:
        d = out_dir(cfg, args.asset, args.spec, "refit_boot", family)
        n = r["n"][family]
        start = time.time()
        joblib.Parallel(n_jobs=args.n_jobs, verbose=5)(
            joblib.delayed(draw_job)(cfg, args.asset, args.spec, family, draw, d / f"b{draw:04d}.json", r["block"]) for draw in range(n))
        print(f"[{args.asset}/{args.spec}] refit bootstrap {family}: {time.time() - start:.0f}s", flush=True)
        records = [json.load(open(p)) for p in sorted(d.glob("b*.json"))]
        rows = []
        for source in ("fit_only", "combined", "repeat"):
            for key in sorted(records[0][source]):
                values = np.array([rec[source].get(key, np.nan) for rec in records], float)
                lo, hi = percentile_ci(values)
                rows.append({"source": source, "statistic": key, "n": len(values), "mean": float(np.nanmean(values)),
                             "sd": float(np.nanstd(values)), "ci_lo": lo, "ci_hi": hi, "q95": float(np.nanquantile(values, 0.95)),
                             "p_zero": boot_p_two_sided(values)})
        write_csv(tdir / f"refit_boot_{family}.csv", pd.DataFrame(rows))
        if family == "har":
            coef = pd.DataFrame([{"draw": rec["draw"], **c} for rec in records for c in rec["coef"]])
            write_csv(tdir / "refit_boot_har_coefs.csv",
                      coef.drop(columns="draw").groupby("fold").agg(["mean", "std"]).pipe(
                          lambda t: t.set_axis(["_".join(c) for c in t.columns], axis=1)).reset_index())
    write_manifest(cfg, tdir, "10_refit_bootstrap")


if __name__ == "__main__":
    main()
