#!/usr/bin/env python3
"""WP4: conditional IV surprises and negative controls.

Every placebo refits f1 with the observed f1 complexity on a signal that is independent of
future Y given X (or misaligned in time), and goes through the same evaluation path as the
observed fit.  Run 04, 05 and 07 first: the observed statistics come from their outputs.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from ivbehav.config import load_config, out_dir, table_dir, write_csv, write_json, write_manifest
from ivbehav.inference import placebo_p
from ivbehav.models import qlike
from ivbehav.pipeline import run_placebo, setup
from ivbehav.stats import RunStats


def placebo_job(cfg, asset, spec, family, kind, draw, path):
    if path.exists():
        return
    write_json(path, run_placebo(cfg, asset, spec, family, kind, draw))


def observed_stats(cfg, asset, spec, family, oos_pos) -> dict:
    bdir = out_dir(cfg, asset, spec, "behavior", family)
    origin = pd.read_parquet(bdir / "origin.parquet")
    response = None
    if (bdir / "response.parquet").exists():
        response = pd.read_parquet(bdir / "response.parquet")
        response = response[response["feature"].isin(cfg["response"]["primary_features"])]
    pairs = pd.read_parquet(bdir / "pairs.parquet")
    return RunStats(oos_pos, cfg, origin, response, pairs).point()


def surprise_strata(origin: pd.DataFrame) -> pd.DataFrame:
    """Surprise quantile strata from the training law's PIT, so no OOS information sets the cuts."""
    f = origin[origin["sup_xv"]].copy()
    f["stratum"] = pd.cut(f["pit_a"], [-0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
                          labels=["q00-10", "q10-25", "q25-50", "q50-75", "q75-90", "q90-100"])
    f["dq"] = qlike(f["y"], f["f1"]) - qlike(f["y"], f["f0"])
    f["err0_rel"] = (f["y"] - f["f0"]) / f["f0"]
    f["isd_rel"] = f["ISD_a"] / f["f0"]
    return (f.groupby("stratum", observed=True)
            .agg(n=("y", "size"), mean_eps=("eps_a", "mean"), mean_ISD=("ISD_a", "mean"), mean_ISD_rel=("isd_rel", "mean"),
                 mean_err0_rel=("err0_rel", "mean"), mean_dqlike=("dq", "mean"), mean_y=("y", "mean"), mean_f0=("f0", "mean"),
                 mean_f1=("f1", "mean"))
            .reset_index())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--asset", default="SPX")
    ap.add_argument("--spec", default="primary")
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--n-jobs", type=int, default=24)
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = cfg["placebo"]
    _, _, oos_pos = setup(cfg, args.asset, args.spec)
    tdir = table_dir(cfg, args.asset, args.spec)
    for family in args.families or cfg["core_families"]:
        jobs = []
        for kind, n_draws in p["n_draws"].items():
            if not n_draws:
                continue
            d = out_dir(cfg, args.asset, args.spec, "placebo", family, kind)
            jobs += [(kind, draw, d / f"r{draw:04d}.json") for draw in range(n_draws)]
        for lag in p["displaced_lags"]:
            d = out_dir(cfg, args.asset, args.spec, "placebo", family, f"displaced_{lag}")
            jobs.append((f"displaced_{lag}", 0, d / "r0000.json"))
        start = time.time()
        joblib.Parallel(n_jobs=args.n_jobs, verbose=5)(
            joblib.delayed(placebo_job)(cfg, args.asset, args.spec, family, kind, draw, path) for kind, draw, path in jobs)
        print(f"[{args.asset}/{args.spec}] placebo {family}: {time.time() - start:.0f}s", flush=True)

        observed = observed_stats(cfg, args.asset, args.spec, family, oos_pos)
        rows = []
        for kind_dir in sorted(out_dir(cfg, args.asset, args.spec, "placebo", family).iterdir()):
            draws = [json.load(open(path)) for path in sorted(kind_dir.glob("r*.json"))]
            if not draws:
                continue
            for key in sorted(observed):
                values = np.array([d.get(key, np.nan) for d in draws], float)
                if not np.isfinite(values).any():
                    continue
                rows.append({"kind": kind_dir.name, "statistic": key, "observed": observed[key], "n_draws": len(draws),
                             "placebo_mean": float(np.nanmean(values)), "placebo_sd": float(np.nanstd(values)),
                             "placebo_q05": float(np.nanquantile(values, 0.05)), "placebo_q95": float(np.nanquantile(values, 0.95)),
                             "p_upper": placebo_p(observed[key], values, "upper"), "p_lower": placebo_p(observed[key], values, "lower"),
                             "p_abs": placebo_p(observed[key], values, "abs")})
        write_csv(tdir / f"placebo_{family}.csv", pd.DataFrame(rows))
        origin = pd.read_parquet(out_dir(cfg, args.asset, args.spec, "behavior", family) / "origin.parquet")
        write_csv(tdir / f"surprise_strata_{family}.csv", surprise_strata(origin))
    write_manifest(cfg, tdir, "06_placebo", n_draws=p["n_draws"])


if __name__ == "__main__":
    main()
