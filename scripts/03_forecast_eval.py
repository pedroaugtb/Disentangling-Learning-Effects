#!/usr/bin/env python3
"""Validity checks only: forecast accuracy never organises the behavioral conclusions."""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from ivbehav.config import load_config, out_dir, table_dir, write_csv, write_manifest
from ivbehav.inference import block_bootstrap, boot_p_two_sided, dm_test, gw_test, hac_slope, percentile_ci
from ivbehav.models import qlike, se
from ivbehav.pipeline import setup


def scores(y, f, label) -> dict:
    return {"model": label, "n": len(y), "qlike": float(np.mean(qlike(y, f))), "mse": float(np.mean(se(y, f))),
            "mae": float(np.mean(np.abs(y - f))), "level_ratio": float(f.mean() / y.mean()),
            "mean_y_over_f": float(np.mean(y / f))}


def mincer_zarnowitz(y, f, lag) -> dict:
    A = np.column_stack([np.ones(len(f)), f])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return {"mz_intercept": float(beta[0]), "mz_slope": float(beta[1]),
            "mz_slope_se": hac_slope(y, f, lag, intercept=True)["se"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--asset", default="SPX")
    ap.add_argument("--spec", default="primary")
    args = ap.parse_args()

    cfg = load_config(args.config)
    b = cfg["bootstrap"]
    fdir = out_dir(cfg, args.asset, args.spec, "forecasts")
    data, _, _ = setup(cfg, args.asset, args.spec)
    rows, tests, yearly, reliability = [], [], [], []
    for path in sorted(fdir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        family = path.stem
        y = frame["y"].to_numpy()
        if family.startswith("bench_"):
            rows.append({"family": family, **scores(y, frame["f"].to_numpy(), "bench")})
            continue
        f0, f1 = frame["f0"].to_numpy(), frame["f1"].to_numpy()
        for label, f in (("f0", f0), ("f1", f1)):
            rows.append({"family": family, **scores(y, f, label), **mincer_zarnowitz(y, f, 21)})
            # Model-free reliability: realised mean within forecast deciles.
            bins = pd.qcut(f, 10, labels=False, duplicates="drop")
            for decile in np.unique(bins):
                m = bins == decile
                reliability.append({"family": family, "model": label, "decile": int(decile), "n": int(m.sum()),
                                    "mean_forecast": float(f[m].mean()), "mean_y": float(y[m].mean())})
        pos = frame["pos"].to_numpy()
        rv = np.log(data.X[pos, 2])
        crisis = (frame["date"] >= "2020-02-15") & (frame["date"] <= "2020-04-30")
        for loss_name, loss in (("qlike", qlike), ("mse", se)):
            d = loss(y, f1) - loss(y, f0)
            for sample, mask in (("all", np.ones(len(d), bool)), ("ex_covid", ~crisis.to_numpy())):
                dm = d[mask]
                for lag in b["hac_lags"]:
                    out = dm_test(dm, horizon=22, lag=lag)
                    H = np.column_stack([np.ones(mask.sum() - 1), dm[:-1], rv[mask][1:]])
                    tests.append({"family": family, "loss": loss_name, "sample": sample, "lag": lag,
                                  "mean_d": out["mean_diff"], "dm": out["dm"], "dm_p": out["p"],
                                  "gw_p": gw_test(dm[1:], H, lag)["p"]})
                boot = block_bootstrap(lambda w: {"v": float((w * dm).sum() / w.sum())}, len(dm),
                                       b["primary_block"], b["n"], cfg["seed"])["v"]
                lo, hi = percentile_ci(boot)
                tests.append({"family": family, "loss": loss_name, "sample": sample, "lag": -1, "mean_d": float(dm.mean()),
                              "boot_ci_lo": lo, "boot_ci_hi": hi, "boot_p": boot_p_two_sided(boot)})
        years = frame["date"].dt.year
        for year in sorted(years.unique()):
            m = (years == year).to_numpy()
            yearly.append({"family": family, "year": int(year), "n": int(m.sum()),
                           "qlike0": float(qlike(y[m], f0[m]).mean()), "qlike1": float(qlike(y[m], f1[m]).mean()),
                           "dqlike": float((qlike(y[m], f1[m]) - qlike(y[m], f0[m])).mean())})
    tdir = table_dir(cfg, args.asset, args.spec)
    write_csv(tdir / "forecast_scores.csv", pd.DataFrame(rows))
    write_csv(tdir / "forecast_tests.csv", pd.DataFrame(tests))
    write_csv(tdir / "forecast_by_year.csv", pd.DataFrame(yearly))
    write_csv(tdir / "forecast_reliability.csv", pd.DataFrame(reliability))
    write_manifest(cfg, tdir, "03_forecast_eval")


if __name__ == "__main__":
    main()
