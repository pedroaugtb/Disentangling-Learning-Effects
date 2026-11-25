#!/usr/bin/env python3
"""Design stage, outcome-blind: support and matched-pair counts from (X, V) alone.

Fits no forecaster and reads no label value, so it can run on the test period before any
result exists; matching calipers are judged here, in the spirit of design before outcomes.
"""
import _bootstrap  # noqa: F401
import argparse

import joblib
import numpy as np
import pandas as pd

from ivbehav.config import load_config, table_dir, write_csv
from ivbehav.core import match_rules, select_pairs
from ivbehav.pipeline import fold_laws, setup


def fold_job(data, fold, cfg, oos_pos):
    laws = fold_laws(data, fold, cfg)
    law_a, law_b, support = laws
    te = fold.test_idx
    c, m = cfg["conditional"], cfg["matching"]
    rules = match_rules(data.X[fold.train_idx], law_a, support, cfg)
    neff = law_b.n_eff(data.X[te])
    knn_ok = support.in_support(data.X[te])
    pit = law_a.pit(data.X[te], data.V[te])
    pit_ok = (pit >= c["pit_lo"]) & (pit <= c["pit_hi"])

    max_date = data.dates[te[-1]] + pd.DateOffset(years=m["max_gap_years"])
    partners = oos_pos[(data.dates[oos_pos] > data.dates[te[0]]) & (data.dates[oos_pos] <= max_date)]
    n_pairs, anchors_with_pair = 0, 0
    partner_counts = np.zeros(0)
    if len(partners):
        usable = lambda pos: support.in_support(data.X[pos]) & (lambda p: (p >= c["pit_lo"]) & (p <= c["pit_hi"]))(law_a.pit(data.X[pos], data.V[pos]))
        a, p = te[usable(te)], partners[usable(partners)]
        if len(a) and len(p):
            max_dates = data.dates[a].values + np.timedelta64(int(m["max_gap_years"] * 366), "D")
            allowed = ((data.dates[p].values[None, :] > data.label_end[a].values[:, None])
                       & (data.dates[p].values[None, :] <= max_dates[:, None]))
            chosen = select_pairs(support.basis_.z(data.X[a]), law_a.standardized(data.X[a], data.V[a]),
                                  support.basis_.z(data.X[p]), law_a.standardized(data.X[p], data.V[p]),
                                  allowed, rules["caliper"], rules["delta"], m["max_partners"])
            n_pairs = len(chosen)
            anchors_with_pair = len({x[0] for x in chosen})
    return {"fold": fold.k, "fit_date": fold.fit_date, "n_train": len(fold.train_idx), "n_test": len(te),
            "share_sup_x": knn_ok.mean(), "share_sup_b": (knn_ok & (neff >= c["min_neff"])).mean(),
            "share_pit_ok": pit_ok.mean(),
            "median_neff": float(np.median(neff)), "bandwidth": law_b.h_,
            "caliper": rules["caliper"], "delta": rules["delta"], "n_pairs": n_pairs, "anchors_with_pair": anchors_with_pair}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--asset", default="SPX")
    ap.add_argument("--spec", default="primary")
    ap.add_argument("--n-jobs", type=int, default=24)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data, folds, oos_pos = setup(cfg, args.asset, args.spec)
    data.y = np.where(np.isfinite(data.y), 1.0, np.nan)      # label values stay out of this script
    rows = joblib.Parallel(n_jobs=args.n_jobs)(joblib.delayed(fold_job)(data, f, cfg, oos_pos) for f in folds)
    frame = pd.DataFrame(rows)
    write_csv(table_dir(cfg, args.asset, args.spec) / "design_check.csv", frame)
    print(f"folds={len(frame)}  origins={frame.n_test.sum()}  train n: {frame.n_train.min()}-{frame.n_train.max()}")
    print("mean share of origins  sup_x={:.3f}  sup_b={:.3f}  pit_ok={:.3f}".format(
        np.average(frame.share_sup_x, weights=frame.n_test), np.average(frame.share_sup_b, weights=frame.n_test),
        np.average(frame.share_pit_ok, weights=frame.n_test)))
    print(f"pairs total={frame.n_pairs.sum()}  anchors with a pair={frame.anchors_with_pair.sum()} of {frame.n_test.sum()}")
    print(f"caliper {frame.caliper.min():.2f}-{frame.caliper.max():.2f}  delta {frame.delta.min():.2f}-{frame.delta.max():.2f}  "
          f"bandwidth {frame.bandwidth.min():.2f}-{frame.bandwidth.max():.2f}  median neff {frame.median_neff.median():.0f}")
    by_year = frame.assign(year=pd.to_datetime(frame.fit_date).dt.year).groupby("year").agg(
        origins=("n_test", "sum"), sup_x=("share_sup_x", "mean"), sup_b=("share_sup_b", "mean"), pit_ok=("share_pit_ok", "mean"), pairs=("n_pairs", "sum"), anchors=("anchors_with_pair", "sum"))
    print(by_year.round(3).to_string())


if __name__ == "__main__":
    main()
