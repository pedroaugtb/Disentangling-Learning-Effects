#!/usr/bin/env python3
"""WP5: matched-history state differentiation, the outcome validation behind P3."""
import _bootstrap  # noqa: F401
import argparse

import joblib
import numpy as np
import pandas as pd

from ivbehav.conditional import fit_laws
from ivbehav.config import load_config, out_dir, table_dir, write_csv, write_json, write_manifest, write_parquet
from ivbehav.core import pair_rows
from ivbehav.pipeline import load_pair, setup
from ivbehav.stats import PairStats, bootstrap_table


def fold_job(data, fold, cfg, asset, spec, family, oos_pos, k_caliper=None):
    pair = load_pair(cfg, asset, spec, family, fold.k)
    tr = fold.train_idx
    laws = fit_laws(data.X[tr], data.V[tr], cfg)
    return pair_rows(data, fold, pair["m0"], pair["m1"], (laws[0], None, laws[2]), cfg, oos_pos, k_caliper=k_caliper)


def stitch_pairs(frames):
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--asset", default="SPX")
    ap.add_argument("--spec", default="primary")
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--n-jobs", type=int, default=24)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data, folds, oos_pos = setup(cfg, args.asset, args.spec)
    tdir = table_dir(cfg, args.asset, args.spec)
    for family in args.families or cfg["core_families"]:
        frames = joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(fold_job)(data, f, cfg, args.asset, args.spec, family, oos_pos) for f in folds)
        pairs = stitch_pairs(frames)
        bdir = out_dir(cfg, args.asset, args.spec, "behavior", family)
        write_parquet(bdir / "pairs.parquet", pairs)
        stat = PairStats(pairs, oos_pos)
        point = {**stat.compute(), **stat.design()}
        write_json(bdir / "pairs_point.json", point)
        print(f"[{args.asset}/{args.spec}] matching {family}: {point.get('pairs:n', 0):.0f} pairs on "
              f"{point.get('pairs:n_dates', 0):.0f} dates", flush=True)
        if stat.empty:
            continue
        write_csv(tdir / f"pairs_boot_{family}.csv", bootstrap_table(stat, len(oos_pos), cfg, cfg["seed"]))
        # Influence check: the same statistics without pairs touching the COVID shock.
        hi_d, lo_d = data.dates[pairs["hi"]], data.dates[pairs["lo"]]
        start, end = np.datetime64("2020-02-15"), np.datetime64("2020-04-30")
        mask = ~(((hi_d >= start) & (hi_d <= end)) | ((lo_d >= start) & (lo_d <= end)))
        ex = PairStats(pairs[mask].reset_index(drop=True), oos_pos)
        write_json(bdir / "pairs_point_ex_covid.json", {**ex.compute(), **ex.design()})
        write_csv(tdir / f"pairs_boot_ex_covid_{family}.csv", bootstrap_table(ex, len(oos_pos), cfg, cfg["seed"]))
        # Caliper sensitivity: the balance-versus-count trade-off around the primary rule.
        for k in cfg["matching"]["k_caliper_sensitivity"]:
            alt = stitch_pairs(joblib.Parallel(n_jobs=args.n_jobs)(
                joblib.delayed(fold_job)(data, f, cfg, args.asset, args.spec, family, oos_pos, k) for f in folds))
            alt_stat = PairStats(alt, oos_pos)
            write_json(bdir / f"pairs_point_k{k}.json", {**alt_stat.compute(), **alt_stat.design()})
            if not alt_stat.empty:
                write_csv(tdir / f"pairs_boot_k{k}_{family}.csv", bootstrap_table(alt_stat, len(oos_pos), cfg, cfg["seed"]))
        by_fold = pairs.groupby("fold").agg(n=("hi", "size"), caliper=("caliper", "first"), delta=("delta", "first"), dist=("dist", "mean"))
        write_csv(tdir / f"pairs_design_{family}.csv", by_fold.reset_index())
    write_manifest(cfg, tdir, "07_matching")


if __name__ == "__main__":
    main()
