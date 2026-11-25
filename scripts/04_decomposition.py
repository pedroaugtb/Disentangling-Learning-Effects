#!/usr/bin/env python3
"""WP2: paired displacement, HFS and ISD at OOS origins, with conditional-law diagnostics."""
import _bootstrap  # noqa: F401
import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kstest

from ivbehav.conditional import fit_laws, weighted_crps
from ivbehav.config import load_config, out_dir, table_dir, write_csv, write_json, write_manifest, write_parquet
from ivbehav.core import origin_rows
from ivbehav.models import qlike
from ivbehav.pipeline import load_pair, setup
from ivbehav.stats import OriginStats, bootstrap_table


def fold_job(data, fold, cfg, asset, spec, family):
    pair = load_pair(cfg, asset, spec, family, fold.k)
    tr, te = fold.train_idx, fold.test_idx
    laws = fit_laws(data.X[tr], data.V[tr], cfg)
    origin = origin_rows(data, fold, pair["m0"], pair["m1"], laws, cfg)

    X, logv = data.X[te], np.log(data.V[te])
    va, wa = laws[0].draws(X)
    vb, wb = laws[1].draws(X)
    marg = np.log(np.repeat(data.V[tr][None, :], len(te), axis=0))
    origin["crps_a"] = weighted_crps(np.log(va), wa, logv)
    origin["crps_b"] = weighted_crps(np.log(vb), wb, logv)
    origin["crps_marginal"] = weighted_crps(marg, np.full(marg.shape, 1.0 / marg.shape[1]), logv)
    return origin


def law_table(origin: pd.DataFrame) -> pd.DataFrame:
    """Credibility of the conditional laws on held-out origins, pooled and by year."""
    rows = []
    groups = [("all", origin)] + [(str(y), g) for y, g in origin.groupby(origin["date"].dt.year)]
    for label, g in groups:
        for law, pit_col, crps_col in (("A_locscale", "pit_a", "crps_a"), ("B_kernel", "pit_b", "crps_b")):
            pit = g[pit_col].to_numpy()
            row = {"sample": label, "law": law, "n": len(g), "ks_stat": float(kstest(pit, "uniform").statistic),
                   "crps": float(g[crps_col].mean()), "crps_marginal": float(g["crps_marginal"].mean()),
                   "crps_skill": float(1 - g[crps_col].mean() / g["crps_marginal"].mean())}
            for level in (0.5, 0.8, 0.9, 0.98):
                row[f"cov{int(level * 100)}"] = float(np.mean(np.abs(pit - 0.5) <= level / 2))
            rows.append(row)
    return pd.DataFrame(rows)


def state_table(origin: pd.DataFrame) -> pd.DataFrame:
    """D / HFS / ISD by historical variance rank and recent-return sign, supported states only."""
    f = origin[origin["sup_x"]].copy()
    f["HFS_b"], f["ISD_b"] = f["HFS_b"].where(f["sup_b"]), f["ISD_b"].where(f["sup_b"])
    f["dq"] = qlike(f["y"], f["f1"]) - qlike(f["y"], f["f0"])
    f["rv_state"] = pd.cut(f["rv_rank"], [-0.001, 1 / 3, 2 / 3, 1.0001], labels=["low", "mid", "high"])
    f["ret_sign"] = np.where(f["ret_5"] < 0, "down", "up")
    f["rel_D"], f["rel_HFS"], f["rel_ISD"] = f["D"] / f["f0"], f["HFS_a"] / f["f0"], f["ISD_a"] / f["f0"]
    return (f.groupby(["rv_state", "ret_sign"], observed=True)
            .agg(n=("D", "size"), D=("D", "mean"), HFS=("HFS_a", "mean"), ISD=("ISD_a", "mean"),
                 HFS_b=("HFS_b", "mean"), ISD_b=("ISD_b", "mean"), rel_D=("rel_D", "mean"), rel_HFS=("rel_HFS", "mean"),
                 rel_ISD=("rel_ISD", "mean"), rms_rel_ISD=("rel_ISD", lambda s: float(np.sqrt(np.mean(s**2)))),
                 mean_u=("u_a", "mean"), dqlike=("dq", "mean"))
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
    data, folds, oos_pos = setup(cfg, args.asset, args.spec)
    tdir = table_dir(cfg, args.asset, args.spec)
    for family in args.families or cfg["core_families"]:
        origin = pd.concat(joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(fold_job)(data, f, cfg, args.asset, args.spec, family) for f in folds), ignore_index=True)
        bdir = out_dir(cfg, args.asset, args.spec, "behavior", family)
        write_parquet(bdir / "origin.parquet", origin)
        stat = OriginStats(origin, oos_pos)
        write_json(bdir / "origin_point.json", {**stat.compute(), **stat.unweighted_extras()})
        write_csv(tdir / f"decomp_boot_{family}.csv", bootstrap_table(stat, len(oos_pos), cfg, cfg["seed"]))
        write_csv(tdir / f"law_diagnostics_{family}.csv", law_table(origin))
        write_csv(tdir / f"decomp_states_{family}.csv", state_table(origin))
        print(f"[{args.asset}/{args.spec}] decomposition {family}: {len(origin)} origins, "
              f"{origin['sup_x'].mean():.1%} supported in x", flush=True)
    write_manifest(cfg, tdir, "04_decomposition")


if __name__ == "__main__":
    main()
