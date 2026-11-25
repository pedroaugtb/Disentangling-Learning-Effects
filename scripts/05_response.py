#!/usr/bin/env python3
"""WP1 and WP3: historical response functions, interactions, and dependence-aware reliance."""
import _bootstrap  # noqa: F401
import argparse

import joblib
import numpy as np
import pandas as pd

from ivbehav.conditional import fit_laws
from ivbehav.config import load_config, out_dir, table_dir, write_csv, write_json, write_manifest, write_parquet
from ivbehav.core import ale2d_rows, response_rows
from ivbehav.data import HIST_COLS
from ivbehav.explain import conditional_group_shapley, conditional_reliance
from ivbehav.models import _HAR_IDX
from ivbehav.models import SEQUENCE_FAMILIES
from ivbehav.pipeline import load_pair, setup
from ivbehav.stats import ResponseStats, bootstrap_table


def fold_job(data, fold, cfg, asset, spec, family, with_shapley):
    pair = load_pair(cfg, asset, spec, family, fold.k)
    m0, m1 = pair["m0"], pair["m1"]
    tr, te = fold.train_idx, fold.test_idx
    laws = fit_laws(data.X[tr], data.V[tr], cfg)
    laws = (laws[0], None, laws[2])
    r = cfg["response"]
    out = {
        "response": response_rows(data, fold, m0, m1, laws, cfg, r["primary_features"] + r["secondary_features"]),
        "ale2d": ale2d_rows(data, fold, m0, m1, cfg),
        "reliance": conditional_reliance(data, fold, m0, m1, cfg, np.random.default_rng([cfg["seed"], fold.k])),
    }
    if with_shapley:
        positions = te[np.isin(te, data.shapley_positions)]
        out["shapley"] = pd.concat([conditional_group_shapley(data, fold, m0, positions, cfg, False, "f0"),
                                    conditional_group_shapley(data, fold, m1, positions, cfg, True, "f1")], ignore_index=True)
    if family in ("har", "har_ols"):
        # Omitted-variable accounting of the HAR shift: b0 = b1 + b_V * gamma, exact under OLS.
        A = np.column_stack([np.ones(len(tr)), data.X[tr][:, _HAR_IDX]])
        gamma = np.linalg.lstsq(A, data.V[tr], rcond=None)[0][1:]
        c0, c1 = m0.coef_raw(), m1.coef_raw()
        s0, s1 = m0.coef_std(), m1.coef_std()
        out["coef"] = pd.DataFrame([{"fold": fold.k, "fit_date": fold.fit_date, "term": name, "b0": c0[name],
                                     "b1": c1[name], "b0_std": s0[name], "b1_std": s1[name], "b_V": c1["V"],
                                     "b_V_std": s1["V"], "gamma_V_on_term": gamma[j],
                                     "ovb_prediction": c1[name] + c1["V"] * gamma[j]}
                                    for j, name in enumerate(c0)])
    return out


def ale2d_table(frame: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Double accumulation of mean second differences, with main effects removed."""
    rows = []
    kept = frame[frame["keep"]]
    for pair, g in kept.groupby("pair"):
        for col in [c for c in g.columns if c.startswith("d2_") and g[c].notna().any()]:
            mean = g.groupby(["bin_a", "bin_b"])[col].mean().unstack().reindex(index=range(n_bins), columns=range(n_bins)).fillna(0.0)
            count = g.groupby(["bin_a", "bin_b"]).size().unstack().reindex(index=range(n_bins), columns=range(n_bins)).fillna(0.0)
            acc = mean.to_numpy().cumsum(axis=0).cumsum(axis=1)
            w = count.to_numpy()
            total = max(w.sum(), 1.0)
            row_mean = (acc * w).sum(1, keepdims=True) / np.maximum(w.sum(1, keepdims=True), 1.0)
            col_mean = (acc * w).sum(0, keepdims=True) / np.maximum(w.sum(0, keepdims=True), 1.0)
            pure = acc - row_mean - col_mean + (acc * w).sum() / total
            for a in range(n_bins):
                for b in range(n_bins):
                    rows.append({"pair": pair, "function": col[3:], "bin_a": a, "bin_b": b, "ale2": float(pure[a, b]), "count": float(w[a, b])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--asset", default="SPX")
    ap.add_argument("--spec", default="primary")
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--no-shapley", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data, folds, oos_pos = setup(cfg, args.asset, args.spec)
    data.shapley_positions = oos_pos[:: cfg["response"]["shapley_every"]]
    tdir = table_dir(cfg, args.asset, args.spec)
    for family in args.families or cfg["core_families"]:
        if family in SEQUENCE_FAMILIES:
            print(f"[{args.asset}/{args.spec}] response {family}: skipped (reads the return path, not the summaries)")
            continue
        results = joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(fold_job)(data, f, cfg, args.asset, args.spec, family, not args.no_shapley) for f in folds)
        bdir = out_dir(cfg, args.asset, args.spec, "behavior", family)
        tables = {k: pd.concat([r[k] for r in results if k in r], ignore_index=True) for k in results[0]}
        for name, frame in tables.items():
            write_parquet(bdir / f"{name}.parquet", frame)

        primary = tables["response"]["feature"].isin(cfg["response"]["primary_features"])
        stat = ResponseStats(tables["response"][primary], oos_pos, cfg)
        write_json(bdir / "response_point.json", stat.compute())
        write_csv(tdir / f"response_boot_{family}.csv", bootstrap_table(stat, len(oos_pos), cfg, cfg["seed"]))
        # WP1: the same responses in calm and stressed states (training rank of the 22-day variance).
        states = {"calm": tables["response"]["rv_rank"] <= 1 / 3, "stressed": tables["response"]["rv_rank"] >= 2 / 3}
        write_json(tdir / f"response_by_state_{family}.json",
                   {name: ResponseStats(tables["response"][mask], oos_pos, cfg).compute() for name, mask in states.items()})
        full = ResponseStats(tables["response"], oos_pos, cfg)
        write_csv(tdir / f"response_curves_{family}.csv", full.curves())
        write_json(tdir / f"response_withheld_{family}.json", full.withheld)
        write_csv(tdir / f"ale2d_{family}.csv", ale2d_table(tables["ale2d"], cfg["response"]["ale2d_bins"]))
        rel = tables["reliance"]
        write_csv(tdir / f"reliance_{family}.csv", rel.groupby(["function", "group"])[["dqlike", "dse"]].mean().reset_index())
        if "shapley" in tables:
            phi = tables["shapley"].assign(abs_phi=lambda d: d["phi"].abs())
            write_csv(tdir / f"shapley_{family}.csv", phi.groupby(["function", "group"])[["phi", "abs_phi"]].mean().reset_index())
        if "coef" in tables:
            write_csv(tdir / f"har_coefs_{family}.csv", tables["coef"])
        print(f"[{args.asset}/{args.spec}] response {family}: done", flush=True)
    write_manifest(cfg, tdir, "05_response")


if __name__ == "__main__":
    main()
