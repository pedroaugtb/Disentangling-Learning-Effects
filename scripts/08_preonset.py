#!/usr/bin/env python3
"""WP6 (exploratory): forecast paths before large daily-variance onsets, at common alert rates.

Shock thresholds and alert levels come from each fold's training segment.  Onsets are defined
retrospectively, so this describes forecast paths and never measures a tradable warning.
"""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from ivbehav.config import load_config, out_dir, table_dir, write_csv, write_manifest
from ivbehav.conditional import LawBasis
from ivbehav.pipeline import load_pair, setup


def find_onsets(data, folds, cfg) -> pd.DataFrame:
    """A session is an onset if r^2 exceeds the training q99 after a quiet refractory span."""
    p = cfg["preonset"]
    rows = []
    last_shock = -10**9
    fold_of = {}
    for f in folds:
        threshold = np.quantile(data.ret[f.train_idx] ** 2, p["shock_quantile"])
        for pos in f.test_idx:
            fold_of[pos] = (f.k, threshold)
    for pos in sorted(fold_of):
        k, threshold = fold_of[pos]
        if data.ret[pos] ** 2 > threshold:
            if pos - last_shock > p["refractory"]:
                rows.append({"pos": pos, "date": data.dates[pos], "fold": k, "r": data.ret[pos]})
            last_shock = pos
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--asset", default="SPX")
    ap.add_argument("--spec", default="primary")
    ap.add_argument("--families", nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = cfg["preonset"]
    data, folds, oos_pos = setup(cfg, args.asset, args.spec)
    onsets = find_onsets(data, folds, cfg)
    tdir = table_dir(cfg, args.asset, args.spec)
    write_csv(tdir / "preonset_events.csv", onsets)
    if onsets.empty:
        print("no onsets in this sample")
        return
    window = p["window"]
    onset_pos = onsets["pos"].to_numpy()
    near_onset = np.zeros(len(data.dates), bool)
    for pos in onset_pos:
        near_onset[max(0, pos - window) : pos + window + 1] = True

    for family in args.families or cfg["core_families"]:
        fc = pd.read_parquet(out_dir(cfg, args.asset, args.spec, "forecasts") / f"{family}.parquet")
        by_pos = fc.set_index("pos")
        paths, alerts = [], []
        # Matched controls: nearest quiet origins in history space under the anchor's training metric.
        for _, ev in onsets.iterrows():
            start = ev["pos"] - window
            fold = folds[int(ev["fold"])]
            basis = LawBasis().fit(data.X[fold.train_idx])
            quiet = oos_pos[~near_onset[oos_pos]]
            if start not in by_pos.index or len(quiet) == 0:
                continue
            dist = np.linalg.norm(basis.whiten(data.X[quiet]) - basis.whiten(data.X[[start]]), axis=1)
            controls = quiet[np.argsort(dist)[: p["n_controls"]]]
            for tag, anchors in (("event", [start]), ("control", list(controls))):
                for a in anchors:
                    for lead in range(0, window + 1):
                        pos = a + lead
                        if pos in by_pos.index:
                            row = by_pos.loc[pos]
                            paths.append({"event": str(ev["date"].date()), "group": tag, "k": lead - window,
                                          "f0": row["f0"], "f1": row["f1"], "D": row["f1"] - row["f0"],
                                          "rel_D": (row["f1"] - row["f0"]) / row["f0"]})
        # Alert levels are quantiles of each model's fitted values in its own training window.
        thresholds = {}
        for f in folds:
            pair = load_pair(cfg, args.asset, args.spec, family, f.k)
            tr = f.train_idx
            q0, q1 = pair["m0"].predict(data.X[tr]), pair["m1"].predict(data.X[tr], data.V[tr])
            for level in p["alert_quantiles"]:
                thresholds[(f.k, level)] = (np.quantile(q0, level), np.quantile(q1, level))
        fold_ids = fc["fold"].to_numpy()
        pos_arr = fc["pos"].to_numpy()
        valid_alert = np.zeros(len(data.dates), bool)
        for pos in onset_pos:
            valid_alert[max(0, pos - window) : pos] = True
        after_onset = np.zeros(len(data.dates), bool)
        for pos in onset_pos:
            after_onset[pos : pos + window + 1] = True
        for level in p["alert_quantiles"]:
            limits = np.array([thresholds[(k, level)] for k in fold_ids])
            for col, idx in (("f0", 0), ("f1", 1)):
                alert = fc[col].to_numpy() > limits[:, idx]
                alert_pos = pos_arr[alert]
                countable = ~after_onset[pos_arr]
                false_alert = alert & countable & ~valid_alert[pos_arr]
                hit_leads = []
                for pos in onset_pos:
                    leads = alert_pos[(alert_pos >= pos - window) & (alert_pos < pos)]
                    hit_leads.append(pos - leads.min() if len(leads) else np.nan)
                hit_leads = np.asarray(hit_leads, float)
                alerts.append({"family": family, "level": level, "function": col, "n_events": len(onset_pos),
                               "false_alerts_per_100": 100 * false_alert.sum() / max(countable.sum(), 1),
                               "event_recall": float(np.mean(np.isfinite(hit_leads))),
                               "mean_first_lead": float(np.nanmean(hit_leads)) if np.isfinite(hit_leads).any() else np.nan})
        write_csv(tdir / f"preonset_paths_{family}.csv", pd.DataFrame(paths))
        write_csv(tdir / f"preonset_alerts_{family}.csv", pd.DataFrame(alerts))
    write_manifest(cfg, tdir, "08_preonset", exploratory=True, n_events=len(onsets))
    print(f"[{args.asset}/{args.spec}] {len(onsets)} onsets; exploratory by design", flush=True)


if __name__ == "__main__":
    main()
