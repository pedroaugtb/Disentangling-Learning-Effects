#!/usr/bin/env python3
"""WP7: core contrasts across credible alternative fits.  Disagreement here is a result."""
import _bootstrap  # noqa: F401
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from ivbehav.config import load_config, out_dir, table_dir, write_csv, write_json, write_manifest
from ivbehav.models import NEURAL_FAMILIES
from ivbehav.pipeline import hp_for_folds, load_tables, run_refit, setup, tune_family


def variant_list(cfg, family):
    m = cfg["multiplicity"]
    variants = []
    if family in ("xgb", *NEURAL_FAMILIES):
        variants += [{"name": f"bag_{b}", "kind": "seed", "seed_offset": 1000 * (b + 1)} for b in range(m["bag_replicates"])]
    if family in ("enet", "xgb", *NEURAL_FAMILIES):
        variants += [{"name": f"near_rank_{r}", "kind": "near", "rank": r - 1} for r in m["near_equiv_ranks"]]
    if family == "hgb":
        variants += [{"name": "alt_learner", "kind": "alt_learner"}]
    variants += [{"name": f"window_{name}", "kind": "window", "wf": wf} for name, wf in m["alt_windows"].items()] if family != "hgb" else []
    variants.append({"name": "baseline_refit", "kind": "base"})
    return variants


def run_variant(cfg, asset, spec, family, variant, path, n_jobs):
    if path.exists():
        return
    start = time.time()
    data, folds, oos_pos = setup(cfg, asset, spec, wf_override=variant.get("wf"))
    if variant["kind"] in ("window", "alt_learner"):
        tables = tune_family(data, folds, family, cfg, n_jobs)
        hp = hp_for_folds(folds, tables)
    else:
        tables = load_tables(cfg, asset, spec, family)
        hp = hp_for_folds(folds, tables, rank=variant.get("rank", 0), tolerance=cfg["multiplicity"]["near_equiv_tolerance"])
    seed = cfg["seed"] + variant.get("seed_offset", 0)
    stats, results = run_refit(cfg, data, folds, oos_pos, family, hp, seed, with_kernel=True, n_jobs=n_jobs)
    origin = pd.concat([r["origin"] for r in results], ignore_index=True)
    write_json(path, {"variant": variant["name"], "kind": variant["kind"], "family": family, "seconds": time.time() - start,
                      "stats": stats})
    origin[["pos", "f0", "f1", "HFS_a", "ISD_a", "HFS_b", "ISD_b"]].to_parquet(path.with_suffix(".parquet"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--asset", default="SPX")
    ap.add_argument("--spec", default="primary")
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--n-jobs", type=int, default=24)
    args = ap.parse_args()

    cfg = load_config(args.config)
    families = args.families or [*cfg["core_families"], *cfg["multiplicity"].get("extra_families", ["hgb"])]
    tdir = table_dir(cfg, args.asset, args.spec)
    rows = []
    for family in families:
        d = out_dir(cfg, args.asset, args.spec, "multiplicity", family)
        for variant in variant_list(cfg, family):
            if family == "hgb" and variant["kind"] == "base":
                continue
            run_variant(cfg, args.asset, args.spec, family, variant, d / f"{variant['name']}.json", args.n_jobs)
            print(f"[{args.asset}/{args.spec}] {family} {variant['name']} done", flush=True)
        for path in sorted(d.glob("*.json")):
            rec = json.load(open(path))
            for key, value in rec["stats"].items():
                rows.append({"family": family, "variant": rec["variant"], "kind": rec["kind"], "statistic": key, "value": value})
    frame = pd.DataFrame(rows)
    write_csv(tdir / "multiplicity_long.csv", frame)
    # Dispersion by source of variation: fit (seed, tuning, window, learner) vs. law vs. estimator.
    summary = (frame.groupby(["family", "kind", "statistic"])["value"]
               .agg(n="size", mean="mean", sd="std", min="min", max="max").reset_index())
    write_csv(tdir / "multiplicity_summary.csv", summary)
    write_manifest(cfg, tdir, "09_multiplicity")


if __name__ == "__main__":
    main()
