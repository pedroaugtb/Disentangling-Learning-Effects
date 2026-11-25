#!/usr/bin/env python3
"""Addendum C: teacher / input decomposition of the gain from V, for one asset and spec.

    L(f0) - L(f1) = [L(f0) - L(P1)] + [L(P1) - L(f1)]       (teacher + input)

Parts, run in this order; each resumes from finished draws:
  observed  point estimates, moving-block bootstrap (with 90% bounds for the margin rule),
            DM tests, state and year tables.  Reads behavior/<family>/origin.parquet (script 04).
  placebo   the cond_block draws of script 06 (same seeds, same refits), scored on the
            decomposition.  Draws already carrying the statistics are copied, not refitted.
  refit     the refit-bootstrap draws of script 10 (same seeds), for combined intervals.
"""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from ivbehav.config import CONFIG_DIR, RESULTS_DIR, load_config, out_dir, resolve_spec, table_dir, write_csv, write_json, write_manifest
from ivbehav.inference import block_bootstrap, boot_p_two_sided, percentile_ci, placebo_p
from ivbehav.pipeline import run_placebo_teacher, run_refit_teacher, setup
from ivbehav.teacher import PREFIX, TeacherStats, dm_table, horizon_of, state_table, year_table


def load_teacher_config(name: str) -> tuple[dict, str]:
    path = Path(name) if Path(name).exists() else CONFIG_DIR / name
    text = path.read_text()
    return yaml.safe_load(text), hashlib.sha256(text.encode()).hexdigest()[:16]


def families_for(cfg, tcfg, spec, requested):
    if requested:
        return requested
    fams = list(cfg["core_families"])
    if cfg["profile"] == load_config(tcfg["confirmatory"]["config"])["profile"]:
        fams += tcfg["observed"].get("extra_families", {}).get(spec, [])
    return fams


def _quantiles(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    q05, q95 = np.quantile(values, [0.05, 0.95])
    return float(q05), float(q95)


# ------------------------------------------------------------------------------------
def run_observed(cfg, tcfg, asset, spec, family, data, oos_pos, tdir) -> bool:
    bdir = RESULTS_DIR / cfg["profile"] / asset / spec / "behavior" / family
    if not (bdir / "origin.parquet").exists():
        print(f"[{asset}/{spec}] teacher {family}: no origin.parquet (run 04 first), skipped", flush=True)
        return False
    origin = pd.read_parquet(bdir / "origin.parquet")
    label_end = data.label_end[origin["pos"].to_numpy()]
    stat = TeacherStats(origin, oos_pos, label_end=label_end, min_closed=tcfg["recalibration"]["min_closed"])
    point = stat.point()
    write_json(bdir / "teacher_point.json", point)

    boot = cfg["bootstrap"]
    rows = []
    for block in boot["block_lengths"]:
        samples = block_bootstrap(stat.compute, len(oos_pos), block, boot["n"], cfg["seed"] + 7000 + block)
        for key, values in samples.items():
            lo, hi = percentile_ci(values)
            q05, q95 = _quantiles(values)
            rows.append({"statistic": key, "block": block, "estimate": point.get(key, float("nan")),
                         "boot_mean": float(np.nanmean(values)), "boot_sd": float(np.nanstd(values)),
                         "ci_lo": lo, "ci_hi": hi, "q05": q05, "q95": q95, "p_zero": boot_p_two_sided(values)})
    write_csv(tdir / f"teacher_boot_{family}.csv", pd.DataFrame(rows))
    horizon = horizon_of(resolve_spec(cfg, spec)["target"])
    write_csv(tdir / f"teacher_dm_{family}.csv", dm_table(origin, horizon))
    write_csv(tdir / f"teacher_states_{family}.csv", state_table(origin))
    write_csv(tdir / f"teacher_by_year_{family}.csv", year_table(origin))
    print(f"[{asset}/{spec}] teacher {family}: observed done ({len(origin)} origins)", flush=True)
    return True


# ------------------------------------------------------------------------------------
def placebo_job(cfg, asset, spec, family, kind, draw, path, source):
    if path.exists():
        return
    if source.exists():
        with open(source) as fh:
            record = json.load(fh)
        ti = {k: v for k, v in record.items() if k.startswith(PREFIX)}
        if ti:
            write_json(path, ti)
            return
    write_json(path, run_placebo_teacher(cfg, asset, spec, family, kind, draw))


def run_placebo(cfg, tcfg, asset, spec, family, tdir, n_jobs):
    kind = tcfg["placebo"]["kind"]
    n = int(cfg["placebo"]["n_draws"].get(kind, 0) or 0)
    bdir = RESULTS_DIR / cfg["profile"] / asset / spec / "behavior" / family
    if n == 0 or not (bdir / "teacher_point.json").exists():
        print(f"[{asset}/{spec}] teacher placebo {family}: nothing to do (draws={n}, observed missing?)", flush=True)
        return
    d = out_dir(cfg, asset, spec, "placebo_ti", family, kind)
    src = RESULTS_DIR / cfg["profile"] / asset / spec / "placebo" / family / kind
    start = time.time()
    joblib.Parallel(n_jobs=n_jobs, verbose=5)(
        joblib.delayed(placebo_job)(cfg, asset, spec, family, kind, draw, d / f"r{draw:04d}.json", src / f"r{draw:04d}.json")
        for draw in range(n))
    print(f"[{asset}/{spec}] teacher placebo {family}: {time.time() - start:.0f}s", flush=True)

    observed = json.load(open(bdir / "teacher_point.json"))
    draws = [json.load(open(p)) for p in sorted(d.glob("r*.json"))]
    rows = []
    for key in sorted(observed):
        values = np.array([dr.get(key, np.nan) for dr in draws], float)
        if not np.isfinite(values).any():
            continue
        rows.append({"kind": kind, "statistic": key, "observed": observed[key], "n_draws": int(np.isfinite(values).sum()),
                     "placebo_mean": float(np.nanmean(values)), "placebo_sd": float(np.nanstd(values)),
                     "placebo_q05": float(np.nanquantile(values, 0.05)), "placebo_q95": float(np.nanquantile(values, 0.95)),
                     "p_upper": placebo_p(observed[key], values, "upper"), "p_lower": placebo_p(observed[key], values, "lower"),
                     "p_abs": placebo_p(observed[key], values, "abs")})
    write_csv(tdir / f"teacher_placebo_{family}.csv", pd.DataFrame(rows))


# ------------------------------------------------------------------------------------
def refit_job(cfg, asset, spec, family, draw, path, block):
    if path.exists():
        return
    write_json(path, run_refit_teacher(cfg, asset, spec, family, draw, block))


def run_refit(cfg, asset, spec, family, tdir, n_jobs, n_draws=None):
    """Draws are indexed by seed, so a larger ``n_draws`` (Addendum E) keeps draws 0..n-1 and adds new ones."""
    r = cfg["refit_bootstrap"]
    n = int(n_draws) if n_draws else int(r["n"].get(family, 0))
    if n == 0:
        print(f"[{asset}/{spec}] teacher refit {family}: no refit budget, skipped", flush=True)
        return
    d = out_dir(cfg, asset, spec, "refit_ti", family)
    start = time.time()
    joblib.Parallel(n_jobs=n_jobs, verbose=5)(
        joblib.delayed(refit_job)(cfg, asset, spec, family, draw, d / f"b{draw:04d}.json", r["block"]) for draw in range(n))
    print(f"[{asset}/{spec}] teacher refit {family}: {time.time() - start:.0f}s", flush=True)
    records = [json.load(open(p)) for p in sorted(d.glob("b*.json"))]
    rows = []
    for source in ("fit_only", "combined"):
        for key in sorted(records[0][source]):
            values = np.array([rec[source].get(key, np.nan) for rec in records], float)
            lo, hi = percentile_ci(values)
            q05, q95 = _quantiles(values)
            rows.append({"source": source, "statistic": key, "n": int(np.isfinite(values).sum()),
                         "mean": float(np.nanmean(values)), "sd": float(np.nanstd(values)),
                         "ci_lo": lo, "ci_hi": hi, "q05": q05, "q95": q95, "p_zero": boot_p_two_sided(values)})
    write_csv(tdir / f"teacher_refit_{family}.csv", pd.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--teacher-config", default="teacher.yaml")
    ap.add_argument("--asset", default="SPX")
    ap.add_argument("--spec", default="primary")
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--part", choices=["observed", "placebo", "refit", "all"], default="observed")
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--refit-draws", type=int, default=None,
                    help="override the profile's refit budget (Addendum E: configs/teacher_amend.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg, thash = load_teacher_config(args.teacher_config)
    data, _, oos_pos = setup(cfg, args.asset, args.spec)
    tdir = table_dir(cfg, args.asset, args.spec)
    for family in families_for(cfg, tcfg, args.spec, args.families):
        if args.part in ("observed", "all"):
            if not run_observed(cfg, tcfg, args.asset, args.spec, family, data, oos_pos, tdir):
                continue
        if args.part in ("placebo", "all"):
            run_placebo(cfg, tcfg, args.asset, args.spec, family, tdir, args.n_jobs)
        if args.part in ("refit", "all"):
            run_refit(cfg, args.asset, args.spec, family, tdir, args.n_jobs, args.refit_draws)
    write_manifest(cfg, tdir, f"12_teacher_input_{args.part}", teacher_config_hash=thash,
                   refit_draws_override=args.refit_draws)


if __name__ == "__main__":
    main()
