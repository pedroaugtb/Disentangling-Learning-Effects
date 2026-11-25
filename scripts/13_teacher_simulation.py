#!/usr/bin/env python3
"""Addendum C: Monte Carlo audit of the teacher / input decomposition.

Reads only pre-out-of-sample SPX rows (DGP calibration, as script 02).  Writes chunked,
resumable replicate files under results/<profile>/simulation/teacher/ and summary_teacher.csv.
"""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml

from ivbehav.config import CONFIG_DIR, RESULTS_DIR, check_config_stamp, load_config, write_csv, write_manifest, write_parquet
from ivbehav.data import load_spec_data
from ivbehav.simulation import calibrate
from ivbehav.teacher_sim import summarize, teacher_replicate


def designs(cfg):
    s = cfg["simulation"]
    return [("redundant", 0.0)] + [(m, b) for m in ("constant", "state") for b in s["informative_b"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--teacher-config", default="teacher.yaml")
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--n-reps", type=int, default=None, help="override (smoke tests only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    check_config_stamp(cfg)
    tpath = Path(args.teacher_config) if Path(args.teacher_config).exists() else CONFIG_DIR / args.teacher_config
    ttext = tpath.read_text()
    tcfg = yaml.safe_load(ttext)
    tsim = dict(tcfg["simulation"])
    if args.n_reps is not None:
        tsim.update(n_reps=args.n_reps, chunk=args.n_reps, placebo_reps=min(tsim["placebo_reps"], args.n_reps),
                    placebo_draws=min(tsim["placebo_draws"], 3), n_pop=min(tsim["n_pop"], 2000))
    conf = tcfg["confirmatory"]

    directory = RESULTS_DIR / cfg["profile"] / "simulation" / "teacher"
    directory.mkdir(parents=True, exist_ok=True)
    data = load_spec_data(cfg, "SPX", "primary")
    cal = calibrate(data, cfg["sample"]["oos_start"])

    frames = []
    for mode, b_bar in designs(cfg):
        for start in range(0, tsim["n_reps"], tsim["chunk"]):
            stop = min(start + tsim["chunk"], tsim["n_reps"])
            path = directory / f"reps_{mode}_b{b_bar:g}_{start:04d}_{stop:04d}.parquet"
            if not path.exists():
                t0 = time.time()
                rows = joblib.Parallel(n_jobs=args.n_jobs)(
                    joblib.delayed(teacher_replicate)(rep, mode, b_bar, cal, cfg, tsim, cfg["seed"],
                                                      rep < tsim["placebo_reps"])
                    for rep in range(start, stop))
                write_parquet(path, pd.DataFrame([r for chunk in rows for r in chunk]))
                print(f"teacher sim {mode} b={b_bar} reps {start}-{stop}: {time.time() - t0:.0f}s", flush=True)
            frames.append(pd.read_parquet(path))
    reps = pd.concat(frames, ignore_index=True)
    out = RESULTS_DIR / cfg["profile"] / "simulation" / "summary_teacher.csv"
    write_csv(out, summarize(reps, conf["alpha"], conf["margin_rel"]))
    write_manifest(cfg, directory, "13_teacher_simulation",
                   teacher_config_hash=hashlib.sha256(ttext.encode()).hexdigest()[:16], n_reps=tsim["n_reps"])
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
