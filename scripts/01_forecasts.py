#!/usr/bin/env python3
"""Stage 1: leak-free walk-forward tuning and fitting of the paired families and benchmarks."""
import _bootstrap  # noqa: F401
import argparse
import time

from ivbehav.config import load_config, out_dir, write_manifest, write_parquet
from ivbehav.pipeline import run_benchmark, run_forecasts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--asset", default="SPX")
    ap.add_argument("--spec", default="primary")
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paired = args.families or list(cfg["core_families"])
    if args.families is None and args.spec == "primary":
        paired += cfg.get("extra_forecast_families", ["har_ols", *cfg["log_target_families"]])
    directory = out_dir(cfg, args.asset, args.spec, "forecasts")
    for family in paired:
        start = time.time()
        write_parquet(directory / f"{family}.parquet", run_forecasts(cfg, args.asset, args.spec, family, args.n_jobs, args.force))
        print(f"[{args.asset}/{args.spec}] {family}: {time.time() - start:.0f}s", flush=True)
    if args.spec == "primary" and args.families is None:
        for family in cfg["benchmarks"]:
            write_parquet(directory / f"bench_{family}.parquet", run_benchmark(cfg, args.asset, args.spec, family, args.n_jobs))
    write_manifest(cfg, directory, "01_forecasts", families=paired)


if __name__ == "__main__":
    main()
