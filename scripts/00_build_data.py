#!/usr/bin/env python3
"""Snapshot raw sources (once), run the cross-source QC, and build per-asset panels."""
import _bootstrap  # noqa: F401
import argparse
import json

from ivbehav.config import load_config
from ivbehav.data import download_raw, save_panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    download_raw(cfg, force=args.force_download)
    for asset in args.assets or list(cfg["assets"]):
        qc = save_panel(cfg, asset)
        print(json.dumps(qc, indent=1, default=str))


if __name__ == "__main__":
    main()
