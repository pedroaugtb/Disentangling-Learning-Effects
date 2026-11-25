"""Configuration loading, spec resolution and output paths."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in over.items():
        # An empty mapping replaces the parent's (a profile can switch a block off).
        if isinstance(val, dict) and val and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path = "primary.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = path if path.exists() else CONFIG_DIR / path
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if "extends" in cfg:
        base = load_config(path.parent / cfg.pop("extends"))
        cfg = _merge(base, cfg)
    cfg.pop("extends", None)
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def resolve_spec(cfg: dict, spec: str) -> dict:
    """Walk-forward settings for one spec: primary settings plus the spec's departures."""
    if spec not in cfg["specs"]:
        raise KeyError(f"unknown spec {spec!r}; defined: {sorted(cfg['specs'])}")
    return {**cfg["walk_forward"], **cfg["specs"][spec]}


def out_dir(cfg: dict, asset: str, spec: str = "primary", *parts: str) -> Path:
    path = RESULTS_DIR / cfg["profile"] / asset / spec
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path


def table_dir(cfg: dict, asset: str, spec: str = "primary") -> Path:
    return out_dir(cfg, asset, spec, "tables")


def atomic_write(path: Path, write_fn) -> None:
    """Write through a temp file so an interrupted run never leaves a half-written output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=path.suffix)
    os.close(fd)
    try:
        write_fn(tmp)
        os.chmod(tmp, 0o664)      # mkstemp defaults to 0600; match the rest of the project
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path: Path, obj) -> None:
    def _write(tmp):
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2, default=float, sort_keys=True)
    atomic_write(path, _write)


def write_parquet(path: Path, frame) -> None:
    atomic_write(path, lambda tmp: frame.to_parquet(tmp, index=False))


def write_csv(path: Path, frame) -> None:
    atomic_write(path, lambda tmp: frame.to_csv(tmp, index=False))


def check_config_stamp(cfg: dict) -> None:
    """Refuse to mix results from different configs under one profile.

    Stages skip finished work, so an edited config would otherwise combine old and new outputs
    without any warning.  The first run stamps the profile; a later mismatch is an error.
    """
    path = RESULTS_DIR / cfg["profile"] / "config_stamp.json"
    current = config_hash(cfg)
    if not path.exists():
        write_json(path, {"config_hash": current, "config": cfg})
        return
    with open(path) as fh:
        stamped = json.load(fh)["config_hash"]
    if stamped != current and not os.environ.get("IVBEHAV_ALLOW_CONFIG_CHANGE"):
        raise RuntimeError(
            f"config hash {current} differs from {stamped}, which produced results/{cfg['profile']}. "
            f"Use a new `profile`, or set IVBEHAV_ALLOW_CONFIG_CHANGE=1 if the edit cannot affect finished work.")


def write_manifest(cfg: dict, directory: Path, script: str, **extra) -> None:
    write_json(directory / f"manifest_{script}.json",
               {"config_hash": config_hash(cfg), "profile": cfg["profile"], "script": script, **extra})
