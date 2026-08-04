"""Per-stage / per-run provenance manifests.

Writes `manifest.json` capturing:
  - pipeline_version (from config)
  - stage name, timestamp, host
  - git SHA if available, else 'not_in_git'
  - SHA-256 of the config file
  - SHA-256 of every input file referenced by the stage
  - SHA-256 of every output file produced by the stage

One manifest per stage under `outputs/<dataset>/<stage>/manifest.json`.
A top-level `outputs/<dataset>/manifest.json` aggregates all stage manifests
after `run_pipeline.sh` completes.

Usage inside a stage:
    from lib.manifest import write_stage_manifest
    write_stage_manifest(
        stage="2_repp_align",
        config_path=Path(args.config),
        cfg=cfg,
        inputs=[merged_csv, wav_dir, gt_map_path],
        outputs_dir=repp_out,
    )
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Iterable, Union, List

PathLike = Union[str, Path]


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_file_entry(p: Path) -> dict:
    try:
        p = p.resolve()
    except Exception:
        pass
    entry = {"path": str(p)}
    if not p.exists():
        entry["status"] = "missing"
        return entry
    if p.is_file():
        try:
            entry["size_bytes"] = p.stat().st_size
            entry["sha256"] = _sha256_file(p)
        except Exception as e:
            entry["status"] = f"error:{e}"
    elif p.is_dir():
        entry["status"] = "directory"
        # List top-level files only (not recursive — keeps manifest manageable)
        files = sorted([q for q in p.iterdir() if q.is_file()])[:50]
        entry["n_files"] = len(files)
        entry["sample"] = [str(q.name) for q in files]
    return entry


def _git_sha() -> str:
    for root in [Path.cwd(), *Path.cwd().parents]:
        if (root / ".git").exists():
            try:
                out = subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                return out
            except Exception:
                pass
    return "not_in_git"


def write_stage_manifest(
    stage: str,
    config_path: PathLike,
    cfg: dict,
    inputs: Iterable[PathLike],
    outputs_dir: PathLike,
    extra: dict | None = None,
) -> Path:
    config_path = Path(config_path)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stage": stage,
        "pipeline_version": cfg.get("pipeline_version", "unknown"),
        "dataset": cfg.get("dataset", "unknown"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_sha": _git_sha(),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _sha256_file(config_path),
            "snapshot": cfg,
        },
        "inputs": [_safe_file_entry(Path(p)) for p in inputs],
        "outputs": [_safe_file_entry(q) for q in sorted(outputs_dir.iterdir())
                    if q.is_file()] if outputs_dir.exists() else [],
    }
    if extra:
        manifest["extra"] = extra

    out_path = outputs_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    return out_path


def aggregate_run_manifest(outputs_root: PathLike, pipeline_version: str) -> Path:
    """Collect all stage manifests into a single run-level manifest."""
    outputs_root = Path(outputs_root)
    stages = []
    for mp in sorted(outputs_root.rglob("manifest.json")):
        if mp.parent == outputs_root:
            continue  # skip the aggregate itself
        try:
            stages.append({
                "stage_dir": str(mp.parent.relative_to(outputs_root)),
                "manifest": json.loads(mp.read_text()),
            })
        except Exception as e:
            stages.append({"stage_dir": str(mp.parent), "error": str(e)})

    run = {
        "pipeline_version": pipeline_version,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "git_sha": _git_sha(),
        "n_stages": len(stages),
        "stages": stages,
    }
    out_path = outputs_root / "manifest.json"
    out_path.write_text(json.dumps(run, indent=2, default=str))
    return out_path
