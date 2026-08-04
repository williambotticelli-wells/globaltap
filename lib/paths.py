"""Path-resolution helpers shared by all pipeline stages.

Configs may reference resource paths (gt_source_map, stimuli_manifest,
audio_dir, outputs_root, tap_csv_sources) using:

  1. An absolute path       ->  used as-is.
  2. A path relative to     ->  try <pipeline_root>/<p> first (works for the
     <pipeline_root>            in-repo configs).
  3. A path relative to     ->  fallback to <data_root>/<p> (the typical
     <data_root>                deployment, where maps live next to data).

`data_root` may contain environment-variable references such as
``${PROJECT_ROOT}``; these are expanded by ``load_config`` below before any
path resolution happens.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def _expand(value: Any) -> Any:
    """Recursively expand ${VAR} environment variables in str/list/dict."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path: PathLike) -> dict:
    """Load a pipeline JSON config and expand all environment variables.

    Configs typically set ``data_root`` to ``${PROJECT_ROOT}``; this loader
    expands it (and any other env vars) so downstream code sees concrete
    filesystem paths.
    """
    with open(path) as fh:
        cfg = json.load(fh)
    return _expand(cfg)


def resolve_config_path(p: PathLike, pipeline_root: Path, data_root: Path,
                        must_exist: bool = True) -> Path:
    """Resolve ``p`` by the 3-tier rule above.

    Returns the first candidate that exists. If none exists and
    ``must_exist`` is False, returns the pipeline_root candidate (useful for
    output paths).
    """
    p = Path(os.path.expandvars(str(p)))
    if p.is_absolute():
        return p
    c_pipe = pipeline_root / p
    c_data = data_root / p
    if c_pipe.exists():
        return c_pipe
    if c_data.exists():
        return c_data
    if must_exist:
        raise FileNotFoundError(
            f"Could not resolve {p}: tried {c_pipe} and {c_data}."
        )
    return c_pipe


def resolve_outputs_root(cfg: dict, pipeline_root: Path | None = None) -> Path:
    """Standard resolution for ``cfg['outputs_root']``: relative to data_root."""
    data_root = Path(os.path.expandvars(str(cfg["data_root"])))
    out = Path(os.path.expandvars(str(cfg["outputs_root"])))
    if out.is_absolute():
        return out
    return data_root / out
