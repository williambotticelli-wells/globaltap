"""Reference-annotation beat-time loader.

For every stem listed in the dataset's `gt_source_map.json`, this module
resolves the annotator-original beat file, applies any `row_filter`, and
returns beat times measured from the start of a given 30-second excerpt,
along with position labels.

The same loader serves the `canonical240` validation set, `globalmood200`
(where a missing reference annotation returns `None`), and any future corpus
that supplies a `gt_source_map`.

Public API:
    load_gt_for_stem(stem_id, gt_map, data_root) -> np.ndarray | None
        Returns an (n, 2) array of (time_in_clip_seconds, position_label)
        for all reference beats falling inside the 30-s clip window, or
        None if no reference annotation is available for this stem (e.g.
        the cross-cultural benchmark).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np


CLIP_DUR = 30.0


def load_beats_raw(path: Path) -> np.ndarray:
    """Load a whitespace-separated `<time>\\t<position>` beats file.

    Lines that can't be parsed as float-first-column are silently skipped.
    Returns an (n, 2) float array; second column is -1 when position is
    missing.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        pos = int(float(parts[1])) if len(parts) > 1 else -1
        rows.append((t, pos))
    return np.array(rows, dtype=float) if rows else np.zeros((0, 2))


def apply_row_filter(beats: np.ndarray, filter_expr: Optional[str]) -> np.ndarray:
    """Apply a simple row filter expression on position labels.

    Supported forms:
        None                -> no filter (identity)
        "pos>=1"            -> keep rows where pos >= 1
        "pos>=0"            -> keep rows where pos >= 0 (drops unlabeled)
        "pos in {1,3,5,7}"  -> keep rows whose pos is in the given set
        "pos in {1,3,5,8}"  -> same
        "pos==1"            -> keep only pos == 1
    """
    if filter_expr is None or beats.size == 0:
        return beats
    expr = filter_expr.replace(" ", "")
    if expr.startswith("pos>="):
        thr = int(expr.split("=", 1)[1])
        return beats[beats[:, 1] >= thr]
    if expr.startswith("pos=="):
        val = int(expr.split("==", 1)[1])
        return beats[beats[:, 1] == val]
    if expr.startswith("posin{") and expr.endswith("}"):
        vals = {int(v) for v in expr[len("posin{"):-1].split(",") if v}
        mask = np.array([int(p) in vals for p in beats[:, 1]], dtype=bool)
        return beats[mask]
    raise ValueError(f"Unsupported row_filter expression: {filter_expr!r}")


def load_gt_for_stem(
    stem_id: str,
    gt_map: dict,
    data_root: Path,
) -> Optional[np.ndarray]:
    """Return (n, 2) array of (t_local, pos) for the 30-s clip, or None
    if no GT is mapped (e.g. cross-cultural datasets).

    Parameters
    ----------
    stem_id : str
        Key in gt_map.
    gt_map : dict
        Parsed contents of `gt_source_map_*.json`.
    data_root : Path
        Project root; used to resolve workspace-relative paths.
    """
    if stem_id not in gt_map:
        raise KeyError(f"{stem_id!r} not found in GT source map.")
    entry = gt_map[stem_id]
    if entry.get("gt") is None or entry.get("path") is None:
        return None

    p_raw = entry["path"]
    path = Path(p_raw) if Path(p_raw).is_absolute() else data_root / p_raw
    raw = load_beats_raw(path)
    filt = apply_row_filter(raw, entry.get("row_filter"))
    offset = float(entry.get("clip_offset_s", 0.0))
    t_local = filt[:, 0] - offset
    # 1 ms tolerance at the t=0 edge: clip_offset_s is stored to 3 decimals in
    # gt_map, so the first beat (by construction at t_annotator = clip_offset)
    # can round to a few-tenths-of-a-ms below zero. Keep it.
    EDGE_TOL = 1e-3
    in_clip = (t_local >= -EDGE_TOL) & (t_local < CLIP_DUR)
    out = filt[in_clip].copy()
    out[:, 0] = np.clip(t_local[in_clip], 0.0, None)
    return out
