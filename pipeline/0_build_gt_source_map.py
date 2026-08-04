"""Stage 0 — Build `gt_source_map_<dataset>.json`.

For every stem in the dataset manifest, resolve the canonical upstream
annotator beat file, determine the clip offset, and record any row
filter required to produce the "annotator tactus" for that stem.

Sources:
    asap, beatles, candombe, groove_midi, gtzan, guitarset, hainsworth,
    harmonix, rwc, tapcorrect   -> beat_this_annotations/<corpus>/annotations/beats/
    carnatic, turkish, cretan   -> ~/Downloads/downbeat_examples/*.txt
                                   (MIREX 2021)

Row-filter and offset overrides (locked policy 2026-04-19 — see
`next-steps/benchmark100-validation/ANNOTATION_SOURCES.md`):

    rwc_classical_CD1_01   : path=audio/benchmark_30s/... , row_filter=pos>=1
                             (bm30_filt; matches the Beat This! source in
                             the 30-s window; Beat This! source has a
                             known conversion bug for this track)
    turkish (all 6 main)   : row_filter keeps the subset of pulse-grid
                             positions that constitute the main usul
                             beats (corpus-author-designated tactus):
                               aksak{1,2}   : pos in {1,3,5,7}
                               duyek{1,2}   : pos in {1,3,5,7}
                               curcuna{1,2} : pos in {1,3,5,8}
    sofyan{1,2}            : no filter (4/4, pos already at tactus)

All other stems: upstream verbatim, offset found by best-match to bm30.

Usage:
    python stages/0_build_gt_source_map.py \
        --config config/benchmark100.yaml
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("${PROJECT_ROOT}")
CLIP_DUR = 30.0

MANIFEST = ROOT / "next-steps/mixed-benchmark/sampled_stimuli.json"
BM30 = ROOT / "audio/benchmark_30s"
BEAT_THIS = ROOT / "next-steps/mixed-benchmark/beat_this_annotations"
MIREX = Path.home() / "Downloads/downbeat_examples"

BEAT_THIS_CORPORA = {
    "asap", "beatles", "candombe", "groove_midi", "gtzan",
    "guitarset", "hainsworth", "harmonix", "rwc", "tapcorrect",
}
MIREX_CORPORA = {"carnatic", "turkish", "cretan"}

# Locked per-stem overrides (see header docstring).
TURKISH_MAIN_FILTER = {
    "aksak1":   "pos in {1,3,5,7}",
    "aksak2":   "pos in {1,3,5,7}",
    "duyek1":   "pos in {1,3,5,7}",
    "duyek2":   "pos in {1,3,5,7}",
    "curcuna1": "pos in {1,3,5,8}",
    "curcuna2": "pos in {1,3,5,8}",
    # sofyan1/2 have pos {1,2,3,4} already at tactus — no filter
}
CD1_01_OVERRIDE = {
    "source": "bm30_filt",
    "path_rel": "audio/benchmark_30s/rwc/rwc_classical_CD1_01.beats",
    "row_filter": "pos>=1",
    "note": (
        "bm30_filt (drop conversion-bug pos==0 rows). Matches the "
        "Beat This! source in the 30-s window "
        "(offset 5.87s). Full-track source has a two-regime "
        "conversion bug flagged in the Beat This! README."
    ),
}


def load_beats(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
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


def upstream_path_for(corpus: str, stem: str):
    if corpus in BEAT_THIS_CORPORA:
        p = BEAT_THIS / corpus / "annotations" / "beats" / f"{stem}.beats"
        return p if p.exists() else None
    if corpus in MIREX_CORPORA:
        p = MIREX / f"{stem}.txt"
        return p if p.exists() else None
    return None


def find_clip_offset(bm_local_times: np.ndarray, up_abs_times: np.ndarray) -> float:
    """Given bm30 clip-local times and upstream absolute times, return
    the t_start (upstream absolute) such that [t_start, t_start+30] best
    matches bm.
    """
    if bm_local_times.size == 0 or up_abs_times.size == 0:
        return 0.0
    t_first = float(bm_local_times[0])
    best = (-1, 0.0)
    for u in up_abs_times:
        cand = float(u - t_first)
        if cand < -0.5:
            continue
        mask = (up_abs_times >= cand - 0.01) & (up_abs_times <= cand + CLIP_DUR + 0.01)
        ul = up_abs_times[mask] - cand
        if len(ul) == 0:
            continue
        n_match = int(sum(1 for t in bm_local_times
                          if np.abs(ul - t).min() < 0.030))
        if n_match > best[0]:
            best = (n_match, cand)
    return best[1]


def build_map_entry(corpus: str, stem: str) -> dict:
    """Return the gt_source_map entry for a single stem."""
    bm_path = BM30 / corpus / f"{stem}.beats"
    if not bm_path.exists():
        return {"gt": None, "path": None, "reason": "missing bm30 file"}
    bm = load_beats(bm_path)

    # Locked special case: RWC CD1_01 uses bm30_filt (= upstream in-window).
    if stem == "rwc_classical_CD1_01":
        return {
            "gt": "annotator",
            "corpus": corpus,
            "path": CD1_01_OVERRIDE["path_rel"],
            "clip_offset_s": 0.0,
            "row_filter": CD1_01_OVERRIDE["row_filter"],
            "note": CD1_01_OVERRIDE["note"],
            "source_label": "bm30_filt (== upstream in-window)",
        }

    up_path = upstream_path_for(corpus, stem)
    if up_path is None:
        return {"gt": None, "path": None, "reason": f"no upstream source for {corpus}"}
    up = load_beats(up_path)
    offset = find_clip_offset(bm[:, 0], up[:, 0])

    # Absolute paths for MIREX (outside workspace), relative for beat_this
    try:
        path_str = str(up_path.relative_to(ROOT))
    except ValueError:
        path_str = str(up_path)

    entry = {
        "gt": "annotator",
        "corpus": corpus,
        "path": path_str,
        "clip_offset_s": round(offset, 3),
        "row_filter": TURKISH_MAIN_FILTER.get(stem),
        "source_label": "beat_this_upstream" if corpus in BEAT_THIS_CORPORA
                         else "mirex_downbeat_examples",
    }
    if stem in TURKISH_MAIN_FILTER:
        entry["note"] = (
            "Turkish main usul tactus — subset of pulse grid at "
            "annotator-designated main pulse positions."
        )
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/benchmark100.yaml",
                    help="Config file (relative to pipeline folder)")
    ap.add_argument("--out", default="config/gt_source_map_benchmark100.json",
                    help="Where to write the GT source map JSON")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent.parent
    out = here / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text())
    gt_map = {}
    stats = {"annotator": 0, "missing": 0, "filt": 0}
    for e in manifest:
        stem = e["stem_id"]
        corpus = e["dataset"]
        entry = build_map_entry(corpus, stem)
        gt_map[stem] = entry
        if entry["gt"] == "annotator":
            stats["annotator"] += 1
            if entry.get("row_filter"):
                stats["filt"] += 1
        else:
            stats["missing"] += 1
        if args.verbose:
            print(f"  [{corpus:12s}] {stem:55s}  offset={entry.get('clip_offset_s', 0):.2f}s"
                  f"  filter={entry.get('row_filter')}")

    out.write_text(json.dumps(gt_map, indent=2))
    print(f"\nWrote {len(gt_map)} stems → {out}")
    print(f"  with annotator GT: {stats['annotator']}")
    print(f"    of which row-filtered (Turkish main + RWC CD1_01 filt): {stats['filt']}")
    print(f"  missing GT: {stats['missing']}")


if __name__ == "__main__":
    main()
