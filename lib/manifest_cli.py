"""CLI wrapper for writing a stage manifest after a stage has run.
Invoked by run_pipeline.sh:

    python -m lib.manifest_cli \
        --stage 2_repp_align \
        --config config/canonical240.json \
        --outputs_dir outputs/canonical240/repp \
        --inputs outputs/canonical240/merged/... audio/canonical240_30s/...

Or to write the aggregate top-level manifest after all stages:
    python -m lib.manifest_cli --aggregate --config config/canonical240.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from lib.manifest import write_stage_manifest, aggregate_run_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--outputs_dir", default=None)
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--aggregate", action="store_true",
                    help="Write run-level aggregate manifest under outputs_root.")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    data_root = Path(cfg["data_root"])
    out_root = Path(cfg["outputs_root"])
    if not out_root.is_absolute():
        out_root = data_root / out_root

    if args.aggregate:
        p = aggregate_run_manifest(out_root, cfg.get("pipeline_version", "unknown"))
        print(f"[manifest] aggregate → {p}")
        return

    if not args.stage or not args.outputs_dir:
        raise SystemExit("Need --stage and --outputs_dir (or --aggregate).")
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.is_absolute():
        # Could be "outputs/canonical240/compare" (repo-relative), "canonical240/compare"
        # (outputs-root-relative), or just "compare" (also outputs-root-relative).
        # Canonicalize to absolute.
        # Try treating as repo-relative first; if that doesn't exist, join with out_root.
        pipeline_root = Path(__file__).resolve().parent.parent
        cand = pipeline_root / outputs_dir
        if cand.exists():
            outputs_dir = cand
        else:
            outputs_dir = out_root / outputs_dir.name

    inputs = []
    for i in args.inputs:
        p = Path(i)
        if not p.is_absolute():
            p = data_root / p
        inputs.append(p)

    mpath = write_stage_manifest(
        stage=args.stage,
        config_path=Path(args.config),
        cfg=cfg,
        inputs=inputs,
        outputs_dir=outputs_dir,
    )
    print(f"[manifest] {args.stage} → {mpath}")


if __name__ == "__main__":
    main()
