#!/usr/bin/env bash
# Run the paper's pipeline from a single config.
#
# Usage:
#   PROJECT_ROOT=/path/to/data ./run_pipeline.sh [config/canonical240.json] [stages]
#
# Examples:
#   ./run_pipeline.sh                              # all validation-set stages
#   ./run_pipeline.sh config/globalmood200.json    # GlobalTap benchmark (1,3,4)
#   ./run_pipeline.sh config/canonical240.json 4,5 # only stages 4 and 5
#   ./run_pipeline.sh config/canonical240.json 1,2,3,3b,4,5,6,7  # with Beat This!
#   SKIP_MADMOM=1 ./run_pipeline.sh                # without algorithmic baseline
#
# Stages 2, 5, 6 and 7 read the reference-annotation map named by the config.
# Datasets collected without external references set gt_source_map to null, so
# those stages are skipped automatically. For the GlobalTap benchmark, this
# leaves stages 1, 3 and 4. Its Stage-2 KDE peaks are rebuilt from the released
# taps with scripts/kde_peaks_from_taps.py instead.
#
# Both configs also name a stimulus manifest (data/sampled_stimuli_*.json) that
# lives in the raw-data tree under ${PROJECT_ROOT}, not in this repository.
#
# Each stage writes outputs under ${PROJECT_ROOT}/outputs/<dataset>/<stage>/
# plus a per-stage manifest.json. After all stages complete, a top-level
# outputs/<dataset>/manifest.json aggregates them for provenance.

set -e
cd "$(dirname "$0")"

CONFIG="${1:-config/canonical240.json}"
STAGES="${2:-1,2,3,4,5,6,7}"
PY="${PYTHON_BIN:-./.venv/bin/python}"

if [ -z "$PROJECT_ROOT" ]; then
  echo "[run_pipeline] ERROR: set PROJECT_ROOT to the absolute path of your data tree." >&2
  echo "  e.g. export PROJECT_ROOT=/path/to/globaltap/data" >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "[run_pipeline] ERROR: config $CONFIG not found." >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "[run_pipeline] ERROR: python interpreter $PY not found." >&2
  echo "  Set PYTHON_BIN=/path/to/python or create ./.venv." >&2
  exit 1
fi

# Resolve config values, expanding ${PROJECT_ROOT} in data_root.
read_cfg() { $PY -c "
import json, os, sys
c = json.load(open('$CONFIG'))
v = c.get('$1', '')
if isinstance(v, str):
    v = os.path.expandvars(v)
print(v)
"; }

DATA_ROOT=$(read_cfg data_root)
DATASET=$(read_cfg dataset)
OUT_REL=$(read_cfg outputs_root)
PVER=$(read_cfg pipeline_version)
GT_MAP=$(read_cfg gt_source_map)
OUT_ROOT="$DATA_ROOT/$OUT_REL"

# Reference-free datasets (gt_source_map: null) cannot run the stages that
# resolve that map. Drop them from the requested list rather than failing part
# way through.
if [ -z "$GT_MAP" ] || [ "$GT_MAP" = "None" ]; then
  KEPT=""
  for s in ${STAGES//,/ }; do
    case "$s" in
      2|5|6|7)
        echo "[run_pipeline] NOTE: skipping stage $s: $CONFIG has no gt_source_map." ;;
      *)
        KEPT="${KEPT:+$KEPT,}$s" ;;
    esac
  done
  STAGES="$KEPT"
  if [ -z "$STAGES" ]; then
    echo "[run_pipeline] ERROR: none of the requested stages can run without a" >&2
    echo "  reference-annotation map. See scripts/kde_peaks_from_taps.py for the" >&2
    echo "  benchmark Stage-2 path." >&2
    exit 1
  fi
fi

echo "==================================================================="
echo "[run_pipeline] dataset:          $DATASET"
echo "[run_pipeline] pipeline_version: $PVER"
echo "[run_pipeline] config:           $CONFIG"
echo "[run_pipeline] data_root:        $DATA_ROOT"
echo "[run_pipeline] outputs_root:     $OUT_ROOT"
echo "[run_pipeline] stages to run:    $STAGES"
echo "[run_pipeline] python:           $PY"
echo "==================================================================="

mkdir -p "$OUT_ROOT"

stage_has() { [[ ",$STAGES," == *",$1,"* ]]; }

write_manifest() {
  local stage="$1"
  local subdir="$2"
  shift 2
  local inputs=("$@")
  $PY -m lib.manifest_cli \
    --config "$CONFIG" \
    --stage "$stage" \
    --outputs_dir "$OUT_ROOT/$subdir" \
    --inputs "${inputs[@]}"
}

start_ts=$(date +%s)

if stage_has 0; then
  echo "[run_pipeline] --- Stage 0: build GT source map ---"
  $PY pipeline/0_build_gt_source_map.py --config "$CONFIG"
fi

if stage_has 1; then
  echo "[run_pipeline] --- Stage 1: merge taps ---"
  $PY pipeline/1_merge_exports.py --config "$CONFIG"
  write_manifest 1_merge_exports merged
fi

if stage_has 2; then
  echo "[run_pipeline] --- Stage 2: REPP alignment + KDE peaks ---"
  $PY pipeline/2_repp_align.py --config "$CONFIG"
  write_manifest 2_repp_align repp
fi

if stage_has 3 && [ -z "$SKIP_MADMOM" ]; then
  echo "[run_pipeline] --- Stage 3: madmom DBN ---"
  $PY pipeline/3_run_madmom.py --config "$CONFIG"
  write_manifest 3_run_madmom madmom
fi

# Stage 3b needs the separate beat_this install (see requirements.txt), so it is
# opt-in: pass it in the stage list, e.g. ./run_pipeline.sh <config> 1,2,3,3b,4.
if stage_has 3b; then
  echo "[run_pipeline] --- Stage 3b: Beat This! transformer ---"
  $PY pipeline/3_run_beat_this.py --config "$CONFIG"
  write_manifest 3_run_beat_this beat_this
fi

if stage_has 4; then
  echo "[run_pipeline] --- Stage 4: crowd_gs optimizer (main) ---"
  $PY pipeline/4_run_crowd_gs.py --config "$CONFIG" --variant crowd_gs
  write_manifest 4_run_crowd_gs crowd_gs
fi

if stage_has 5; then
  echo "[run_pipeline] --- Stage 5: compare variants ---"
  $PY pipeline/5_compare_variants.py --config "$CONFIG"
  write_manifest 5_compare_variants compare
fi

if stage_has 6; then
  echo "[run_pipeline] --- Stage 6: reference/crowd period comparison ---"
  $PY pipeline/6_period_audit.py --config "$CONFIG"
  write_manifest 6_period_audit compare
fi

if stage_has 7; then
  echo "[run_pipeline] --- Stage 7: render click tracks ---"
  $PY pipeline/7_render_clicks.py --config "$CONFIG"
  write_manifest 7_render_clicks clicks
fi

echo "[run_pipeline] --- Aggregating run manifest ---"
$PY -m lib.manifest_cli --config "$CONFIG" --aggregate

elapsed=$(( $(date +%s) - start_ts ))
echo "[run_pipeline] DONE in ${elapsed}s.  Outputs: $OUT_ROOT"
