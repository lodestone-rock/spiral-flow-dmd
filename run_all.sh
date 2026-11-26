#!/usr/bin/env bash
# End-to-end reproduction: spiral CFG flow-matching teacher + DMD/DMD2 + PDD.
# Usage: ./run_all.sh [device] [teacher_run] [dmd_run] [pdd_run]
#   CLASS_ALPHA=0.5 CLASS_SEED=4 ./run_all.sh ...   # random class-imbalanced data
set -euo pipefail
cd "$(dirname "$0")"
DEVICE="${1:-cuda}"
TEACHER_RUN="${2:-teacher}"
DMD_RUN="${3:-dmd}"
PDD_RUN="${4:-pdd}"
PYTHON="${PYTHON:-python}"
CLASS_ALPHA="${CLASS_ALPHA:-0}"
CLASS_SEED="${CLASS_SEED:-0}"
CLASS_FLOOR="${CLASS_FLOOR:-0.02}"

"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" train_flow.py all --device "$DEVICE" --run-name "$TEACHER_RUN" \
    --class-alpha "$CLASS_ALPHA" --class-seed "$CLASS_SEED" --class-floor "$CLASS_FLOOR"
"$PYTHON" dmd.py all --device "$DEVICE" --run-name "$DMD_RUN" --teacher-run "$TEACHER_RUN"
"$PYTHON" pdd.py all --device "$DEVICE" --run-name "$PDD_RUN" --teacher-run "$TEACHER_RUN" \
    --scales 2 --updates 50000 --dmd-run-name "$DMD_RUN" \
    --controls-from "$DMD_RUN/evaluation_s40_n4096_seed1234"
"$PYTHON" interpolate.py --device "$DEVICE" --teacher-run "$TEACHER_RUN"

echo "Done. Figures:"
echo "  runs/$TEACHER_RUN/teacher_samples.png"
echo "  runs/$TEACHER_RUN/interpolation_*/clockwise_paths_w1.png"
echo "  runs/$DMD_RUN/comparison_w2.png"
echo "  runs/$DMD_RUN/comparison_w4.png"
echo "  runs/$DMD_RUN/evaluation_*/<arm>/trajectory_progression.png"
echo "  runs/$PDD_RUN/pdd_samples_w2.png"
echo "  runs/$PDD_RUN/pdd_nfe_sweep_w2.png"
echo "  runs/$PDD_RUN/pdd_vs_dmd_w2.png"
