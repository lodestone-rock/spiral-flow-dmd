#!/usr/bin/env bash
# End-to-end reproduction: spiral CFG flow-matching teacher + DMD/DMD2 distillation.
# Usage: ./run_all.sh [device] [teacher_run] [dmd_run]
set -euo pipefail
cd "$(dirname "$0")"
DEVICE="${1:-cuda}"
TEACHER_RUN="${2:-teacher}"
DMD_RUN="${3:-dmd}"
PYTHON="${PYTHON:-python}"

"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" train_flow.py all --device "$DEVICE" --run-name "$TEACHER_RUN"
"$PYTHON" dmd.py all --device "$DEVICE" --run-name "$DMD_RUN" --teacher-run "$TEACHER_RUN"
"$PYTHON" interpolate.py --device "$DEVICE" --teacher-run "$TEACHER_RUN"

echo "Done. Figures:"
echo "  runs/$TEACHER_RUN/teacher_samples.png"
echo "  runs/$TEACHER_RUN/interpolation_*/clockwise_paths_w1.png"
echo "  runs/$DMD_RUN/comparison_w2.png"
echo "  runs/$DMD_RUN/comparison_w4.png"
echo "  runs/$DMD_RUN/evaluation_*/<arm>/trajectory_progression.png"
