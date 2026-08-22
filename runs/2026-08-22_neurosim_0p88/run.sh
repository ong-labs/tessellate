#!/bin/sh
# Capture run for the presentation demo (slide 10 / slide 11 report).
#   --ignore-learned-state : literature calibration + a real LHS warm-up.
#                            Without it energy lands on a learned scale
#                            (166.93x vs 51.85x) and warm-up is skipped.
#   --auto-hitl-max-rounds : default is 2, which would end a 150-iteration
#                            run early with low_confidence_conclusion.
export AUTOCIM_QAT_TRAIN_SIZE=4096
./.venv/Scripts/python.exe -m main \
  --model-id resnet18 \
  --hw-config examples/hw_configs/calibrated_neurosim_v1_5.json \
  --target-accuracy 0.88 \
  --search-budget-iterations 150 \
  --auto-hitl --auto-hitl-max-rounds 30 \
  --ignore-learned-state \
  --seed 0 \
  --thread-id demo-neurosim-0p88 \
  --dashboard-out presentation/tessellate_report.html
echo "EXIT=$?"
