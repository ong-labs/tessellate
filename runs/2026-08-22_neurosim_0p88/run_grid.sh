#!/bin/sh
export TESSELLATE_QAT_TRAIN_SIZE=4096
./.venv/Scripts/python.exe _grid_probe.py \
  examples/hw_configs/calibrated_neurosim_v1_5.json 0 2>&1 | tee _grid_run.log
echo "EXIT=$?"
