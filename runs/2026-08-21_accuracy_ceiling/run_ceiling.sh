#!/bin/sh
out=_ceiling_results.txt
: > "$out"
for n in 4096 8192 16384; do
  echo "=== TRAIN_SIZE=$n ===" >> "$out"
  TESSELLATE_QAT_TRAIN_SIZE=$n ./.venv/Scripts/python.exe _ceiling_probe.py \
    examples/hw_configs/calibrated_neurosim_v1_5.json 0 2>&1 \
    | grep -E "PROBE_RESULT|Error" >> "$out"
done
echo "DONE" >> "$out"
