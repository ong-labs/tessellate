"""Measure the attainable-accuracy ceiling at several QAT train-set sizes.

The naive baseline (uniform weight_bits=8, no pruning) is the closest thing
this repo has to "no compression at all", so its accuracy is the ceiling
that any searched candidate has to live under. README line 81 names
TESSELLATE_QAT_TRAIN_SIZE as the lever that moves this ceiling; this measures
how far 4096 -> 8192 -> 16384 actually moves it, before a 3-hour capture run
commits to a target that may sit above it.

TESSELLATE_QAT_TRAIN_SIZE is read when the QAT backend builds its loaders, so
each size runs in its own subprocess (see run_probe.py) rather than being
re-set in-process.
"""
import json
import os
import sys
import time

import main
from tools.baseline_compare import run_naive_baseline_replicate

hw_path = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

hw_config = main.load_hw_config(hw_path)
t0 = time.time()
res = run_naive_baseline_replicate("resnet18", hw_config, seed)
res["train_size"] = os.environ.get("TESSELLATE_QAT_TRAIN_SIZE")
res["seconds"] = round(time.time() - t0, 1)
res["hw_spec_id"] = hw_config.hw_spec_id
print("PROBE_RESULT " + json.dumps(res))
