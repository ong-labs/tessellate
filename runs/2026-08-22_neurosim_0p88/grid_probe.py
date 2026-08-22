"""Grid baseline on the same hardware the 150-iteration capture run used.

The 2026-08-20 grid comparison in README.md was measured on
default_cim_v1_128x128; slides 13/14 now report the 2026-08-22 run on
calibrated_neurosim_v1_5, so the two sat on different hardware. This
re-measures only the 15 grid points -- the search side is already done and
does not need re-running.

The grid deliberately keeps its default weight_bits (4, 6, 8) even though
the search on this hw_spec_id is bounded to 2..7 by min(adc_bits,
dac_bits): the grid is the "no search at all, sweep it systematically"
reference, and letting it reach a precision the search may not use makes
the comparison harder on us, not easier.
"""
import json
import sys
import time

import main
from tools.baseline_compare import best_grid_point, run_grid_search_baseline

hw_config = main.load_hw_config(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

t0 = time.time()
results = run_grid_search_baseline("resnet18", hw_config, seed)
best = best_grid_point(results)

out = {
    "hw_spec_id": hw_config.hw_spec_id,
    "run_seed": seed,
    "n_points": len(results),
    "seconds": round(time.time() - t0, 1),
    "best": best,
    "points": results,
}
print("GRID_RESULT " + json.dumps(out, ensure_ascii=False))
