"""Docker-bridge entrypoint for `tools.precision_check.simulate_precise_energy`.

`tools/precision_check.py`'s `_call_docker_precision_bridge` (`AUTOCIM_PRECISION_BACKEND=docker`)
invokes this via `docker run ... python -m tools.precision_check_cli` so a
local (Windows, no C++ toolchain) run can still get real
`hwcomponents_neurosim`/`hwcomponents_cacti` numbers for the rare converged
candidate `@precision_verifier` checks, without running QAT/search inside
the (CPU-only) container too. Reads one `{"hw_config": ..., "layer_configs":
[...]}` JSON payload from stdin, writes `simulate_precise_energy`'s result
JSON to stdout -- the same schema-validated shapes `tools/simulators.py`'s
`precision_check_tool` already passes in-process, just round-tripped
through JSON instead of native Python objects."""

import json
import sys

from schemas.config import HWConfig
from schemas.tools import LayerBitConfig
from tools.precision_check import simulate_precise_energy


def main() -> None:
    payload = json.load(sys.stdin)
    hw = HWConfig(**payload["hw_config"])
    layer_configs = [LayerBitConfig(**lc) for lc in payload["layer_configs"]]
    result = simulate_precise_energy(hw, layer_configs)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
