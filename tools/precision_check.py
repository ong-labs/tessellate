"""Stage-2 CIM verification (docs/coding_guidelines.md 5.D "Mock First" -- graduating in
place): the ADC energy term is a real, literature-fit model
(`hwcomponents_adc`, https://arxiv.org/abs/2404.06553); DAC and crossbar-MAC
energy are now real NeuroSim device models (`hwcomponents_neurosim`) too,
when that package is installed; SRAM buffer-staging energy is a real
CACTI model (`hwcomponents_cacti`) when *that* package is installed --
everywhere else, all four terms fall back to tools/cim_physics.py's
analytical placeholder.

`nodes/precision_verifier.py` calls this only for a candidate @evaluator's
fast-approximation already judged converged (graph.py routes evaluator's
"Converged (Done)" edge to @precision_verifier instead of straight to END)
-- never every candidate a run searches. A real precise simulator call is
far more expensive than tools/cim_physics.py's analytical formulas;
re-checking only the (rare) candidates a run is about to stop on keeps
that cost bounded regardless of how many candidates the search tries.

A local process without `hwcomponents_neurosim`/`hwcomponents_cacti`
importable (this project's own Windows dev environment included) can still
get their real numbers for these rare converged-candidate checks without
running the whole search/QAT pipeline inside a container: opt into
`AUTOCIM_PRECISION_BACKEND=docker` (default `local`, same local-first-by-
default/opt-in-cloud pattern `llm.py`'s `AUTOCIM_PLANNER_MODEL` already
uses) and `simulate_precise_energy` shells out to this project's own Docker
image (`docker build -t autocim-agent .`) via `_call_docker_precision_bridge`
for just this one computation. Never required and never blocking: any
failure (Docker not installed, the image missing, the daemon not running, a
timeout) silently falls back to this module's own local per-term real/
analytical split, exactly as if the env var were unset.
`_call_docker_precision_bridge` keeps a long-lived container running
(`_ensure_docker_container_running`, `docker exec` per call) across an
entire session -- and across separate `python main.py` processes, since
it's left running rather than torn down -- instead of paying a fresh
`docker run`'s container-creation cost on every call; falls back to a
one-shot `docker run --rm` if warm-keeping itself fails.

Why `hwcomponents_neurosim`/`hwcomponents_cacti` are optional, not plain
dependencies: `hwcomponents-neurosim` bundles NeuroSim's real C++ source
and compiles it via `make`/`g++` at `pip install` time -- this project's
own Windows dev environment has no C++ toolchain, so it can't be installed
there at all. `hwcomponents-cacti` ships a pre-built Linux binary and needs
no compile step, but imports the POSIX-only `fcntl` module, so it fails to
even import on Windows. Both are Docker-only extras (requirements-docker.txt,
installed on top of requirements.txt inside the Dockerfile's build-essential
Linux image) instead of requirements.txt entries every environment must
satisfy. `hwcomponents-cacti` was initially evaluated for this module's
DAC/crossbar-MAC gap and rejected for that purpose -- it doesn't model DAC
or crossbar-MAC at all, its `SRAM`/`Cache`/`DDR3` components are buffer/DRAM
memory-array energy -- but that's exactly what `HWConfig.sram_buffer_kb`
needed once `tools.cim_physics.simulate_cim_profile` grew a `buffer_energy_pj`
term for it, so it's used for that instead (`_real_sram_buffer_energy_fj`).
CiMLoop itself (the tool this project's roadmap originally named) is no
longer maintained and only ever shipped via Docker, and wasn't evaluated
further once NeuroSim/CACTI's own components covered every remaining term.

`_real_dac_conversion_energy_fj` (NeuroSim's `RowDrivers`, a voltage DAC
driving crossbar rows) and `_real_crossbar_mac_energy_fj_per_pass`
(NeuroSim's `MemoryCell`, the analog array itself) return `None` when
`hwcomponents_neurosim` isn't importable -- `simulate_precise_energy` reads
that the same way it already reads `_real_adc_conversion_energy_fj`
returning `None` below `_MIN_REAL_ADC_BITS`: keep the analytical term for
that one candidate rather than crash or fabricate a number.

Both real NeuroSim calls select a device-specific `.cell` config via
`HWConfig.device_type` (`_neurosim_cell_config_filename`) -- RRAM/SRAM/PCM/
STTRAM map to one of NeuroSim's bundled NVMExplorer-fit cells; `None`
(no device technology stated) falls back to NeuroSim's generic placeholder
cell instead of claiming a specific device this run never stated. Without
a real device modeled, even the Docker-only NeuroSim numbers describe a
fictional generic cell, not a citable real chip -- setting `device_type`
is what turns them into (still approximate, but device-grounded) numbers.

Also doubles as the feed for tools/calibration.py's calibration_factors:
one precision check both (a) judges this one candidate against the real
targets using a more-trustworthy energy number, and (b) derives an updated
calibration_factor from the gap between the analytical prediction and that
more-precise number.
"""

import json
import os
import subprocess
from typing import Dict, List, Optional

from hwcomponents_adc import ADC

from schemas.config import CIMDeviceType, HWConfig
from schemas.tools import LayerBitConfig
from tools.cim_physics import CIM_TECH_PARAMS, simulate_cim_profile

try:
    from hwcomponents_neurosim import ColDrivers, MemoryCell, RowDrivers
    from hwcomponents_neurosim.main import CACHE as _NEUROSIM_CACHE
    from hwcomponents_neurosim.main import PERMITTED_TECH_NODES as _NEUROSIM_TECH_NODES_M
    from hwcomponents_neurosim.main import SCRIPT_DIR as _NEUROSIM_SCRIPT_DIR

    _NEUROSIM_AVAILABLE = True
except ImportError:
    _NEUROSIM_AVAILABLE = False

try:
    from hwcomponents_cacti import SRAM as CactiSRAM

    _CACTI_AVAILABLE = True
except ImportError:
    _CACTI_AVAILABLE = False

# hwcomponents_adc.ADC requires a technology node (meters) to scale its
# fit against -- HWConfig.tech_node_nm is optional (not every hardware spec
# in this project's examples/hw_configs/ states one), so this is the same
# "documented placeholder, never a bare literal at the call site" pattern
# device_noise_sigma/sram_buffer_kb already use. 45nm is a common node for
# published CIM/ADC prototypes in the ADC survey this model is fit against,
# not a claim about any specific hw_spec_id's real process.
_DEFAULT_TECH_NODE_NM = 45.0


# hwcomponents_adc.ADC's own real-model floor: it asserts n_bits >= 4,
# since the published ADC survey it's fit against (arXiv:2404.06553) has
# too little sub-4-bit data to fit a reliable curve. HWConfig.adc_bits
# itself allows down to 1 (schemas/config.py), so this project's own
# examples/hw_configs/ can genuinely go below what the real model covers.
_MIN_REAL_ADC_BITS = 4


# hwcomponents_neurosim's RowDrivers/MemoryCell (via _NeurosimPIMComponent)
# take cols_active_at_once/cell_config/average_input_value/
# average_cell_value/read_pulse_width/cycle_period as *required* constructor
# args -- no Python-level default -- but the package's own PIM_PARAMS table
# documents a recommended default for each (used by its own CLI/docs
# generator). HWConfig has no field for any of these (no examples/
# hw_configs/ entry models column-multiplexing ratio, cell technology,
# input/cell activity statistics, or pulse/cycle timing), so this project
# uses the library's own documented defaults rather than inventing values,
# the same "documented placeholder, never a bare literal at the call site"
# pattern _DEFAULT_TECH_NODE_NM already uses.
_NEUROSIM_COLS_ACTIVE_AT_ONCE = 8  # clamped to hw.crossbar_cols: NeuroSim rejects cols_active_at_once > cols
_NEUROSIM_AVERAGE_INPUT_VALUE = 1.0
_NEUROSIM_AVERAGE_CELL_VALUE = 1.0
_NEUROSIM_READ_PULSE_WIDTH_S = 1e-8  # 10ns
_NEUROSIM_CYCLE_PERIOD_S = 1e-9  # 1ns

# cell_config must be a real filesystem path (the package resolves it with a
# bare `open()`, not a SAMPLE_CELLS name lookup, despite PIM_PARAMS's doc
# string suggesting otherwise) -- these map HWConfig.device_type to one of
# hwcomponents_neurosim's bundled NVMExplorer-fit .cell files. Picked the
# NVMExplorer variant over the older nvsim_* one for each technology where
# both exist (RRAM, PCM); no nvsim_SRAM/PCM-equivalent choice was needed --
# only one SRAM cell ships. STTRAM stands in for CIMDeviceType.STTRAM only
# (see that enum's docstring for why there's no separate generic-MRAM cell).
_NEUROSIM_DEVICE_CELL_FILENAMES = {
    CIMDeviceType.RRAM: "nvmexplorer_RRAM.cell",
    CIMDeviceType.SRAM: "nvmexplorer_SRAM.cell",
    CIMDeviceType.PCM: "nvmexplorer_PCM.cell",
    CIMDeviceType.STTRAM: "nvmexplorer_sttram.cell",
}
_NEUROSIM_PLACEHOLDER_CELL_FILENAME = "placeholder.cell"


# hwcomponents_cacti.SRAM's own real-model range: 22-180nm, a continuous
# range (unlike hwcomponents_neurosim's fixed node list) but still not
# arbitrary -- clamps into it instead of raising, same
# "documented placeholder, never crash" pattern as
# _nearest_neurosim_tech_node_m, using the same _DEFAULT_TECH_NODE_NM
# fallback (45nm, itself within CACTI's supported range) when unset.
_CACTI_MIN_TECH_NODE_NM = 22.0
_CACTI_MAX_TECH_NODE_NM = 180.0


def _neurosim_cell_config_filename(device_type: Optional[CIMDeviceType]) -> str:
    """Maps `hw.device_type` to one of hwcomponents_neurosim's bundled real
    .cell files -- `None` (no device technology stated, HWConfig's default)
    falls back to NeuroSim's generic placeholder cell, the same honesty
    pattern `_DEFAULT_TECH_NODE_NM` already uses. Pure filename lookup,
    independent of `_NEUROSIM_AVAILABLE` and testable without
    `hwcomponents_neurosim` installed; `_real_dac_conversion_energy_fj`/
    `_real_crossbar_mac_energy_fj_per_pass` join the result with the
    package's own install path (`_NEUROSIM_SCRIPT_DIR`, only bound when the
    package is importable) to get a real, openable path."""
    return _NEUROSIM_DEVICE_CELL_FILENAMES.get(device_type, _NEUROSIM_PLACEHOLDER_CELL_FILENAME)


def _nearest_neurosim_tech_node_m(tech_node_nm: Optional[float]) -> float:
    """hwcomponents_neurosim only ships characterized data for a fixed set
    of process nodes (`PERMITTED_TECH_NODES`, 7-130nm) and rejects any other
    value outright -- unlike hwcomponents_adc's continuous fit. Snaps to the
    nearest supported node instead of raising, using the same
    `_DEFAULT_TECH_NODE_NM` fallback as the ADC term when `hw.tech_node_nm`
    is unset, so the DAC/crossbar terms agree with the ADC term on which
    literature process they represent."""
    tech_node_m = (tech_node_nm or _DEFAULT_TECH_NODE_NM) * 1e-9
    return min(_NEUROSIM_TECH_NODES_M, key=lambda node_m: abs(node_m - tech_node_m))


def _clamped_cacti_tech_node_m(tech_node_nm: Optional[float]) -> float:
    """hwcomponents_cacti.SRAM only supports 22-180nm -- clamps into that
    range instead of raising, using the same `_DEFAULT_TECH_NODE_NM`
    fallback as the ADC/DAC/crossbar terms when `hw.tech_node_nm` is unset."""
    nm = tech_node_nm or _DEFAULT_TECH_NODE_NM
    clamped_nm = min(max(nm, _CACTI_MIN_TECH_NODE_NM), _CACTI_MAX_TECH_NODE_NM)
    return clamped_nm * 1e-9


def _real_adc_conversion_energy_fj(hw: HWConfig) -> Optional[float]:
    """One real ADC conversion's energy (femtojoules), from
    `hwcomponents_adc`'s literature-fit model (arXiv:2404.06553) -- `n_bits`
    comes directly from `hw.adc_bits` (never hardcoded, docs/coding_guidelines.md 5.C).
    `throughput` is derived from `tools.cim_physics.CIM_TECH_PARAMS`'s
    existing `adc_cycle_time_ns` (the same per-bit SAR-ADC timing the
    analytical model already assumes) rather than a second, disconnected
    timing assumption -- both models agree on how fast this ADC runs, they
    only disagree on how much energy that costs.

    Returns `None` below `_MIN_REAL_ADC_BITS` -- the real model has no
    valid answer there (it raises rather than extrapolate), and
    `simulate_precise_energy` reads `None` as "keep the analytical term
    for this candidate" rather than crashing or fabricating a number
    outside what the real model was ever fit to represent."""
    if hw.adc_bits < _MIN_REAL_ADC_BITS:
        return None
    conversion_time_ns = hw.adc_bits * CIM_TECH_PARAMS.adc_cycle_time_ns
    throughput_hz = 1e9 / conversion_time_ns
    tech_node_m = (hw.tech_node_nm or _DEFAULT_TECH_NODE_NM) * 1e-9
    adc = ADC(n_bits=hw.adc_bits, tech_node=tech_node_m, throughput=throughput_hz)
    return adc.convert().energy * 1e15  # joules -> femtojoules


def _real_dac_conversion_energy_fj(hw: HWConfig) -> Optional[float]:
    """One layer's real row-driver energy for asserting all
    `hw.crossbar_rows` rows through a `hw.dac_bits`-resolution voltage DAC,
    from `hwcomponents_neurosim`'s `RowDrivers` -- the same "one array-wide
    drive per layer" granularity `simulate_cim_profile`'s own DAC term
    already assumes (`dac_ref_energy_fj * dac_bits * crossbar_rows`), just
    replacing that per-bit-per-row constant with a real characterized
    value for driving the whole array's rows at once.

    Returns `None` when `hwcomponents_neurosim` isn't installed (see module
    docstring) -- `simulate_precise_energy` then keeps the analytical DAC
    term for that candidate, same fallback shape as the ADC term above.

    `_NEUROSIM_CACHE.clear()`: hwcomponents_neurosim caches its underlying
    Crossbar simulation keyed on rows/cols/tech_node/timing/etc. -- but not
    `cell_config` (verified by direct inspection of its cache-key
    construction). Two hw_specs sharing those other dimensions but
    different `device_type` would otherwise silently reuse whichever
    cell's simulation ran first, making device_type a no-op for the
    second one. Clearing before every real call trades away that cache's
    speedup (irrelevant here: precision_check only runs for the rare
    converged candidates a run is about to stop on, never the hot path)
    for actually getting the device-specific number requested."""
    if not _NEUROSIM_AVAILABLE:
        return None
    _NEUROSIM_CACHE.clear()
    drivers = RowDrivers(
        tech_node=_nearest_neurosim_tech_node_m(hw.tech_node_nm),
        cycle_period=_NEUROSIM_CYCLE_PERIOD_S,
        rows=hw.crossbar_rows,
        cols=hw.crossbar_cols,
        cols_active_at_once=min(hw.crossbar_cols, _NEUROSIM_COLS_ACTIVE_AT_ONCE),
        cell_config=os.path.join(_NEUROSIM_SCRIPT_DIR, "cells", _neurosim_cell_config_filename(hw.device_type)),
        average_input_value=_NEUROSIM_AVERAGE_INPUT_VALUE,
        average_cell_value=_NEUROSIM_AVERAGE_CELL_VALUE,
        read_pulse_width=_NEUROSIM_READ_PULSE_WIDTH_S,
        voltage_dac_bits=hw.dac_bits,
    )
    return drivers.read().energy * 1e15  # joules -> femtojoules


def _real_crossbar_mac_energy_fj_per_pass(hw: HWConfig) -> Optional[float]:
    """One full-array read/MAC pass's real energy: `hwcomponents_neurosim`'s
    `MemoryCell` (the analog compute a real crossbar's cells perform when
    its rows are asserted) plus `ColDrivers` with `adc_resolution=0` (the
    column precharge/select energy that same pass also costs, separate from
    both `MemoryCell` and the `RowDrivers`/ADC peripheral energy modeled
    elsewhere). Both are genuinely separate physical contributions to one
    pass, not overlapping ones -- `adc_resolution=0` on `ColDrivers`
    specifically excludes any sense-amp/ADC conversion energy (confirmed by
    direct measurement: `ColDrivers(adc_resolution=N).read().energy` -
    `ColDrivers(adc_resolution=0).read().energy` equals a standalone
    `hwcomponents_neurosim.ADC(adc_resolution=N)` call almost exactly), so
    this never double-counts the ADC term `_real_adc_conversion_energy_fj`
    already models via `hwcomponents_adc` (a different, literature-fit
    package, deliberately preferred over this one's own bundled generic-cell
    ADC model -- see module docstring).

    Precharge is not optional for `MemCellType: SRAM`: NeuroSim's own bundled
    `nvmexplorer_SRAM.cell` carries only cell geometry (`CellArea`/
    `CellAspectRatio`), never resistance/read-voltage/read-power fields --
    `MemoryCell` models cell energy purely as resistive (Ohm's-law) current
    through the cell, a model that is architecturally inapplicable to a
    digital 6T SRAM bitcell (it has no such resistance to characterize), so
    `MemoryCell.read().energy` alone is always exactly `0.0` for SRAM, not a
    genuinely near-zero real measurement -- confirmed against NeuroSim's own
    compiled C++ core (`DNN_NeuroSim_V1.3/.../SubArray.cpp`), whose SRAM
    branch sets `readDynamicEnergyArray = 0 // Just BL discharging` but
    still sums non-zero `precharger`/`senseAmp`/`wlDecoder`/`adder`/`dff`
    energy into its own total. `ColDrivers` is this package's Python-level
    equivalent of that `precharger` contribution, so adding it is what turns
    an architecturally-zero SRAM number into a physically real one, without
    needing (and without SRAM having) any resistance data to plug in.

    `simulate_cim_profile`'s analytical `crossbar_ref_energy_fj * rows *
    cols * weight_bits` term already assumes bit-serial CIM: `weight_bits`
    per-layer passes over the same array. This returns the energy of *one*
    such pass; callers multiply by a layer's `weight_bits` themselves,
    mirroring how the analytical term already scales its own per-pass
    constant.

    Returns `None` when `hwcomponents_neurosim` isn't installed (see module
    docstring) -- same fallback shape as the ADC/DAC terms.

    `_NEUROSIM_CACHE.clear()`: see `_real_dac_conversion_energy_fj`'s
    docstring -- same missing-`cell_config`-in-cache-key issue applies here."""
    if not _NEUROSIM_AVAILABLE:
        return None
    _NEUROSIM_CACHE.clear()
    common_kwargs = dict(
        tech_node=_nearest_neurosim_tech_node_m(hw.tech_node_nm),
        cycle_period=_NEUROSIM_CYCLE_PERIOD_S,
        rows=hw.crossbar_rows,
        cols=hw.crossbar_cols,
        cols_active_at_once=min(hw.crossbar_cols, _NEUROSIM_COLS_ACTIVE_AT_ONCE),
        cell_config=os.path.join(_NEUROSIM_SCRIPT_DIR, "cells", _neurosim_cell_config_filename(hw.device_type)),
        average_input_value=_NEUROSIM_AVERAGE_INPUT_VALUE,
        average_cell_value=_NEUROSIM_AVERAGE_CELL_VALUE,
        read_pulse_width=_NEUROSIM_READ_PULSE_WIDTH_S,
    )
    cell = MemoryCell(**common_kwargs)
    col_drivers = ColDrivers(**common_kwargs, adc_resolution=0)
    return (cell.read().energy + col_drivers.read().energy) * 1e15  # joules -> femtojoules


def _real_sram_buffer_energy_fj(hw: HWConfig) -> Optional[float]:
    """Real one-time SRAM-buffer-fill energy from `hwcomponents_cacti`'s
    `SRAM.write()` -- the same "buffer filled once per profile, not per
    layer" granularity `tools.cim_physics.simulate_cim_profile`'s own
    `buffer_energy_pj` term already assumes for this same field. `width`
    is left to CACTI's own default (derived from `size` -- no project
    convention exists for on-chip buffer access width), so the number of
    word-writes needed to fill the whole buffer is `size / sram.width`
    (the resolved value, an attribute after construction).

    Returns `None` when `hw.sram_buffer_kb` is unset (nothing to model --
    `HWConfig.sram_buffer_kb`'s own `gt=0` constraint means this is exactly
    the "not every hardware spec states one" case, never zero/negative) or
    `hwcomponents_cacti` isn't installed (Docker-only, see module
    docstring) -- `simulate_precise_energy` reads either the same way it
    reads the ADC/DAC/crossbar terms returning `None`: keep the analytical
    buffer term for that candidate rather than crash or fabricate a number."""
    if not hw.sram_buffer_kb:
        return None
    if not _CACTI_AVAILABLE:
        return None
    buffer_bits = int(round(hw.sram_buffer_kb * 8 * 1024))
    sram = CactiSRAM(tech_node=_clamped_cacti_tech_node_m(hw.tech_node_nm), size=buffer_bits)
    n_accesses = buffer_bits / sram.width
    return sram.write().energy * 1e15 * n_accesses  # joules -> femtojoules, scaled to the whole buffer


_PRECISION_BACKEND_ENV_VAR = "AUTOCIM_PRECISION_BACKEND"
_DOCKER_IMAGE_ENV_VAR = "AUTOCIM_PRECISION_DOCKER_IMAGE"
_DEFAULT_DOCKER_IMAGE = "autocim-agent"
_DOCKER_CONTAINER_ENV_VAR = "AUTOCIM_PRECISION_DOCKER_CONTAINER"
_DEFAULT_DOCKER_CONTAINER = "autocim-precision-bridge"
# precision_check_tool (tools/simulators.py) only runs for the rare
# candidates @evaluator already judged converged, never the search hot
# path (see this module's docstring) -- a call every few minutes at most,
# so a generous timeout costs nothing in practice but avoids a hung `docker
# run`/`exec` silently stalling the whole agent loop.
_DOCKER_BRIDGE_TIMEOUT_S = 120.0
# Container lifecycle calls (inspect/rm/start) are local Docker-daemon
# bookkeeping, not the actual precision computation -- a short timeout is
# enough, and keeps a hung daemon from blocking as long as a real
# NeuroSim/CACTI-backed run would need.
_DOCKER_LIFECYCLE_TIMEOUT_S = 15.0


def _ensure_docker_container_running(image: str, container: str) -> bool:
    """Starts (or confirms already running) a long-lived `container` from
    `image` -- kept alive via `tail -f /dev/null` overriding the image's own
    `ENTRYPOINT ["python", "main.py"]` -- so repeated
    `_call_docker_precision_bridge` calls, across this whole session (not
    just one call), can `docker exec` into it instead of every single call
    paying `docker run`'s full container-creation cost. The container is
    otherwise stateless (each `docker exec ... tools.precision_check_cli`
    invocation is independent), so leaving it running between separate
    `python main.py` processes is intentional, not a leak -- reuse across
    back-to-back sessions is exactly the point; `docker stop/rm
    autocim-precision-bridge` (or the `AUTOCIM_PRECISION_DOCKER_CONTAINER`
    override name) cleans it up manually when no longer wanted.

    Returns `False` on any failure (docker missing, daemon down, a
    `--name`/`docker run -d` race against another process starting the same
    container concurrently, etc.) -- never raises, so
    `_call_docker_precision_bridge` can always fall back to the one-shot
    `docker run --rm` path exactly as if warm-keeping were never attempted."""
    try:
        inspected = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            capture_output=True,
            text=True,
            timeout=_DOCKER_LIFECYCLE_TIMEOUT_S,
        )
        if inspected.returncode == 0:
            if inspected.stdout.strip() == "true":
                return True
            # Exists but stopped/exited (e.g. a previous crash) -- clear it
            # so `docker run --name` below doesn't collide with the dead one.
            subprocess.run(
                ["docker", "rm", "-f", container], capture_output=True, text=True, timeout=_DOCKER_LIFECYCLE_TIMEOUT_S
            )
        started = subprocess.run(
            ["docker", "run", "-d", "--name", container, "--entrypoint", "tail", image, "-f", "/dev/null"],
            capture_output=True,
            text=True,
            timeout=_DOCKER_LIFECYCLE_TIMEOUT_S,
        )
        return started.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _call_docker_precision_bridge(
    hw: HWConfig, layer_configs: List[LayerBitConfig]
) -> Optional[Dict[str, float]]:
    """Opt-in (`AUTOCIM_PRECISION_BACKEND=docker`, default `local`, same
    local-first-by-default/opt-in-cloud pattern `llm.py`'s
    `AUTOCIM_PLANNER_MODEL` already uses) escape hatch for running this
    module's real computation inside the project's own Docker image
    (`docker build -t autocim-agent .`, `AUTOCIM_PRECISION_DOCKER_IMAGE` to
    override the tag) when the *local* process doesn't have
    `hwcomponents_neurosim`/`hwcomponents_cacti` importable -- e.g. this
    project's own Windows dev environment (no C++ toolchain, POSIX-only
    `fcntl`; see module docstring) -- without needing to run the entire
    search/QAT pipeline inside that (CPU-only) container too.

    Tries `_ensure_docker_container_running`'s warm container first
    (`docker exec`, cheap) and only falls back to a fresh one-shot `docker
    run --rm` (pays full container-creation cost) when warm-keeping itself
    failed for any reason -- both are still subject to the same "never
    raises" contract below.

    Never raises: any failure (`docker` not on PATH, the image not built,
    the daemon not running, a timeout, or a malformed response) returns
    `None` so `simulate_precise_energy` falls back to its own already-
    existing per-term analytical/real split exactly as if the bridge were
    never attempted -- docs/coding_guidelines.md 5.D's `wrap_tool_call` contract (never
    break the agent loop over a backend tool's exception) applies here too,
    one level up from the individual `_real_*_fj` functions that already
    follow it for a missing package."""
    image = os.environ.get(_DOCKER_IMAGE_ENV_VAR, _DEFAULT_DOCKER_IMAGE)
    container = os.environ.get(_DOCKER_CONTAINER_ENV_VAR, _DEFAULT_DOCKER_CONTAINER)
    payload = json.dumps(
        {
            "hw_config": hw.model_dump(mode="json"),
            "layer_configs": [lc.model_dump(mode="json") for lc in layer_configs],
        }
    )
    command = (
        ["docker", "exec", "-i", container, "python", "-m", "tools.precision_check_cli"]
        if _ensure_docker_container_running(image, container)
        else ["docker", "run", "--rm", "-i", "--entrypoint", "python", image, "-m", "tools.precision_check_cli"]
    )
    try:
        proc = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            text=True,
            timeout=_DOCKER_BRIDGE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def simulate_precise_energy(hw: HWConfig, layer_configs: List[LayerBitConfig]) -> Dict[str, float]:
    """Recomputes `tools.cim_physics.simulate_cim_profile`'s energy with its
    analytical ADC/DAC/crossbar-MAC/SRAM-buffer terms each independently
    replaced by a real model when one is available -- ADC via
    `_real_adc_conversion_energy_fj` (always, `hwcomponents_adc` is a plain
    dependency), DAC and crossbar-MAC via `_real_dac_conversion_energy_fj`/
    `_real_crossbar_mac_energy_fj_per_pass` (only where `hwcomponents_neurosim`
    is installed), and SRAM buffer energy via `_real_sram_buffer_energy_fj`
    (only where `hwcomponents_cacti` is installed -- see module docstring).
    Each term keeps `simulate_cim_profile`'s own per-layer/per-column/
    per-pass/per-buffer multiplicity convention, so this only swaps *which*
    number represents each physical quantity, never the quantity itself.
    Any term whose real model returns `None` (below `_MIN_REAL_ADC_BITS`,
    `hw.sram_buffer_kb` unset, or the relevant package unavailable) keeps
    its unmodified analytical value instead of crashing or fabricating a
    number.

    Returns both the total (`raw_energy_pj`/`precise_energy_pj`, ADC+DAC+
    crossbar+buffer -- what @evaluator's Pareto ranking and target-checking
    use) and the compute-only component (`raw_compute_energy_pj`/
    `precise_compute_energy_pj`, ADC+DAC+crossbar only) -- the latter is
    what `compute_updated_calibration_factor` must be evaluated against
    (see that function and `tools.calibration.compute_calibration_factor`
    for why buffer energy is excluded from calibration).

    When `AUTOCIM_PRECISION_BACKEND=docker` and this process doesn't already
    have `hwcomponents_neurosim`/`hwcomponents_cacti` importable, tries
    `_call_docker_precision_bridge` first and returns its result verbatim if
    it succeeds -- already running inside a container with both real
    (`_NEUROSIM_AVAILABLE and _CACTI_AVAILABLE`) skips the bridge entirely
    (no docker-in-docker), and any bridge failure falls through to this
    function's own local computation below exactly as if the env var were
    never set, per `_call_docker_precision_bridge`'s own docstring."""
    if os.environ.get(_PRECISION_BACKEND_ENV_VAR, "local") == "docker" and not (
        _NEUROSIM_AVAILABLE and _CACTI_AVAILABLE
    ):
        bridged = _call_docker_precision_bridge(hw, layer_configs)
        if bridged is not None:
            return bridged

    raw_profile = simulate_cim_profile(hw, layer_configs)
    raw_energy_pj = raw_profile["energy_pj"]
    raw_compute_energy_pj = raw_profile["compute_energy_pj"]
    raw_buffer_energy_pj = raw_profile["buffer_energy_pj"]

    n_layers = len(layer_configs)
    compute_delta_fj = 0.0

    n_conversions = hw.crossbar_cols * n_layers
    real_adc_conversion_fj = _real_adc_conversion_energy_fj(hw)
    if real_adc_conversion_fj is not None:
        analytical_adc_fj = CIM_TECH_PARAMS.adc_ref_energy_fj * (2**hw.adc_bits) * n_conversions
        compute_delta_fj += real_adc_conversion_fj * n_conversions - analytical_adc_fj

    real_dac_fj_per_layer = _real_dac_conversion_energy_fj(hw)
    if real_dac_fj_per_layer is not None:
        analytical_dac_fj = CIM_TECH_PARAMS.dac_ref_energy_fj * hw.dac_bits * hw.crossbar_rows * n_layers
        compute_delta_fj += real_dac_fj_per_layer * n_layers - analytical_dac_fj

    real_crossbar_fj_per_pass = _real_crossbar_mac_energy_fj_per_pass(hw)
    if real_crossbar_fj_per_pass is not None:
        analytical_crossbar_fj = sum(
            CIM_TECH_PARAMS.crossbar_ref_energy_fj * hw.crossbar_rows * hw.crossbar_cols * lc.weight_bits
            for lc in layer_configs
        )
        real_crossbar_fj = sum(real_crossbar_fj_per_pass * lc.weight_bits for lc in layer_configs)
        compute_delta_fj += real_crossbar_fj - analytical_crossbar_fj

    precise_compute_energy_pj = round(raw_compute_energy_pj + compute_delta_fj / 1000.0, 6)

    buffer_delta_fj = 0.0
    real_buffer_fj = _real_sram_buffer_energy_fj(hw)
    if real_buffer_fj is not None:
        analytical_buffer_fj = CIM_TECH_PARAMS.sram_buffer_ref_energy_fj_per_bit * (hw.sram_buffer_kb * 8 * 1024)
        buffer_delta_fj = real_buffer_fj - analytical_buffer_fj
    precise_buffer_energy_pj = round(raw_buffer_energy_pj + buffer_delta_fj / 1000.0, 6)

    precise_energy_pj = round(precise_compute_energy_pj + precise_buffer_energy_pj, 6)

    return {
        "raw_energy_pj": raw_energy_pj,
        "precise_energy_pj": precise_energy_pj,
        "raw_compute_energy_pj": raw_compute_energy_pj,
        "precise_compute_energy_pj": precise_compute_energy_pj,
    }


def compute_updated_calibration_factor(raw_energy_pj: float, precise_energy_pj: float) -> float:
    """Same ratio `tools.calibration.compute_calibration_factor` computes
    from a published literature reference -- here from this run's own
    precision check instead of a citation. `raw_energy_pj` must be the
    *pre*-calibration analytical prediction (`simulate_precise_energy`'s
    own fresh `simulate_cim_profile` call), never the already-calibrated
    number @profiler reported (`ProfilerToolOutput.data['energy_pj']`) --
    that already has an old factor baked in, and dividing by it would
    apply that factor twice."""
    if raw_energy_pj <= 0:
        return 1.0
    return precise_energy_pj / raw_energy_pj
