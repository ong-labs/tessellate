"""Configuration schemas for AutoCIM-Agent.

Hardware constraints and run-scoped identifiers live here. Node/tool logic
must load these values through `HWConfig`/`ExecutionContext` instances
rather than hardcoding hardware numbers or model architecture details
(see docs/coding_guidelines.md 5.C).
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NoCTopology(str, Enum):
    MESH = "mesh"
    TORUS = "torus"
    RING = "ring"
    CROSSBAR_BUS = "crossbar_bus"


class CIMDeviceType(str, Enum):
    """Crossbar cell device technology -- used only by tools.precision_check's
    real hwcomponents_neurosim-based DAC/crossbar-MAC energy models
    (Docker-only, see that module's docstring) to pick one of NeuroSim's
    bundled NVMExplorer-fit .cell configs instead of its generic placeholder
    cell. Restricted to the four technologies hwcomponents_neurosim actually
    ships a real cell config for -- not every device technology
    tools/calibration.py's KNOWN_REFERENCES cites (e.g. FeFET) has one.
    STTRAM stands in for MRAM here: NeuroSim ships no separate generic-MRAM
    cell, and STT-MRAM is the dominant real MRAM implementation."""

    RRAM = "rram"
    SRAM = "sram"
    PCM = "pcm"
    STTRAM = "sttram"


class HWConfig(BaseModel):
    """Hardware constraint specification for a target CIM chip.

    An instance is resolved (e.g. from a hardware spec store) using
    `ExecutionContext.hw_spec_id` and passed into tools explicitly --
    values are never hardcoded in node/tool logic.
    """

    hw_spec_id: str = Field(..., description="Unique identifier for this hardware spec, e.g. 'cim_v1_128x128'")

    # Crossbar array
    crossbar_rows: int = Field(..., gt=0, description="Rows per crossbar tile")
    crossbar_cols: int = Field(..., gt=0, description="Columns per crossbar tile")
    num_tiles: int = Field(..., gt=0, description="Number of crossbar tiles available on-chip")

    # Converters
    adc_bits: int = Field(..., gt=0, le=16, description="ADC resolution in bits")
    dac_bits: int = Field(..., gt=0, le=16, description="DAC resolution in bits")

    # Interconnect
    noc_topology: NoCTopology = Field(..., description="Network-on-Chip topology")
    noc_link_bandwidth_gbps: float = Field(..., gt=0, description="Per-link NoC bandwidth in Gbps")

    # Device non-idealities (optional: not every hardware spec models these)
    wire_resistance_ohm_per_um: Optional[float] = Field(
        default=None, ge=0, description="Wire resistance used for IR-drop estimation"
    )
    device_noise_sigma: Optional[float] = Field(
        default=None, ge=0, description="Std-dev of device-level conductance noise"
    )
    sram_buffer_kb: Optional[float] = Field(default=None, gt=0, description="Per-tile SRAM buffer size in KB")
    tech_node_nm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Process technology node in nanometers, used by tools.precision_check's real "
        "hwcomponents_adc-based ADC energy model. None falls back to a documented placeholder "
        "(tools/precision_check.py's _DEFAULT_TECH_NODE_NM), same honesty pattern as "
        "device_noise_sigma/sram_buffer_kb -- not every hardware spec states its process node.",
    )
    device_type: Optional[CIMDeviceType] = Field(
        default=None,
        description="Crossbar cell device technology (RRAM/SRAM/PCM/STTRAM), used by "
        "tools.precision_check's real hwcomponents_neurosim-based DAC/crossbar-MAC energy models "
        "(Docker-only) to pick a real NVMExplorer-fit cell config. None falls back to NeuroSim's "
        "generic placeholder cell, same honesty pattern as tech_node_nm -- not every hardware spec "
        "states a device technology, and hwcomponents_neurosim doesn't ship a real config for every "
        "technology tools/calibration.py's KNOWN_REFERENCES cites (e.g. FeFET).",
    )

    model_config = {"frozen": True}


class ExecutionContext(BaseModel):
    """Run-scoped metadata injected into tools via ToolRuntime.

    Carries identifiers only. Tools resolve `model_id` -> the actual
    nn.Module/ONNX graph and `hw_spec_id` -> an `HWConfig` instance via
    their own registries/stores -- this model never carries the resolved
    objects itself, keeping tool logic model-/hardware-agnostic.
    """

    session_id: str = Field(..., description="Unique id for this LangGraph run/thread")
    model_id: str = Field(..., description="Identifier for the target model, resolved externally")
    dataset_id: Optional[str] = Field(
        default=None,
        description="Identifier for the target dataset, resolved externally (tools.qat's registry); "
        "None uses the model's default dataset (built-in 'cifar10')",
    )
    hw_spec_id: str = Field(..., description="Identifier for the target HWConfig, resolved externally")
    run_seed: int = Field(
        default=0,
        description="This session's seed (state.py's run_seed) -- tools.qat.run_qat_tuning seeds torch's "
        "RNG from it so training stochasticity (dropout, batch shuffling) is reproducible per (candidate, "
        "run_seed), and independent replicates (tools/repeat_search.py) get genuinely different training "
        "trajectories, not just a different label on the same one",
    )
    iteration_count: int = Field(default=0, ge=0)
    user_id: Optional[str] = Field(default=None, description="Requesting researcher/user id, if applicable")
