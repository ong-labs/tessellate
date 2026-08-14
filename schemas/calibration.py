"""Schema for real calibration reference points (tools/calibration.py).

Each `CalibrationReference` is a genuine measured/validated data point from
published CIM accelerator literature -- not a value invented to make the
analytical model in tools/cim_physics.py look accurate. `source` carries the
citation; `note` documents any unit-conversion assumptions made when
translating the paper's reported metric into pJ-per-MAC, so uncertainty is
visible rather than hidden behind a clean-looking number.
"""

from typing import Optional

from pydantic import BaseModel, Field

from schemas.config import CIMDeviceType


class CalibrationReference(BaseModel):
    crossbar_rows: int = Field(..., gt=0)
    crossbar_cols: int = Field(..., gt=0)
    adc_bits: int = Field(..., gt=0)
    weight_bits: int = Field(..., gt=0)
    reference_energy_pj_per_mac: float = Field(..., gt=0)
    source: str = Field(..., min_length=1)
    note: Optional[str] = None
    device_type: Optional[CIMDeviceType] = Field(
        default=None,
        description="This reference's real device technology, when it's one of CIMDeviceType's four -- "
        "None for a reference whose device (e.g. FeFET, DARAM) isn't one of them, same honest-gap meaning "
        "as HWConfig.device_type=None. tools.calibration.find_matching_reference uses this to disambiguate "
        "when multiple references share the same (crossbar_rows, crossbar_cols, adc_bits) key.",
    )
