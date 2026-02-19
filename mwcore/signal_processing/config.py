from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import numpy as np


class ProcessingMode(Enum):
    MODE_2D = "2D"  # Azimuth only (Z = 0)
    MODE_3D = "3D"  # Azimuth + Elevation


@dataclass
class RadarConfig:
    """Holds all hardware and processing parameters."""
    # Processing mode (accept Enum or string)
    mode: Union[ProcessingMode, str] = ProcessingMode.MODE_3D
    
    # Hardware/Capture Params
    num_tx: int = 3
    num_rx: int = 4
    loops_per_frame: int = 128
    adc_samples: int = 256

    # RF Params (used for derived resolutions)
    start_freq_ghz: float = 77.0
    freq_slope_mhz_us: float = 60.012
    sample_rate_ksps: int = 4400
    idle_time_us: float = 7.0
    ramp_end_time_us: float = 65.0

    # Optional calibration (kept here because it's device-specific, not step-specific)
    range_bias_m: float = 0.0

    # Derived parameters
    range_resolution: float = field(init=False)
    doppler_resolution: float = field(init=False)
    max_range: float = field(init=False)
    max_doppler: float = field(init=False)

    def __post_init__(self):
        # Coerce mode from string
        if isinstance(self.mode, str):
            s = self.mode.strip().upper()
            if s in ("2D", "MODE_2D"):
                self.mode = ProcessingMode.MODE_2D
            elif s in ("3D", "MODE_3D"):
                self.mode = ProcessingMode.MODE_3D
            else:
                raise ValueError(f"Invalid mode string: {self.mode}")

        # Derived RF parameters
        c = 3e8
        # Range resolution (meters/bin)
        self.range_resolution = (c * self.sample_rate_ksps * 1e3) / (
            2 * self.freq_slope_mhz_us * 1e12 * self.adc_samples
        )
        self.max_range = (300 * self.sample_rate_ksps) / (2 * self.freq_slope_mhz_us * 1e3)
        t_chirp = (self.idle_time_us + self.ramp_end_time_us) * 1e-6
        self.doppler_resolution = c / (2 * self.start_freq_ghz * 1e9 * t_chirp * self.loops_per_frame * self.num_tx)
        self.max_doppler = c / (4 * self.start_freq_ghz * 1e9 * t_chirp * self.num_tx)