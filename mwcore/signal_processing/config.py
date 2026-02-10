from dataclasses import dataclass, field
from enum import Enum



class ProcessingMode(Enum):
    MODE_2D = "2D"  # Azimuth only (Z = 0)
    MODE_3D = "3D"  # Azimuth + Elevation


@dataclass
class RadarConfig:
    """Holds all hardware and processing parameters."""
    # Processing Mode
    mode: ProcessingMode = ProcessingMode.MODE_3D
    
    # Hardware/Capture Params
    num_tx: int = 3
    num_rx: int = 4
    loops_per_frame: int = 128
    adc_samples: int = 256
    
    # Physics/RF Params
    start_freq_ghz: float = 77
    freq_slope_mhz_us: float = 60.012
    sample_rate_ksps: int = 4400
    idle_time_us: float = 7
    ramp_end_time_us: float = 65
    
    # CFAR Params
    cfar_guard_len: int = 4
    cfar_noise_len: int = 8
    cfar_threshold_scale: float = 15.0
    
    # Derived parameters
    range_resolution: float = field(init=False)
    doppler_resolution: float = field(init=False)
    max_range: float = field(init=False)
    max_doppler: float = field(init=False)

    def __post_init__(self):
        c = 3e8
        self.range_resolution = (c * self.sample_rate_ksps * 1e3) / (2 * self.freq_slope_mhz_us * 1e12 * self.adc_samples)
        self.max_range = (300 * self.sample_rate_ksps) / (2 * self.freq_slope_mhz_us * 1e3)
        t_chirp = (self.idle_time_us + self.ramp_end_time_us) * 1e-6
        self.doppler_resolution = c / (2 * self.start_freq_ghz * 1e9 * t_chirp * self.loops_per_frame * self.num_tx)
        self.max_doppler = c / (4 * self.start_freq_ghz * 1e9 * t_chirp * self.num_tx)