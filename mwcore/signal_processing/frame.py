from typing import Optional

import numpy as np
from .config import RadarConfig


class RadarFrame:
    """Central Data Structure."""
    def __init__(self, raw_bytes: bytes, config: Optional[RadarConfig] = None):
        self._config = config
        self.raw_bytes = raw_bytes
        self.raw_complex: Optional[np.ndarray] = None
        self.radar_cube: Optional[np.ndarray] = None    # (Tx, Rx, Loops, Samples)
        self.range_fft: Optional[np.ndarray] = None
        self.doppler_fft: Optional[np.ndarray] = None
        self.energy_map: Optional[np.ndarray] = None
        self.detected_points: Optional[np.ndarray] = None # (N, 3) [RangeIdx, DopIdx, Val]
        self.point_cloud: Optional[np.ndarray] = None   # (6, N) [x,y,z,v,snr,r]
    
    @property
    def config(self) -> RadarConfig:
        if self._config is None:
            raise ValueError("RadarConfig not set yet.")
        return self._config