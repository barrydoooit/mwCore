from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from mwcore.registry import ADCPROCESSORS
from mwcore.signal_processing.frame import RadarFrame
from .utils import _cov_inv, _mvdr_spectrum, _steering_vec_nu
from ..base import BaseSignalProcess


@ADCPROCESSORS.register_module()
class DynamicCaponRAHeatmapWM(BaseSignalProcess):
    """
    Generate dynamic range-azimuth heatmap using MVDR/Capon on azimuth-only antennas.

    Inputs:
      virt_cube_cr: (12, Loops, Range)

    Outputs:
      ra_heatmap:         (Naz, Range) float32  (linear power)
      ra_peak_per_range:  (Range,) float32      max over azimuth at each range
      nu_grid:            (Naz,) float32        direction cosine grid (nu = sin(phi) when theta=0)

    Parameters (from TI dynamicRangeAngleCfg):
      angle_search_step_deg: 0.75 (default ISK wall mount)
      diag_loading:          0.0010

    Extra parameters (needed in Python; TI stores them in board config):
      fov_az_deg: (-60,60) typical; MUST match your use-case.
      virt_ant_idx_az: indices of 8 azimuth antennas (default Tx0+Tx2 -> [0..3, 8..11])
      ant_geom_m: (12,) m-index for each virtual antenna (board geometry)
    """
    requires = {"virt_cube_cr"}
    provides = {"ra_heatmap", "ra_peak_per_range", "nu_grid"}

    def __init__(
        self,
        angle_search_step_deg: float = 0.75,
        diag_loading: float = 0.0010,
        fov_az_deg: Tuple[float, float] = (-60.0, 60.0),
        virt_ant_idx_az: Tuple[int, ...] = (0, 1, 2, 3, 8, 9, 10, 11),
        ant_geom_m: Optional[Tuple[int, ...]] = None,
        name: str = "DynamicCaponRAHeatmapWM",
    ):
        super().__init__(name)
        self.angle_search_step_deg = float(angle_search_step_deg)
        self.diag_loading = float(diag_loading)
        self.fov_az_deg = (float(fov_az_deg[0]), float(fov_az_deg[1]))
        self.virt_ant_idx_az = tuple(int(x) for x in virt_ant_idx_az)
        self.ant_geom_m = None if ant_geom_m is None else np.asarray(ant_geom_m, dtype=np.float32)

    def _process_generic(self, frame: RadarFrame) -> None:
        V = self.read(frame, "virt_cube_cr")  # (12, Loops, Range)
        if V.ndim != 3 or V.shape[0] != 12:
            raise ValueError(f"[{self.name}] virt_cube_cr must be (12,Loops,Range). Got {V.shape}")

        nant, loops, rng = V.shape

        if self.ant_geom_m is None:
            # If you don't have board geometry yet, fall back to a simple ULA index.
            # This is NOT a correct ISK geometry; supply correct m-index for accurate angles.
            m_all = np.arange(12, dtype=np.float32)
        else:
            if self.ant_geom_m.shape[0] != 12:
                raise ValueError(f"[{self.name}] ant_geom_m must have length 12.")
            m_all = self.ant_geom_m

        az_idx = np.asarray(self.virt_ant_idx_az, dtype=np.int32)
        if az_idx.size != 8:
            raise ValueError(f"[{self.name}] virt_ant_idx_az must have 8 indices. Got {az_idx.size}")

        # Build azimuth nu-grid
        az_min, az_max = self.fov_az_deg
        if az_max <= az_min:
            raise ValueError(f"[{self.name}] invalid fov_az_deg={self.fov_az_deg}")
        step = self.angle_search_step_deg
        n_az = int(np.floor((az_max - az_min) / step)) + 1
        phi = (az_min + step * np.arange(n_az, dtype=np.float32)) * (np.pi / 180.0)
        nu = np.sin(phi).astype(np.float32)  # at theta=0, nu = sin(phi)
        self.write(frame, "nu_grid", nu)

        # Steering vectors for the 8 azimuth antennas
        A8 = _steering_vec_nu(m_all[az_idx], nu)  # (Naz,8)

        # Heatmap: (Naz,Range)
        H = np.zeros((n_az, rng), dtype=np.float32)

        # per-range maximum
        per_rng_max = np.zeros((rng,), dtype=np.float32)

        for r in range(rng):
            # snapshots: Y (8, Loops)
            Y = V[az_idx, :, r]  # (8,Loops)
            invR = _cov_inv(Y, self.diag_loading)  # (8,8)
            P = _mvdr_spectrum(invR, A8)           # (Naz,)
            H[:, r] = P
            per_rng_max[r] = float(np.max(P))

        self.write(frame, "ra_heatmap", H)
        self.write(frame, "ra_peak_per_range", per_rng_max)