from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import maximum_filter

from mwcore.registry import ADCPROCESSORS
from mwcore.signal_processing.frame import RadarFrame
from mwcore.signal_processing.processors.base import BaseSignalProcess


def _etcm_alpha(omega: float, p_fa: float) -> float:
    """
    Closed-form threshold factor α (Eq. 16 in paper).
    Threshold T = α * x_hat  (Eq. 17 in paper).
    """
    w = float(omega)
    m = float(p_fa)
    if not (0.0 < w < 1.0):
        raise ValueError(f"omega must be in (0,1), got {omega}")
    if not (0.0 < m < 1.0):
        raise ValueError(f"p_fa must be in (0,1), got {p_fa}")

    # α = ((2-ω) - (2-ω)*sqrt(1 + (2ω ln m)/(2-ω))) / ω
    # (ln m < 0, so the sqrt term is slightly < 1; α becomes positive and typically ~[4..20].)
    two_minus_w = 2.0 - w
    radicand = 1.0 + (2.0 * w * np.log(m)) / two_minus_w
    if radicand <= 0.0:
        raise ValueError(
            f"Invalid parameters produce negative radicand in Eq16: radicand={radicand}. "
            f"Try smaller |ln(p_fa)| or smaller omega."
        )
    alpha = (two_minus_w - two_minus_w * np.sqrt(radicand)) / w
    return float(alpha)


def _power_map_from_doppler_fft(dfft: np.ndarray) -> np.ndarray:
    """Sum power over (Tx,Rx) -> (Doppler,Range)."""
    p = np.abs(dfft) ** 2
    return np.sum(p, axis=(0, 1)).astype(np.float64)


@ADCPROCESSORS.register_module()
class ETCMCFAR(BaseSignalProcess):
    """
    ETCM-CFAR (temporal clutter-map CFAR).

    Inputs/outputs:
      - requires: doppler_fft (Tx,Rx,D,R)
      - provides:
          etcm_cfar_mask      : (D,R) bool
          detected_points     : (N,3) float32 [range_idx, doppler_idx, snr_db]
          etcm_energy_map     : (D,R) float32 linear power (RDS)
          etcm_noise_map      : (D,R) float32 CM (noise estimate)
          etcm_threshold_map  : (D,R) float32 threshold T

    Notes:
      - Uses Algorithm 1 update rule: update CM only where Mask==0.
      - Uses Eq. 16/17 for threshold.
    """
    requires = {"doppler_fft"}
    provides = {
        "etcm_cfar_mask",
        "detected_points",
        "etcm_energy_map",
        "etcm_noise_map",
        "etcm_threshold_map",
    }

    def __init__(
        self,
        *,
        omega: float = 0.001,
        p_fa: float = 1e-6,
        alpha: Optional[float] = None,
        # initialization
        init_mode: str = "mean_first_n",   # "first_frame" | "mean_first_n"
        init_frames: int = 30,
        freeze_until_initialized: bool = True,
        # post filters (optional)
        min_snr_db: Optional[float] = None,
        fov_range_m: Optional[Tuple[float, float]] = None,
        fov_doppler_mps: Optional[Tuple[float, float]] = None,
        # peak grouping (optional)
        peak_grouping: bool = False,
        peak_grouping_size: Tuple[int, int] = (3, 3),
        # numerical
        eps: float = 1e-12,
        name: str = "ETCMCFAR",
    ):
        super().__init__(name)

        self.omega = float(omega)
        self.p_fa = float(p_fa)
        self.alpha_override = None if alpha is None else float(alpha)

        self.init_mode = (init_mode or "").strip().lower()
        self.init_frames = int(init_frames)
        self.freeze_until_initialized = bool(freeze_until_initialized)

        self.min_snr_db = None if min_snr_db is None else float(min_snr_db)
        self.fov_range_m = fov_range_m
        self.fov_doppler_mps = fov_doppler_mps

        self.peak_grouping = bool(peak_grouping)
        self.peak_grouping_size = (int(peak_grouping_size[0]), int(peak_grouping_size[1]))

        self.eps = float(eps)

        # persistent state
        self._cm: Optional[np.ndarray] = None
        self._boot_sum: Optional[np.ndarray] = None
        self._boot_count: int = 0

    def reset(self) -> None:
        """Manually clear CM (useful when the environment changes)."""
        self._cm = None
        self._boot_sum = None
        self._boot_count = 0

    def _ensure_initialized(self, p_map: np.ndarray) -> bool:
        """Return True iff CM is ready."""
        if self._cm is not None:
            return True

        if self.init_mode == "first_frame":
            self._cm = p_map.copy()
            return True

        if self.init_mode == "mean_first_n":
            if self._boot_sum is None:
                self._boot_sum = np.zeros_like(p_map, dtype=np.float64)
                self._boot_count = 0

            self._boot_sum += p_map
            self._boot_count += 1

            if self._boot_count >= max(1, self.init_frames):
                self._cm = self._boot_sum / float(self._boot_count)
                self._boot_sum = None
                self._boot_count = 0
                return True

            return False

        raise ValueError(f"[{self.name}] Unsupported init_mode: {self.init_mode}")

    def _process_generic(self, frame: RadarFrame) -> None:
        dfft = self.read(frame, "doppler_fft")  # (Tx,Rx,D,R)
        if dfft.ndim != 4:
            raise ValueError(f"[{self.name}] Expected doppler_fft (Tx,Rx,D,R). Got {dfft.shape}")

        p_map = _power_map_from_doppler_fft(dfft)  # (D,R) linear power (RDS)
        self.write(frame, "etcm_energy_map", p_map.astype(np.float32))

        ready = self._ensure_initialized(p_map)
        if not ready and self.freeze_until_initialized:
            # Output “empty detection” until CM is ready
            cm_est = (self._boot_sum / max(1.0, float(self._boot_count))) if self._boot_sum is not None else p_map
            self.write(frame, "etcm_noise_map", cm_est.astype(np.float32))
            self.write(frame, "etcm_threshold_map", np.zeros_like(p_map, dtype=np.float32))
            self.write(frame, "etcm_cfar_mask", np.zeros_like(p_map, dtype=bool))
            self.write(frame, "detected_points", np.zeros((0, 3), dtype=np.float32))
            return

        if self._cm is None:
            # If freeze_until_initialized=False, fall back to first-frame CM
            self._cm = p_map.copy()

        omega = self.omega
        alpha = self.alpha_override if self.alpha_override is not None else _etcm_alpha(omega, self.p_fa)

        # Threshold T = α * x_hat (per cell).
        thr = alpha * self._cm
        mask = p_map > thr

        # Optional peak grouping (usually OFF for “denser” point clouds)
        if self.peak_grouping:
            local_max = maximum_filter(p_map, size=self.peak_grouping_size, mode="nearest")
            mask = mask & (p_map == local_max)
        
        print("[ETCMCFAR] Alpha: {:.2f}".format(alpha))
        print("[ETCMCFAR] Mask Ratio: {:.2f}%".format(100.0 * np.sum(mask) / mask.size))
        print("[ETCMCFAR] Median Noise Power: {:.2e}".format(np.median(p_map / (self._cm + self.eps))))
        # Update CM only where Mask == 0 (Algorithm 1).
        upd = ~mask
        self._cm[upd] = (1.0 - omega) * self._cm[upd] + omega * p_map[upd]

        self.write(frame, "etcm_noise_map", self._cm.astype(np.float32))
        self.write(frame, "etcm_threshold_map", thr.astype(np.float32))
        self.write(frame, "etcm_cfar_mask", mask)

        # Build detected_points = [range_idx, doppler_idx, snr_db]
        det_idx = np.argwhere(mask)  # [doppler_idx, range_idx]
        if det_idx.size == 0:
            self.write(frame, "detected_points", np.zeros((0, 3), dtype=np.float32))
            return

        d_idx = det_idx[:, 0].astype(np.int32)
        r_idx = det_idx[:, 1].astype(np.int32)

        # SNR estimate from CM (linear -> dB)
        snr_db = 10.0 * np.log10((p_map[d_idx, r_idx] + self.eps) / (self._cm[d_idx, r_idx] + self.eps))
        snr_db = snr_db.astype(np.float32)

        # Optional min SNR filter
        if self.min_snr_db is not None:
            keep = snr_db >= float(self.min_snr_db)
            d_idx, r_idx, snr_db = d_idx[keep], r_idx[keep], snr_db[keep]

        # Optional FOV filtering (range/doppler bounds)
        cfg = frame.config
        if self.fov_range_m is not None and r_idx.size:
            r_m = r_idx.astype(np.float32) * cfg.range_resolution
            keep = (r_m >= self.fov_range_m[0]) & (r_m <= self.fov_range_m[1])
            d_idx, r_idx, snr_db = d_idx[keep], r_idx[keep], snr_db[keep]

        if self.fov_doppler_mps is not None and d_idx.size:
            nd = p_map.shape[0]
            # Doppler FFT is fftshifted
            d_signed = d_idx.astype(np.int32) - (nd // 2)
            v = d_signed.astype(np.float32) * cfg.doppler_resolution
            keep = (v >= self.fov_doppler_mps[0]) & (v <= self.fov_doppler_mps[1])
            d_idx, r_idx, snr_db = d_idx[keep], r_idx[keep], snr_db[keep]

        if d_idx.size == 0:
            self.write(frame, "detected_points", np.zeros((0, 3), dtype=np.float32))
            return

        det_points = np.column_stack([r_idx, d_idx, snr_db]).astype(np.float32)
        self.write(frame, "detected_points", det_points)