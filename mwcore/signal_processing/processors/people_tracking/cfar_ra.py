from __future__ import annotations

import numpy as np

from mwcore.registry import ADCPROCESSORS
from mwcore.signal_processing.frame import RadarFrame
from .utils import _caso_noise_1d, _cov_inv, _local_max_2d, _mvdr_spectrum, _steering_vec_nu
from ..base import BaseSignalProcess



@ADCPROCESSORS.register_module()
class DynamicRACfar2PassWM(BaseSignalProcess):
    """
    2-pass CFAR-CASO on dynamic range-azimuth heatmap (linear power).

    Inputs:
      ra_heatmap: (Naz, Range) float32
      ra_peak_per_range: (Range,) float32

    Outputs:
      ra_detections: (N,4) float32 columns:
        [range_idx, az_idx, snr_linear, noise_linear]
      ra_det_mask: (Naz,Range) bool

    Parameters (from TI dynamicRACfarCfg):
      - cfarDiscardLeftRange, cfarDiscardRightRange
      - cfarDiscardLeftAngle, cfarDiscardRightAngle
      - refWinSizeRange, guardWinSizeRange
      - refWinSizeAngle, guardWinSizeAngle
      - rangeThre, angleThre (linear scale multipliers) :contentReference[oaicite:13]{index=13}
      - sidelobeThre (linear ratio of per-range max) :contentReference[oaicite:14]{index=14}
      - enable2ndPass (1 recommended for dynamic) :contentReference[oaicite:15]{index=15}
      - dynamicFlag (0/1 off, 2 range-dynamic, 3 range+angle dynamic) :contentReference[oaicite:16]{index=16}

    Note:
      TI's dynamicFlag scaling law is NOT specified in the tuning guide; we implement only fixed thresholds
      for dynamicFlag in {0,1}. For 2/3 we leave a hook.
    """
    requires = {"ra_heatmap", "ra_peak_per_range"}
    provides = {"ra_detections", "ra_det_mask"}

    def __init__(
        self,
        cfar_discard_left_range: int = 4,
        cfar_discard_right_range: int = 4,
        cfar_discard_left_angle: int = 2,
        cfar_discard_right_angle: int = 2,
        ref_win_size_range: int = 8,
        guard_win_size_range: int = 4,
        ref_win_size_angle: int = 12,
        guard_win_size_angle: int = 8,
        range_thre: float = 5.0,
        angle_thre: float = 8.0,
        sidelobe_thre: float = 0.40,
        enable_2nd_pass: int = 1,
        dynamic_flag: int = 1,
        name: str = "DynamicRACfar2PassWM",
    ):
        super().__init__(name)
        self.cfar_discard_left_range = int(cfar_discard_left_range)
        self.cfar_discard_right_range = int(cfar_discard_right_range)
        self.cfar_discard_left_angle = int(cfar_discard_left_angle)
        self.cfar_discard_right_angle = int(cfar_discard_right_angle)
        self.ref_win_size_range = int(ref_win_size_range)
        self.guard_win_size_range = int(guard_win_size_range)
        self.ref_win_size_angle = int(ref_win_size_angle)
        self.guard_win_size_angle = int(guard_win_size_angle)
        self.range_thre = float(range_thre)
        self.angle_thre = float(angle_thre)
        self.sidelobe_thre = float(sidelobe_thre)
        self.enable_2nd_pass = int(enable_2nd_pass)
        self.dynamic_flag = int(dynamic_flag)

    def _dyn_scale_range(self, r_idx: int, n_range: int) -> float:
        # TI dynamic scaling law is not specified in the guide.
        # Hook only: keep 1.0 for disabled modes (0/1).
        if self.dynamic_flag in (0, 1):
            return 1.0
        return 1.0

    def _dyn_scale_angle(self, a_idx: int, n_az: int) -> float:
        if self.dynamic_flag in (0, 1, 2):
            return 1.0
        return 1.0

    def _process_generic(self, frame: RadarFrame) -> None:
        H = self.read(frame, "ra_heatmap").astype(np.float32)  # (Naz,Range)
        per_rng_max = self.read(frame, "ra_peak_per_range").astype(np.float32)  # (Range,)
        if H.ndim != 2:
            raise ValueError(f"[{self.name}] ra_heatmap must be 2D (Az,Range). Got {H.shape}")

        n_az, n_range = H.shape
        det_mask = np.zeros((n_az, n_range), dtype=bool)

        # First-pass detections list: (range, az, noise)
        tmp = []

        a_lo = self.cfar_discard_left_angle
        a_hi = n_az - self.cfar_discard_right_angle
        r_lo = self.cfar_discard_left_range
        r_hi = n_range - self.cfar_discard_right_range

        for a in range(a_lo, a_hi):
            x = H[a, :]  # (Range,)
            for r in range(r_lo, r_hi):
                noise = _caso_noise_1d(
                    x, r,
                    ref=self.ref_win_size_range,
                    guard=self.guard_win_size_range,
                    cyclic=False,
                )
                if not np.isfinite(noise):
                    continue
                thr = noise * self.range_thre * self._dyn_scale_range(r, n_range)
                if x[r] > thr:
                    tmp.append((r, a, float(noise)))

        out = []
        for (r, a, noise) in tmp:
            cut = float(H[a, r])

            # Sidelobe guard relative to max peak at this range
            if cut < float(per_rng_max[r]) * self.sidelobe_thre:
                continue

            if self.enable_2nd_pass == 1:
                # 2nd pass: CFAR across angle at fixed range r (cyclic wrap)
                col = H[:, r]  # (Az,)
                noise2 = _caso_noise_1d(
                    col, a,
                    ref=self.ref_win_size_angle,
                    guard=self.guard_win_size_angle,
                    cyclic=True,   # TI uses wrap in angle domain
                )
                if not np.isfinite(noise2):
                    continue
                thr2 = noise2 * self.angle_thre * self._dyn_scale_angle(a, n_az)
                confirmed = (cut > thr2)
            else:
                confirmed = True

            # Additional local-peak confirmation (TI mentions local-peak checks in logic)
            if confirmed and not _local_max_2d(H, a, r):
                confirmed = False

            if confirmed:
                det_mask[a, r] = True
                snr_lin = cut / max(noise, 1e-12)
                out.append((r, a, float(snr_lin), float(noise)))

        if len(out) == 0:
            det = np.zeros((0, 4), dtype=np.float32)
        else:
            det = np.asarray(out, dtype=np.float32)

        self.write(frame, "ra_det_mask", det_mask)
        self.write(frame, "ra_detections", det)