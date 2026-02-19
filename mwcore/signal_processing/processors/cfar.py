import math
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import convolve1d, maximum_filter

from mwcore.registry import ADCPROCESSORS
from mwcore.signal_processing.frame import RadarFrame
from mwcore.signal_processing.processors.base import BaseSignalProcess


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _auto_div_shift(mode: str, noise_win: int) -> int:
    """TI rule-of-thumb for divShift.

    - CA:         divShift = ceil(log2(2*noiseWin))
    - CASO/CAGO:  divShift = ceil(log2(noiseWin))

    TI uses divShift as a cheap power-of-two divisor in fixed-point.
    """
    m = mode.upper()
    if m == "CA":
        return int(math.ceil(math.log2(max(1, 2 * noise_win))))
    if m in ("CASO", "CAGO"):
        return int(math.ceil(math.log2(max(1, noise_win))))
    raise ValueError(f"Unsupported CFAR mode: {mode}")


def _cfar_noise_sum_db(
    x_db: np.ndarray,
    axis: int,
    mode: str,
    noise_win: int,
    guard_len: int,
    cyclic: bool,
) -> np.ndarray:
    """Noise *sum* around each CUT in a log-domain (dB) detection matrix.

    This matches TI's "log input" CFAR compare structure:
        CUT_log > (noise_sum / 2^divShift) + thresholdScale

    We implement the window sum via 1D convolution along the chosen axis.
    For non-cyclic mode, use a very low dB padding so padding never becomes a peak.
    """
    m = mode.upper()
    conv_mode = "wrap" if cyclic else "constant"
    cval = -1e9  # very low dB floor for padding

    k_len = 2 * (noise_win + guard_len) + 1
    if k_len < 3:
        raise ValueError("Invalid CFAR window: need noise_win + guard_len >= 1")

    if m == "CA":
        k = np.ones(k_len, dtype=np.float64)
        c = noise_win + guard_len
        # zero out guard region + CUT
        k[c - guard_len : c + guard_len + 1] = 0.0
        return convolve1d(x_db, k, axis=axis, mode=conv_mode, cval=cval)

    if m in ("CASO", "CAGO"):
        k_left = np.zeros(k_len, dtype=np.float64)
        k_right = np.zeros(k_len, dtype=np.float64)
        k_left[:noise_win] = 1.0
        k_right[-noise_win:] = 1.0
        l_sum = convolve1d(x_db, k_left, axis=axis, mode=conv_mode, cval=cval)
        r_sum = convolve1d(x_db, k_right, axis=axis, mode=conv_mode, cval=cval)
        return np.minimum(l_sum, r_sum) if m == "CASO" else np.maximum(l_sum, r_sum)

    raise ValueError(f"Unsupported CFAR mode: {mode}")


def _suppress_edges(mask: np.ndarray, axis: int, edge: int) -> np.ndarray:
    """Disable detections within 'edge' bins of both ends along axis."""
    if edge <= 0:
        return mask
    out = mask.copy()
    slicer = [slice(None)] * out.ndim

    slicer[axis] = slice(0, edge)
    out[tuple(slicer)] = False

    slicer[axis] = slice(out.shape[axis] - edge, out.shape[axis])
    out[tuple(slicer)] = False
    return out


def _peak_group(
    det_mask: np.ndarray,
    x_db: np.ndarray,
    *,
    cyclic: bool,
    group_in_doppler: bool,
    group_in_range: bool,
) -> np.ndarray:
    """TI-style peak grouping approximation (flag-based) on a 2D matrix.

    - group doppler + range: 3x3
    - group doppler only:    3x1
    - group range only:      1x3
    """
    if x_db.ndim != 2:
        return det_mask

    if group_in_doppler and group_in_range:
        size = (3, 3)
    elif group_in_doppler and not group_in_range:
        size = (3, 1)
    elif (not group_in_doppler) and group_in_range:
        size = (1, 3)
    else:
        return det_mask

    local_max = maximum_filter(
        x_db,
        size=size,
        mode="wrap" if cyclic else "constant",
        cval=-1e9,  # padding should never become a peak
    )
    return det_mask & (x_db == local_max)


def _power_to_db(p: np.ndarray, eps: float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(p, 0.0) + eps)


# -----------------------------------------------------------------------------
# Base CFAR
# -----------------------------------------------------------------------------

class BaseCFAR(BaseSignalProcess):
    """Two-stage TI-style CFAR (demo-faithful semantics).

    Core choice (to match TI CLI numbers like 20 / 15 directly):
      - Build a detection matrix in *log domain* (dB) as "log input"
      - Compute noise sum/avg in that same log domain
      - Apply threshold as an *additive* offset:
            CUT_db > noise_avg_db + threshold_db

    Peak grouping and padding/edge-handling are kept as previously improved.
    """

    def __init__(
        self,
        name: str = "BaseCFAR",
        *,
        input_key: str,
        process_axis: int,
        averaging_mode: str = "CA",
        noise_win: int = 8,
        guard_len: int = 4,
        threshold_db: float = 15.0,
        div_shift: Optional[int] = None,
        cyclic: bool = False,
        # Pre-mask support (two-stage CFAR)
        pre_mask_key: Optional[str] = None,
        # Output naming
        mask_key: str = "cfar_mask",
        noise_key: str = "noise_db",
        energy_key: str = "energy_map",
        energy_db_key: Optional[str] = None,  # optional alias output
        det_points_key: str = "detected_points",
        # Peak grouping (usually enabled for final stage)
        peak_grouping: bool = False,
        group_in_doppler: bool = True,
        group_in_range: bool = True,
        # FOV / post filters (usually only final stage)
        fov_range_m: Optional[Tuple[float, float]] = None,
        fov_doppler_mps: Optional[Tuple[float, float]] = None,
        min_snr_db: Optional[float] = None,
        # Numerical
        eps: float = 1e-12,
    ):
        super().__init__(name)

        self.input_key = input_key
        self.process_axis = int(process_axis)
        self.averaging_mode = averaging_mode.upper()
        self.noise_win = int(noise_win)
        self.guard_len = int(guard_len)
        self.threshold_db = float(threshold_db)
        self.div_shift = div_shift
        self.cyclic = bool(cyclic)

        self.pre_mask_key = pre_mask_key
        self.mask_key = mask_key
        self.noise_key = noise_key
        self.energy_key = energy_key
        self.energy_db_key = energy_db_key
        self.det_points_key = det_points_key

        self.peak_grouping = bool(peak_grouping)
        self.group_in_doppler = bool(group_in_doppler)
        self.group_in_range = bool(group_in_range)

        self.fov_range_m = fov_range_m
        self.fov_doppler_mps = fov_doppler_mps
        self.min_snr_db = min_snr_db

        self.eps = float(eps)

    def _compute_energy_db(self, x: np.ndarray) -> np.ndarray:
        """Return a 2D log-domain detection matrix in dB.

        Common cases:
          - complex doppler FFT cube (Tx,Rx,Doppler,Range): sum power over Tx/Rx -> (D,R) -> dB
          - complex 2D (D,R): abs^2 -> dB
          - real 2D: assumed already in dB (e.g., energy_map forwarded between stages)
        """
        if np.iscomplexobj(x):
            # Reduce everything down to last 2 dims (Doppler, Range) by summing power on leading dims.
            p = np.abs(x) ** 2
            if p.ndim == 4:
                # expected (Tx,Rx,D,R)
                p = np.sum(p, axis=(0, 1))
            elif p.ndim > 2:
                p = np.sum(p, axis=tuple(range(p.ndim - 2)))
            # now p should be (D,R)
            return _power_to_db(np.asarray(p, dtype=np.float64), self.eps)

        # Real input: treat as already "log input" in dB
        return np.asarray(x, dtype=np.float64)

    def _process_generic(self, frame: RadarFrame):
        # 1) Read input
        x = self.read(frame, self.input_key)

        # 2) Build log-domain detection matrix (dB)
        energy_db = self._compute_energy_db(x)

        # Optional: write energy map (dB). RangeCFAR produces it; DopplerCFAR consumes it.
        if self.energy_key in self.provides:
            self.write(frame, self.energy_key, energy_db.astype(np.float32))

        # Optional alias output
        if self.energy_db_key is not None and (self.energy_db_key in self.provides):
            self.write(frame, self.energy_db_key, energy_db.astype(np.float32))

        # 3) Optional pre-mask (RangeCFAR mask gating into DopplerCFAR)
        pre_mask = None
        if self.pre_mask_key is not None:
            pre_mask = self.read(frame, self.pre_mask_key).astype(bool)

        # 4) Noise sum/avg in log domain + TI-style divShift
        div_shift = self.div_shift
        if div_shift is None:
            div_shift = _auto_div_shift(self.averaging_mode, self.noise_win)

        noise_sum_db = _cfar_noise_sum_db(
            energy_db,
            axis=self.process_axis,
            mode=self.averaging_mode,
            noise_win=self.noise_win,
            guard_len=self.guard_len,
            cyclic=self.cyclic,
        )
        noise_avg_db = noise_sum_db / (2.0 ** div_shift)

        # 5) TI CLI thresholdScale is in dB for "log input" -> additive
        thr_db = noise_avg_db + float(self.threshold_db)

        det_mask = energy_db > thr_db
        if pre_mask is not None:
            det_mask &= pre_mask

        # Non-cyclic: suppress edge bins (partial windows)
        if not self.cyclic:
            edge = self.noise_win + self.guard_len
            det_mask = _suppress_edges(det_mask, axis=self.process_axis, edge=edge)

        # 6) Peak grouping (flag-based)
        if self.peak_grouping:
            det_mask = _peak_group(
                det_mask,
                energy_db,
                cyclic=self.cyclic,
                group_in_doppler=self.group_in_doppler,
                group_in_range=self.group_in_range,
            )

        # 7) Write mask/noise
        if self.mask_key in self.provides:
            self.write(frame, self.mask_key, det_mask)

        if self.noise_key in self.provides:
            self.write(frame, self.noise_key, noise_avg_db.astype(np.float32))

        # 8) Detected points (N,3) = [range_idx, doppler_idx, snr_db]
        # SNR (dB) in this log-input scheme is simply CUT_db - noise_avg_db
        if self.det_points_key in self.provides:
            if energy_db.ndim != 2:
                raise ValueError(
                    f"[{self.name}] detected_points extraction assumes 2D (Doppler,Range). Got {energy_db.shape}"
                )

            det_idx = np.argwhere(det_mask)  # [doppler_idx, range_idx]
            if det_idx.size == 0:
                self.write(frame, self.det_points_key, np.zeros((0, 3), dtype=np.float32))
                return

            d_idx = det_idx[:, 0].astype(np.int32)
            r_idx = det_idx[:, 1].astype(np.int32)

            snr_db = (energy_db[d_idx, r_idx] - noise_avg_db[d_idx, r_idx]).astype(np.float32)

            # Optional min SNR filter
            if self.min_snr_db is not None:
                keep = snr_db >= float(self.min_snr_db)
                d_idx, r_idx, snr_db = d_idx[keep], r_idx[keep], snr_db[keep]

            # Optional cfarFovCfg-like filtering (range/doppler bounds)
            cfg = frame.config

            if self.fov_range_m is not None and r_idx.size:
                r_m = r_idx.astype(np.float32) * cfg.range_resolution
                keep = (r_m >= self.fov_range_m[0]) & (r_m <= self.fov_range_m[1])
                d_idx, r_idx, snr_db = d_idx[keep], r_idx[keep], snr_db[keep]

            if self.fov_doppler_mps is not None and d_idx.size:
                nd = energy_db.shape[0]
                # Your DopplerFFT is fftshifted in the pipeline; keep signed mapping.
                d_signed = d_idx.astype(np.int32) - (nd // 2)
                v = d_signed.astype(np.float32) * cfg.doppler_resolution
                keep = (v >= self.fov_doppler_mps[0]) & (v <= self.fov_doppler_mps[1])
                d_idx, r_idx, snr_db = d_idx[keep], r_idx[keep], snr_db[keep]

            if d_idx.size == 0:
                self.write(frame, self.det_points_key, np.zeros((0, 3), dtype=np.float32))
                return

            det_points = np.column_stack([r_idx, d_idx, snr_db]).astype(np.float32)
            self.write(frame, self.det_points_key, det_points)


# -----------------------------------------------------------------------------
# Thin wrappers (axis + I/O contracts)
# -----------------------------------------------------------------------------

@ADCPROCESSORS.register_module()
class RangeCFAR(BaseCFAR):
    """Stage-1: Range-direction CFAR (procDirection=0 in TI CLI)."""
    requires = {"doppler_fft"}
    provides = {"energy_map", "range_cfar_mask", "range_noise_db"}

    def __init__(
        self,
        name: str = "RangeCFAR",
        averaging_mode: str = "CASO",
        noise_win: int = 8,
        guard_len: int = 4,
        threshold_db: float = 20.0,  # TI CLI number, used directly
        div_shift: Optional[int] = None,
        cyclic: bool = False,
    ):
        super().__init__(
            name=name,
            input_key="doppler_fft",
            process_axis=1,  # range axis in (Doppler,Range)
            averaging_mode=averaging_mode,
            noise_win=noise_win,
            guard_len=guard_len,
            threshold_db=threshold_db,
            div_shift=div_shift,
            cyclic=cyclic,
            pre_mask_key=None,
            mask_key="range_cfar_mask",
            noise_key="range_noise_db",
            energy_key="energy_map",   # energy_map is now dB log-input
            energy_db_key=None,
            det_points_key="detected_points",  # not in provides, so not emitted
            peak_grouping=False,
        )


@ADCPROCESSORS.register_module()
class DopplerCFAR(BaseCFAR):
    """Stage-2: Doppler-direction CFAR (procDirection=1 in TI CLI)."""
    requires = {"energy_map", "range_cfar_mask"}
    provides = {"final_cfar_mask", "doppler_noise_db", "detected_points"}

    def __init__(
        self,
        name: str = "DopplerCFAR",
        averaging_mode: str = "CA",
        noise_win: int = 4,
        guard_len: int = 2,
        threshold_db: float = 15.0,  # TI CLI number, used directly
        div_shift: Optional[int] = None,
        cyclic: bool = True,
        peak_grouping: bool = True,
        group_in_doppler: bool = True,
        group_in_range: bool = True,
        fov_range_m: Optional[Tuple[float, float]] = None,
        fov_doppler_mps: Optional[Tuple[float, float]] = None,
        min_snr_db: Optional[float] = None,
    ):
        super().__init__(
            name=name,
            input_key="energy_map",     # energy_map is dB log-input from RangeCFAR
            process_axis=0,             # doppler axis in (Doppler,Range)
            averaging_mode=averaging_mode,
            noise_win=noise_win,
            guard_len=guard_len,
            threshold_db=threshold_db,
            div_shift=div_shift,
            cyclic=cyclic,
            pre_mask_key="range_cfar_mask",
            mask_key="final_cfar_mask",
            noise_key="doppler_noise_db",
            energy_key="energy_map",    # not in provides here, so not rewritten
            energy_db_key=None,
            det_points_key="detected_points",
            peak_grouping=peak_grouping,
            group_in_doppler=group_in_doppler,
            group_in_range=group_in_range,
            fov_range_m=fov_range_m,
            fov_doppler_mps=fov_doppler_mps,
            min_snr_db=min_snr_db,
        )


# -----------------------------------------------------------------------------
# Optional wrappers (kept for API completeness)
# -----------------------------------------------------------------------------

@ADCPROCESSORS.register_module()
class AzimuthCFAR(BaseCFAR):
    requires = {"azimuth_fft"}
    provides = {"energy_map", "cfar_mask", "noise_db", "detected_points"}

    def __init__(
        self,
        name: str = "AzimuthCFAR",
        averaging_mode: str = "CASO",
        noise_win: int = 8,
        guard_len: int = 4,
        threshold_db: float = 15.0,
        div_shift: Optional[int] = None,
        cyclic: bool = False,
        peak_grouping: bool = True,
    ):
        super().__init__(
            name=name,
            input_key="azimuth_fft",
            process_axis=-1,
            averaging_mode=averaging_mode,
            noise_win=noise_win,
            guard_len=guard_len,
            threshold_db=threshold_db,
            div_shift=div_shift,
            cyclic=cyclic,
            pre_mask_key=None,
            mask_key="cfar_mask",
            noise_key="noise_db",
            energy_key="energy_map",
            energy_db_key=None,
            det_points_key="detected_points",
            peak_grouping=peak_grouping,
        )


@ADCPROCESSORS.register_module()
class ElevationCFAR(BaseCFAR):
    requires = {"elevation_fft"}
    provides = {"energy_map", "cfar_mask", "noise_db", "detected_points"}

    def __init__(
        self,
        name: str = "ElevationCFAR",
        averaging_mode: str = "CASO",
        noise_win: int = 8,
        guard_len: int = 4,
        threshold_db: float = 15.0,
        div_shift: Optional[int] = None,
        cyclic: bool = False,
        peak_grouping: bool = True,
    ):
        super().__init__(
            name=name,
            input_key="elevation_fft",
            process_axis=-1,
            averaging_mode=averaging_mode,
            noise_win=noise_win,
            guard_len=guard_len,
            threshold_db=threshold_db,
            div_shift=div_shift,
            cyclic=cyclic,
            pre_mask_key=None,
            mask_key="cfar_mask",
            noise_key="noise_db",
            energy_key="energy_map",
            energy_db_key=None,
            det_points_key="detected_points",
            peak_grouping=peak_grouping,
        )
