from __future__ import annotations

import numpy as np
from mwcore.registry import ADCPROCESSORS
from ..frame import RadarFrame
from .base import BaseSignalProcess


def _get_window(name: str, n: int) -> np.ndarray:
    s = (name or "hann").strip().lower()
    if s in ("hann", "hanning"):
        return np.hanning(n)
    if s in ("hamming",):
        return np.hamming(n)
    if s in ("rect", "rectangular", "boxcar", "none"):
        return np.ones(n, dtype=np.float64)
    raise ValueError(f"Unsupported window: {name}")


@ADCPROCESSORS.register_module()
class RangeFFT(BaseSignalProcess):
    """1D FFT over ADC samples -> range bins."""
    requires = {"radar_cube"}
    provides = {"range_fft"}

    def __init__(self, window: str = "hann", name: str = ""):
        super().__init__(name)
        self.window = window

    def _process_generic(self, frame: RadarFrame) -> None:
        cube = self.read(frame, "radar_cube")  # (Tx,Rx,Loops,Samples)
        n = frame.config.adc_samples
        w = _get_window(self.window, n).reshape(1, 1, 1, -1)
        r_fft = np.fft.fft(cube * w, axis=3)
        self.write(frame, "range_fft", r_fft)


@ADCPROCESSORS.register_module()
class CalibDcRangeSig(BaseSignalProcess):
    """
    TI calibDcRangeSig (DC range signature removal)

    - Accumulate/average bins around DC for first num_avg "chirps"
    - From the next chirps onward, subtract the estimated signature

    Notes for your data layout:
    - range_fft shape: (Tx,Rx,Loops,RangeBins)
    - "chirps" in TI are total chirps across all Tx. Loops axis is per-Tx.
      We approximate TI's numAvg chirps by using num_avg_loops = num_avg_chirps // num_tx (at least 1).
      This matches TI intent without changing intermediate structures.
    """
    requires = {"range_fft"}
    provides = {"range_fft"}

    def __init__(
        self,
        enabled: bool = True,
        negative_bin_idx: int = 0,
        positive_bin_idx: int = 0,
        num_avg_chirps: int = 256,
        subtract_during_estimation: bool = False,
        name: str = "",
    ):
        super().__init__(name)
        self.enabled = enabled
        self.negative_bin_idx = int(negative_bin_idx)
        self.positive_bin_idx = int(positive_bin_idx)
        self.num_avg_chirps = int(num_avg_chirps)
        self.subtract_during_estimation = subtract_during_estimation

    def _selected_bins(self, n_range: int) -> np.ndarray:
        # positiveBinIdx: includes bins [0..positiveBinIdx]
        pos = np.arange(0, min(self.positive_bin_idx, n_range - 1) + 1, dtype=np.int32)

        # negativeBinIdx is expressed as negative count (e.g. -5 => last 5 bins) :contentReference[oaicite:17]{index=17}
        neg = np.array([], dtype=np.int32)
        if self.negative_bin_idx < 0:
            k = min(-self.negative_bin_idx, n_range)
            neg = np.arange(n_range - k, n_range, dtype=np.int32)

        if pos.size == 0 and neg.size == 0:
            return np.array([], dtype=np.int32)
        return np.unique(np.concatenate([pos, neg]))

    def _process_generic(self, frame: RadarFrame) -> None:
        if not self.enabled:
            return

        r_fft = self.read(frame, "range_fft")  # (Tx,Rx,Loops,RangeBins)
        cfg = frame.config
        n_tx, n_rx, n_loops, n_range = r_fft.shape

        bins = self._selected_bins(n_range)
        if bins.size == 0:
            return

        # Approx TI "numAvg chirps" -> per-Tx loops count
        num_avg_loops = max(1, self.num_avg_chirps // max(1, cfg.num_tx))
        num_avg_loops = min(num_avg_loops, n_loops)

        # Estimate DC signature from first num_avg_loops
        # Signature shape: (Tx,Rx,len(bins))
        sig = r_fft[:, :, :num_avg_loops, :][:, :, :, bins].mean(axis=2)

        # Decide which loops to subtract on (TI: apply after the averaging period) :contentReference[oaicite:18]{index=18}
        start_sub = 0 if self.subtract_during_estimation else num_avg_loops
        if start_sub >= n_loops:
            return

        r_fft2 = r_fft.copy()
        r_fft2[:, :, start_sub:, :][:, :, :, bins] -= sig[:, :, None, :]
        self.write(frame, "range_fft", r_fft2)


@ADCPROCESSORS.register_module()
class DopplerFFT(BaseSignalProcess):
    """2D FFT over loops -> Doppler bins (fftshifted)."""
    requires = {"range_fft"}
    provides = {"doppler_fft"}

    def __init__(self, window: str = "hann", clutter_removal: bool = False, name: str = ""):
        super().__init__(name)
        self.window = window
        self.clutter_removal = clutter_removal

    def _process_generic(self, frame: RadarFrame) -> None:
        r_fft = self.read(frame, "range_fft")  # (Tx,Rx,Loops,Range)
        if self.clutter_removal:
            # Subtract mean prior to 2D FFT
            r_fft = r_fft - r_fft.mean(axis=2, keepdims=True)

        n = frame.config.loops_per_frame
        w = _get_window(self.window, n).reshape(1, 1, -1, 1)

        fft_out = np.fft.fft(r_fft * w, axis=2)
        d_fft = np.fft.fftshift(fft_out, axes=2)  # IMPORTANT: shifted Doppler indexing
        self.write(frame, "doppler_fft", d_fft)
 
@ADCPROCESSORS.register_module()
class DopplerCompensation(BaseSignalProcess):
    """
    Doppler compensation for TDM-MIMO.
    """
    requires = {"doppler_fft"}
    provides = {"doppler_fft"}

    def __init__(
        self,
        enabled: bool = True,
        *,
        tx_spacing_chirps: float = 1.0,
        doppler_fftshifted: bool = True,
        reference_tx: int = 0,
        name: str = "DopplerCompensation",
    ):
        super().__init__(name)
        self.enabled = bool(enabled)
        self.tx_spacing_chirps = float(tx_spacing_chirps)
        self.doppler_fftshifted = bool(doppler_fftshifted)
        self.reference_tx = int(reference_tx)

    def _process_generic(self, frame: RadarFrame) -> None:
        if not self.enabled:
            return

        dfft = self.read(frame, "doppler_fft")  # (Tx,Rx,D,R)
        cfg = frame.config
        n_tx = int(cfg.num_tx)

        if n_tx <= 1:
            return

        if dfft.ndim != 4:
            raise ValueError(f"[{self.name}] Expected doppler_fft with 4 dims (Tx,Rx,D,R). Got {dfft.shape}")

        tx, rx, nd, nr = dfft.shape
        if tx != n_tx:
            # Not fatal, but usually indicates an upstream mismatch
            raise ValueError(f"[{self.name}] doppler_fft Tx dim={tx} does not match cfg.num_tx={n_tx}")

        # Doppler signed bin index (your DopplerFFT is fftshifted by default) :contentReference[oaicite:5]{index=5}
        if self.doppler_fftshifted:
            d_signed = (np.arange(nd, dtype=np.float64) - (nd // 2))
        else:
            # If not shifted, signed mapping: [0..nd-1] -> [0..nd/2-1, -nd/2..-1]
            d_signed = np.arange(nd, dtype=np.float64)
            d_signed[d_signed >= (nd // 2)] -= nd

        # TX index relative to reference (TX0 is typically reference)
        tx_ids = (np.arange(n_tx, dtype=np.float64) - float(self.reference_tx)).reshape(-1, 1)

        # Compensation phase:
        #   exp(-j * 2π * d_signed * tx_id * tx_spacing_chirps / (nd * num_tx))
        # (Derived for your per-TX-loop Doppler FFT representation.)
        phase = np.exp(
            -1j
            * 2.0
            * np.pi
            * (tx_ids * (self.tx_spacing_chirps * d_signed.reshape(1, -1)))
            / (float(nd) * float(n_tx))
        )  # (Tx, D)

        dfft_comp = dfft * phase[:, None, :, None]
        self.write(frame, "doppler_fft", dfft_comp)