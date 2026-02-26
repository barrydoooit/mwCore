from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from mwcore.registry import ADCPROCESSORS
from mwcore.signal_processing.frame import RadarFrame
from .utils import _cov_inv, _mvdr_spectrum, _nextpow2, _steering_vec_mu
from ..base import BaseSignalProcess



@ADCPROCESSORS.register_module()
class DynamicCaponElevDopplerWM(BaseSignalProcess):
    """
    For each (range_idx, az_idx) detection:
      - compute elevation via 1D MVDR spectrum (single peak)
      - compute Doppler via MVDR beamforming + FFT peak pick
      - output point_cloud (6,N) in DATA_FORMAT.md format :contentReference[oaicite:17]{index=17}

    Inputs:
      virt_cube:       (12, Loops, Range) complex64   (raw)
      virt_cube_cr:    (12, Loops, Range) complex64   (clutter removed)
      ra_detections:   (N,4) float32 [range_idx, az_idx, snr_lin, noise_lin]
      nu_grid:         (Naz,) float32

    Outputs:
      point_cloud:     (6,N) float32

    Params (from TI dynamic2DAngleCfg + Doppler notes):
      elev_search_step_deg: 1.5
      diag_loading:         0.0300
      fov_el_deg:           (-30,30) typical; MUST match your use-case.
      doppler_fft_size:     optional; if None uses nextpow2(Loops)
    Board params:
      ant_geom_m (12,), ant_geom_n (12,)
    """
    requires = {"virt_cube", "virt_cube_cr", "ra_detections", "nu_grid"}
    provides = {"point_cloud"}

    def __init__(
        self,
        elev_search_step_deg: float = 1.5,
        diag_loading: float = 0.0300,
        fov_el_deg: Tuple[float, float] = (-30.0, 30.0),
        ant_geom_m: Optional[Tuple[int, ...]] = None,
        ant_geom_n: Optional[Tuple[int, ...]] = None,
        doppler_fft_size: Optional[int] = None,
        points_first: bool = True,
        name: str = "DynamicCaponElevDopplerWM",
    ):
        super().__init__(name)
        self.elev_search_step_deg = float(elev_search_step_deg)
        self.diag_loading = float(diag_loading)
        self.fov_el_deg = (float(fov_el_deg[0]), float(fov_el_deg[1]))
        self.ant_geom_m = None if ant_geom_m is None else np.asarray(ant_geom_m, dtype=np.float32)
        self.ant_geom_n = None if ant_geom_n is None else np.asarray(ant_geom_n, dtype=np.float32)
        self.doppler_fft_size = None if doppler_fft_size is None else int(doppler_fft_size)
        self.points_first = bool(points_first)

    def _process_generic(self, frame: RadarFrame) -> None:
        V_raw = self.read(frame, "virt_cube").astype(np.complex64)      # (12,Loops,Range)
        V_cr  = self.read(frame, "virt_cube_cr").astype(np.complex64)   # (12,Loops,Range)
        det   = self.read(frame, "ra_detections").astype(np.float32)    # (N,4)
        nu    = self.read(frame, "nu_grid").astype(np.float32)          # (Naz,)

        if det.size == 0:
            self.write(frame, "point_cloud", np.zeros((6, 0), dtype=np.float32))
            return

        if V_raw.ndim != 3 or V_raw.shape[0] != 12:
            raise ValueError(f"[{self.name}] virt_cube must be (12,Loops,Range). Got {V_raw.shape}")

        nant, loops, n_range = V_raw.shape

        # Geometry indices (must be supplied for correct angles)
        if self.ant_geom_m is None or self.ant_geom_n is None:
            # Fallback (NOT accurate): simple indices
            m_all = np.arange(12, dtype=np.float32)
            n_all = np.zeros((12,), dtype=np.float32)
        else:
            if self.ant_geom_m.shape[0] != 12 or self.ant_geom_n.shape[0] != 12:
                raise ValueError(f"[{self.name}] ant_geom_m/n must have length 12.")
            m_all = self.ant_geom_m
            n_all = self.ant_geom_n

        # Elevation mu-grid
        el_min, el_max = self.fov_el_deg
        if el_max <= el_min:
            raise ValueError(f"[{self.name}] invalid fov_el_deg={self.fov_el_deg}")
        step = self.elev_search_step_deg
        n_el = int(np.floor((el_max - el_min) / step)) + 1
        theta = (el_min + step * np.arange(n_el, dtype=np.float32)) * (np.pi / 180.0)
        mu = np.sin(theta).astype(np.float32)  # mu = sin(theta)

        # Precompute elevation-only steering vectors b(mu) (Nel,12)
        B = _steering_vec_mu(n_all, mu)  # (Nel,12)

        # Doppler FFT size
        n_fft_d = loops if self.doppler_fft_size is None else int(self.doppler_fft_size)
        if self.doppler_fft_size is None:
            n_fft_d = _nextpow2(loops)

        cfg = frame.config
        # We assume cfg.doppler_resolution already matches your DopplerFFT convention (fftshifted)
        # and cfg.range_resolution exists. DATA_FORMAT.md uses these. :contentReference[oaicite:18]{index=18}

        pts = []
        for i in range(det.shape[0]):
            r_idx = int(det[i, 0])
            az_idx = int(det[i, 1])
            snr_lin = float(det[i, 2])

            if not (0 <= r_idx < n_range):
                continue
            if not (0 <= az_idx < nu.shape[0]):
                continue

            nu_det = float(nu[az_idx])

            # Build mixed steering u(mu_j, nu_det) = a_az(nu_det) ∘ b(mu_j)
            # a_az(nu_det): exp(j*pi*m*nu_det)
            a_az = np.exp(1j * np.pi * m_all * nu_det).astype(np.complex64)  # (12,)
            U = (B * a_az[None, :]).astype(np.complex64)  # (Nel,12)

            # 12x12 covariance inverse at this range using clutter-removed snapshots
            Y12 = V_cr[:, :, r_idx]  # (12,Loops)
            invR12 = _cov_inv(Y12, self.diag_loading)  # (12,12)

            # Elevation MVDR spectrum
            P_el = _mvdr_spectrum(invR12, U)  # (Nel,)
            el_bin = int(np.argmax(P_el))
            mu_det = float(mu[el_bin])

            # Convert direction cosines (nu,mu) -> angles:
            # mu = sin(theta) -> theta
            theta_det = float(np.arcsin(np.clip(mu_det, -1.0, 1.0)))
            # nu = cos(theta)*sin(phi) -> phi
            cth = float(np.cos(theta_det))
            if abs(cth) < 1e-6:
                phi_det = 0.0
            else:
                arg = np.clip(nu_det / cth, -1.0, 1.0)
                phi_det = float(np.arcsin(arg))

            # MVDR weights for Doppler beamforming: w = invR u / (u^H invR u)
            u_det = U[el_bin, :]  # (12,)
            num = invR12 @ u_det
            den = (u_det.conj().T @ num).real
            den = max(float(den), 1e-12)
            w = (num / den).astype(np.complex64)

            # Beamform raw (non-clutter-removed) slow-time sequence at this range
            X = V_raw[:, :, r_idx]  # (12,Loops)
            y = (w.conj().T @ X).astype(np.complex64)  # (Loops,)

            # Doppler FFT + fftshift
            y_pad = np.zeros((n_fft_d,), dtype=np.complex64)
            y_pad[:loops] = y
            Yd = np.fft.fftshift(np.fft.fft(y_pad, axis=0), axes=0)
            mag2 = (np.abs(Yd) ** 2).astype(np.float32)
            d_idx = int(np.argmax(mag2))

            # signed doppler bin relative to fftshifted center
            d_signed = d_idx - (n_fft_d // 2)

            # Convert to m/s using cfg.doppler_resolution (assumed per-bin after fftshift)
            v_mps = float(d_signed) * float(getattr(cfg, "doppler_resolution", 0.0))

            # Range meters
            r_m = float(r_idx) * float(getattr(cfg, "range_resolution", 0.0)) - float(getattr(cfg, "range_bias_m", 0.0))

            # Convert angles to Cartesian in sensor frame (x right, y forward, z up):
            # direction cosines:
            # mu = sin(theta) -> z component fraction
            # nu = cos(theta)*sin(phi) -> x component fraction
            # y = sqrt(1 - x^2 - z^2)
            x_sin = float(np.sin(phi_det) * np.cos(theta_det))  # == nu_det
            z_sin = float(np.sin(theta_det))                    # == mu_det
            y2 = 1.0 - x_sin * x_sin - z_sin * z_sin
            if y2 <= 0.0:
                continue
            y_sin = float(np.sqrt(y2))

            x = x_sin * r_m
            yv = y_sin * r_m
            z = z_sin * r_m

            pts.append([x, yv, z, v_mps, snr_lin, r_m])

        if len(pts) == 0:
            pc = np.zeros((6, 0), dtype=np.float32)
        else:
            pc = np.asarray(pts, dtype=np.float32).T  # (6,N)
        pc = pc.T if self.points_first else pc  # (N,6) if points_first else (6,N)
        self.write(frame, "point_cloud", pc)