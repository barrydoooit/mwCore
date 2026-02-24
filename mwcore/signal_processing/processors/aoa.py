from __future__ import annotations

import numpy as np
from mwcore.registry import ADCPROCESSORS
from mwcore.signal_processing.frame import RadarFrame
from mwcore.signal_processing.processors.base import BaseSignalProcess, Supports2D, Supports3D
from .utils import apply_doppler_compensation

def _signed_angle_bin(k: np.ndarray, n: int) -> np.ndarray:
    """Map FFT bin index to signed integer in [-n/2, n/2)."""
    ks = k.astype(np.int32).copy()
    ks[ks > (n // 2) - 1] -= n
    return ks


def _bin_to_sin(ks: np.ndarray, n: int) -> np.ndarray:
    """TI-style mapping used in your existing AoA/math: sin(theta) ≈ 2*k/N."""
    return (2.0 * ks.astype(np.float32)) / float(n)


@ADCPROCESSORS.register_module()
class AoA_TI_DPU(BaseSignalProcess, Supports2D, Supports3D):
    """
    TI-style AoA DPU behavior with:
      - Doppler compensation (TDM MIMO)
      - multiObjBeamForming (2nd azimuth peak)
      - aoaFovCfg filtering

    Output point_cloud follows DATA_FORMAT.md

    Note that it currently only supports the typical TI 3 TX antenna configuration, where
    two antennas are used for azimuth and one for elevation. 
    """
    requires = {"detected_points", "doppler_fft"}
    provides = {"point_cloud"}

    def __init__(
        self,
        num_angle_bins: int = 64,
        points_first: bool = True,
        # Doppler compensation:
        apply_doppler_comp: bool = True,
        tx_offsets: list[int] | None = None,  # e.g. [0,1,2] means Tx0 first, Tx1 second, Tx2 third
        azimuth_tx_indices: tuple[int, int] = (0, 1), 
        elevation_tx_index: int = 2,
        # multiObjBeamForming:
        multi_obj_enable: bool = True,
        multi_obj_thresh: float = 0.5,  # thresholdScale in [0..1]
        multi_obj_exclusion: int = 2,   # bins to exclude around peak1 when searching peak2
        # aoaFovCfg:
        aoa_fov_az_deg: tuple[float, float] = (-90.0, 90.0),
        aoa_fov_el_deg: tuple[float, float] = (-90.0, 90.0),
        name: str = "",
    ):
        super().__init__(name)
        self.n_ang = int(num_angle_bins)
        self.points_first = bool(points_first)

        self.apply_doppler_comp = bool(apply_doppler_comp)
        self.tx_offsets = tx_offsets  # resolved at runtime if None

        self.azimuth_tx_indices = azimuth_tx_indices
        self.elevation_tx_index = int(elevation_tx_index)

        self.multi_obj_enable = bool(multi_obj_enable)
        self.multi_obj_thresh = float(multi_obj_thresh)
        self.multi_obj_exclusion = int(multi_obj_exclusion)

        self.aoa_fov_az_deg = aoa_fov_az_deg
        self.aoa_fov_el_deg = aoa_fov_el_deg

    def _doppler_signed(self, d_idx_shifted: np.ndarray, n_doppler: int) -> np.ndarray:
        # doppler FFT is fftshifted in DopplerFFT
        return d_idx_shifted.astype(np.int32) - (n_doppler // 2)

    def _gather_and_compensate(self, frame: RadarFrame):
        det = self.read(frame, "detected_points")
        if det is None or len(det) == 0:
            return None

        cfg = frame.config
        d_fft = self.read(frame, "doppler_fft")  # (Tx,Rx,Doppler,Range)

        r_idx = det[:, 0].astype(np.int32)
        d_idx = det[:, 1].astype(np.int32)  # shifted indices
        snr_db = det[:, 2].astype(np.float32)

        data = d_fft[:, :, d_idx, r_idx]  # (Tx,Rx,N)

        n_doppler = cfg.loops_per_frame
        d_signed = self._doppler_signed(d_idx, n_doppler)

        if self.apply_doppler_comp:
            data = apply_doppler_compensation(
                            data=data,
                            d_idxs=d_idx,
                            num_doppler_bins=n_doppler,
                            num_tx=cfg.num_tx,
                            tx_offsets=self.tx_offsets
                        )

        return data, r_idx, d_idx, d_signed, snr_db

    def _azimuth_fft(self, az_ant: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        az_ant: (N_az, N)
        Returns:
          k1 (N,), peak1_complex (N,), spectrum_mag2 (N_ang, N)
        """
        n_obj = az_ant.shape[1]
        x = np.zeros((self.n_ang, n_obj), dtype=np.complex64)
        x[: az_ant.shape[0], :] = az_ant.astype(np.complex64)
        X = np.fft.fft(x, axis=0)
        mag2 = (np.abs(X) ** 2).astype(np.float32)
        k1 = np.argmax(mag2, axis=0).astype(np.int32)
        peak1 = X[k1, np.arange(n_obj)]
        return k1, peak1, mag2

    def _second_peak(self, mag2: np.ndarray, k1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Find 2nd peak excluding neighborhood around k1.
        Returns (k2, ok_mask)
        """
        mag2c = mag2.copy()
        n = mag2.shape[0]
        for i in range(k1.size):
            c = int(k1[i])
            lo = max(0, c - self.multi_obj_exclusion)
            hi = min(n, c + self.multi_obj_exclusion + 1)
            mag2c[lo:hi, i] = -np.inf
        k2 = np.argmax(mag2c, axis=0).astype(np.int32)
        p1 = mag2[k1, np.arange(k1.size)]
        p2 = mag2[k2, np.arange(k1.size)]
        ok = p2 >= (self.multi_obj_thresh * p1)  # TI: thresholdScale * firstPeakHeight
        return k2, ok

    def _apply_aoa_fov(self, x_sin: np.ndarray, z_sin: np.ndarray) -> np.ndarray:
        # Convert sin to degrees safely
        az = np.degrees(np.arcsin(np.clip(x_sin, -1.0, 1.0)))
        el = np.degrees(np.arcsin(np.clip(z_sin, -1.0, 1.0)))
        ok = (az >= self.aoa_fov_az_deg[0]) & (az <= self.aoa_fov_az_deg[1]) & \
             (el >= self.aoa_fov_el_deg[0]) & (el <= self.aoa_fov_el_deg[1])
        return ok

    def process_2d(self, frame: RadarFrame) -> None:
        out = self._gather_and_compensate(frame)
        if out is None:
            self.write(frame, "point_cloud", np.zeros((0, 6), dtype=np.float32) if self.points_first else np.zeros((6, 0), dtype=np.float32))
            return

        data, r_idx, d_idx, d_signed, snr_db = out
        cfg = frame.config

        # Azimuth antennas: Tx0 + Tx2 => 8 virtual azimuth channels (matches your current approach)
        idx1, idx2 = self.azimuth_tx_indices
        az_ant = np.concatenate([data[idx1], data[idx2]], axis=0)  # (8, N)

        k1, peak1, mag2 = self._azimuth_fft(az_ant)
        k1s = _signed_angle_bin(k1, self.n_ang)
        x_sin = _bin_to_sin(k1s, self.n_ang)

        # 2D: z=0
        z_sin = np.zeros_like(x_sin, dtype=np.float32)

        # Geometry
        y2 = 1.0 - x_sin ** 2
        valid = y2 > 0
        y_sin = np.zeros_like(x_sin, dtype=np.float32)
        y_sin[valid] = np.sqrt(y2[valid]).astype(np.float32)

        # aoaFovCfg filtering
        valid &= self._apply_aoa_fov(x_sin, z_sin)

        r_m = r_idx.astype(np.float32) * cfg.range_resolution - float(cfg.range_bias_m)
        v_mps = d_signed.astype(np.float32) * cfg.doppler_resolution

        x = x_sin * r_m
        y = y_sin * r_m
        z = z_sin * r_m

        keep = valid & (r_m > 0)
        pc = np.stack([x[keep], y[keep], z[keep], v_mps[keep], snr_db[keep], r_m[keep]], axis=0).astype(np.float32)
        pc = pc.T if self.points_first else pc
        self.write(frame, "point_cloud", pc)

    def process_3d(self, frame: RadarFrame) -> None:
        out = self._gather_and_compensate(frame)
        if out is None:
            self.write(frame, "point_cloud", np.zeros((0, 6), dtype=np.float32) if self.points_first else np.zeros((6, 0), dtype=np.float32))
            return

        data, r_idx, d_idx, d_signed, snr_db = out
        cfg = frame.config

        # Azimuth antennas: Tx0 + Tx2
        idx1, idx2 = self.azimuth_tx_indices
        az_ant = np.concatenate([data[idx1], data[idx2]], axis=0)  # (8, N)
        k1, peak1, mag2 = self._azimuth_fft(az_ant)

        # Optional multi-object: duplicate detections at same (r,d) with 2nd az peak
        k_list = [k1]
        peak_list = [peak1]
        r_list = [r_idx]
        d_list = [d_idx]
        ds_list = [d_signed]
        snr_list = [snr_db]

        if self.multi_obj_enable:
            k2, ok = self._second_peak(mag2, k1)
            if np.any(ok):
                # Gather peak2 complex
                n_obj = az_ant.shape[1]
                xpad = np.zeros((self.n_ang, n_obj), dtype=np.complex64)
                xpad[: az_ant.shape[0], :] = az_ant.astype(np.complex64)
                X = np.fft.fft(xpad, axis=0)
                peak2 = X[k2, np.arange(n_obj)]

                k_list.append(k2[ok])
                peak_list.append(peak2[ok])
                r_list.append(r_idx[ok])
                d_list.append(d_idx[ok])
                ds_list.append(d_signed[ok])
                snr_list.append(snr_db[ok])

        # Concatenate (possibly extended list)
        k_all = np.concatenate(k_list, axis=0)
        peak_az_all = np.concatenate(peak_list, axis=0)
        r_all = np.concatenate(r_list, axis=0)
        d_all = np.concatenate(d_list, axis=0)
        ds_all = np.concatenate(ds_list, axis=0)
        snr_all = np.concatenate(snr_list, axis=0)

        # Rebuild azimuth sin(x)
        k_all_s = _signed_angle_bin(k_all, self.n_ang)
        wx = (2.0 * np.pi * k_all_s.astype(np.float32)) / float(self.n_ang)
        x_sin = (wx / np.pi).astype(np.float32)  # equivalent to 2*k/N

        # Elevation antennas: Tx1 (your current convention)
        # We estimate elevation via phase compare like your mmMesh-style approach.
        # Extract elevation channel data at the same detections.
        # NOTE: data is still only N original objects; for duplicated multiobj items we reuse Tx1 samples by index.
        # We'll map by reconstructing elevation peaks from the original pool:
        # For simplicity, re-gather elevation spectrum for the original N then index into it.

        # Original elevation FFT peaks per original detection
        el_idx = self.elevation_tx_index
        elev_ant = data[el_idx]
        n0 = elev_ant.shape[1]
        el_pad = np.zeros((self.n_ang, n0), dtype=np.complex64)
        el_pad[: elev_ant.shape[0], :] = elev_ant.astype(np.complex64)
        EL = np.fft.fft(el_pad, axis=0)
        # If multiobj duplicated, build mapping indices to the original objects
        base_idx = np.arange(n0, dtype=np.int32)
        map_idx = [base_idx]
        if self.multi_obj_enable:
            _, ok = self._second_peak(mag2, k1)
            if np.any(ok):
                map_idx.append(base_idx[ok])
        map_idx_all = np.concatenate(map_idx, axis=0)

        # Extract elevation complex value at the EXACT azimuth bin 
        # k_all contains the azimuth bin (k1 or k2), map_idx_all contains the object index
        peak_el_all = EL[k_all, map_idx_all]

        # TI-style elevation phase usage (mirrors your mmMesh-style wz expression)
        wz = np.angle(peak_az_all * np.conj(peak_el_all) * np.exp(1j * 2.0 * wx))
        z_sin = (wz / np.pi).astype(np.float32)

        # Geometry
        y2 = 1.0 - x_sin ** 2 - z_sin ** 2
        valid = y2 > 0
        y_sin = np.zeros_like(x_sin, dtype=np.float32)
        y_sin[valid] = np.sqrt(y2[valid]).astype(np.float32)

        # aoaFovCfg filtering :contentReference[oaicite:45]{index=45}
        valid &= self._apply_aoa_fov(x_sin, z_sin)

        r_m = r_all.astype(np.float32) * cfg.range_resolution - float(cfg.range_bias_m)
        v_mps = ds_all.astype(np.float32) * cfg.doppler_resolution

        x = x_sin * r_m
        y = y_sin * r_m
        z = z_sin * r_m

        keep = valid & (r_m > 0)
        pc = np.stack([x[keep], y[keep], z[keep], v_mps[keep], snr_all[keep], r_m[keep]], axis=0).astype(np.float32)
        pc = pc.T if self.points_first else pc
        self.write(frame, "point_cloud", pc)
