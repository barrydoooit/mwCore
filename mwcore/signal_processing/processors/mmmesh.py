import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.ndimage import convolve1d

from mwcore.registry import ADCPROCESSORS
from mwcore.signal_processing.frame import RadarFrame
from mwcore.signal_processing.processors.base import BaseSignalProcess, Supports2D, Supports3D
from .utils import apply_doppler_compensation


@ADCPROCESSORS.register_module()
class TopKDetector(BaseSignalProcess):
    """
    The standard mmMesh detection strategy.
    Instead of CFAR, it selects the top K peaks from the energy map.
    Reference: 'EnergyTop128' in pc_generation.py
    """
    requires = {'doppler_fft'}
    provides = {'detected_points', 'energy_map'}

    def __init__(
        self,
        top_k: int = 128,
        range_cut_idx: Tuple[int, int] = (25, 125),
        floor_db: float = -100.0,
        name: str = "TopKDetector",
    ):
        super().__init__(name)
        self.top_k = int(top_k)
        self.min_range_idx, self.max_range_idx = int(range_cut_idx[0]), int(range_cut_idx[1])
        self.floor_db = float(floor_db)

    def _process_generic(self, frame: RadarFrame) -> None:
        doppler_fft = self.read(frame, "doppler_fft")  # (Tx, Rx, DopplerBins, RangeBins)

        # (DopplerBins, RangeBins): coherent integration across antennas (phase-sensitive)
        coh = np.sum(doppler_fft, axis=(0, 1))

        # Uses log10(|.|) (not 10log10(power))
        # Zeros become -inf, which is also consistent with the reference behavior.
        score_map = np.log10(np.abs(coh))

        # mmMesh range cut (reference uses -100)
        if self.min_range_idx > 0:
            score_map[:, : self.min_range_idx] = self.floor_db
        score_map[:, self.max_range_idx :] = self.floor_db

        flat = score_map.ravel()
        if flat.size < 2:
            detected_points = np.zeros((0, 3), dtype=np.float32)

            p = np.sum(np.abs(doppler_fft) ** 2, axis=(0, 1))
            energy_map_db = 10.0 * np.log10(p + 1e-12)
            if self.min_range_idx > 0:
                energy_map_db[:, : self.min_range_idx] = self.floor_db
            energy_map_db[:, self.max_range_idx :] = self.floor_db

            self.write(frame, "detected_points", detected_points)
            self.write(frame, "energy_map", energy_map_db.astype(np.float32, copy=False))
            return

        # Replicate reference thresholding: idx = total_bins - top_k - 1
        k = min(max(self.top_k, 1), flat.size - 1)
        idx = flat.size - k - 1
        threshold = np.partition(flat, idx)[idx]

        mask = score_map > threshold  # strict > like reference

        det_indices = np.argwhere(mask)  # (N, 2): [doppler_idx, range_idx]
        d_idxs = det_indices[:, 0].astype(np.int32, copy=False)
        r_idxs = det_indices[:, 1].astype(np.int32, copy=False)
        p = np.sum(np.abs(doppler_fft) ** 2, axis=(0, 1))  # (DopplerBins, RangeBins)
        energy_map_db = 10.0 * np.log10(p + 1e-12)

        # Apply the same range-cut floor for consistency in output
        if self.min_range_idx > 0:
            energy_map_db[:, : self.min_range_idx] = self.floor_db
        energy_map_db[:, self.max_range_idx :] = self.floor_db

        peaks_vals = energy_map_db[d_idxs, r_idxs].astype(np.float32, copy=False)
        # column order: [range_idx, doppler_idx, peak_value]
        detected_points = np.column_stack((r_idxs, d_idxs, peaks_vals)).astype(np.float32, copy=False)

        self.write(frame, "detected_points", detected_points)
        self.write(frame, "energy_map", energy_map_db.astype(np.float32, copy=False))


@ADCPROCESSORS.register_module()
class NaiveAoA(BaseSignalProcess, Supports2D, Supports3D):
    """
    mmMesh Standard 'Naive' AoA.
    Refactored to support explicit 2D/3D modes.
    """
    requires = {'detected_points', 'doppler_fft'}
    provides = {'point_cloud'}

    def __init__(self, 
                 fft_size: int = 64, 
                 points_first: bool = True,
                 apply_doppler_comp: bool = True,
                 tx_offsets: Optional[List[int]] = None,
                 azimuth_tx_indices: tuple[int, int] = (0, 1), 
                 elevation_tx_index: int = 2,
                 name: str = "NaiveAoA"):
        super().__init__(name)
        self.points_first = points_first
        self.fft_size = fft_size
        self.apply_doppler_comp = apply_doppler_comp
        self.tx_offsets = tx_offsets
        self.azimuth_tx_indices = azimuth_tx_indices
        self.elevation_tx_index = elevation_tx_index

    def _get_vectors_2d(self, azimuth_ant):
        """Helper to get X vector only."""
        num_detected = azimuth_ant.shape[1]
        az_padded = np.zeros((self.fft_size, num_detected), dtype=np.complex128)

        num_az_ant = azimuth_ant.shape[0]
        az_padded[:num_az_ant, :] = azimuth_ant
        # az_padded[:8, :] = azimuth_ant
        az_fft = np.fft.fft(az_padded, axis=0)
        k_max = np.argmax(np.abs(az_fft), axis=0)
        
        # Logic to extract peak complex value (for 3D phase usage later)
        peak_complex = az_fft[k_max, np.arange(num_detected)]
        
        k_max_signed = k_max.copy()
        k_max_signed[k_max > (self.fft_size // 2) - 1] -= self.fft_size
        
        wx = 2 * np.pi * k_max_signed / self.fft_size
        x_vec = wx / np.pi
        return x_vec, peak_complex, wx

    def process_2d(self, frame: RadarFrame) -> None:
        # Safely read inputs
        det_points = self.read(frame, 'detected_points')
        doppler_fft = self.read(frame, 'doppler_fft')
        
        if len(det_points) == 0:
            pc_dim_first = np.zeros((6, 0))
            self.write(frame, 'point_cloud', pc_dim_first.T if self.points_first else pc_dim_first)
            return

        r_idxs = det_points[:, 0].astype(int)
        d_idxs = det_points[:, 1].astype(int)
        
        # Shape: (Tx, Rx, N) -> (2, 4, N)
        idx1, idx2 = self.azimuth_tx_indices
        data = doppler_fft[[idx1, idx2], :, d_idxs, r_idxs].copy()
        if self.apply_doppler_comp:
            data = apply_doppler_compensation(
                data=data,
                d_idxs=d_idxs,
                num_doppler_bins=frame.config.loops_per_frame,
                num_tx=frame.config.num_tx,
                tx_offsets=self.tx_offsets
            )
        azimuth_ant = np.concatenate((data[0], data[1]), axis=0)
        
        x_vec, _, _ = self._get_vectors_2d(azimuth_ant)
        
        # 2D Geometry
        y_sq = 1 - x_vec**2
        valid = y_sq > 0
        
        y_vec = np.zeros_like(x_vec)
        y_vec[valid] = np.sqrt(y_sq[valid])
        z_vec = np.zeros_like(x_vec) # Z=0
        
        # Build cloud and write it
        pc = self._build_cloud(frame, x_vec, y_vec, z_vec, valid, r_idxs, d_idxs)
        self.write(frame, 'point_cloud', pc)

    def process_3d(self, frame: RadarFrame) -> None:
        # Safely read inputs
        det_points = self.read(frame, 'detected_points')
        doppler_fft = self.read(frame, 'doppler_fft')
        
        if len(det_points) == 0:
            pc_dim_first = np.zeros((6, 0))
            self.write(frame, 'point_cloud', pc_dim_first.T if self.points_first else pc_dim_first)
            return

        r_idxs = det_points[:, 0].astype(int)
        d_idxs = det_points[:, 1].astype(int)
        
        # Extract All Txs
        data = doppler_fft[:, :, d_idxs, r_idxs].copy()
        if self.apply_doppler_comp:
            data = apply_doppler_compensation(
                data=data,
                d_idxs=d_idxs,
                num_doppler_bins=frame.config.loops_per_frame,
                num_tx=frame.config.num_tx,
                tx_offsets=self.tx_offsets
            )
            
        idx1, idx2 = self.azimuth_tx_indices
        el_idx = self.elevation_tx_index
        azimuth_ant = np.concatenate((data[idx1], data[idx2]), axis=0)
        elevation_ant = data[el_idx] # Tx1
        
        # Get X
        x_vec, peak_az, wx = self._get_vectors_2d(azimuth_ant)
        
        # Get Z (Elevation Phase Diff)
        # Pad elevation to same size
        el_padded = np.zeros((self.fft_size, data.shape[2]), dtype=np.complex128)
        # el_padded[:4, :] = elevation_ant # 4 Rx
        el_padded[:elevation_ant.shape[0], :] = elevation_ant
        el_fft = np.fft.fft(el_padded, axis=0)
        el_max_idx = np.argmax(np.abs(el_fft), axis=0)
        peak_el = el_fft[el_max_idx, np.arange(data.shape[2])]
        
        # Phase Compare
        wz = np.angle(peak_az * np.conj(peak_el) * np.exp(1j * 2 * wx))
        z_vec = wz / np.pi
        
        # 3D Geometry
        y_sq = 1 - x_vec**2 - z_vec**2
        valid = y_sq > 0
        y_vec = np.zeros_like(x_vec)
        y_vec[valid] = np.sqrt(y_sq[valid])
        
        # Build cloud and write it
        pc = self._build_cloud(frame, x_vec, y_vec, z_vec, valid, r_idxs, d_idxs)
        self.write(frame, 'point_cloud', pc)
        
    def _build_cloud(self, frame, x_vec, y_vec, z_vec, valid, r_idxs, d_idxs):
        """Helper to assemble final array."""
        cfg = frame.config
        r_vals = r_idxs[valid] * cfg.range_resolution
        
        # Doppler FFT is fftshift()'d (see DopplerFFT.execute), so center index is N/2.
        d_signed = d_idxs[valid].astype(np.int32) - (cfg.loops_per_frame // 2)
        v_vals = d_signed * cfg.doppler_resolution
        
        # Read the detected points safely to get the SNR column
        det_points = self.read(frame, 'detected_points')
        snr = det_points[valid, 2]
        
        x = x_vec[valid] * r_vals
        y = y_vec[valid] * r_vals
        z = z_vec[valid] * r_vals
        
        pc_dim_first = np.stack((x, y, z, v_vals, snr, r_vals), axis=0)
        
        # Simply return the matrix instead of writing directly to frame
        return pc_dim_first.T if self.points_first else pc_dim_first