import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.ndimage import convolve1d

from mwcore.registry import ADCPROCESSORS
from signal_processing.frame import RadarFrame
from signal_processing.processors.base import BaseSignalProcess, Supports2D, Supports3D



@ADCPROCESSORS.register_module()
class TopKDetector(BaseSignalProcess):
    """
    The standard mmMesh detection strategy.
    Instead of CFAR, it selects the top K peaks from the energy map.
    Reference: 'EnergyTop128' in pc_generation.py
    """
    def __init__(self, top_k: int = 128, range_cut_idx: Tuple[int, int] = (25, 125), name: str = "TopKDetector"):
        super().__init__(name)
        self.top_k = top_k
        self.min_range_idx, self.max_range_idx = range_cut_idx

    def execute(self, frame: RadarFrame):
        if frame.doppler_fft is None:
            raise ValueError("Doppler FFT missing")

        # 1. Calculate Energy Map (Sum of Mags across antennas)
        # Shape: (Loops, Samples) -> (Doppler, Range)
        energy_map = np.sum(np.abs(frame.doppler_fft), axis=(0, 1))
        
        # 2. Hardcoded Range Cut (Specific to mmMesh logic to remove near/far noise)
        # Note: mmMesh typically zeroes out the edges
        energy_map[:, :self.min_range_idx] = 0
        energy_map[:, self.max_range_idx:] = 0

        # 3. Find Top K Threshold
        flat_energy = energy_map.ravel()
        # Safety check if K is larger than total bins
        k = min(self.top_k, flat_energy.size - 1)
        
        # np.partition moves the K-th largest element to the pivot position
        # We want the indices of the largest K elements
        partition_idx = flat_energy.size - k
        threshold = np.partition(flat_energy, partition_idx)[partition_idx]
        
        # 4. Generate Mask & Indices
        mask = energy_map >= threshold
        det_indices = np.argwhere(mask)  # Shape: (N, 2) -> [Doppler, Range]
        
        # Limit to exactly K if we got more due to duplicate values
        if len(det_indices) > k:
            # Sort by energy to get strictly top K
            # (Optional refinement, mmMesh might just take whatever comes)
            vals = energy_map[mask]
            sort_order = np.argsort(vals)[::-1] # Descending
            det_indices = det_indices[sort_order[:k]]

        # 5. Extract Values
        # det_indices is (Doppler, Range)
        d_idxs = det_indices[:, 0]
        r_idxs = det_indices[:, 1]
        peaks_vals = energy_map[d_idxs, r_idxs]
        
        # Store in Frame: [RangeIdx, DopplerIdx, PeakVal]
        frame.detected_points = np.column_stack((r_idxs, d_idxs, peaks_vals))

@ADCPROCESSORS.register_module()
class NaiveAoA(BaseSignalProcess, Supports2D, Supports3D):
    """
    mmMesh Standard 'Naive' AoA.
    Refactored to support explicit 2D/3D modes.
    """
    def __init__(self, fft_size: int = 64, name: str = "NaiveAoA"):
        super().__init__(name)
        self.fft_size = fft_size

    def _get_vectors_2d(self, azimuth_ant):
        """Helper to get X vector only."""
        num_detected = azimuth_ant.shape[1]
        az_padded = np.zeros((self.fft_size, num_detected), dtype=np.complex_)
        # Assuming 4 Rx, 2 Tx for Azimuth -> 8 lines
        az_padded[:8, :] = azimuth_ant
        
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
        det_points = frame.detected_points
        if det_points is None or len(det_points) == 0:
            frame.point_cloud = np.zeros((6, 0))
            return

        r_idxs = det_points[:, 0].astype(int)
        d_idxs = det_points[:, 1].astype(int)
        
        # Extract Tx0 and Tx2 (Azimuth)
        # Shape: (Tx, Rx, N) -> (2, 4, N)
        data = frame.doppler_fft[[0, 1], :, d_idxs, r_idxs]
        azimuth_ant = np.concatenate((data[0], data[1]), axis=0)
        
        x_vec, _, _ = self._get_vectors_2d(azimuth_ant)
        
        # 2D Geometry
        y_sq = 1 - x_vec**2
        valid = y_sq > 0
        
        y_vec = np.zeros_like(x_vec)
        y_vec[valid] = np.sqrt(y_sq[valid])
        z_vec = np.zeros_like(x_vec) # Z=0
        
        self._build_cloud(frame, x_vec, y_vec, z_vec, valid, r_idxs, d_idxs)

    def process_3d(self, frame: RadarFrame) -> None:
        det_points = frame.detected_points
        if det_points is None or len(det_points) == 0:
            frame.point_cloud = np.zeros((6, 0))
            return

        r_idxs = det_points[:, 0].astype(int)
        d_idxs = det_points[:, 1].astype(int)
        
        # Extract All Txs
        data = frame.doppler_fft[:, :, d_idxs, r_idxs]
        azimuth_ant = np.concatenate((data[0], data[1]), axis=0)
        elevation_ant = data[2] # Tx1
        
        # Get X
        x_vec, peak_az, wx = self._get_vectors_2d(azimuth_ant)
        
        # Get Z (Elevation Phase Diff)
        # Pad elevation to same size
        el_padded = np.zeros((self.fft_size, data.shape[2]), dtype=np.complex_)
        el_padded[:4, :] = elevation_ant # 4 Rx
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
        
        self._build_cloud(frame, x_vec, y_vec, z_vec, valid, r_idxs, d_idxs)
        
    def _build_cloud(self, frame, x_vec, y_vec, z_vec, valid, r_idxs, d_idxs):
        """Helper to assemble final array."""
        cfg = frame.config
        r_vals = r_idxs[valid] * cfg.range_resolution
        
        d_signed = d_idxs[valid].copy()
        # Fix doppler wrapping if needed, depends on FFT Shift state. 
        # Assuming doppler_fft is already shifted centrally, indices are 0..127. 
        # But for velocity calculation we need signed relative to center.
        if np.max(d_signed) > cfg.loops_per_frame // 2:
             d_signed[d_signed >= cfg.loops_per_frame // 2] -= cfg.loops_per_frame
             
        v_vals = d_signed * cfg.doppler_resolution
        snr = frame.detected_points[valid, 2]
        
        x = x_vec[valid] * r_vals
        y = y_vec[valid] * r_vals
        z = z_vec[valid] * r_vals
        
        frame.point_cloud = np.stack((x, y, z, v_vals, snr, r_vals), axis=0)