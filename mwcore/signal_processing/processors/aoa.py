import numpy as np

from mwcore.registry import ADCPROCESSORS
from mwcore.signal_processing.frame import RadarFrame
from mwcore.signal_processing.processors.base import BaseSignalProcess, Supports2D, Supports3D



@ADCPROCESSORS.register_module()
class AoA_DopplerCompensated(BaseSignalProcess, Supports2D, Supports3D):
    """
    Professional Beamforming AoA.
    Supports both 2D (Azimuth only) and 3D (Azimuth + Elevation).
    """
    def __init__(self, name: str = "AoA_DopplerCompensated"):
        super().__init__(name)
    
    def _prepare_data(self, frame: RadarFrame):
        """Common data extraction and Doppler compensation."""
        det_points = frame.detected_points
        if det_points is None or len(det_points) == 0:
            return None, None, None

        cfg = frame.config
        r_idxs = det_points[:, 0].astype(int)
        d_idxs = det_points[:, 1].astype(int)
        
        # Extract data: (Tx, Rx, N_det)
        antenna_data = frame.doppler_fft[:, :, d_idxs, r_idxs]
        
        # Doppler Compensation (TDM Correction)
        # Tx1 fires 1 unit later, Tx2 fires 2 units later (relative to Tx0)
        # Note: Check specific board layout. Usually Tx0->Tx2->Tx1
        N_doppler = cfg.loops_per_frame
        d_idxs_signed = d_idxs.copy()
        d_idxs_signed[d_idxs_signed >= N_doppler // 2] -= N_doppler

        for i_obj in range(len(d_idxs)):
            dop_idx = d_idxs_signed[i_obj]
            # Phase correction per unit time delay
            phase_corr = np.exp(-1j * 2 * np.pi * dop_idx / (N_doppler * cfg.num_tx))
            
            # Apply to Tx2 (Delay 1) and Tx1 (Delay 2) - assuming 0-2-1 order
            antenna_data[1, :, i_obj] *= phase_corr       # Tx2
            antenna_data[2, :, i_obj] *= (phase_corr ** 2) # Tx1

        return antenna_data, r_idxs, d_idxs_signed

    def process_2d(self, frame: RadarFrame) -> None:
        """2D Implementation: Calculate X, Y. Force Z=0."""
        data, r_idxs, d_idxs_signed = self._prepare_data(frame)
        if data is None:
            frame.point_cloud = np.zeros((6, 0))
            return
            
        # 1. Select Azimuth Antennas Only (Tx0 + Tx2) -> 8 Antennas
        # We ignore Tx1 (Elevation) completely in 2D mode
        azimuth_ant = np.concatenate((data[0], data[1]), axis=0)
        
        # 2. Beamforming (FFT)
        num_angle_bins = 64
        az_fft_in = np.zeros((num_angle_bins, data.shape[2]), dtype=np.complex_)
        az_fft_in[:8, :] = azimuth_ant
        
        az_fft = np.fft.fft(az_fft_in, axis=0)
        peak_idxs = np.argmax(np.abs(az_fft), axis=0)
        
        # 3. Angle to Coordinate
        peak_idxs[peak_idxs > num_angle_bins//2] -= num_angle_bins
        x_vec = (2 * peak_idxs) / num_angle_bins
        
        # 4. Geometry (2D: y = sqrt(1 - x^2))
        y_sq = 1 - x_vec**2
        valid = y_sq > 0
        
        y_vec = np.zeros_like(x_vec)
        y_vec[valid] = np.sqrt(y_sq[valid])
        
        # 5. Build Cloud
        cfg = frame.config
        r_vals = r_idxs[valid] * cfg.range_resolution
        x = x_vec[valid] * r_vals
        y = y_vec[valid] * r_vals
        z = np.zeros_like(x) # Z is explicitly 0
        v = d_idxs_signed[valid] * cfg.doppler_resolution
        snr = frame.detected_points[valid, 2]
        
        frame.point_cloud = np.stack((x, y, z, v, snr, r_vals), axis=0)

    def process_3d(self, frame: RadarFrame) -> None:
        """3D Implementation: Calculate X, Y, Z."""
        data, r_idxs, d_idxs_signed = self._prepare_data(frame)
        if data is None:
            frame.point_cloud = np.zeros((6, 0))
            return

        # 1. Azimuth FFT (Same as 2D)
        azimuth_ant = np.concatenate((data[0], data[1]), axis=0)
        num_angle_bins = 64
        az_fft_in = np.zeros((num_angle_bins, data.shape[2]), dtype=np.complex_)
        az_fft_in[:8, :] = azimuth_ant
        az_fft = np.fft.fft(az_fft_in, axis=0)
        peak_idxs = np.argmax(np.abs(az_fft), axis=0)
        
        peak_idxs[peak_idxs > num_angle_bins//2] -= num_angle_bins
        x_vec = (2 * peak_idxs) / num_angle_bins
        
        # 2. Elevation (Phase Difference)
        # Compare Tx1 (Elev) vs Tx0 (Azim reference)
        # Note: Professional code often matches specific overlapping arrays.
        # Simple Phase Diff approach:
        elevation_ant = data[2] # Tx1
        # Correlate Elevation row with Azimuth row (Tx0)
        phase_diffs = np.angle(elevation_ant * np.conj(data[0]))
        wz_avg = np.mean(phase_diffs, axis=0)
        z_vec = wz_avg / np.pi
        
        # 3. Geometry (3D: y = sqrt(1 - x^2 - z^2))
        y_sq = 1 - x_vec**2 - z_vec**2
        valid = y_sq > 0
        
        y_vec = np.zeros_like(x_vec)
        y_vec[valid] = np.sqrt(y_sq[valid])
        z_vec = z_vec[valid] # Keep Z where valid
        
        # 4. Build Cloud
        cfg = frame.config
        r_vals = r_idxs[valid] * cfg.range_resolution
        x = x_vec[valid] * r_vals
        y = y_vec[valid] * r_vals
        z = z_vec[valid] * r_vals
        v = d_idxs_signed[valid] * cfg.doppler_resolution
        snr = frame.detected_points[valid, 2]
        
        frame.point_cloud = np.stack((x, y, z, v, snr, r_vals), axis=0)