from typing import Union
import numpy as np
from .configs import ProcessConfig


class PointCloudGenerator:
    class _FrameConfig:
        def __init__(self, config: ProcessConfig):
            self.numTxAntennas = config.tx_antennas
            self.numRxAntennas = config.rx_antennas
            self.numLoopsPerFrame = config.chirps_per_frame
            self.numADCSamples = config.samples_per_chirp
            self.numAngleBins = config.num_angle_bins

            self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
            self.numRangeBins = self.numADCSamples
            self.numDopplerBins = self.numLoopsPerFrame

    def __init__(self, config: Union[dict, ProcessConfig]):
        self.config = ProcessConfig.from_dict(config)
        self._frame_cfg = self._FrameConfig(config)
        self._pos_offset_arr = np.array(self.config.mmwave_radar_loc)
        self._range_window = np.hamming(self.config.samples_per_chirp)
        self._doppler_window = np.reshape(np.hamming(self.config.num_doppler_bins), (1, 1, -1, 1))
        self.enableStaticClutterRemoval = self.config.enable_static_clutter_removal
        self.RangeCut = self.config.range_cut
        self.EnergyTop128 = self.config.energy_top_128

    def transform(self, frame_data: np.ndarray) -> np.ndarray:
        np_frame = self._bin2np_frame(frame_data)
        point_cloud = self._frame2pointcloud(np_frame)
        if point_cloud.shape[0] == 0 or point_cloud.shape[1] == 0:
            return np.array([]).reshape(0, 6)
        raw_points = np.transpose(point_cloud, (1, 0))
        raw_points[:, :3] = raw_points[:, :3] + self._pos_offset_arr
        final_point_cloud = raw_points[:, [0, 1, 2, 3, 4, 5]]
        return final_point_cloud

    def _bin2np_frame(self, bin_frame):
        """Converts the int16 raw data to complex."""
        np_frame = np.zeros(shape=(len(bin_frame) // 2), dtype=np.complex_)
        # This implicitly handles iq_format=2 and bytes_per_sample_part=2
        np_frame[0::2] = bin_frame[0::4] + 1j * bin_frame[2::4]
        np_frame[1::2] = bin_frame[1::4] + 1j * bin_frame[3::4]
        return np_frame

    def _frameReshape(self, frame):
        frameConfig = self._frame_cfg
        frameWithChirp = np.reshape(frame, (frameConfig.numLoopsPerFrame, 
                                            frameConfig.numTxAntennas, 
                                            frameConfig.numRxAntennas, -1))
        return frameWithChirp.transpose(1, 2, 0, 3)

    def _rangeFFT(self, reshapedFrame):
        windowedBins1D = reshapedFrame * self._range_window
        rangeFFTResult = np.fft.fft(windowedBins1D)
        return rangeFFTResult

    def _clutter_removal(self, input_val, axis=2):
        reordering = np.arange(len(input_val.shape))
        reordering[0] = axis
        reordering[axis] = 0
        input_val_reordered = input_val.transpose(reordering)
        
        mean = input_val_reordered.mean(0)
        output_val = input_val_reordered - mean
        
        return output_val.transpose(reordering)

    def _dopplerFFT(self, rangeResult):
        windowedBins2D = rangeResult * self._doppler_window
        dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2)
        dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)
        return dopplerFFTResult

    def _naive_xyz(self, virtual_ant):
        num_tx = self.config.tx_antennas
        num_rx = self.config.rx_antennas
        fft_size = self.config.num_angle_bins # 64
        
        num_detected_obj = virtual_ant.shape[1]
        azimuth_ant = virtual_ant[:2 * num_rx, :]
        azimuth_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
        azimuth_ant_padded[:2 * num_rx, :] = azimuth_ant

        azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
        k_max = np.argmax(np.abs(azimuth_fft), axis=0)
        
        peak_1 = azimuth_fft[k_max, np.arange(num_detected_obj)]

        k_max[k_max > (fft_size // 2) - 1] = k_max[k_max > (fft_size // 2) - 1] - fft_size
        wx = 2 * np.pi / fft_size * k_max
        x_vector = wx / np.pi

        elevation_ant = virtual_ant[2 * num_rx:, :]
        elevation_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
        elevation_ant_padded[:num_rx, :] = elevation_ant

        elevation_fft = np.fft.fft(elevation_ant_padded, axis=0)
        elevation_max = np.argmax(np.log2(np.abs(elevation_fft) + 1e-9), axis=0)
        
        peak_2 = elevation_fft[elevation_max, np.arange(num_detected_obj)]

        wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * 2 * wx))
        z_vector = wz / np.pi
        ypossible = 1 - x_vector ** 2 - z_vector ** 2
        
        y_vector = np.zeros_like(ypossible)
        valid_indices = ypossible >= 0
        y_vector[valid_indices] = np.sqrt(ypossible[valid_indices])
        
        x_vector[~valid_indices] = 0
        z_vector[~valid_indices] = 0
        
        return x_vector, y_vector, z_vector
    
    def _frame2pointcloud(self, frame):
        frameConfig = self._frame_cfg
        
        reshapedFrame = self._frameReshape(frame)
        rangeResult = self._rangeFFT(reshapedFrame)
        
        if self.enableStaticClutterRemoval:
            rangeResult = self._clutter_removal(rangeResult, axis=2)
            
        dopplerResult = self._dopplerFFT(rangeResult)

        dopplerResultSumAllAntenna = np.sum(dopplerResult, axis=(0, 1))
        dopplerResultInDB = 10 * np.log10(np.absolute(dopplerResultSumAllAntenna) + 1e-9)

        if self.RangeCut:
            dopplerResultInDB[:, :25] = -100
            dopplerResultInDB[:, 125:] = -100

        cfarResult = np.zeros(dopplerResultInDB.shape, bool)
        if self.EnergyTop128:
            top_size = 128
            try:
                energyThre128 = np.partition(dopplerResultInDB.ravel(), -top_size)[-top_size]
                cfarResult[dopplerResultInDB > energyThre128] = True
            except (ValueError, IndexError):
                pass

        det_peaks_indices = np.argwhere(cfarResult == True)
        if det_peaks_indices.shape[0] == 0:
             return np.array([]).reshape(6, 0)
            
        R = det_peaks_indices[:, 1].astype(np.float64)
        V = (det_peaks_indices[:, 0] - frameConfig.numDopplerBins // 2).astype(np.float64)
        
        # Use config-derived physics values
        R *= self.config.range_resolution_m
        V *= self.config.doppler_resolution_mps
        
        energy = dopplerResultInDB[cfarResult == True]

        AOAInput = dopplerResult[:, :, cfarResult == True]
        AOAInput = AOAInput.reshape(frameConfig.numTxAntennas * frameConfig.numRxAntennas, -1)

        if AOAInput.shape[1] == 0:
            return np.array([]).reshape(6, 0)
            
        x_vec, y_vec, z_vec = self._naive_xyz(AOAInput)
        
        x, y, z = x_vec * R, y_vec * R, z_vec * R
        
        pointCloud = np.stack((x, y, z, V, energy, R), axis=0)
        
        pointCloud = pointCloud[:, y_vec != 0]
        
        return pointCloud