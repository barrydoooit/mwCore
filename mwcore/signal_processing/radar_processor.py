
import struct
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Protocol
from abc import ABC, abstractmethod
import math

# Use dataclass for cleaner config
from dataclasses import dataclass, field

@dataclass
class RadarConfig:
    """Configuration for radar sensor parameters."""
    num_tx: int = 3
    num_rx: int = 4
    loops_per_frame: int = 128
    adc_samples: int = 256
    num_angle_bins: int = 64
    range_resolution: float = 0.044
    doppler_resolution: float = 0.03
    radar_loc: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))

    @property
    def num_chirps_per_frame(self) -> int:
        return self.num_tx * self.loops_per_frame

    @property
    def num_range_bins(self) -> int:
        return self.adc_samples

    @property
    def num_doppler_bins(self) -> int:
        return self.loops_per_frame
    
    @property
    def chirp_size(self) -> int:
        return self.num_rx * self.adc_samples
    
    @property
    def chirp_loop_size(self) -> int:
        return self.chirp_size * self.num_tx
        
    @property
    def frame_size(self) -> int:
        return self.chirp_loop_size * self.loops_per_frame

    # Helper property for binary reading
    @property
    def frame_bytes(self) -> int:
        return self.frame_size * 4  # 4 bytes per complex sample (2 * int16)


class RadarBinFileReader:
    """Reads raw radar data from .bin files."""
    
    def __init__(self, file_path: str, config: RadarConfig):
        self.file_path = file_path
        self.config = config
        self.file_handle = open(self.file_path, 'rb')
        self.file_size = self._get_file_size()
        self.total_frames = self.file_size // self.config.frame_bytes

    def _get_file_size(self) -> int:
        current_pos = self.file_handle.tell()
        self.file_handle.seek(0, 2)
        size = self.file_handle.tell()
        self.file_handle.seek(current_pos)
        return size

    def __iter__(self):
        self.file_handle.seek(0)
        return self

    def __next__(self) -> np.ndarray:
        data = self.file_handle.read(self.config.frame_bytes)
        if len(data) < self.config.frame_bytes:
            raise StopIteration
        
        # Convert raw bytes to int16 array
        return np.frombuffer(data, dtype=np.int16)

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Reads a specific frame by index."""
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        
        offset = frame_idx * self.config.frame_bytes
        self.file_handle.seek(offset)
        data = self.file_handle.read(self.config.frame_bytes)
        
        if len(data) != self.config.frame_bytes:
            return None
            
        return np.frombuffer(data, dtype=np.int16)

    def close(self):
        self.file_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class BaseRadarProcessor(ABC):
    """Abstract base class for radar signal processing."""
    
    def __init__(self, config: RadarConfig):
        self.config = config

    @abstractmethod
    def process(self, raw_frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a raw frame and return results.
        
        Args:
            raw_frame: Raw int16 numpy array from file/stream.
            
        Returns:
            Dictionary containing processed artefacts (e.g. 'point_cloud', 'micro_doppler').
        """
        pass


class StandardRadarProcessor(BaseRadarProcessor):
    """
    Standard processing pipeline:
    Raw -> Reshape -> RangeFFT -> ClutterRemoval -> DopplerFFT -> CFAR -> AoA -> PointCloud
    """
    
    def __init__(self, config: RadarConfig, 
                 enable_static_clutter_removal: bool = True,
                 energy_top_128: bool = True,
                 range_cut: bool = True):
        super().__init__(config)
        self.enable_static_clutter_removal = enable_static_clutter_removal
        self.energy_top_128 = energy_top_128
        self.range_cut = range_cut

    def process(self, raw_frame: np.ndarray) -> Dict[str, Any]:
        
        # 1. Convert to Complex
        complex_frame = self._int16_to_complex(raw_frame)
        
        # 2. Reshape to (Loops, Tx, Rx, Samples) then transpose to (Tx, Rx, Loops, Samples)
        reshaped_frame = self._reshape_frame(complex_frame)
        
        # 3. Range FFT
        range_res = self._range_fft(reshaped_frame)
        
        # 4. Static Clutter Removal
        if self.enable_static_clutter_removal:
            range_res = self._clutter_removal(range_res, axis=2) # Axis 2 is Loops/Doppler
            
        # 5. Doppler FFT
        doppler_res = self._doppler_fft(range_res)
        
        # 6. Generate Point Cloud
        point_cloud = self._generate_point_cloud(doppler_res)
        
        return {
            "point_cloud": point_cloud,
            # We could add other intermediate results here if needed, like heatmaps
            # "range_doppler": ...
        }

    def _int16_to_complex(self, bin_frame: np.ndarray) -> np.ndarray:
        # Assuming interleaved I/Q
        # [I1, I2, Q1, Q2, ...] is NOT the pattern in pc_generation.py
        # pc_generation.py: 
        #   np_frame[0::2] = bin_frame[0::4]+1j*bin_frame[2::4] (I1 + jQ1)
        #   np_frame[1::2] = bin_frame[1::4]+1j*bin_frame[3::4] (I2 + jQ2)
        # This means 4 samples (I1, I2, Q1, Q2) -> 2 complex numbers.
        
        # NOTE: Using exactly the logic from pc_generation.py for compatibility
        np_frame = np.zeros(shape=(len(bin_frame)//2), dtype=complex)
        np_frame[0::2] = bin_frame[0::4] + 1j * bin_frame[2::4]
        np_frame[1::2] = bin_frame[1::4] + 1j * bin_frame[3::4]
        return np_frame

    def _reshape_frame(self, frame: np.ndarray) -> np.ndarray:
        # (Loops, Tx, Rx, Samples)
        frame_with_chirp = np.reshape(frame, (self.config.loops_per_frame, 
                                              self.config.num_tx, 
                                              self.config.num_rx, 
                                              -1))
        # Transpose to (Tx, Rx, Loops, Samples)
        return frame_with_chirp.transpose(1, 2, 0, 3)

    def _range_fft(self, reshaped_frame: np.ndarray) -> np.ndarray:
        windowed_bins_1d = reshaped_frame * np.hamming(self.config.adc_samples)
        return np.fft.fft(windowed_bins_1d)

    def _clutter_removal(self, input_val: np.ndarray, axis: int = 0) -> np.ndarray:
        # Matches pc_generation.py logic
        reordering = np.arange(len(input_val.shape))
        reordering[0] = axis
        reordering[axis] = 0
        input_val = input_val.transpose(reordering)
        mean = input_val.mean(0)
        output_val = input_val - mean
        return output_val.transpose(reordering)

    def _doppler_fft(self, range_result: np.ndarray) -> np.ndarray:
        windowed_bins_2d = range_result * np.reshape(np.hamming(self.config.loops_per_frame), (1, 1, -1, 1))
        doppler_fft_res = np.fft.fft(windowed_bins_2d, axis=2)
        return np.fft.fftshift(doppler_fft_res, axes=2)

    def _generate_point_cloud(self, doppler_result: np.ndarray) -> np.ndarray:
        # Sum energy across all antennas
        doppler_sum = np.sum(doppler_result, axis=(0, 1))
        doppler_db = np.log10(np.absolute(doppler_sum)) # Note: pc_generation uses absolute, not abs**2 

        if self.range_cut:
            doppler_db[:, :25] = -100
            doppler_db[:, 125:] = -100

        cfar_result = np.zeros(doppler_db.shape, bool)
        
        if self.energy_top_128:
            top_size = 128
            flat_idx = 128 * 256 - top_size - 1
            # Safety for different sizes
            total_elements = doppler_db.size
            if flat_idx < total_elements:
                energy_thresh = np.partition(doppler_db.ravel(), flat_idx)[flat_idx]
                cfar_result[doppler_db > energy_thresh] = True
            
        det_peaks_indices = np.argwhere(cfar_result == True)
        if len(det_peaks_indices) == 0:
            return np.array([]).reshape(6, 0) # Return empty compatible shape

        r_idx = det_peaks_indices[:, 1].astype(np.float64)
        v_idx = (det_peaks_indices[:, 0] - self.config.num_doppler_bins // 2).astype(np.float64)

        r_val = r_idx * self.config.range_resolution
        v_val = v_idx * self.config.doppler_resolution
        
        energy = doppler_db[cfar_result == True]
        
        # AoA
        aoa_input = doppler_result[:, :, cfar_result == True]
        # Reshape to (VirtualAntennas, NumPeaks) -> (12, N)
        aoa_input = aoa_input.reshape(self.config.num_tx * self.config.num_rx, -1)
        
        if aoa_input.shape[1] == 0:
             return np.array([]).reshape(6, 0)
             
        x_vec, y_vec, z_vec = self._naive_xyz(aoa_input)
        
        x = x_vec * r_val
        y = y_vec * r_val
        z = z_vec * r_val
        
        # Stack: x, y, z, v, energy, r
        # Note: pc_generation.py concatenates then reshapes.
        # It results in shape (6, N)
        
        # We want to match that output format exactly for verification
        pc = np.concatenate((x, y, z, v_val, energy, r_val))
        pc = np.reshape(pc, (6, -1))
        
        # Filter where y != 0
        pc = pc[:, y_vec != 0]
        
        return pc

    def _naive_xyz(self, virtual_ant: np.ndarray, fft_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_tx = self.config.num_tx
        num_rx = self.config.num_rx
        
        num_detected_obj = virtual_ant.shape[1]
        
        # Azimuth: First 2*Rx antennas (8)
        azimuth_ant = virtual_ant[:2 * num_rx, :]
        azimuth_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=complex)
        azimuth_ant_padded[:2 * num_rx, :] = azimuth_ant
        
        azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
        k_max = np.argmax(np.abs(azimuth_fft), axis=0)
        
        peak_1 = np.take_along_axis(azimuth_fft, k_max[None, :], axis=0).squeeze(0)
        
        k_max[k_max > (fft_size // 2) - 1] -= fft_size
        wx = 2 * np.pi / fft_size * k_max
        x_vector = wx / np.pi
        
        # Elevation: Remaining antennas (last Rx antennas, usually 4)
        elevation_ant = virtual_ant[2 * num_rx:, :]
        elevation_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=complex)
        elevation_ant_padded[:num_rx, :] = elevation_ant
        
        elevation_fft = np.fft.fft(elevation_ant_padded, axis=0)
        elevation_max = np.argmax(np.log2(np.abs(elevation_fft) + 1e-9), axis=0) # Added epsilon for safety
        
        peak_2 = np.take_along_axis(elevation_fft, elevation_max[None, :], axis=0).squeeze(0)
        
        wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * 2 * wx))
        z_vector = wz / np.pi
        
        y_possible = 1 - x_vector ** 2 - z_vector ** 2
        y_vector = y_possible.copy()
        
        # Filter invalid
        mask = y_possible < 0
        x_vector[mask] = 0
        z_vector[mask] = 0
        y_vector[mask] = 0
        y_vector = np.sqrt(y_vector)
        
        return x_vector, y_vector, z_vector
