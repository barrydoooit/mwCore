from ..frame import RadarFrame
from .base import BaseSignalProcess
import numpy as np



class RangeFFT(BaseSignalProcess):
    """Range FFT with Hamming window."""
    def __init__(self, name: str = "RangeFFT"):
        super().__init__(name)

    def execute(self, frame: RadarFrame):
        # Window along last axis (Samples)
        window = np.hamming(frame.config.adc_samples)
        frame.range_fft = np.fft.fft(frame.radar_cube * window, axis=3)

class DopplerFFT(BaseSignalProcess):
    """Doppler FFT with Clutter Removal."""
    def __init__(self, clutter_removal: bool = True, name: str = "DopplerFFT"):
        super().__init__(name)
        self.clutter_removal = clutter_removal

    def execute(self, frame: RadarFrame):
        # 1. Clutter Removal (Static filtering)
        # Subtract mean across loops (axis 2)
        if self.clutter_removal:
            frame.range_fft = frame.range_fft - frame.range_fft.mean(axis=2, keepdims=True)
            
        # 2. Windowing
        window = np.hamming(frame.config.loops_per_frame).reshape(1, 1, -1, 1)
        
        # 3. FFT and Shift
        fft_out = np.fft.fft(frame.range_fft * window, axis=2)
        frame.doppler_fft = np.fft.fftshift(fft_out, axes=2)