from mwcore.registry import ADCPROCESSORS
from ..frame import RadarFrame
from .base import BaseSignalProcess
import numpy as np



@ADCPROCESSORS.register_module()
class RangeFFT(BaseSignalProcess):
    """Range FFT with Hamming window."""
    requires = {'radar_cube'}
    provides = {'range_fft'}

    def _process_generic(self, frame: RadarFrame) -> None:
        cube = self.read(frame, 'radar_cube')
        window = np.hamming(frame.config.adc_samples)
        r_fft = np.fft.fft(cube * window, axis=3)
        self.write(frame, 'range_fft', r_fft)

@ADCPROCESSORS.register_module()
class DopplerFFT(BaseSignalProcess):
    """Doppler FFT with Clutter Removal."""
    requires = {'range_fft'}
    provides = {'doppler_fft'}

    def __init__(self, clutter_removal: bool = True, name: str = ""):
        super().__init__(name)
        self.clutter_removal = clutter_removal

    def _process_generic(self, frame: RadarFrame) -> None:
        r_fft = self.read(frame, 'range_fft')
        if self.clutter_removal:
            r_fft = r_fft - r_fft.mean(axis=2, keepdims=True)
            
        window = np.hamming(frame.config.loops_per_frame).reshape(1, 1, -1, 1)
        fft_out = np.fft.fft(r_fft * window, axis=2)
        d_fft = np.fft.fftshift(fft_out, axes=2)

        self.write(frame, 'doppler_fft', d_fft)