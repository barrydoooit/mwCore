

from mwcore.registry import ADCPROCESSORS
from ..frame import RadarFrame
from .base import BaseSignalProcess
import numpy as np


@ADCPROCESSORS.register_module()
class FrameReshaper(BaseSignalProcess):
    """Parses raw bytes into the (Tx, Rx, Loops, Samples) Radar Cube."""
    requires = {'raw_bytes'}
    provides = {'raw_complex', 'radar_cube'}

    def _process_generic(self, frame: RadarFrame) -> None:
        raw_bytes = self.read(frame, 'raw_bytes')
        # ADC raw stream is defined as little-endian int16.
        raw_int16 = np.frombuffer(raw_bytes, dtype="<i2")
        
        raw_complex = np.zeros(len(raw_int16)//2, dtype=np.complex128)
        raw_complex[0::2] = raw_int16[0::4] + 1j * raw_int16[2::4]
        raw_complex[1::2] = raw_int16[1::4] + 1j * raw_int16[3::4]
        
        cfg = frame.config
        try:
            reshaped = np.reshape(raw_complex, (cfg.loops_per_frame, cfg.num_tx, cfg.num_rx, cfg.adc_samples))
            radar_cube = reshaped.transpose(1, 2, 0, 3)
        except ValueError as e:
            print(f"Reshape Error: Expected {cfg.loops_per_frame*cfg.num_tx*cfg.num_rx*cfg.adc_samples} items, got {raw_complex.size}")
            raise e

        self.write(frame, 'raw_complex', raw_complex)
        self.write(frame, 'radar_cube', radar_cube)


@ADCPROCESSORS.register_module()
class StaticClutterRemoval(BaseSignalProcess):
    """Removes static objects by subtracting the mean across the Doppler axis."""
    requires = {'range_fft'}
    provides = {'range_fft'}

    def __init__(self, active: bool = True, name: str = ""):
        super().__init__(name)
        self.active = active

    def _process_generic(self, frame: RadarFrame) -> None:
        if not self.active:
            return
        
        r_fft = self.read(frame, 'range_fft')
        r_fft_clean = r_fft - r_fft.mean(axis=2, keepdims=True)
        self.write(frame, 'range_fft', r_fft_clean)
