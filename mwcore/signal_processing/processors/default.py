

from mwcore.registry import ADCPROCESSORS
from ..frame import RadarFrame
from .base import BaseSignalProcess
import numpy as np


@ADCPROCESSORS.register_module()
class FrameReshaper(BaseSignalProcess):
    """
    Parses raw bytes into the (Tx, Rx, Loops, Samples) Radar Cube.
    """
    def __init__(self, name: str = "FrameReshaper"):
        super().__init__(name)

    def execute(self, frame: RadarFrame):
        # 1. Bytes to Int16
        raw_int16 = np.frombuffer(frame.raw_bytes, dtype=np.int16)
        
        # 2. Int16 to Complex (Interleaved: Real=0,4.. Imag=2,6..)
        # This matches both mmMesh and OpenRadar (adc.py) logic
        frame.raw_complex = np.zeros(len(raw_int16)//2, dtype=np.complex_)
        frame.raw_complex[0::2] = raw_int16[0::4] + 1j * raw_int16[2::4]
        frame.raw_complex[1::2] = raw_int16[1::4] + 1j * raw_int16[3::4]
        
        # 3. Reshape to (Loops, Tx, Rx, Samples)
        cfg = frame.config
        try:
            # mmMesh standard shape handling
            reshaped = np.reshape(frame.raw_complex, (cfg.loops_per_frame, cfg.num_tx, cfg.num_rx, cfg.adc_samples))
            # Transpose to (Tx, Rx, Loops, Samples) for antenna-first processing
            frame.radar_cube = reshaped.transpose(1, 2, 0, 3)
        except ValueError as e:
            print(f"Reshape Error: Expected {cfg.loops_per_frame*cfg.num_tx*cfg.num_rx*cfg.adc_samples} items, got {frame.raw_complex.size}")
            raise e

@ADCPROCESSORS.register_module()
class StaticClutterRemoval(BaseSignalProcess):
    """Removes static objects by subtracting the mean across the Doppler axis."""
    def __init__(self, active: bool = True, name: str = "StaticClutterRemoval"):
        super().__init__(name)
        self.active = active

    def execute(self, frame: RadarFrame):
        if not self.active or frame.range_fft is None:
            return

        # axis 2 is loops/doppler
        mean_val = frame.range_fft.mean(axis=2, keepdims=True)
        frame.range_fft = frame.range_fft - mean_val