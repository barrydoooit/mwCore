import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.ndimage import convolve1d

from mwcore.registry import ADCPROCESSORS
from signal_processing.frame import RadarFrame
from signal_processing.processors.base import BaseSignalProcess



@ADCPROCESSORS.register_module()
class CFAR_CA(BaseSignalProcess):
    """
    Cell-Averaging CFAR (CA-CFAR) implementation derived from OpenRadar `cfar.py`.
    Uses convolution for efficient sliding window calculation.
    """
    def __init__(self, name = "CFAR_CA"):
        super().__init__(name)
        
    def execute(self, frame: RadarFrame):
        # 1. Collapse to Energy Map (Non-Coherent Integration)
        # Sum magnitude squared across antennas
        energy = np.sum(np.abs(frame.doppler_fft)**2, axis=(0, 1)) # (Loops, Samples)
        
        # 2. Setup CFAR Kernel
        guard = frame.config.cfar_guard_len
        noise = frame.config.cfar_noise_len
        
        # Create kernel: [1/N ... 1/N, 0...0 (guard), 0 (CUT), 0 (guard), 1/N ... 1/N]
        kernel_1d = np.ones(1 + (2*guard) + (2*noise)) / (2 * noise)
        kernel_1d[noise : noise + 2*guard + 1] = 0
        
        # 3. Perform CFAR along Range Dimension (Fast axis)
        # We apply 1D CFAR on each Doppler bin
        threshold_map = convolve1d(energy, kernel_1d, axis=1, mode='nearest')
        
        # 4. Apply Threshold
        # Convert scale to linear factor if needed, or additive if log. 
        # Here we use linear data, so multiplicative scale.
        det_mask = energy > (threshold_map * frame.config.cfar_threshold_scale)
        
        # 5. Extract Peaks
        det_indices = np.argwhere(det_mask)
        
        if len(det_indices) == 0:
            frame.detected_points = np.zeros((0, 3))
            return
            
        # 6. (Optional) Peak Grouping / Filtering
        # For simplicity, we just take the detected points and their values
        peaks_vals = energy[det_mask]
        
        # Format: [RangeIdx, DopplerIdx, Value]
        # det_indices is (Doppler, Range), so flip col 0 and 1
        frame.detected_points = np.column_stack((det_indices[:, 1], det_indices[:, 0], peaks_vals))