import os
import re
import struct
import numpy as np
from typing import List, Optional, Union, Generator, Tuple, TYPE_CHECKING
from pathlib import Path

from mwcore.radario.readers.base import BaseReader
from mwcore.signal_processing.dsp_ppl import DspPipeline
from mwcore.signal_processing.frame import RadarFrame, RadarConfig
from mwcore.registry import READERS



@READERS.register_module()
class OfflineAdcDataReader(BaseReader):
    """
    Reads raw binary radar data from a set of files and optionally processes them.
    
    Features:
    - Regex file filtering.
    - Automatic parameter extraction from DspPipeline (if provided).
    - Sequential frame access.
    - Yields either raw bytes or processed RadarFrames.
    """
    def __init__(self, 
                 data_dir: str, 
                 file_pattern: str = r".*\.bin", 
                 frame_rate: Optional[float] = None,
                 pipeline: Optional[Union[DspPipeline, dict]] = None,
                 # Fallback params if pipeline is not provided
                 num_chirps: int = 128,
                 num_rx: int = 4,
                 num_tx: int = 3,
                 num_samples: int = 256,
                 is_complex: bool = True, # IQ
                 bytes_per_sample: int = 2):
        
        self.data_dir = Path(data_dir)
        self.frame_rate = frame_rate
        self.file_pattern = re.compile(file_pattern)
        if isinstance(pipeline, dict):
            self.pipeline = DspPipeline.from_cfg(pipeline)
        else:
            self.pipeline = pipeline
        
        if self.pipeline is not None:
            cfg = self.pipeline.config
            self.num_chirps = cfg.loops_per_frame
            self.num_rx = cfg.num_rx
            self.num_tx = cfg.num_tx
            self.num_samples = cfg.adc_samples
        else:
            self.num_chirps = num_chirps
            self.num_rx = num_rx
            self.num_tx = num_tx
            self.num_samples = num_samples
            
        self.is_complex = is_complex
        self.bytes_per_sample = bytes_per_sample
        
        iq_factor = 2 if self.is_complex else 1
        self.frame_size_bytes = (self.num_chirps * self.num_tx * self.num_rx * self.num_samples * iq_factor * self.bytes_per_sample)
        
        # 2. Discovery Files
        self.file_list = sorted([
            f for f in self.data_dir.iterdir() 
            if f.is_file() and self.file_pattern.match(f.name)
        ])
        
        if not self.file_list:
            print(f"Warning: No files found in {data_dir} matching {file_pattern}")
            
        self.current_file_idx = 0
        self.current_frame_idx = 0 # Global frame count
        self.frames_in_current_file = 0
        self._file_handle = None
        
        self._open_next_file()

    def _open_next_file(self) -> bool:
        """Closes current and opens next file. Returns False if no more files."""
        if self._file_handle:
            self._file_handle.close()
            self.current_file_idx += 1
            
        if self.current_file_idx >= len(self.file_list):
            return False
            
        current_path = self.file_list[self.current_file_idx]
        print(f"Opening file: {current_path.name}")
        self._file_handle = open(current_path, "rb")
        self.frames_in_current_file = 0
        return True

    def __iter__(self):
        return self

    def __next__(self) -> Union[RadarFrame, bytes]:
        """Iterator protocol support."""
        result = self.read()
        if result is None:
            raise StopIteration
        return result

    def read(self) -> Optional[Union[RadarFrame, bytes]]:
        """
        Reads the next frame.
        
        Returns:
            RadarFrame: If pipeline is set.
            bytes: If pipeline is None.
            None: If end of data.
        """
        if not self._file_handle:
            return None
            
        raw_data = self._file_handle.read(self.frame_size_bytes)
        
        # Handle End of File
        if len(raw_data) < self.frame_size_bytes:
            if self._open_next_file():
                return self.read()
            else:
                # End of all data
                return None
        
        self.frames_in_current_file += 1
        self.current_frame_idx += 1
        
        if self.pipeline:
            return self.pipeline.run(raw_data)
        else:
            return raw_data

    def close(self):
        if self._file_handle:
            self._file_handle.close()

    def get_progress(self) -> dict:
        """Returns current reading statistics."""
        return {
            "current_file": self.file_list[self.current_file_idx].name if self.current_file_idx < len(self.file_list) else "Done",
            "file_index": self.current_file_idx,
            "total_files": len(self.file_list),
            "frame_global_index": self.current_frame_idx,
            "frame_in_file": self.frames_in_current_file
        }