
from typing import Any, Dict, Optional
import numpy as np

from mwcore.registry import READERS
from mwcore.radario.readers.base import BaseReader
from mwcore.signal_processing.radar_processor import RadarBinFileReader, RadarConfig, StandardRadarProcessor

@READERS.register_module()
class RawBinReader(BaseReader):
    """
    Reader that processes raw .bin radar files using software DSP.
    Acts as a virtual sensor for mwCore.
    """
    
    def __init__(self, file_path: str, radar_config: Optional[RadarConfig] = None):
        self.file_path = file_path
        self.config = radar_config if radar_config else RadarConfig()
        
        # Initialize internal processing modules
        self.reader = RadarBinFileReader(self.file_path, self.config)
        self.processor = StandardRadarProcessor(self.config)
        
        self.current_frame_idx = 0
        
    def read(self) -> Any:
        # Read raw frame
        raw_frame = self.reader.read_frame(self.current_frame_idx)
        
        if raw_frame is None:
            # End of file or error
            return None
            
        # DSP Processing
        result = self.processor.process(raw_frame)
        
        # Increment frame
        self.current_frame_idx += 1
        
        # Format output for mwCore consumption
        # mwCore trackers usually expect a dictionary or numpy array.
        # Returning the point cloud directly for now.
        return result['point_cloud']

    def close(self):
        self.reader.close()
