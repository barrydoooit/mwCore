
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
    
    def __init__(self, file_path: str, radar_config: Optional[RadarConfig] = None, has_timestamp: bool = True):
        self.file_path = file_path
        self.config = radar_config if radar_config else RadarConfig()
        self.has_timestamp = has_timestamp
        
        # Initialize internal processing modules
        self.reader = RadarBinFileReader(self.file_path, self.config, has_timestamp=self.has_timestamp)
        self.processor = StandardRadarProcessor(self.config)
        
        self.current_frame_idx = 0
        
    def read(self) -> Any:
        # Read raw frame
        read_result = self.reader.read_frame(self.current_frame_idx)
        
        if read_result is None:
            # End of file or error
            return None
            
        if self.has_timestamp:
            timestamp, raw_frame = read_result
        else:
            timestamp = 0.0
            raw_frame = read_result
            
        # DSP Processing
        result = self.processor.process(raw_frame)
        
        # Increment frame
        self.current_frame_idx += 1
        
        # Format output for mwCore consumption
        # mwCore trackers usually expect a dictionary or numpy array.
        # Returning the point cloud directly for now.
        if self.has_timestamp:
            return timestamp, result['point_cloud']
        return result['point_cloud']

    def close(self):
        self.reader.close()
