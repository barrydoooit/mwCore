import os
import numpy as np
from typing import Dict, Tuple, List, Union, Optional
import logging
import pickle
import csv
import random

from mwcore.registry import READERS

log = logging.getLogger(__name__)

@READERS.register_module()
class OfflineReader:
    """Reader for offline point cloud data using the OfflineManager class"""
    
    def __init__(self, data: str = None, source_file: str = None, source_dir: str = None, **kwargs):
        """
        Initialize the offline reader
        
        Parameters
        ----------
        data : str
            Path to the directory with data
        source_file : str
            Path to a specific file to read (alternative to data)
        source_dir : str
            Path to a directory with data files (alternative to data)
        """
        
        # Use the first available path
        self.path = data or source_file or source_dir
        if not self.path:
            raise ValueError("One of data, source_file, or source_dir must be provided")

        log.info(f"Creating offline reader for path: {self.path}")
        

        self._config = None
        self.current_frame = 0
        self.frame_data = None
        
    def register_config(self, config):
        """Store config data if needed"""
        self._config = config

    @property
    def config(self):
        return self._config
        
    def read(self) -> Tuple[int, int, Dict, Optional[np.ndarray]]:
        raise NotImplementedError("The read method must be implemented in subclasses")
    
    def is_finished(self) -> bool:
        """Check if all frames have been read"""
        return self.current_frame == -1
        
    def close(self):
        """Clean up resources"""
        log.info("Closing offline reader")
        # No need to close serial ports in offline mode
        pass



