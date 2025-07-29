from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List, Union, Optional
import logging

from mwcore.registry import READERS
from mwcore.radario.readers.base import BaseReader

log = logging.getLogger(__name__)

@READERS.register_module()
class OfflineReader(BaseReader):    
    def __init__(self):
        self._current_frame = 0
        self._finished = False
        self.frame_data = None
        
    def read(self) -> Tuple[int, int, Dict, Optional[np.ndarray]]:
        raise NotImplementedError("The read method must be implemented in subclasses")
    
    def is_finished(self) -> bool:
        return self._finished
    
    def frame_inc(self):
        self._current_frame += 1
    
    def close(self):
        log.info("Closing offline reader")
        pass



