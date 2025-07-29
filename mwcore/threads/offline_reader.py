import numpy as np
from mwcore.registry import THREADS, READERS
from PySide6.QtCore import QThread, Signal
import logging
import time
from typing import TYPE_CHECKING, Dict, Optional, TypeVar, Union

from mwcore.radario.readers.offline.base import OfflineReader
    
log = logging.getLogger(__name__)
T = TypeVar('T', bound=OfflineReader)



@THREADS.register_module()
class OfflineReaderThread(QThread):
    raw_data = Signal(dict)  # Will emit the det_obj dictionary
    array_data = Signal(np.ndarray)  # Will emit the processed numpy array
    ground_truth_data = Signal(object)  # Will emit ground truth data if available

    def __init__(self, 
                 reader: Union[dict, T], 
                 playback_speed: Optional[float] = None):
        super().__init__()
        if isinstance(reader, dict):
            reader = READERS.build(reader)
        self.reader = reader
        self.playback_speed = playback_speed

    @property
    def sleep_time(self):
        if self.playback_speed is None:
            return None
        if not hasattr(self, '_sleep_time'):
            frame_time_ms = 1000 / self.reader.frame_rate if hasattr(self.reader, 'frame_rate') else 50
            self._sleep_time = frame_time_ms / 1000 / (self.playback_speed or 1)
        return self._sleep_time
    
    def run(self):
        log.info("Starting OfflineReaderThread")
        
        while not self.isInterruptionRequested():
            start_time = time.time()
            data_ok, frame_number, det_obj, ground_truth = self.reader.read()
            
            if not data_ok or self.reader.is_finished():
                log.debug("Data not found or finished reading, continuing to next frame")
                if self.reader.is_finished():
                    break
                continue

            self.raw_data.emit(det_obj)
            self.array_data.emit(self._raw_to_numpy(det_obj))

            #Emit ground truth if available
            if ground_truth is not None and len(ground_truth) > 0:
                # We need to send an array of arrays, if we only receive a single ground truth ndarray, we wrap it in a list
                if isinstance(ground_truth, np.ndarray):
                    ground_truth = [ground_truth]
                self.ground_truth_data.emit(ground_truth)
            
            if self.sleep_time is not None:
                elapsed = time.time() - start_time
                remaining_sleep = max(0, self.sleep_time - elapsed)
                time.sleep(remaining_sleep)

        self.reader.close()
    
    def _raw_to_numpy(self, det_obj: Dict[str, np.ndarray]):
        data = np.stack([det_obj[key] for key in ['x', 'y', 'z', 'doppler', 'peakVal']], axis=-1)
        return data
