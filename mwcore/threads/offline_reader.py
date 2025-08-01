import numpy as np
from tqdm import tqdm
from mwcore.registry import THREADS, READERS
from PySide6.QtCore import QThread, Signal
import logging
import time
from typing import TYPE_CHECKING, Dict, Optional, TypeVar, Union

from mwcore.radario.readers.offlineReaders.base import OfflineReader
from mwcore.radario.readers.offlineReaders.framedata import FrameData
    
log = logging.getLogger(__name__)



T = TypeVar('T', bound=OfflineReader)
@THREADS.register_module()
class OfflineReaderThread(QThread):
    raw_data = Signal(dict)  # Will emit the det_obj dictionary
    array_data = Signal(np.ndarray)  # Will emit the processed numpy array
    ground_truth_data = Signal(object)  # Will emit ground truth data if available
    signal_framedata = Signal(FrameData)
    signal_finished = Signal(int) # Will emit the final frame idx

    def __init__(self, 
                 reader: Union[dict, T], 
                 playback_speed: Optional[float] = None):
        super().__init__()
        if isinstance(reader, dict):
            reader = READERS.build(reader)
        self.reader = reader
        self.playback_speed = playback_speed
        self._last_frame_number: int = -1

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
        total = self.reader.total_num_frames
        pbar = tqdm(total=total, unit='frames', desc="Reading Frames")

        while not self.isInterruptionRequested():
            start_time = time.time()
            res: tuple = self.reader.read()
            if len(res) == 4:
                data_ok, frame_number, det_obj, ground_truth = res
                framedata = None
            elif len(res) == 5:
                data_ok, frame_number, det_obj, ground_truth, framedata = res
            
            if not data_ok or self.reader.is_finished():
                log.debug("Data not found or finished reading, continuing to next frame")
                if self.reader.is_finished():
                    self.signal_finished.emit(self._last_frame_number)
                    break
                continue
            self._last_frame_number = frame_number
            self.raw_data.emit(det_obj)
            self.array_data.emit(self._raw_to_numpy(det_obj))
            if framedata is not None:
                self.signal_framedata.emit(framedata)

            #Emit ground truth if availableW
            if ground_truth is not None and len(ground_truth) > 0:
                self.ground_truth_data.emit(ground_truth)
            
            if self.sleep_time is not None:
                elapsed = time.time() - start_time
                remaining_sleep = max(0, self.sleep_time - elapsed)
                time.sleep(remaining_sleep)
            pbar.update(1)
        self.reader.close()
    
    def _raw_to_numpy(self, det_obj: Dict[str, np.ndarray]):
        data = np.stack([det_obj[key] for key in ['x', 'y', 'z', 'doppler', 'peakVal']], axis=-1)
        return data
