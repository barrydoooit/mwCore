import numpy as np
from mwcore.radario.readers.base import SerialReader
from mwcore.registry import THREADS, READERS
from PySide6.QtCore import QThread, Signal
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mwcore.radario.readers.offline.base import OfflineReader
    

log = logging.getLogger(__name__)

@THREADS.register_module()
class OfflineReaderThread(QThread):
    raw_data = Signal(object)  # Will emit the det_obj dictionary
    array_data = Signal(np.ndarray)  # Will emit the processed numpy array
    ground_truth_data = Signal(object)  # Will emit ground truth data if available

    def __init__(self, reader='OfflineReader', playback_speed=1.0):
        super().__init__()
        if isinstance(reader, dict):
            reader = READERS.build(reader)
        self.reader = reader
        self.playback_speed = playback_speed  # Multiplier for playback speed
        log.info("Initialized OfflineReaderThread")

    def run(self):
        log.info("Starting OfflineReaderThread")
        frame_time_ms = 1000 / self.reader.frame_rate if hasattr(self.reader, 'frame_rate') else 50
        sleep_time = frame_time_ms / 1000 / self.playback_speed  
        
        while not self.isInterruptionRequested():
            start_time = time.time()
            
            # Use the read method from OfflineReader
            data_ok, frame_number, det_obj, ground_truth = self.reader.read()
            
            if not data_ok or self.reader.is_finished():
                log.info("Data not found or finished reading, continuing to next frame")
                if self.reader.is_finished():
                    break  # Exit the loop if we're done reading
                continue
                
            # Emit signals for raw data
            self.raw_data.emit(det_obj)
            
            # Convert to array format expected by visualizer
            array_data = self._raw_to_numpy(det_obj)
            self.array_data.emit(array_data)

            #Emit ground truth if available
            if len(ground_truth) > 0:
                # We need to send an array of arrays, if we only receive a single ground truth ndarray, we wrap it in a list
                if isinstance(ground_truth, np.ndarray):
                    ground_truth = [ground_truth]
                self.ground_truth_data.emit(ground_truth)
            
            # Control playback speed
            elapsed = time.time() - start_time
            remaining_sleep = max(0, sleep_time - elapsed)
            time.sleep(remaining_sleep)
            
    def set_playback_speed(self, speed):
        """Set the playback speed multiplier"""
        self.playback_speed = max(0.1, speed)  # Ensure minimum speed
        log.info(f"Playback speed set to {self.playback_speed}x")
    
    def _raw_to_numpy(self, det_obj):
        """Convert detection objects to numpy array format"""
        # Check if we have valid data
        if det_obj.get("numObj", 0) > 0:
            # Stack the features in the same way as OnlineReaderThread
            data = np.stack([
                det_obj["x"], 
                det_obj["y"], 
                det_obj["z"], 
                det_obj["doppler"], 
                det_obj["peakVal"]
            ], axis=-1)
            
            log.info(f"Offline frame - Num Points: {data.shape[0]}")
            return data
        else:
            # Return empty array if no points
            return np.empty((0, 5))
            
    def terminate(self):
        """Clean up when thread is terminated"""
        self.reader.close()
        return super().terminate()
