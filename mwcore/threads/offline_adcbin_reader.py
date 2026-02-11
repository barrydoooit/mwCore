import time
import logging
from typing import Tuple, Optional, Any
from PySide6.QtCore import QObject, QThread, Signal, Slot

from mwcore.registry import THREADS, READERS
# Adjust import path based on your project structure
from mwcore.radario.readers.offlineReaders.adcbin_reader import OfflineAdcDataReader

log = logging.getLogger(__name__)

class OfflineAdcDataReaderWorker(QObject):
    finished = Signal()
    frame_signal = Signal(object)
    progress_signal = Signal(dict) 

    def __init__(self, reader, playback_speed=1.0):
        super().__init__()
        self.reader = reader
        self.playback_speed = playback_speed
        self._sleep_time = 0.0
        self._fast_forward = False  # NEW: State flag

        if self.playback_speed > 0 and hasattr(self.reader, 'frame_rate') and self.reader.frame_rate:
             self._sleep_time = (1.0 / self.reader.frame_rate) / self.playback_speed
        self._last_file_idx = -1

    @Slot(bool)
    def set_fast_forward(self, enabled: bool):
        self._fast_forward = enabled

    def process(self):
        log.info("OfflineAdcReaderWorker: Started.")
        
        while not QThread.currentThread().isInterruptionRequested():
            start_time = time.time()
            
            try:
                if hasattr(self.reader, 'current_file_idx'):
                    if self.reader.current_file_idx != self._last_file_idx:
                        self._last_file_idx = self.reader.current_file_idx
                        # Use the existing get_progress method
                        if hasattr(self.reader, 'get_progress'):
                            self.progress_signal.emit(self.reader.get_progress())

                if hasattr(self.reader, 'read'):
                    data = self.reader.read()
                else:
                    data = next(self.reader)
                
                # Handle End of Data
                if data is None:
                    log.info("OfflineAdcReaderWorker: End of data stream (None).")
                    break
                    
                self.frame_signal.emit(data)
                
            except StopIteration:
                log.info("OfflineAdcReaderWorker: StopIteration reached.")
                break
            except Exception as e:
                log.error(f"Error: {e}")
                break
            
            if self._sleep_time > 0 and not self._fast_forward:
                elapsed = time.time() - start_time
                sleep_needed = max(0, self._sleep_time - elapsed)
                if sleep_needed > 0:
                    time.sleep(sleep_needed)
                    
        self.finished.emit()

    @staticmethod
    @THREADS.register_module(name="OfflineAdcDataReaderWorker")
    def build_with_thread(reader: dict, 
                          playback_speed: float = 1.0, 
                          **kwargs) -> Tuple['OfflineAdcDataReaderWorker', QThread]:
        """
        Factory method to build the Worker and Thread pair.
        This is registered in THREADS, so config['type'] = 'OfflineAdcDataReaderWorker' calls this.
        """
        
        # 1. Build the Reader
        reader_instance = READERS.build(reader)
        if not isinstance(reader_instance, OfflineAdcDataReader):
             # Optional: strict check, or allow duck typing
             log.warning(f"Worker expects OfflineAdcDataReader, got {type(reader_instance)}")

        # 2. Instantiate Worker
        worker = OfflineAdcDataReaderWorker(reader_instance, playback_speed)
        
        # 3. Create Thread
        thread = QThread()
        worker.moveToThread(thread)
        
        # 4. Connect Worker Lifecycle to Thread
        thread.started.connect(worker.process)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        
        return worker, thread