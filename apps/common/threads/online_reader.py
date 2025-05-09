import numpy as np
from apps.common.threads import THREADS
from PySide2.QtCore import QThread, Signal

from radario.base import READERS

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from radario.base import BaseBufferedReader



@THREADS.register_module()
class OnlineReaderThread(QThread):
    raw_data = Signal(np.ndarray)
    array_data = Signal(np.ndarray)
    
    def __init__(self, reader: 'BaseBufferedReader'):
        super().__init__()
        self.reader = reader

    def run(self):
        while not self.isInterruptionRequested():
            data_ok, frame_number, det_obj = self.reader.read()
            if data_ok:
                self.raw_data.emit(det_obj)
                self.array_data.emit(self._raw_to_numpy(det_obj))
                
    def _raw_to_numpy(self, det_obj: Dict[str, np.ndarray]):
        data = np.stack([det_obj[key] for key in ['x', 'y', 'z', 'doppler', 'peakVal']], axis=-1)
        print("Num Points Detected: ", data.shape[0])
        return data
        
    def terminate(self):
        self.reader.Data_port.close()
        return super().terminate()