import numpy as np
from mwcore.radario.readers.base import SerialReader
from mwcore.registry import READERS, THREADS

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:
    print("Failed to import QT, mocking...")
    class QThread:
        def __init__(*args, **kwargs):
            pass
    class Signal:
        def __init__(*args, **kwargs):
            pass

from typing import TYPE_CHECKING, Dict, Generic, TypeVar, Union

if TYPE_CHECKING:
    from ..radario.readers.TI.base import BaseTIBufferedReader



T = TypeVar('T', bound=SerialReader)
@THREADS.register_module()
class OnlineReaderThread(QThread, Generic[T]):
    raw_data = Signal(dict)
    array_data = Signal(np.ndarray)
    
    def __init__(self, reader: Union[T, dict], sensor_started: bool = False):
        super().__init__()
        if isinstance(reader, dict):
            reader = READERS.build(reader)
        self._sensor_started = sensor_started
        self.reader: T = reader

    @property
    def sensor_started(self) -> bool: return self._sensor_started

    def run(self):
        if not self.sensor_started:
            self.reader.connect()
            self._sensor_started = True

        while not self.isInterruptionRequested():
            data_ok, frame_number, det_obj = self.reader.read()
            if data_ok:
                self.raw_data.emit(det_obj)
                self.array_data.emit(self._raw_to_numpy(det_obj))
                
    def _raw_to_numpy(self, det_obj: Dict[str, np.ndarray]):
        data = np.stack([det_obj[key] for key in ['x', 'y', 'z', 'doppler', 'peakVal']], axis=-1)
        return data
        
    def terminate(self):
        self.reader.Data_port.close()
        return super().terminate()
