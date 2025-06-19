import numpy as np
from mwcore.radario.readers.base import SerialReader
from mwcore.registry import READERS, THREADS, TRACKERS
from PySide6.QtCore import QThread, Signal

from typing import TYPE_CHECKING, Dict, Generic, TypeVar, Union

from mwcore.threads.online_reader import OnlineReaderThread

if TYPE_CHECKING:
    from mwcore.radario.readers.TI.base import BaseTIBufferedReader
    from mwcore.tracking.api.base import BaseTracker


R = TypeVar('R', bound=SerialReader)
K = TypeVar('K', bound=BaseTracker)
@THREADS.register_module()
class OnlineReadTrackThread(OnlineReaderThread, Generic[R, K]):
    tracking_data = Signal(object)
    
    def __init__(
        self,
        reader: Union[R, dict],
        tracker: Union[K, dict],
        sensor_started: bool = False
    ):
        # initialize reader & sensor_started in base class
        super().__init__(reader, sensor_started)
        if isinstance(tracker, dict):
            tracker = TRACKERS.build(tracker)
        self.tracker: K = tracker

    def run(self):
        if not self.sensor_started:
            self.reader.connect()
            self._sensor_started = True

        while not self.isInterruptionRequested():
            data_ok, frame_number, det_obj = self.reader.read()
            if not data_ok:
                continue

            self.raw_data.emit(det_obj)
            self.array_data.emit(self._raw_to_numpy(det_obj))
            locations = self.tracker.consume(det_obj)
            self.tracking_data.emit(locations)
            # print(locations)