import time
import numpy as np
from mwcore.registry import THREADS
from PySide6.QtCore import QThread, Signal, Slot
from mwcore.tracking.api.base import BaseTracker
from typing import TYPE_CHECKING, Dict, Generic, TypeVar, Union

from mwcore.radario.readers.offlineReaders.framedata import FrameData



K = TypeVar('K', bound=BaseTracker)
@THREADS.register_module()
class OnlineTrackingThread(QThread, Generic[K]):
    tracking_data = Signal(np.ndarray)
    tracking_framedata = Signal(FrameData)

    def __init__(self, 
                 tracker: Union[K, dict],
                 use_framedata: bool = False,
                 parent=None):

        super().__init__(parent=parent)
        if isinstance(tracker, dict):
            from mwcore.registry import TRACKERS
            tracker = TRACKERS.build(tracker)
        self.tracker: K = tracker
        self.use_framedata = use_framedata

    @Slot(object)
    def process_frame(self, det_obj: Union[dict, FrameData]):
        locations = []
        if det_obj is not None:
            if self.use_framedata and isinstance(det_obj, FrameData):
                det_obj_dict = det_obj.input_data
                start_time = time.perf_counter()
                locations = self.tracker.consume(det_obj=det_obj_dict, sort_metric='size')
                latency = time.perf_counter() - start_time
                det_obj.prediction = np.array(locations)
                det_obj.latency = latency # I added this line to store latency. Perhaps an if statement with a check for latency evaluator would be better
                self.tracking_data.emit(np.array(locations))
                self.tracking_framedata.emit(det_obj)
            else:
                locations = self.tracker.consume(det_obj=det_obj, sort_metric='size')
                self.tracking_data.emit(np.array(locations))

    def run(self):
        print("Online Tracking Thread started")
        self.exec_()
