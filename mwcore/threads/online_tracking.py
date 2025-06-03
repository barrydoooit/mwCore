import time
import numpy as np
from mwcore.registry import THREADS
from PySide2.QtCore import QThread, Signal, Slot



from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from mwcore.tracking.api.base import BaseTracker


@THREADS.register_module()
class OnlineTrackingThread(QThread):
    tracking_data = Signal(np.ndarray)
    
    def __init__(self, tracker: 'BaseTracker', parent=None):

        super().__init__(parent=parent)
        self.tracker = tracker

    @Slot(object)
    def process_frame(self, det_obj):
        locations = []
        if det_obj is not None:
            locations = self.tracker.consume(det_obj)
            self.tracking_data.emit(locations)
            # print(locations)

    def run(self):
        print("Online Tracking Thread started")
        self.exec_()
    
    