from copy import deepcopy
from functools import partial
import sys
from typing import TYPE_CHECKING

from mwcore.registry import READERS, TRACKERS, THREADS

from PySide2.QtWidgets import QApplication

from apps.__base__.runner import APPS, MWAppRunner
from mwcore.utils.tranforms import dev2standard, standard2dev
from mwcore.visualization.visualizer import MainVisualizer

if TYPE_CHECKING:
    from mwcore.radario.readers.TI.base import BaseTIBufferedReader
    from mwcore.tracking.api.base import BaseTracker



@APPS.register_module()
class VisAndTrackApp(MWAppRunner):
    def __init__(self,
                 reader_cfg: dict,
                 tracker_cfg: dict,
                 vis_cfg: dict = dict()):
        self.sensor_started = reader_cfg.pop('sensor_started', False)
        self.reader = self._make_reader(reader_cfg)
        self.reader_thread = THREADS.build(dict(
            type="OnlineReaderThread",
            reader=self.reader
        ))
        self.tracker_cfg = deepcopy(tracker_cfg)
        self.tracker =  self._make_tracker(tracker_cfg)
        self.tracker_thread = THREADS.build(dict(
            type="OnlineTrackingThread",
            tracker=self.tracker
        ))
        
        self.vis_cfg = deepcopy(vis_cfg)
        radar_cfg = tracker_cfg.get("radar_cfg", dict())
        self.radar2world = dev2standard(
            device_tilt=radar_cfg.get("sensor_tilt", 0),
            device_height=radar_cfg.get("sensor_height", 0)
        )
        self.world2radar = standard2dev(
            device_tilt=radar_cfg.get("sensor_tilt", 0),
            device_height=radar_cfg.get("sensor_height", 0)
        )

    
    def _make_reader(self, reader_cfg: dict) -> 'BaseTIBufferedReader':
        assert reader_cfg.get("type") == "BufferedPcdReaderIWR6843", "We only tested the IWR6843 reader. For other readers, comment this assertion."
        reader = READERS.build(reader_cfg)
        return reader

    def _make_tracker(self, tracker_cfg: dict) -> 'BaseTracker':
        tracker = TRACKERS.build(tracker_cfg)
        return tracker

    def start(self):
        if not self.sensor_started:
            self.reader.connect()
            self.sensor_started = True

        self.app = QApplication(sys.argv)
        def _on_close(event):
            self.reader_thread.requestInterruption()
            self.reader_thread.wait()

        self.main_window = MainVisualizer(
            on_close=_on_close,
            tracking_mode=self.vis_cfg.get("tracking_mode", "dot"),
        )
        self.reader_thread.array_data.connect(partial(self.main_window.update_point_cloud, 
                                                      trans_matrix=self.radar2world if self.tracker_cfg.get("do_dev2standard", True) else None))
        self.reader_thread.raw_data.connect(self.tracker_thread.process_frame)
        self.tracker_thread.tracking_data.connect(self.main_window.update_tracking)

        self.reader_thread.start()
        self.tracker_thread.start()
        self.main_window.show()
        sys.exit(self.app.exec_())
    
    @classmethod
    def from_cfg(cls, cfg: dict):
        tracker_cfg = dict(
                           cfg.get("tracker_cfg"),
                           radar_cfg=cfg.get("radar_cfg", dict(
                               sensor_tilt=0,
                               sensor_height=0
                           )))
        vis_cfg = cfg.get("vis_cfg", dict())
        return cls(
            reader_cfg=cfg.get("reader_cfg"),
            tracker_cfg=tracker_cfg,
            vis_cfg=vis_cfg
        )