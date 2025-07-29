from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Type, cast
from typing_extensions import override
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from mwcore.apps.base import ConfigType
from mwcore.radario.readers.base import SerialReader
from mwcore.radario.readers.offline.base import OfflineReader


from . import BaseMWOnlineApp
from mwcore.visualization.visualizers.online_tracking import OnlineTrackingVisualizer

if TYPE_CHECKING:
    from mwcore.radario.readers.TI.base import BaseTIBufferedReader
    from mwcore.threads.online_reader import OnlineReaderThread
    from mwcore.threads.offline_reader import OfflineReaderThread
    from mwcore.threads.error_measurement import ErrorMeasurementThread
    from mwcore.threads.online_tracking import OnlineTrackingThread
    from mwcore.visualization.visualizers.online_pointcloud import OnlinePointCloudVisualizer
    from PySide6.QtWidgets import QMainWindow

from mwcore.registry import READERS, THREADS, VISUALIZERS, APPS



@APPS.register_module()
class TrackingApp(BaseMWOnlineApp):
    def __init__(self,
                 reader_cfg: dict,
                 tracker_cfg: dict,
                 vis_cfg: Optional[dict] = None,
                 error_cfg: Optional[dict] = None,
                 cfg: Optional[ConfigType] = None):
        super().__init__(reader_cfg, vis_cfg, cfg)
        self.tracker_cfg = deepcopy(tracker_cfg)
        self.error_cfg = deepcopy(error_cfg) if error_cfg is not None else None
        
    @property
    def tracker_thread(self) -> 'OnlineTrackingThread':
        if not hasattr(self, '_tracker_thread'):
            self._tracker_thread = THREADS.build(dict(
                type="OnlineTrackingThread",
                tracker=self.tracker_cfg
            ))
        return self._tracker_thread

    @property
    def error_thread(self) -> 'ErrorMeasurementThread':
        if self.error_cfg is None:
            return None
        if not hasattr(self, '_error_thread'):
            self._error_thread = THREADS.build(dict(
                type="ErrorMeasurementThread",
                tracker_name=self.tracker_cfg.get("type", "default_experiment"),
                dataset_name=self.reader_cfg.get("type", "default_dataset") +
                             self.reader_cfg.get("data", "default_dataset").replace("data", "").replace("/", "_").replace("?", ""),
                error_cfg=self.error_cfg
            ))
        return self._error_thread

    @override
    @property
    def visualizer(self) -> OnlineTrackingVisualizer:
        return cast(OnlineTrackingVisualizer, super().visualizer)
    
    @override
    def _make_visualizer(self, vis_cfg: dict) -> OnlineTrackingVisualizer:
        viz = VISUALIZERS.build(
            dict(vis_cfg, on_close=self.on_visualizer_close)
        )
        assert isinstance(viz, OnlineTrackingVisualizer)
        return viz
    
    def on_visualizer_close(self, event):
        if self.error_thread is not None:
            self.error_thread.requestInterruption()
            self.error_thread.wait()
        self.reader_thread.requestInterruption()
        self.tracker_thread.requestInterruption()

    def start(self):
        self.app = QApplication(sys.argv)
        self.reader_thread.raw_data.connect(self.tracker_thread.process_frame)
        if self.error_thread is not None:
            self.reader_thread.ground_truth_data.connect(self.error_thread.update_ground_truth)
            self.tracker_thread.tracking_data.connect(self.error_thread.update_tracking)

        if self.visualizer is not None:
            self.reader_thread.array_data.connect(
                self.visualizer.on_new_cloud, Qt.ConnectionType.QueuedConnection
            )
            self.tracker_thread.tracking_data.connect(
                self.visualizer.update_tracking, Qt.ConnectionType.QueuedConnection
            )
            self.visualizer.show()

        if self.error_thread is not None:
            self.error_thread.start()
        self.reader_thread.start()
        self.tracker_thread.start()
        sys.exit(self.app.exec())

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            reader_cfg=cfg.get("reader_cfg"),
            tracker_cfg=cfg.get("tracker_cfg"),
            vis_cfg=cfg.get("vis_cfg", None),
            error_cfg=cfg.get("error_cfg", None),
            cfg=cfg
        )