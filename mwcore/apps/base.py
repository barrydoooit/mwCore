from copy import deepcopy
from typing import TYPE_CHECKING, Optional
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


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
class BaseMWApp:
    @classmethod
    def from_cfg(cls, cfg: dict):
        return NotImplementedError
    
    def start(self):
        return NotImplementedError

@APPS.register_module()
class BaseMWOnlineApp(BaseMWApp):
    def __init__(self,
                 reader_cfg: dict,
                 vis_cfg: Optional[dict] = None):
        self.reader_cfg = deepcopy(reader_cfg)
        self.vis_cfg = deepcopy(vis_cfg) if vis_cfg is not None else None

    @property
    def reader_thread(self) -> 'OnlineReaderThread':
        if not hasattr(self, '_reader_thread'):
            self._reader_thread = self._make_reader_thread(self.reader_cfg)
        return self._reader_thread
    
    @property
    def visualizer(self) -> 'OnlinePointCloudVisualizer':
        if not hasattr(self, '_visualizer'):
            self._visualizer = self._make_visualizer(self.vis_cfg)
        return self._visualizer
    
    def _make_reader_thread(self, reader_cfg: dict) -> 'OnlineReaderThread':
        assert reader_cfg.get("type") == "BufferedPcdReaderIWR6843", \
            "By default we use BufferedPcdReaderIWR6843. \
                Override _make_reader method in your app class to allow readers for custom radars."
        sensor_started = reader_cfg.pop('sensor_started', False)
        reader_thread = THREADS.build(dict(
            type="OnlineReaderThread",
            reader=reader_cfg,
            sensor_started=sensor_started
        ))
        return reader_thread
    
    def _make_visualizer(self, vis_cfg: dict) -> 'OnlinePointCloudVisualizer':
        def _on_close(event):
            self.reader_thread.requestInterruption()
            self.reader_thread.wait()
        visualizer = VISUALIZERS.build(dict(
            vis_cfg,
            on_close=_on_close
        ))
        assert isinstance(visualizer, OnlinePointCloudVisualizer)
        return visualizer
    
    def start(self):
        self.app = QApplication(sys.argv)
        if self.vis_cfg is not None:
            self.reader_thread.array_data.connect(
                self.visualizer.on_new_cloud, Qt.ConnectionType.QueuedConnection
            )
            self.visualizer.show()
        self.reader_thread.start()
        sys.exit(self.app.exec())
    
    @classmethod
    def from_cfg(cls, cfg):
        vis_cfg = cfg.get("vis_cfg", None)
        return cls(
            reader_cfg=cfg.get("reader_cfg", dict(type="BufferedPcdReaderIWR6843")),
            vis_cfg=vis_cfg
        )
    
@APPS.register_module()
class BaseMWOfflineApp(BaseMWApp):
    def __init__(self,
                 reader_cfg: dict,
                 tracker_cfg: dict,
                 vis_cfg: Optional[dict] = None,
                 error_cfg: Optional[dict] = None):
        self.reader_cfg = deepcopy(reader_cfg)
        self.tracker_cfg = deepcopy(tracker_cfg)
        self.vis_cfg = deepcopy(vis_cfg) if vis_cfg is not None else {}
        self.error_cfg = deepcopy(error_cfg) if error_cfg is not None else {}
        

    @property
    def reader_thread(self) -> 'OfflineReaderThread':
        if not hasattr(self, '_reader_thread'):
            self._reader_thread = THREADS.build(dict(
                type="OfflineReaderThread",
                reader=self.reader_cfg,
                playback_speed=self.reader_cfg.get("playback_speed", 0.4),
            ))
        return self._reader_thread

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
        if not hasattr(self, '_error_thread'):
            self._error_thread = THREADS.build(dict(
                type="ErrorMeasurementThread",
                tracker_name=self.tracker_cfg.get("type", "default_experiment"),
                dataset_name=self.reader_cfg.get("type", "default_dataset") +
                             self.reader_cfg.get("data", "default_dataset").replace("data", "").replace("/", "_").replace("?", ""),
                error_cfg=self.reader_cfg.get("error_cfg", {}),
            ))
        return self._error_thread

    @property
    def visualizer(self) -> 'OnlineTrackingVisualizer':
        if not hasattr(self, '_visualizer'):
            self._visualizer = self._make_visualizer(self.vis_cfg)
        return self._visualizer
    
    def _make_visualizer(self, vis_cfg: dict) -> 'OnlineTrackingVisualizer':
        def _on_close(event):
            self.error_thread.requestInterruption()
            self.error_thread.wait()
            self.reader_thread.requestInterruption()
            self.tracker_thread.requestInterruption()
        visualizer = VISUALIZERS.build(dict(
            vis_cfg,
            on_close=_on_close
        ))
        assert isinstance(visualizer, OnlineTrackingVisualizer)
        return visualizer
    

    def start(self):
        self.app = QApplication(sys.argv)
        if self.error_cfg.save_stats:
            # Prompt for experiment name if not provided
            from PySide6.QtWidgets import QInputDialog
            exp_name, ok = QInputDialog.getText(None, "Experiment Name", "Enter experiment name to save error statistics:")
            if ok and exp_name:
                self.error_cfg["experiment_name"] = exp_name
            else:
                self.error_cfg["experiment_name"] = "default_experiment"

        # Connect signals
        self.reader_thread.raw_data.connect(self.tracker_thread.process_frame)
        self.reader_thread.ground_truth_data.connect(self.error_thread.update_ground_truth)
        self.tracker_thread.tracking_data.connect(self.error_thread.update_tracking)
        # self.tracker_thread.tracking_data.connect(self.visualizer.update_tracking)
        # self.error_thread.error_data.connect(self.visualizer.update_error_metrics)

        if self.visualizer is not None:
            self.reader_thread.array_data.connect(
                self.visualizer.on_new_cloud, Qt.ConnectionType.QueuedConnection
            )
            self.tracker_thread.tracking_data.connect(
                self.visualizer.update_tracking, Qt.ConnectionType.QueuedConnection
            )
            self.visualizer.show()

        self.error_thread.start()
        self.reader_thread.start()
        self.tracker_thread.start()
        # Run the event loop
        sys.exit(self.app.exec())




    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            reader_cfg=cfg.get("reader_cfg", {}),
            tracker_cfg=cfg.get("tracker_cfg", {}),
            vis_cfg=cfg.get("vis_cfg", None),
            error_cfg=cfg.get("error_cfg", None)
        )