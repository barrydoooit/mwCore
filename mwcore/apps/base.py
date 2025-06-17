from copy import deepcopy
from typing import TYPE_CHECKING
import sys
from PySide6.QtWidgets import QApplication

from mwcore.visualization.visualizers.online_pointcloud import OnlinePointCloudVisualizer

if TYPE_CHECKING:
    from mwcore.radario.readers.TI.base import BaseTIBufferedReader
    from mwcore.threads.online_reader import OnlineReaderThread
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
                 vis_cfg: dict = dict()):
        self.reader_cfg = deepcopy(reader_cfg)
        self.vis_cfg = deepcopy(vis_cfg)

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
        self.reader_thread.array_data.connect(
            self.visualizer.update_point_cloud
        )
        self.reader_thread.start()
        self.visualizer.show()
        sys.exit(self.app.exec())
    
    @classmethod
    def from_cfg(cls, cfg):
        vis_cfg = cfg.get("vis_cfg", dict(type="OnlinePointCloudVisualizer"))
        return cls(
            reader_cfg=cfg.get("reader_cfg", dict(type="BufferedPcdReaderIWR6843")),
            vis_cfg=vis_cfg
        )