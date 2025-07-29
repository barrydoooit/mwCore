from copy import deepcopy
import time
from typing import TYPE_CHECKING, Dict, Optional, Type, Union
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
import os.path as osp

from mmengine.registry import DefaultScope
from mmengine.config import Config, ConfigDict
from mwcore.visualization.visualizers.online_pointcloud import OnlinePointCloudVisualizer
from mwcore.radario.readers.base import SerialReader
from mwcore.radario.readers.offline.base import OfflineReader

if TYPE_CHECKING:
    from mwcore.radario.readers.TI.base import BaseTIBufferedReader
    from mwcore.threads.online_reader import OnlineReaderThread
    from mwcore.threads.offline_reader import OfflineReaderThread
    from mwcore.threads.error_measurement import ErrorMeasurementThread
    from mwcore.threads.online_tracking import OnlineTrackingThread
    from PySide6.QtWidgets import QMainWindow

from mwcore.registry import READERS, THREADS, VISUALIZERS, APPS



ConfigType = Union[Dict, Config, ConfigDict]

@APPS.register_module()
class BaseMWApp:
    def __init__(self, cfg: Optional[ConfigType] = None):
        self.cfg = deepcopy(cfg) if cfg is not None else {}
        
        self._timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        experiment_name = cfg.get('experiment_name', None)
        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._timestamp}'
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
        else:
            self._experiment_name = self.timestamp
        
        if cfg.get('default_scope', None) is not None:
            self.default_scope = DefaultScope.get_instance(  # type: ignore
                self._experiment_name,
                scope_name=cfg['default_scope'])
        else:
            self.default_scope = DefaultScope.get_instance(self._experiment_name)
        
    @classmethod
    def from_cfg(cls, cfg: dict):
        return NotImplementedError
    
    def start(self):
        return NotImplementedError

@APPS.register_module()
class BaseMWOnlineApp(BaseMWApp):
    def __init__(self,
                 reader_cfg: dict,
                 vis_cfg: Optional[dict] = None,
                 cfg: Optional[ConfigType] = None):
        super().__init__(cfg)
        self.reader_cfg = deepcopy(reader_cfg)
        self.vis_cfg = deepcopy(vis_cfg) if vis_cfg is not None else None

    @property
    def reader_thread(self) -> Union['OnlineReaderThread', 'OfflineReaderThread']:
        if not hasattr(self, '_reader_thread'):
            reader_cfg = deepcopy(self.reader_cfg)
            if 'dataloader' in self.cfg:
                reader_cfg['dataloader'] = self.cfg['dataloader']
            self._reader_thread = self._make_reader_thread(reader_cfg)
        return self._reader_thread
    
    @property
    def visualizer(self) -> 'OnlinePointCloudVisualizer':
        if not hasattr(self, '_visualizer'):
            self._visualizer = self._make_visualizer(self.vis_cfg)
        return self._visualizer
    
    def _make_reader_thread(self, reader_cfg):
        reader_cls: Type = READERS.get(reader_cfg.get("type"))
        if issubclass(reader_cls, OfflineReader):
            playback_speed = reader_cfg.pop("playback_speed", None)
            return THREADS.build(dict(
                type="OfflineReaderThread",
                reader=reader_cfg, 
                playback_speed=playback_speed
            ))
        elif issubclass(reader_cls, SerialReader):
            assert reader_cfg.get("type") == "BufferedPcdReaderIWR6843", \
            "By default we use BufferedPcdReaderIWR6843. \
                Override _make_reader method in your app class to allow readers for custom radars."
            sensor_started = reader_cfg.pop("sensor_started", False)
            return THREADS.build(dict(
                type="OnlineReaderThread",
                reader=reader_cfg,
                sensor_started=sensor_started
            ))
    
    def on_visualizer_close(self, event):
        self.reader_thread.requestInterruption()
        self.reader_thread.wait()
    
    def _make_visualizer(self, vis_cfg: dict) -> 'OnlinePointCloudVisualizer':
        visualizer = VISUALIZERS.build(dict(
            vis_cfg,
            on_close=self.on_visualizer_close
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
            vis_cfg=vis_cfg,
            cfg=cfg
        )
