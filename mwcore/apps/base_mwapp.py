
from copy import deepcopy
import warnings
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QThread, QObject
import os.path as osp

from mmengine.registry import DefaultScope
from mmengine.config import Config, ConfigDict
from mwcore.apps.base import BaseApp, ConfigType
from mwcore.visualization.visualizers.online_pointcloud import OnlinePointCloudVisualizer

from mwcore.registry import READERS, THREADS, VISUALIZERS, APPS



@APPS.register_module()
class BaseMWApp(BaseApp):
    def __init__(self,
                 thread_cfg: dict,
                 vis_cfg: Optional[dict] = None,
                 connections: Optional[List[dict]] = None,
                 cfg: Optional[ConfigType] = None):
        super().__init__(cfg)
        self.thread_cfg = deepcopy(thread_cfg)
        self.vis_cfg = deepcopy(vis_cfg) if vis_cfg is not None else None
        self.connection_cfgs = deepcopy(connections) if connections is not None else []
        
        self._reader_worker_obj = None
        self._reader_thread_obj = None
        self._visualizer_obj = None

    @property
    def reader_worker(self) -> QObject:
        """
        Exposes the Worker Object. 
        Use this in config connections (e.g., 'reader_worker.frame_signal').
        """
        if self._reader_worker_obj is None:
            self._build_reader_system()
        return self._reader_worker_obj

    @property
    def reader_thread(self) -> QThread:
        """
        Exposes the QThread Object.
        Use this for lifecycle management (start, quit, wait).
        """
        if self._reader_thread_obj is None:
            self._build_reader_system()
        return self._reader_thread_obj
    
    @property
    def visualizer(self) -> Optional['OnlinePointCloudVisualizer']:
        if self._visualizer_obj is None and self.vis_cfg:
             self._visualizer_obj = self._make_visualizer(self.vis_cfg)
        return self._visualizer_obj
    
    def _build_reader_system(self):
        """
        Builds both Worker and Thread and assigns them to instance attributes.
        """
        cfg_copy = deepcopy(self.thread_cfg)
        
        # Inject 'sensor_started' if present in App config
        sensor_started = self.cfg.get('sensor_started')
        if sensor_started is not None:
            if 'reader' in cfg_copy and isinstance(cfg_copy['reader'], dict):
                 if 'sensor_started' not in cfg_copy['reader']:
                     cfg_copy['reader']['sensor_started'] = sensor_started
            elif 'sensor_started' not in cfg_copy:
                cfg_copy['sensor_started'] = sensor_started

        # Build via registry
        # We expect the factory to return Tuple[Worker, QThread]
        result = THREADS.build(cfg_copy)
        
        if isinstance(result, tuple) and len(result) == 2:
            self._reader_worker_obj, self._reader_thread_obj = result
        else:
            # Fallback or Error for legacy types if strictly ignoring them
            raise TypeError(f"THREADS.build expected (Worker, Thread) tuple, got {type(result)}")

    def _make_visualizer(self, vis_cfg: dict):
        visualizer = VISUALIZERS.build(dict(
            vis_cfg,
            on_close=self.on_visualizer_close
        ))
        return visualizer

    def on_visualizer_close(self, event):
        self._cleanup()

    def _cleanup(self):
        """Stops the QThread."""
        if self._reader_thread_obj and self._reader_thread_obj.isRunning():
            self._reader_thread_obj.requestInterruption()
            self._reader_thread_obj.quit()
            self._reader_thread_obj.wait(2000)
            if self._reader_thread_obj.isRunning():
                self._reader_thread_obj.terminate()

    def _resolve_attr(self, attr_path: str):
        """
        Resolves string path starting from 'self'.
        Examples:
        - "reader_worker.frame_signal" -> self.reader_worker.frame_signal
        - "visualizer.on_new_frame" -> self.visualizer.on_new_frame
        """
        obj = self
        try:
            for part in attr_path.split('.'):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            raise AttributeError(f"Could not resolve path '{attr_path}' in BaseMWApp.")

    def _setup_connections(self):
        for conn in self.connection_cfgs:
            signal_str = conn.get('signal')
            slot_str = conn.get('slot')
            conn_type_str = conn.get('type', 'AutoConnection')
            
            try:
                # 1. Resolve Signal (e.g., from reader_worker)
                signal_obj = self._resolve_attr(signal_str)
                
                # 2. Resolve Slot (e.g., from visualizer)
                slot_obj = self._resolve_attr(slot_str)
                
                # 3. Connect
                conn_type = getattr(Qt.ConnectionType, conn_type_str, Qt.ConnectionType.AutoConnection)
                signal_obj.connect(slot_obj, conn_type)
                
                print(f"[BaseMWApp] Connected: {signal_str} -> {slot_str}")
            except Exception as e:
                print(f"[BaseMWApp] Connection Error ({signal_str} -> {slot_str}): {e}")

    def start(self):
        self.app = QApplication(sys.argv)
        
        # 1. Ensure Components are Built
        # Accessing properties triggers _build_reader_system if needed
        worker = self.reader_worker 
        thread = self.reader_thread
        vis = self.visualizer
        
        # 2. Setup Connections
        # This relies on properties (like reader_worker) being available
        self._setup_connections()
        
        if vis:
            vis.show()
            
        # 3. Start Execution
        # We start the THREAD, not the worker
        thread.start()
        
        sys.exit(self.app.exec())
    
    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            thread_cfg=cfg.get("thread_cfg"),
            vis_cfg=cfg.get("vis_cfg", None),
            connections=cfg.get("connections", []),
            cfg=cfg
        )