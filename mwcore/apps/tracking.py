from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Type, cast
from typing_extensions import override
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from mwcore.apps.base import ConfigType
from mwcore.threads.evaluator_thread import EvaluationWorker


from . import BaseMWOnlineApp
from mwcore.visualization.visualizers.online_tracking import OnlineTrackingVisualizer

if TYPE_CHECKING:
    from mwcore.threads.online_tracking import OnlineTrackingThread

from mwcore.registry import READERS, THREADS, VISUALIZERS, APPS



@APPS.register_module()
class TrackingApp(BaseMWOnlineApp):
    def __init__(self,
                 reader_cfg: dict,
                 tracker_cfg: dict,
                 vis_cfg: Optional[dict] = None,
                 evaluators: Optional[list] = None,
                 cfg: Optional[ConfigType] = None):
        super().__init__(reader_cfg, vis_cfg, cfg)
        self.tracker_cfg = deepcopy(tracker_cfg)
        self.evaluator_worker, self.evaluator_thread = EvaluationWorker.build_with_thread(evaluators) if evaluators else (None, None)

    @property
    def tracker_thread(self) -> 'OnlineTrackingThread':
        if not hasattr(self, '_tracker_thread'):
            self._tracker_thread = THREADS.build(dict(
                type="OnlineTrackingThread",
                tracker=self.tracker_cfg,
                use_framedata=True
            ))
        return self._tracker_thread

    @override
    @property
    def visualizer(self) -> Optional[OnlineTrackingVisualizer]:
        if not self.vis_cfg.get('is_on', True):
            return None
        else:
            self.vis_cfg.pop('is_on', None)
        return cast(OnlineTrackingVisualizer, super().visualizer)
    
    @override
    def _make_visualizer(self, vis_cfg: dict) -> OnlineTrackingVisualizer:
        viz = VISUALIZERS.build(
            dict(vis_cfg, on_close=self.on_visualizer_close)
        )
        assert isinstance(viz, OnlineTrackingVisualizer)
        return viz
    
    def on_visualizer_close(self, event):
        if self.evaluator_thread is not None:
            self.evaluator_thread.quit()
        self.reader_thread.requestInterruption()
        self.tracker_thread.requestInterruption()

    def start(self):
        self.app = QApplication(sys.argv)
        import signal
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        from PySide6.QtCore import QTimer
        _signal_timer = QTimer(self.app)
        _signal_timer.start(1000)  # check every 100 ms
        _signal_timer.timeout.connect(lambda: None)

        if hasattr(self.reader_thread, 'signal_framedata'):
            self.reader_thread.signal_framedata.connect(self.tracker_thread.process_frame, Qt.ConnectionType.QueuedConnection)
        else:
            self.reader_thread.raw_data.connect(self.tracker_thread.process_frame, Qt.ConnectionType.QueuedConnection)
        if self.visualizer is not None:
            self.reader_thread.array_data.connect(
                self.visualizer.on_new_cloud, Qt.ConnectionType.QueuedConnection
            )
            self.tracker_thread.tracking_data.connect(
                self.visualizer.update_tracking, Qt.ConnectionType.QueuedConnection
            )
            self.visualizer.show()

        if self.evaluator_worker is not None:
            self.tracker_thread.tracking_framedata.connect(self.evaluator_worker.process_framedata, Qt.ConnectionType.QueuedConnection)
            if hasattr(self.reader_thread, 'signal_finished'):
                self.reader_thread.signal_finished.connect(self.evaluator_worker.update_final_frame_number, Qt.ConnectionType.QueuedConnection)
            self.evaluator_worker.signal_evaluation_complete.connect(self.app.quit)
            self.evaluator_thread.start()
        else:
            if hasattr(self.reader_thread, 'signal_finished'):
                self.reader_thread.signal_finished.connect(lambda x: self.app.quit())
        
        self.reader_thread.start()
        self.tracker_thread.start()
        try:
            sys.exit(self.app.exec())
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down…")
            sys.exit(0)

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            reader_cfg=cfg.get("reader_cfg"),
            tracker_cfg=cfg.get("tracker_cfg"),
            vis_cfg=cfg.get("vis_cfg", None),
            evaluators=cfg.get("evaluators", None),
            cfg=cfg
        )