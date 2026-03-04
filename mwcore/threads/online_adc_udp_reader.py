import logging
import queue
import threading
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from mwcore.registry import READERS, THREADS
from mwcore.radario.readers.TI.DCA1000EVM.adc_flow_reader import RawAdcUdpFlowReader
from mwcore.signal_processing.frame import RadarFrame

log = logging.getLogger(__name__)


def _to_points_first(point_cloud: np.ndarray) -> np.ndarray:
    if point_cloud.ndim != 2:
        return np.zeros((0, 6), dtype=np.float32)
    if point_cloud.shape[1] == 6:
        return point_cloud
    if point_cloud.shape[0] == 6:
        return point_cloud.T
    return np.zeros((0, 6), dtype=np.float32)


class RawAdcUdpFlowReaderWorker(QObject):
    finished = Signal()
    frame_signal = Signal(object)
    raw_data = Signal(dict)
    array_data = Signal(np.ndarray)
    progress_signal = Signal(dict)

    def __init__(
        self,
        reader: RawAdcUdpFlowReader,
        idle_sleep_s: float = 0.001,
        max_pending_frames: int = 256,
    ):
        super().__init__()
        self.reader = reader
        self.idle_sleep_s = float(idle_sleep_s)
        self._fast_forward = False
        self._stop_event = threading.Event()
        self._raw_queue: "queue.Queue[Tuple[int, np.ndarray, Optional[float]]]" = queue.Queue(
            maxsize=max_pending_frames
        )

    def _emit_output(self, data: Union[Dict, RadarFrame]):
        self.frame_signal.emit(data)
        if isinstance(data, RadarFrame):
            if hasattr(data, "point_cloud"):
                points = _to_points_first(np.asarray(data.point_cloud))
                self.array_data.emit(points[:, :5] if points.shape[1] >= 5 else np.zeros((0, 5), dtype=np.float32))
            else:
                self.array_data.emit(np.zeros((0, 5), dtype=np.float32))
        elif isinstance(data, dict):
            self.raw_data.emit(data)
            try:
                arr = np.stack([data[key] for key in ["x", "y", "z", "doppler", "peakVal"]], axis=-1)
            except Exception:
                arr = np.zeros((0, 5), dtype=np.float32)
            self.array_data.emit(arr)
        else:
            self.array_data.emit(np.zeros((0, 5), dtype=np.float32))

    def _process_queue_loop(self):
        while not self._stop_event.is_set() or not self._raw_queue.empty():
            try:
                frame_number, frame_data, frame_start_timestamp_ms = self._raw_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            data_ok, data = self.reader.process_raw_frame(
                frame_data=frame_data,
                frame_num=frame_number,
                frame_start_timestamp_ms=frame_start_timestamp_ms,
            )
            if data_ok:
                self._emit_output(data)
                if frame_number % 100 == 0:
                    self.progress_signal.emit({
                        "current_file": "UDP_STREAM",
                        "file_index": 0,
                        "total_files": 1,
                        "frame_global_index": frame_number,
                        "frame_in_file": frame_number,
                    })

    def process(self):
        log.info("RawAdcUdpFlowReaderWorker: Started.")
        self.reader.connect()
        self.progress_signal.emit({
            "current_file": "UDP_STREAM",
            "file_index": 0,
            "total_files": 1,
            "frame_global_index": 0,
            "frame_in_file": 0,
        })
        processor = threading.Thread(target=self._process_queue_loop, daemon=True)
        processor.start()
        try:
            while not QThread.currentThread().isInterruptionRequested():
                data_ok, frame_number, frame_data, frame_start_timestamp_ms = self.reader.poll_raw_frame()
                if not data_ok:
                    if self.idle_sleep_s > 0:
                        time.sleep(self.idle_sleep_s)
                    continue

                if frame_data is None:
                    continue
                try:
                    self._raw_queue.put(
                        (frame_number, frame_data, frame_start_timestamp_ms),
                        timeout=0.001,
                    )
                except queue.Full:
                    # Backpressure protection: drop oldest pending frame, enqueue newest.
                    try:
                        _ = self._raw_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._raw_queue.put_nowait((frame_number, frame_data, frame_start_timestamp_ms))
                    except queue.Full:
                        pass
        finally:
            self._stop_event.set()
            processor.join(timeout=2.0)
            self.reader.close()
            self.finished.emit()

    @Slot(bool)
    def set_fast_forward(self, enabled: bool):
        # UDP live stream has no playback speed control; kept for API compatibility.
        self._fast_forward = bool(enabled)

    @staticmethod
    @THREADS.register_module(name="RawAdcUdpFlowReaderWorker")
    def build_with_thread(
        reader: Union[dict, RawAdcUdpFlowReader],
        idle_sleep_s: float = 0.001,
        max_pending_frames: int = 256,
        **kwargs,
    ) -> Tuple["RawAdcUdpFlowReaderWorker", QThread]:
        reader_instance = READERS.build(reader) if isinstance(reader, dict) else reader
        if not isinstance(reader_instance, RawAdcUdpFlowReader):
            log.warning("Worker expects RawAdcUdpFlowReader, got %s", type(reader_instance))

        worker = RawAdcUdpFlowReaderWorker(
            reader_instance,
            idle_sleep_s=idle_sleep_s,
            max_pending_frames=max_pending_frames,
        )
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.process)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        return worker, thread
