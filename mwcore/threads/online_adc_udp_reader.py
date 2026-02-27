import logging
import time
from typing import Dict, Tuple, Union

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

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

    def __init__(self, reader: RawAdcUdpFlowReader, idle_sleep_s: float = 0.001):
        super().__init__()
        self.reader = reader
        self.idle_sleep_s = float(idle_sleep_s)

    def process(self):
        log.info("RawAdcUdpFlowReaderWorker: Started.")
        self.reader.connect()
        try:
            while not QThread.currentThread().isInterruptionRequested():
                data_ok, frame_number, data = self.reader.read()
                if not data_ok:
                    if self.idle_sleep_s > 0:
                        time.sleep(self.idle_sleep_s)
                    continue

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
        finally:
            self.reader.close()
            self.finished.emit()

    @staticmethod
    @THREADS.register_module(name="RawAdcUdpFlowReaderWorker")
    def build_with_thread(
        reader: Union[dict, RawAdcUdpFlowReader],
        idle_sleep_s: float = 0.001,
        **kwargs,
    ) -> Tuple["RawAdcUdpFlowReaderWorker", QThread]:
        reader_instance = READERS.build(reader) if isinstance(reader, dict) else reader
        if not isinstance(reader_instance, RawAdcUdpFlowReader):
            log.warning("Worker expects RawAdcUdpFlowReader, got %s", type(reader_instance))

        worker = RawAdcUdpFlowReaderWorker(reader_instance, idle_sleep_s=idle_sleep_s)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.process)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        return worker, thread
