import logging
import time
from typing import Optional, Dict, Any

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from mwcore.signal_processing.radar_processor import StandardRadarProcessor


logger = logging.getLogger(__name__)


class DspProcessingWorker(QObject):
    """
    QObject-based DSP processing worker.

    This worker:
    - Receives completed frames from UdpCaptureWorker via frameReady signal
    - Optionally records raw frames to a .bin file
    - Processes frames through StandardRadarProcessor
    - Emits processedFrame(det_obj, frame_num) on success
    """

    processedFrame = Signal(object, int)  # det_obj, frame_num
    processingFailed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        processor: StandardRadarProcessor,
        save_to_file: Optional[str] = None,
    ):
        super().__init__()
        self._processor = processor
        self._save_to_file_path = save_to_file
        self._file_handle = None

        if save_to_file is not None:
            try:
                self._file_handle = open(save_to_file, "wb")
                logger.info("DspProcessingWorker will save raw data to: %s", save_to_file)
            except Exception as e:
                logger.error("Failed to open %s for writing: %s", save_to_file, e)
                self._file_handle = None

    @Slot(object, int, bool, float)
    def process_frame(
        self,
        frame_data: np.ndarray,
        frame_num: int,
        lost_packets: bool,
        timestamp_ms: float,
    ) -> None:
        """
        Slot to process a single frame.

        Parameters
        ----------
        frame_data : np.ndarray
            Raw int16 ADC frame.
        frame_num : int
            Frame sequence number.
        lost_packets : bool
            Whether packets were lost for this frame.
        timestamp_ms : float
            Capture timestamp in milliseconds.
        """
        if lost_packets:
            logger.warning("Skipping frame %d due to lost packets.", frame_num)
            return

        if not isinstance(frame_data, np.ndarray):
            logger.error("Invalid frame data type in DspProcessingWorker: %s", type(frame_data))
            return

        # Optional recording of raw data
        if self._file_handle is not None:
            try:
                self._file_handle.write(frame_data.tobytes())
                # Do not flush every frame to avoid IO bottlenecks; OS buffer is enough
            except Exception as e:
                logger.error("Error writing raw frame %d to .bin file: %s", frame_num, e)

        try:
            result: Dict[str, Any] = self._processor.process(frame_data)
            point_cloud = result.get("point_cloud", None)
        except Exception as e:
            msg = f"Error processing frame {frame_num}: {e}"
            logger.error(msg, exc_info=True)
            self.processingFailed.emit(msg)
            return

        if point_cloud is None or point_cloud.size == 0 or point_cloud.shape[1] == 0:
            # No points detected
            det_obj = {
                "numObj": 0,
                "x": np.array([]),
                "y": np.array([]),
                "z": np.array([]),
                "doppler": np.array([]),
                "peakVal": np.array([]),
                "timestamp": timestamp_ms if timestamp_ms is not None else time.time() * 1000.0,
            }
        else:
            num_points = point_cloud.shape[1]
            det_obj = {
                "numObj": num_points,
                "x": point_cloud[0, :],
                "y": point_cloud[1, :],
                "z": point_cloud[2, :],
                "doppler": point_cloud[3, :],
                "peakVal": point_cloud[4, :],
                "timestamp": timestamp_ms if timestamp_ms is not None else time.time() * 1000.0,
            }

        self.processedFrame.emit(det_obj, frame_num)

    @Slot()
    def close(self) -> None:
        """Close any resources held by the worker."""
        if self._file_handle is not None:
            try:
                logger.info("Closing DSP worker .bin file")
                self._file_handle.close()
            except Exception as e:
                logger.warning("Error while closing DSP worker file: %s", e)
            finally:
                self._file_handle = None
        self.finished.emit()

