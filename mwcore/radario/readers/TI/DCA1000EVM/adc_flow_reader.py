import logging
import struct
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from mmengine.config import Config, ConfigDict

from mwcore.radario.readers.base import BaseReader
from mwcore.registry import READERS
from mwcore.signal_processing.config import RadarConfig
from mwcore.signal_processing.dsp_ppl import DspPipeline
from mwcore.signal_processing.frame import RadarFrame

from .udp_capture import UdpCaptureThread

logger = logging.getLogger(__name__)

ConfigType = Union[Dict[str, Any], Config, ConfigDict]


def _to_dim_first(point_cloud: np.ndarray) -> np.ndarray:
    if point_cloud.ndim != 2:
        raise ValueError(f"Expected 2D point cloud, got shape {point_cloud.shape}")
    if point_cloud.shape[0] == 6:
        return point_cloud
    if point_cloud.shape[1] == 6:
        return point_cloud.T
    raise ValueError(f"Unsupported point cloud shape {point_cloud.shape}; expected (6, N) or (N, 6)")


@READERS.register_module()
class RawAdcUdpFlowReader(BaseReader):
    def __init__(
        self,
        radar_cfg: Optional[ConfigType],
        pipeline: Optional[ConfigType] = None,
        static_ip: str = "192.168.33.30",
        adc_ip: str = "192.168.33.180",
        data_port: int = 4098,
        config_port: int = 4096,
        buffer_size: int = 1500,
        return_as_radar_frame: bool = True,
        save_to_file: Optional[str] = None,
        save_with_timestamp: bool = False,
    ):  
        self.radar_cfg = RadarConfig(**radar_cfg)
        self.pipeline = DspPipeline.from_cfg(pipeline) if pipeline is not None else None
        self.return_as_radar_frame = return_as_radar_frame
        self.save_with_timestamp = bool(save_with_timestamp)
        self.capture_thread = UdpCaptureThread(
            radar_config=self.radar_cfg,
            static_ip=static_ip,
            adc_ip=adc_ip,
            data_port=data_port,
            config_port=config_port,
            buffer_size=buffer_size,
        )

        self.file_handle = None
        if save_to_file:
            logger.info("Data will be saved: %s", save_to_file)
            self.file_handle = open(save_to_file, "wb")

        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    def connect(self):
        logger.info("Starting UDP capture thread...")
        self.capture_thread.start()
        logger.info("UDP capture thread started - waiting for radar data")

    def _handle_frame_errors(self, frame_num: int, lost_packet_flag: bool) -> Tuple[int, str]:
        if frame_num == -1:
            logger.warning("Buffer overwritten: %s", frame_num)
            return 0, "Buffer overwritten"
        if frame_num == -2:
            return 0, "Waiting for new frame"
        if lost_packet_flag:
            logger.warning("Frame %s had lost packets - skipping", frame_num)
            return 0, "Lost packets"
        return 1, "OK"
    
    def read(self) -> Tuple[int, int, Union[Dict[str, Any], RadarFrame]]:
        frame_data, frame_num, lost_packet_flag, frame_start_timestamp_ms = self.capture_thread.get_frame(
            include_timestamp=True
        )

        status, msg = self._handle_frame_errors(frame_num, lost_packet_flag)
        if status == 0:
            return 0, frame_num, {}

        if self.file_handle is not None:
            try:
                if self.save_with_timestamp:
                    ts_ms = frame_start_timestamp_ms if frame_start_timestamp_ms is not None else time.time() * 1000.0
                    self.file_handle.write(struct.pack("<d", float(ts_ms)))
                # Store as little-endian int16, frame-major contiguous bytes.
                # This matches the format expected by OfflineAdcDataReader.
                raw_bytes = np.ascontiguousarray(frame_data).astype("<i2", copy=False).tobytes()
                self.file_handle.write(raw_bytes)
                self.file_handle.flush()
            except Exception as exc:
                logger.error("Error writing to .bin file: %s", exc)

        data_ok, out = self.gen_point_cloud(
            frame_data=frame_data,
            frame_num=frame_num,
            frame_start_timestamp_ms=frame_start_timestamp_ms,
        )
        if not data_ok:
            return 0, frame_num, {}

        self._frame_count += 1
        if self._frame_count % 100 == 0:
            if isinstance(out, RadarFrame):
                num_obj = 0
                if hasattr(out, "point_cloud"):
                    pc = _to_dim_first(np.asarray(out.point_cloud))
                    num_obj = int(pc.shape[1])
            else:
                num_obj = int(out.get("numObj", 0))
            if num_obj > 0:
                logger.info("Processed %s frames (latest: %s points)", self._frame_count, num_obj)

        return 1, frame_num, out

    def gen_point_cloud(
        self,
        frame_data: np.ndarray,
        frame_num: int,
        frame_start_timestamp_ms: Optional[float],
    ) -> Tuple[int, Union[Dict[str, Any], RadarFrame]]:
        timestamp_ms = frame_start_timestamp_ms if frame_start_timestamp_ms is not None else time.time() * 1000.0

        if not isinstance(frame_data, np.ndarray):
            logger.error("Invalid frame data type: %s", type(frame_data))
            return 0, {}

        frame = RadarFrame(
            raw_bytes=frame_data.tobytes(),
            config=self.radar_cfg,
            frame_start_timestamp_ms=timestamp_ms,
        )
        frame.frame_number = frame_num

        if self.pipeline is None:
            frame.point_cloud = np.zeros((0, 6), dtype=np.float32)
            if self.return_as_radar_frame:
                return 1, frame
            return 1, self._empty_det_obj(timestamp_ms)

        try:
            frame = self.pipeline.run(
                start_with_this_frame=frame,
            )
        except Exception as exc:
            logger.error("Error processing frame %s: %s", frame_num, exc)
            return 0, {}

        if self.return_as_radar_frame:
            return 1, frame

        point_cloud = getattr(frame, "point_cloud", None)
        if point_cloud is None:
            return 1, self._empty_det_obj(timestamp_ms)

        point_cloud = _to_dim_first(np.asarray(point_cloud))
        if point_cloud.size == 0 or point_cloud.shape[1] == 0:
            return 1, self._empty_det_obj(timestamp_ms)

        frame_ts = frame.frame_start_timestamp_ms
        if frame_ts is not None:
            timestamp_ms = frame_ts

        return 1, {
            "numObj": int(point_cloud.shape[1]),
            "x": point_cloud[0, :],
            "y": point_cloud[1, :],
            "z": point_cloud[2, :],
            "doppler": point_cloud[3, :],
            "peakVal": point_cloud[4, :],
            "timestamp": timestamp_ms,
        }

    def _empty_det_obj(self, timestamp_ms: float) -> Dict[str, Any]:
        return {
            "numObj": 0,
            "x": np.array([], dtype=np.float32),
            "y": np.array([], dtype=np.float32),
            "z": np.array([], dtype=np.float32),
            "doppler": np.array([], dtype=np.float32),
            "peakVal": np.array([], dtype=np.float32),
            "timestamp": timestamp_ms,
        }
        
    def close(self):
        logger.info("Closing UdpRawAdcDataReader...")
        if hasattr(self, "capture_thread"):
            self.capture_thread.stop()
            if self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
                if self.capture_thread.is_alive():
                    logger.warning("Capture thread did not terminate gracefully")
            self.capture_thread.close()

        if self.file_handle is not None:
            logger.info("Closing .bin file (wrote %s frames)", self._frame_count)
            self.file_handle.close()
            self.file_handle = None

        logger.info("UdpRawAdcDataReader closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
