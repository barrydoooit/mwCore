import socket
import traceback
from typing import Tuple, Union
from PySide6.QtCore import QObject, Signal, Slot, QThread
import numpy as np
import logging

from .frameprocess import PointCloudGenerator


logger = logging.getLogger(__name__)
from .eth_adc_reader import AdcConfig, EthAdcReader



class AdcReadingWorker(QObject):
    frameReady = Signal(object, int, bool, float)
    finished = Signal()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def recent_cap_num(self) -> int:
        return self._recent_cap_num

    def __init__(self, reader: Union[dict, EthAdcReader]):
            super().__init__()

            if isinstance(reader, dict):
                self.reader = EthAdcReader.from_cfg(**reader)
            else:
                self.reader = reader
            self._running = False
            self._recent_cap_num = 0

    @Slot()
    def run(self):
         self._running = True
         BYTES_IN_PACKET = self.reader.config.data_bytes_per_packet
         BYTES_IN_FRAME = self.reader.config.bytes_in_frame
         UINT16_IN_FRAME = self.reader.config.uint16_in_frame

         lost_packets = False
         recent_frame = np.zeros(UINT16_IN_FRAME, dtype=np.int16)
         last_packet_num = -1
         last_recv_time = 0.0
         recentframe_collect_count = 0

         while self._running: # First capture loop: Find the beginning of a frame
            try:
                packet_num, byte_count, packet_data, recv_time = self.reader.read()
            except socket.timeout:
                logger.debug("Socket timeout, continuing read loop.")
                continue

            after_packet_count = (byte_count + BYTES_IN_PACKET) % BYTES_IN_FRAME

            if after_packet_count < BYTES_IN_PACKET:
                data_start_index = (BYTES_IN_PACKET - after_packet_count) // 2
                frame_end_index = after_packet_count // 2

                recent_frame[0:frame_end_index] = packet_data[data_start_index:]
                self._recent_cap_num = (byte_count + BYTES_IN_PACKET) // BYTES_IN_FRAME
                recentframe_collect_count = after_packet_count
                last_packet_num = packet_num
                last_recv_time = recv_time
                logger.info(f"Frame found. Starting capture at packet {packet_num}")
                break

            last_packet_num = packet_num

            while self.running:
                if QThread.currentThread().isInterruptionRequested():
                    logger.info("Interruption requested, breaking out of read loop.")
                    self._running = False
                    break
                try:
                    packet_num, byte_count, packet_data, recv_time = self.reader.read()
                    # logger.info(f"Received packet {packet_num} with byte count {byte_count}.")
                except socket.timeout:
                    logger.error("Socket timeout, continuing read loop.")
                    continue

                last_recv_time = recv_time

                if last_packet_num < packet_num - 1:
                    lost_packets = True
                    logger.warning(f"\a--- PACKET LOST --- Expected {last_packet_num + 1}, got {packet_num}")

                last_packet_num = packet_num

                if recentframe_collect_count + BYTES_IN_PACKET >= BYTES_IN_FRAME:
                    bytes_to_complete = BYTES_IN_FRAME - recentframe_collect_count
                    recent_frame[recentframe_collect_count // 2:] = packet_data[:bytes_to_complete // 2]

                    self._process_frame_data(
                        recent_frame.copy(),
                        self.recent_cap_num,
                        lost_packets,
                        last_recv_time
                    )
                    self._recent_cap_num = (byte_count + BYTES_IN_PACKET) // BYTES_IN_FRAME
                    recent_frame = np.zeros(UINT16_IN_FRAME, dtype=np.int16)

                    after_packet_count = (recentframe_collect_count + BYTES_IN_PACKET) % BYTES_IN_FRAME
                    data_start_index = (BYTES_IN_PACKET - after_packet_count) // 2

                    recent_frame[0:after_packet_count // 2] = packet_data[data_start_index:]
                    recentframe_collect_count = after_packet_count

                else:
                    after_packet_count = (recentframe_collect_count + BYTES_IN_PACKET) % BYTES_IN_FRAME
                    recent_frame[recentframe_collect_count // 2: after_packet_count // 2] = packet_data
                    recentframe_collect_count = after_packet_count
            
            self.reader.close_sockets()
            self.finished.emit()
        
    @Slot()
    def stop(self):
        logger.info("Stopping ADC Capture Worker.")
        self._running = False

    def _process_frame_data(self,
                            frame_data,
                            frame_num,
                            lost_packets,
                            timestamp):
        if lost_packets:
            return
        self.frameReady.emit(frame_data, frame_num, lost_packets, timestamp)

class AdcPointCloudGenWorker(AdcReadingWorker):
    array_data = Signal(object) # point cloud array
    processingFailed = Signal(str)

    def __init__(self,
                 reader: Union[dict, EthAdcReader],
                 generator: Union[dict, PointCloudGenerator]):
        super().__init__(reader)
        if not isinstance(generator, PointCloudGenerator):
            self.generator = PointCloudGenerator(generator)
        else:
            self.generator = generator

        self._frame_counter = 0

    @property
    def frame_counter(self) -> int:
        return self._frame_counter
    
    def _process_frame_data(self, frame_data, frame_num, lost_packets, timestamp):
        if lost_packets:
            logger.warning(f"Skipping frame {frame_num} due to lost packets.")
            return
        try:
            logger.info(f"Processing frame {frame_num} into point cloud.")
            point_cloud = self.generator.transform(frame_data)
            self._frame_counter += 1
            if point_cloud.size > 0:
                self.array_data.emit(point_cloud)

        except Exception as e:
            error_str =  f"PointCloud processing failed for frame {frame_num}: {e}\n{traceback.format_exc()}"
            logger.error(error_str)
            self.processingFailed.emit(error_str)
    
    @classmethod
    def build_with_thread(cls,
                          reader: Union[dict, EthAdcReader],
                          generator: Union[dict, PointCloudGenerator],) -> Tuple['AdcPointCloudGenWorker', QThread]:
        thread = QThread()
        worker = cls(reader, generator)

        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        return worker, thread