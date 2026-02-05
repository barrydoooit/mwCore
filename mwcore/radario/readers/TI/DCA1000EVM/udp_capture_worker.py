import socket
import struct
import logging
import numpy as np
from typing import Tuple

from PySide6.QtCore import QObject, Signal, Slot, QThread

from mwcore.signal_processing.radar_processor import RadarConfig
import time


logger = logging.getLogger(__name__)


class UdpCaptureWorker(QObject):
    """
    QObject-based UDP capture worker.

    This worker:
    - Runs inside its own QThread
    - Receives UDP packets from DCA1000
    - Assembles complete frames
    - Emits a frameReady signal when a frame is complete
    """

    frameReady = Signal(object, int, bool, float)
    finished = Signal()

    # Network constants
    MAX_PACKET_SIZE = 4096
    BYTES_IN_PACKET = 1456  # Data payload size (excluding 10-byte header)

    def __init__(
        self,
        radar_config: RadarConfig,
        static_ip: str = "192.168.33.30",
        adc_ip: str = "192.168.33.180",
        data_port: int = 4098,
        config_port: int = 4096,
    ):
        super().__init__()

        self.config = radar_config
        self._running = False

        # Calculate frame size from radar config
        # Frame = chirps * rx * tx * IQ * samples * bytes
        # IQ = 2 (I and Q), bytes = 2 (int16)
        self.bytes_in_frame = (
            self.config.loops_per_frame
            * self.config.num_rx
            * self.config.num_tx
            * 2  # IQ
            * self.config.adc_samples
            * 2  # bytes per sample
        )
        self.uint16_in_frame = self.bytes_in_frame // 2

        logger.info(
            "CaptureWorker frame size: %d bytes (%d uint16 samples)",
            self.bytes_in_frame,
            self.uint16_in_frame,
        )
        logger.info(
            "CaptureWorker packets per frame: %.2f",
            self.bytes_in_frame / self.BYTES_IN_PACKET,
        )

        # Create network destinations
        self.cfg_dest = (adc_ip, config_port)
        self.cfg_recv = (static_ip, config_port)
        self.data_recv = (static_ip, data_port)

        # Create sockets
        self.config_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        self.data_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )

        # Bind data socket
        logger.info("CaptureWorker binding data socket to %s", self.data_recv)
        self.data_socket.bind(self.data_recv)
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**27)

        # Bind config socket
        self.config_socket.bind(self.cfg_recv)

    @Slot()
    def run(self) -> None:
        """
        Main worker execution - receives and assembles frames.

        This method is intended to be invoked when the owning QThread starts.
        """
        logger.info("UdpCaptureWorker started in thread")
        self._running = True

        try:
            self._frame_receiver()
        finally:
            # Always ensure sockets are closed
            self._close_sockets()
            self.finished.emit()
            logger.info("UdpCaptureWorker finished")

    @Slot()
    def stop(self) -> None:
        """Request the worker to stop."""
        logger.info("Stopping UdpCaptureWorker...")
        self._running = False

    def _frame_receiver(self) -> None:
        """
        Receive UDP packets and assemble them into complete frames.

        On each complete frame, emit frameReady(frame, frame_num, lost_packets, timestamp).
        """
        # Set timeout for socket operations
        self.data_socket.settimeout(1)

        # First capture loop: find the beginning of a frame
        recent_frame = np.zeros(self.uint16_in_frame, dtype=np.int16)

        logger.info("CaptureWorker searching for frame boundary...")
        last_packet_num = -1
        while self._running:
            try:
                packet_num, byte_count, packet_data, recv_time = self._read_data_packet()
            except socket.timeout:
                logger.debug("CaptureWorker socket timeout while searching for frame boundary")
                continue
            except Exception as e:
                logger.error("CaptureWorker error reading packet (search phase): %s", e)
                continue

            after_packet_count = (byte_count + self.BYTES_IN_PACKET) % self.bytes_in_frame

            # The recent frame begins in the middle of this packet
            if after_packet_count < self.BYTES_IN_PACKET:
                recent_frame[0 : after_packet_count // 2] = packet_data[
                    (self.BYTES_IN_PACKET - after_packet_count) // 2 :
                ]
                recentframe_collect_count = after_packet_count
                last_packet_num = packet_num
                recent_cap_num = (byte_count + self.BYTES_IN_PACKET) // self.bytes_in_frame
                logger.info(
                    "CaptureWorker frame boundary found at packet %d (frame_num=%d)",
                    packet_num,
                    recent_cap_num,
                )
                break

            last_packet_num = packet_num

        if not self._running:
            return

        # Main capture loop: assemble complete frames
        logger.info("CaptureWorker entering main frame assembly loop")
        lost_packets = False
        recent_cap_num = 0

        while self._running:
            try:
                packet_num, byte_count, packet_data, recv_time = self._read_data_packet()
            except socket.timeout:
                logger.debug("CaptureWorker socket timeout in main loop")
                continue
            except Exception as e:
                logger.error("CaptureWorker error reading packet (main loop): %s", e)
                continue

            # Check for lost packets
            if last_packet_num >= 0 and last_packet_num < packet_num - 1:
                lost_packets = True
                logger.warning(
                    "CaptureWorker PACKET LOST! Expected %d, got %d",
                    last_packet_num + 1,
                    packet_num,
                )

            # Check if frame is complete with this packet
            if recentframe_collect_count + self.BYTES_IN_PACKET >= self.bytes_in_frame:
                # Complete the frame
                bytes_to_complete = self.bytes_in_frame - recentframe_collect_count
                recent_frame[recentframe_collect_count // 2 :] = packet_data[
                    : bytes_to_complete // 2
                ]

                # Compute frame number
                recent_cap_num = (byte_count + self.BYTES_IN_PACKET) // self.bytes_in_frame

                # Emit the completed frame
                if not lost_packets:
                    self.frameReady.emit(recent_frame.copy(), recent_cap_num, False, recv_time)
                else:
                    # Still emit with lost flag so downstream can decide to skip
                    self.frameReady.emit(recent_frame.copy(), recent_cap_num, True, recv_time)

                # Start a new frame
                recent_frame = np.zeros(self.uint16_in_frame, dtype=np.int16)
                after_packet_count = (recentframe_collect_count + self.BYTES_IN_PACKET) % self.bytes_in_frame
                recent_frame[0 : after_packet_count // 2] = packet_data[
                    (self.BYTES_IN_PACKET - after_packet_count) // 2 :
                ]
                recentframe_collect_count = after_packet_count
                lost_packets = False
            else:
                # Accumulate data for current frame
                after_packet_count = (recentframe_collect_count + self.BYTES_IN_PACKET) % self.bytes_in_frame
                recent_frame[
                    recentframe_collect_count // 2 : after_packet_count // 2
                ] = packet_data
                recentframe_collect_count = after_packet_count

            last_packet_num = packet_num

    def _read_data_packet(self) -> Tuple[int, int, np.ndarray, float]:
        """
        Read and parse a single UDP data packet.

        Returns:
            Tuple of (packet_number, byte_count, packet_data, recv_timestamp)
        """
        data, _addr = self.data_socket.recvfrom(self.MAX_PACKET_SIZE)

        # Parse packet header (10 bytes total)
        # Bytes 0-3: packet number (little-endian int32)
        packet_num = struct.unpack("<1l", data[:4])[0]

        # Bytes 4-9: byte count (big-endian, 6 bytes padded to 8)
        byte_count = struct.unpack(">Q", b"\x00\x00" + data[4:10][::-1])[0]

        # Bytes 10+: actual ADC data as uint16
        packet_data = np.frombuffer(data[10:], dtype=np.uint16)

        recv_time = time.time() * 1000.0  # ms timestamp
        return packet_num, byte_count, packet_data, recv_time

    def _close_sockets(self) -> None:
        """Close network sockets."""
        try:
            self.data_socket.close()
            self.config_socket.close()
        except Exception as e:
            logger.warning("CaptureWorker error while closing sockets: %s", e)

