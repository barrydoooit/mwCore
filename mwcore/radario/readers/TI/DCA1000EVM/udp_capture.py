import socket
import struct
import threading
import logging
import time
import numpy as np
from typing import Tuple, Optional

from mwcore.signal_processing.config import RadarConfig

logger = logging.getLogger(__name__)


class UdpCaptureThread(threading.Thread):
    """
    Thread that captures raw ADC data via UDP packets and assembles frames.
    
    This captures raw IQ data from the radar's DCA1000EVM in data forwarding mode,
    assembling UDP packets into complete frames for processing.
    """
    
    # Network constants
    MAX_PACKET_SIZE = 4096
    BYTES_IN_PACKET = 1456  # Data payload size (excluding 10-byte header)
    
    def __init__(
        self,
        radar_config: RadarConfig,
        static_ip: str = '192.168.33.30',
        adc_ip: str = '192.168.33.180',
        data_port: int = 4098,
        config_port: int = 4096,
        buffer_size: int = 1500
    ):
        """
        Initialize UDP capture thread.
        
        Args:
            radar_config: Radar configuration (defines frame structure)
            static_ip: IP address of this computer
            adc_ip: IP address of the radar/DCA1000
            data_port: UDP port for data packets
            config_port: UDP port for configuration
            buffer_size: Size of circular frame buffer
        """
        threading.Thread.__init__(self)
        self.daemon = True
        
        self.config = radar_config
        self.running = False
        self.recent_cap_num = 0
        self.latest_read_num = 0
        self.next_read_buffer_position = 0
        self.next_cap_buffer_position = 0
        self.buffer_overwritten = True
        self.buffer_size = buffer_size
        
        # Calculate frame size from radar config
        # Frame = chirps * rx * tx * IQ * samples * bytes
        # IQ = 2 (I and Q), bytes = 2 (int16)
        self.bytes_in_frame = (
            self.config.loops_per_frame * 
            self.config.num_rx * 
            self.config.num_tx * 
            2 *  # IQ
            self.config.adc_samples * 
            2   # bytes per sample
        )
        self.uint16_in_frame = self.bytes_in_frame // 2
        
        logger.info(f"Frame size: {self.bytes_in_frame} bytes ({self.uint16_in_frame} uint16 samples)")
        logger.info(f"Packets per frame: {self.bytes_in_frame / self.BYTES_IN_PACKET:.2f}")
        
        # Create network destinations
        self.cfg_dest = (adc_ip, config_port)
        self.cfg_recv = (static_ip, config_port)
        self.data_recv = (static_ip, data_port)
        
        # Create sockets
        self.config_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        
        # Bind data socket
        logger.info(f"Binding data socket to {self.data_recv}")
        self.data_socket.bind(self.data_recv)
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**27)
        
        # Bind config socket
        self.config_socket.bind(self.cfg_recv)
        
        # Initialize circular buffer for frames
        self.buffer_array = np.zeros((self.buffer_size, self.uint16_in_frame), dtype=np.int16)
        self.item_num_array = np.zeros(self.buffer_size, dtype=np.int32)
        self.lost_packet_flags = np.zeros(self.buffer_size, dtype=bool)
        self.frame_start_timestamp_ms_array = np.zeros(self.buffer_size, dtype=np.float64)
        
        logger.info("UDP Capture Thread initialized successfully")
    
    def run(self):
        """Main thread execution - receives and assembles frames."""
        self.running = True
        logger.info("UDP Capture Thread started")
        self._frame_receiver()
    
    def stop(self):
        """Stop the capture thread gracefully."""
        logger.info("Stopping UDP Capture Thread...")
        self.running = False
    
    def _frame_receiver(self):
        """
        Receive UDP packets and assemble them into complete frames.
        
        This implements packet alignment and frame assembly logic.
        """
        # Set timeout for socket operations
        self.data_socket.settimeout(1)
        
        # First capture loop: find the beginning of a frame
        recent_frame = np.zeros(self.uint16_in_frame, dtype=np.int16)
        
        logger.info("Searching for frame boundary...")
        while self.running:
            try:
                packet_num, byte_count, packet_data, packet_ts_ms = self._read_data_packet()
            except socket.timeout:
                logger.debug("Socket timeout while searching for frame boundary")
                continue
            except Exception as e:
                logger.error(f"Error reading packet: {e}")
                continue
                
            after_packet_count = (byte_count + self.BYTES_IN_PACKET) % self.bytes_in_frame
            
            # The recent frame begins in the middle of this packet
            if after_packet_count < self.BYTES_IN_PACKET:
                recent_frame[0:after_packet_count//2] = packet_data[(self.BYTES_IN_PACKET - after_packet_count)//2:]
                self.recent_cap_num = (byte_count + self.BYTES_IN_PACKET) // self.bytes_in_frame
                recentframe_collect_count = after_packet_count
                current_frame_start_timestamp_ms = packet_ts_ms
                current_frame_lost_packet_flag = False
                last_packet_num = packet_num
                logger.info(f"Frame boundary found at packet {packet_num}")
                break
            
            last_packet_num = packet_num
        
        # Main capture loop: assemble complete frames
        logger.info("Entering main frame assembly loop")
        while self.running:
            try:
                packet_num, byte_count, packet_data, packet_ts_ms = self._read_data_packet()
            except socket.timeout:
                logger.debug("Socket timeout in main loop")
                continue
            except Exception as e:
                logger.error(f"Error reading packet: {e}")
                continue
            
            # Check for lost packets
            if last_packet_num < packet_num - 1:
                logger.error(f"PACKET LOST! Expected {last_packet_num + 1}, got {packet_num}")
                logger.error("Data integrity compromised. Please discard this capture session.")
                current_frame_lost_packet_flag = True
            
            # Check if frame is complete with this packet
            if recentframe_collect_count + self.BYTES_IN_PACKET >= self.bytes_in_frame:
                # Complete the frame
                recent_frame[recentframe_collect_count//2:] = packet_data[:(self.bytes_in_frame - recentframe_collect_count)//2]
                self._store_frame(
                    recent_frame,
                    frame_start_timestamp_ms=current_frame_start_timestamp_ms,
                    lost_packet_flag=current_frame_lost_packet_flag
                )
                
                # Update frame number
                self.recent_cap_num = (byte_count + self.BYTES_IN_PACKET) // self.bytes_in_frame
                
                # Start new frame
                recent_frame = np.zeros(self.uint16_in_frame, dtype=np.int16)
                after_packet_count = (recentframe_collect_count + self.BYTES_IN_PACKET) % self.bytes_in_frame
                recent_frame[0:after_packet_count//2] = packet_data[(self.BYTES_IN_PACKET - after_packet_count)//2:]
                recentframe_collect_count = after_packet_count
                current_frame_start_timestamp_ms = packet_ts_ms
                current_frame_lost_packet_flag = False
            else:
                # Accumulate data for current frame
                after_packet_count = (recentframe_collect_count + self.BYTES_IN_PACKET) % self.bytes_in_frame
                recent_frame[recentframe_collect_count//2:after_packet_count//2] = packet_data
                recentframe_collect_count = after_packet_count
            
            last_packet_num = packet_num
        
        logger.info("Frame receiver loop ended")
    
    def get_frame(self, include_timestamp: bool = False):
        """
        Get the next available frame from the buffer.
        
        Returns:
            Tuple of (frame_data, frame_number, lost_packet_flag) by default, and
            (frame_data, frame_number, lost_packet_flag, frame_start_timestamp_ms)
            if include_timestamp=True.
            - frame_data: np.ndarray of int16 raw ADC data, or string error message
            - frame_number: Frame sequence number, or -1 (buffer overwritten) or -2 (no new frame)
            - lost_packet_flag: True if packets were lost in this frame
        """
        # Check for buffer overwrite
        if self.latest_read_num != 0:
            if self.buffer_overwritten:
                logger.warning("Buffer overwritten - frames were lost!")
                if include_timestamp:
                    return "bufferOverWritten", -1, False, None
                return "bufferOverWritten", -1, False
        else:
            self.buffer_overwritten = False
        
        # Check if new frame is available
        if self.next_read_buffer_position == self.next_cap_buffer_position:
            if include_timestamp:
                return "wait new frame", -2, False, None
            return "wait new frame", -2, False
        
        # Read frame from buffer
        read_position = self.next_read_buffer_position
        next_read_position = (self.next_read_buffer_position + 1) % self.buffer_size
        read_frame = self.buffer_array[read_position].copy()
        self.latest_read_num = self.item_num_array[read_position]
        lost_packet_flag = self.lost_packet_flags[read_position]
        frame_start_timestamp_ms = self.frame_start_timestamp_ms_array[read_position]
        self.next_read_buffer_position = next_read_position

        if include_timestamp:
            return read_frame, self.latest_read_num, lost_packet_flag, frame_start_timestamp_ms
        return read_frame, self.latest_read_num, lost_packet_flag
    
    def _store_frame(
        self,
        recent_frame: np.ndarray,
        frame_start_timestamp_ms: Optional[float] = None,
        lost_packet_flag: bool = False
    ):
        """Store a completed frame in the circular buffer."""
        self.buffer_array[self.next_cap_buffer_position] = recent_frame
        self.item_num_array[self.next_cap_buffer_position] = self.recent_cap_num
        self.lost_packet_flags[self.next_cap_buffer_position] = lost_packet_flag
        self.frame_start_timestamp_ms_array[self.next_cap_buffer_position] = (
            0.0 if frame_start_timestamp_ms is None else frame_start_timestamp_ms
        )
        
        # Check if we're about to overwrite unread data
        if (self.next_read_buffer_position - 1 + self.buffer_size) % self.buffer_size == self.next_cap_buffer_position:
            self.buffer_overwritten = True
            logger.warning("Circular buffer full - will overwrite unread frames!")
        
        self.next_cap_buffer_position = (self.next_cap_buffer_position + 1) % self.buffer_size
    
    def _read_data_packet(self) -> Tuple[int, int, np.ndarray, float]:
        """
        Read and parse a single UDP data packet.
        
        Returns:
            Tuple of (packet_number, byte_count, packet_data)
            - packet_number: Sequence number from packet header
            - byte_count: Total bytes received so far
            - packet_data: numpy array of uint16 ADC samples
        """
        data, addr = self.data_socket.recvfrom(self.MAX_PACKET_SIZE)
        packet_timestamp_ms = time.time() * 1000.0
        
        # Parse packet header (10 bytes total)
        # Bytes 0-3: packet number (little-endian int32)
        packet_num = struct.unpack('<1l', data[:4])[0]
        
        # Bytes 4-9: byte count (big-endian, 6 bytes padded to 8)
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        
        # Bytes 10+: actual ADC data as uint16
        packet_data = np.frombuffer(data[10:], dtype=np.uint16)
        
        return packet_num, byte_count, packet_data, packet_timestamp_ms
    
    def close(self):
        """Close network sockets."""
        logger.info("Closing UDP sockets")
        self.data_socket.close()
        self.config_socket.close()
