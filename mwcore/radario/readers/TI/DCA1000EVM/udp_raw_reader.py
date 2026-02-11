import time
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import struct

from mwcore.registry import READERS
from mwcore.radario.readers.base import SerialReader
from mwcore.signal_processing.radar_processor import RadarConfig, StandardRadarProcessor
from .udp_capture import UdpCaptureThread

logger = logging.getLogger(__name__)


class _DummyPort:
    """Dummy port object to satisfy SerialReader protocol for OnlineReaderThread compatibility."""
    def close(self):
        pass


@READERS.register_module()
class UdpRawDataReader(SerialReader):
    """
    Reader that captures raw ADC data via UDP and processes it to point clouds.
    
    This reader:
    1. Captures raw ADC data from IWR6843 via UDP (using DCA1000EVM data forwarding)
    2. Processes raw data through StandardRadarProcessor DSP pipeline
    3. Outputs point clouds in the same format as BufferedPcdReaderIWR6843
    4. Optionally saves raw data to .bin file for offline analysis
    
    Example:
        reader = UdpRawDataReader(save_to_file='capture.bin')
        reader.connect()
        while True:
            data_ok, frame_num, det_obj = reader.read()
            if data_ok:
                print(f"Frame {frame_num}: {det_obj['numObj']} points")
    """
    
    def __init__(
        self,
        static_ip: str = '192.168.33.30',
        adc_ip: str = '192.168.33.180',
        data_port: int = 4098,
        config_port: int = 4096,
        buffer_size: int = 1500,
        radar_config: Optional[RadarConfig] = None,
        save_to_file: Optional[str] = None,
        process_point_cloud: bool = False,
        enable_static_clutter_removal: bool = True,
        energy_top_128: bool = True,
        range_cut: bool = True
    ):
        """
        Initialize UDP raw data reader.
        
        Args:
            static_ip: IP address of this computer
            adc_ip: IP address of the radar/DCA1000
            data_port: UDP port for data packets
            config_port: UDP port for configuration
            buffer_size: Size of circular frame buffer
            radar_config: Radar configuration (uses defaults if None)
            save_to_file: Optional path to save raw .bin file
            enable_static_clutter_removal: Enable clutter removal in DSP
            energy_top_128: Use top 128 energy peaks for CFAR
            range_cut: Cut near and far range bins
        """
        self.static_ip = static_ip
        self.adc_ip = adc_ip
        self.data_port = data_port
        self.config_port = config_port
        self.buffer_size = buffer_size
        
        # Add dummy serial port attributes for SerialReader protocol compatibility
        # (OnlineReaderThread expects these when terminating)
        self.Data_port = _DummyPort()
        self.CLI_port = _DummyPort()
        
        # Use provided config or create default
        self.config = radar_config if radar_config is not None else RadarConfig()
        
        # Initialize capture thread
        self.capture_thread = UdpCaptureThread(
            radar_config=self.config,
            static_ip=static_ip,
            adc_ip=adc_ip,
            data_port=data_port,
            config_port=config_port,
            buffer_size=buffer_size
        )
        
        # Initialize DSP processor
        self.processor = StandardRadarProcessor(
            config=self.config,
            enable_static_clutter_removal=enable_static_clutter_removal,
            energy_top_128=energy_top_128,
            range_cut=range_cut
        )
        
        # Optional .bin file recording
        self.save_to_file = save_to_file
        self.file_handle = None
        if save_to_file:
            logger.info(f"Will save raw data to: {save_to_file}")
            self.file_handle = open(save_to_file, 'wb')
        
        self.frame_count = 0
        self.process_point_cloud = process_point_cloud
        logger.info("UdpRawDataReader initialized successfully")
    
    def connect(self):
        """Start the UDP capture thread."""
        logger.info("Starting UDP capture thread...")
        self.capture_thread.start()
        logger.info("UDP capture thread started - waiting for radar data")
    
    def read(self) -> Tuple[int, int, Dict[str, Any]]:
        """
        Read and process one frame from the UDP stream.
        
        Returns:
            Tuple of (data_ok, frame_number, det_obj)
            - data_ok: 1 if valid data, 0 if no data or error
            - frame_number: Frame sequence number
            - det_obj: Dictionary with keys:
                - 'x': np.ndarray of x coordinates
                - 'y': np.ndarray of y coordinates
                - 'z': np.ndarray of z coordinates
                - 'doppler': np.ndarray of doppler velocities
                - 'peakVal': np.ndarray of signal strengths
                - 'numObj': int number of detected points
                - 'timestamp': float timestamp in milliseconds
        """
        # Get raw frame from capture thread
        frame_data, frame_num, lost_packet_flag = self.capture_thread.get_frame()
        timestamp = time.time()
        
        # Handle special return codes
        if frame_num == -1:
            # Buffer overwritten
            logger.warning(f"Buffer overwritten: {frame_data}")
            return 0, 0, {}
        
        if frame_num == -2:
            # No new frame available yet
            return 0, 0, {}
        
        # Handle lost packets
        if lost_packet_flag:
            logger.warning(f"Frame {frame_num} had lost packets - skipping")
            return 0, frame_num, {}
        
        # Validate frame data
        if not isinstance(frame_data, np.ndarray):
            logger.error(f"Invalid frame data type: {type(frame_data)}")
            return 0, frame_num, {}
        
        # Optionally save raw data to .bin file
        if self.file_handle is not None:
            try:
                self.file_handle.write(struct.pack('d', float(timestamp)))
                self.file_handle.write(frame_data.tobytes())
                self.file_handle.flush()
            except Exception as e:
                logger.error(f"Error writing to .bin file: {e}")
        
        if self.process_point_cloud:
            # Process through DSP pipeline
            try:
                result = self.processor.process(frame_data)
                point_cloud = result['point_cloud']
            except Exception as e:
                logger.error(f"Error processing frame {frame_num}: {e}")
                return 0, frame_num, {}
            
            # Convert point cloud to det_obj format
            # point_cloud shape is (6, N) where rows are [x, y, z, doppler, energy, range]
            if point_cloud.size == 0 or point_cloud.shape[1] == 0:
                assert False, "No points detected"
            
            # Extract point cloud components
            # point_cloud format: [x, y, z, doppler, energy, range]
            num_points = point_cloud.shape[1]
            
            det_obj = {
                'numObj': num_points,
                'x': point_cloud[0, :],      # x coordinates
                'y': point_cloud[1, :],      # y coordinates
                'z': point_cloud[2, :],      # z coordinates
                'doppler': point_cloud[3, :],  # doppler velocities
                'peakVal': point_cloud[4, :],  # energy/SNR values
                'timestamp': time.time() * 1000
            }
            
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                logger.info(f"Processed {self.frame_count} frames (latest: {num_points} points)")
            
            return 1, frame_num, det_obj
        else:
            # return dummy point cloud
            det_obj = {
                'numObj': 0,
                'x': np.array([]),
                'y': np.array([]),
                'z': np.array([]),
                'doppler': np.array([]),
                'peakVal': np.array([]),
                'timestamp': timestamp
            }
            return 1, frame_num, det_obj
    
    def close(self):
        """Stop capture thread and close resources."""
        logger.info("Closing UdpRawDataReader...")
        
        # Stop capture thread
        if hasattr(self, 'capture_thread'):
            self.capture_thread.stop()
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                logger.warning("Capture thread did not terminate gracefully")
            self.capture_thread.close()
        
        # Close file handle
        if self.file_handle is not None:
            logger.info(f"Closing .bin file (wrote {self.frame_count} frames)")
            self.file_handle.close()
            self.file_handle = None
        
        logger.info("UdpRawDataReader closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
