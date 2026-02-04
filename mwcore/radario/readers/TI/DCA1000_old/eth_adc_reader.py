# Code Adapted from {mmMesh}
from typing import Optional, Union
from mwcore.radario.readers.base import BaseReader
from mwcore.registry import READERS
from .configs import AdcConfig
import socket
import struct
import time
import numpy as np
import logging
logger = logging.getLogger(__name__)




from dataclasses import dataclass, field


@READERS.register_module()
class EthAdcReader(BaseReader):
    def __init__(self, config: Union[dict, AdcConfig], static_ip: str, adc_ip: str, 
                 data_port: int, config_port: int):
        
        print(f"Initializing AdcReader: Static IP {static_ip}, ADC IP {adc_ip}")
        # Store config object for frame logic
        self.config = config if isinstance(config, AdcConfig) else AdcConfig(**config) 
        
        # Store network params directly
        self.static_ip = static_ip
        self.adc_ip = adc_ip
        self.data_port = data_port
        self.config_port = config_port

        # Create configuration and data destinations
        self.cfg_dest = (self.adc_ip, self.config_port)
        self.cfg_recv = (self.static_ip, self.config_port)
        self.data_recv = (self.static_ip, self.data_port)

        # Create sockets
        self.config_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        # Bind data socket
        print(f"Binding data socket to {self.data_recv}")
        self.data_socket.bind(self.data_recv)
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**27)
        self.data_socket.settimeout(1.0) 

        # Bind config socket
        self.config_socket.bind(self.cfg_recv)

    def read(self):
        data, addr = self.data_socket.recvfrom(self.config.max_packet_size)
        recv_timestamp = time.time()
        # print(f"Received data packet from {addr} at {recv_timestamp}")
        packet_num = struct.unpack('<1l', data[:4])[0]
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]

        packet_data = np.frombuffer(data[10:], dtype=np.int16)

        return packet_num, byte_count, packet_data, recv_timestamp
        
    def close_sockets(self):
        self.data_socket.close()
        self.config_socket.close()

    @classmethod
    def from_cfg(cls,
                 static_ip: str = '192.168.33.30',
                 adc_ip: str = '192.168.33.180',
                 data_port: int = 4098,
                 config_port: int = 4096,
                 adc_config: Optional[dict] = None) -> 'EthAdcReader':
        
        if adc_config is None:
            adc_config = {}
        if not isinstance(adc_config, AdcConfig):
            adc_config = AdcConfig(**adc_config)

        return cls(adc_config, static_ip, adc_ip, data_port, config_port)