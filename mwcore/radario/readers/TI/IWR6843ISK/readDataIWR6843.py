from enum import Enum
import time
import numpy as np
import struct
import logging

import serial

from .parseTLVs6843 import TLVTYPES, tlv2parser
from mwcore.registry import READERS
log = logging.getLogger(__name__)
from datetime import datetime, timezone
from ..base import BaseTIBufferedReader

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from . import ChirpConfigIWR1443


@READERS.register_module()
class BufferedPcdReaderIWR6843(BaseTIBufferedReader):
    MAGIC_STRUCT = "Q"
    HEADER_STRUCT = "8I"
    
    def __init__(self, 
                 CLI_port: Union[str, serial.Serial], 
                 Data_port: Union[str, serial.Serial],
                 config_file_path: str,
                 max_buffer_size: int = 2**20):
        super().__init__(CLI_port, Data_port, config_file_path, max_buffer_size)
    
    def register_config(self, config: "ChirpConfigIWR1443"):
        self._config = config

    @property
    def config(self) -> "ChirpConfigIWR1443":
        assert self._config is not None, "Config has not been registered yet"
        return self._config
    
    def parse_standard_frame(self):
        frame_data = bytearray(b'')
        magic_bytes = self.get_from_buffer(length=struct.calcsize("Q"))
        frame_data += bytearray(magic_bytes)
        header_bytes = self.get_from_buffer(length=struct.calcsize(self.HEADER_STRUCT))
        frame_data += bytearray(header_bytes)
        
        output_dict = {}
        try:
            (version,
             total_packet_len,
             platform,
             frame_number,
             time_cpu_cycles,
             num_detected_obj,
             num_tlvs,
             subframe_num) = struct.unpack(self.HEADER_STRUCT, header_bytes)
            #  subframe_num) = struct.unpack(header_struct, header_data)
            output_dict['error'] = 0
        except:
            log.error('Error: Could not read frame header')
            output_dict['error'] = 1

        output_dict['frame_num'] = frame_number
        data_ok = 0
        det_obj = {}
        if num_detected_obj > 0:
            # print(f"Num detected objects: {num_detected_obj}")
            output_dict['pointCloud'] = np.zeros((num_detected_obj, 7), dtype=np.float64)
            output_dict['pointCloud'][:, 6] = 255
            for i in range(num_tlvs):
                tlv_bytes = self.get_from_buffer(length=8)
                tlv_type, tlv_length = struct.unpack('2I', bytearray(tlv_bytes))
                this_data_ok = self.parse_tlv(tlv_type, 
                                                  tlv_length,
                                                  output_dict)
                data_ok = data_ok or this_data_ok
        if data_ok:
            det_obj = {
                "numObj": output_dict['numDetectedPoints'],
                # "rangeIdx": range_idx,
                # "range": output_dict['pointCloud'][:, 0],
                # "dopplerIdx": doppler_idx,
                "doppler": output_dict['pointCloud'][:, 3],
                "peakVal": output_dict['pointCloud'][:, 4], # NOTE: This is actually SNR in the case of IWR6843
                "x": output_dict['pointCloud'][:, 0],
                "y": output_dict['pointCloud'][:, 1],
                "z": output_dict['pointCloud'][:, 2],
                "timestamp": time.time() * 1000
            }
            # print(f"x, y, z: {det_obj['x'][0]}, {det_obj['y'][0]}, {det_obj['z'][0]}, {det_obj['doppler'][0]}, {det_obj['peakVal'][0]}")
        return data_ok, frame_number, det_obj
    
    def parse_tlv(self, tlv_type: int, tlv_length: int, output_dict: dict):
        data_ok = 0
        # print(f"TLV type: {tlv_type}")
        
        tlv_data = self.get_from_buffer(length=tlv_length)
        
        parse_func = tlv2parser(tlv_type)
        if parse_func is not None:
            parse_func(tlv_data, tlv_length, output_dict)
            data_ok = 1
        else:
            pass
            # print(f"TLV type {tlv_type} not supported")
        
        self._compact_buffer()
        return data_ok
    
    def read(self):
        in_waiting = self.Data_port.in_waiting
        # print(f"Bytes in waiting: {in_waiting}")
        _income_bytes = self.Data_port.read(in_waiting)
        if in_waiting >= self.max_buffer_size:
            raise RuntimeError("Reading Buffer overflow")
            # print("Warning! Buffer overflow")
            # self.byte_buffer[:] = np.frombuffer(_income_bytes[-self.max_buffer_size:], dtype=np.uint8)
            # self.byte_buffer_volume = self.max_buffer_size
            # self._read_ptr = 0
        else:
            potential_volume = self.byte_buffer_volume + in_waiting
            if potential_volume > self.max_buffer_size:
                excess_bytes = potential_volume - self.max_buffer_size
                self.byte_buffer[:-excess_bytes] = self.byte_buffer[excess_bytes:]
                self.byte_buffer[-in_waiting:] = np.frombuffer(_income_bytes, dtype=np.uint8)
                self.byte_buffer_volume = self.max_buffer_size
                self._read_ptr = max(0, self._read_ptr - excess_bytes)
            else:
                _byte_vector = np.frombuffer(_income_bytes, dtype=np.uint8)
                self.byte_buffer[self.byte_buffer_volume:self.byte_buffer_volume+in_waiting] = _byte_vector
                self.byte_buffer_volume += in_waiting
        
        magic_ok, total_packet_len = self.find_and_consume_magic()
        data_ok = 0
        frame_number = 0
        det_obj = {}
        if magic_ok:
            data_ok, frame_number, det_obj = self.parse_standard_frame()
            self._compact_buffer()
        # self.clear()
        return data_ok, frame_number, det_obj

    def close(self):
        self.CLI_port.close()
        self.Data_port.close()
        log.info("Ports closed")
        