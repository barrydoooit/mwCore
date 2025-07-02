from enum import Enum
import numpy as np
import struct
import logging

import serial

log = logging.getLogger(__name__)

from datetime import datetime, timezone
from mwcore.registry import READERS
from ..base import BaseTIBufferedReader, bytes_to_int16

from typing import TYPE_CHECKING, Union



class TLVTYPES(Enum):
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1

@READERS.register_module()
class BufferedPcdReaderIWR1443(BaseTIBufferedReader):
    def __init__(self, 
                 CLI_port: Union[str, serial.Serial], 
                 Data_port: Union[str, serial.Serial],
                 config_file_path: str,
                 max_buffer_size: int = 2**20):
        super().__init__(CLI_port, Data_port, config_file_path, max_buffer_size)
    
    def parse_standard_frame(self):
                    # Read the header
        frame_data = bytearray(b'')
        magic_bytes = self.get_from_buffer(length=struct.calcsize("Q"))
        frame_data += bytearray(magic_bytes)
        header_bytes = self.get_from_buffer(length=struct.calcsize("7I"))
        frame_data += bytearray(header_bytes)
        (version,
         total_packet_len,
         platform,
         frame_number,
         time_cpu_cycles,
         num_detected_obj,
         num_tlvs) = struct.unpack("<7I",  header_bytes)
        data_ok = 0
        det_obj = {}
        if num_detected_obj > 0:
            tlv_bytes = self.get_from_buffer(length=8)
            tlv_type, tlv_length = struct.unpack('<2I', bytearray(tlv_bytes))
            data_ok, det_obj = self.parse_tlv(tlv_type, tlv_length)
            
        return data_ok, frame_number, det_obj
    
    def parse_tlv(self, tlv_type: int, tlv_length: int):
        data_ok = 0
        det_obj = {}
        if tlv_type == TLVTYPES.MMWDEMO_UART_MSG_DETECTED_POINTS.value:
            ntp_stamp = round(datetime.now(timezone.utc).timestamp() * 1e3)
            tlv_num_obj = bytes_to_int16(self.get_from_buffer(length=2))
            tlv_xyz_q_format = 2 ** self.get_from_buffer(length=2, to_int=16) # NOTE: 1443 exclusive

            range_idx = np.zeros(tlv_num_obj, dtype=np.int16)
            doppler_idx = np.zeros(tlv_num_obj, dtype=np.int16)
            peak_val = np.zeros(tlv_num_obj, dtype=np.int16)
            x = np.zeros(tlv_num_obj, dtype=np.int16)
            y = np.zeros(tlv_num_obj, dtype=np.int16)
            z = np.zeros(tlv_num_obj, dtype=np.int16)

            for object_num in range(tlv_num_obj):
                assign_with_overflow_check(range_idx, object_num, self.get_from_buffer(length=2, to_int=16), label="range_idx")
                assign_with_overflow_check(doppler_idx, object_num, self.get_from_buffer(length=2, to_int=16), label="doppler_idx")
                assign_with_overflow_check(peak_val, object_num, self.get_from_buffer(length=2, to_int=16), label="peak_val")
                assign_with_overflow_check(x, object_num, self.get_from_buffer(length=2, to_int=16), label="x")
                assign_with_overflow_check(y, object_num, self.get_from_buffer(length=2, to_int=16), label="y")
                assign_with_overflow_check(z, object_num, self.get_from_buffer(length=2, to_int=16), label="z")
            
            range_val = range_idx * self.config.config_parameters["rangeIdxToMeters"]
            doppler_idx[doppler_idx > (self.config.config_parameters["numDopplerBins"] / 2 - 1)] = (
                doppler_idx[doppler_idx > (self.config.config_parameters["numDopplerBins"] / 2 - 1)] - 65535
            )
            doppler_val = doppler_idx * self.config.config_parameters["dopplerResolutionMps"]
            x = x / tlv_xyz_q_format
            y = y / tlv_xyz_q_format
            z = z / tlv_xyz_q_format
            
            det_obj = {
                "numObj": tlv_num_obj,
                "rangeIdx": range_idx,
                "range": range_val,
                "dopplerIdx": doppler_idx,
                "doppler": doppler_val,
                "peakVal": peak_val,
                "x": x,
                "y": y,
                "z": z,
                "timestamp": ntp_stamp
            }
            data_ok = 1
        else:
            log.error('Error: Unknown tlv type')
        
        # self._compact_buffer()
        return data_ok, det_obj
    
    def read(self):
        _income_bytes = self.Data_port.read(self.Data_port.in_waiting)
        _byte_vector = np.frombuffer(_income_bytes, dtype=np.uint8)
        _byte_count = len(_byte_vector)
        self.byte_buffer[self.byte_buffer_volume:self.byte_buffer_volume+_byte_count] = _byte_vector
        self.byte_buffer_volume += _byte_count
        
        magic_ok, total_packet_len = self.find_and_consume_magic()
        data_ok = 0
        frame_number = 0
        det_obj = {}
        
        if magic_ok:
            data_ok, frame_number, det_obj = self.parse_standard_frame()
            self._compact_buffer()

        return data_ok, frame_number, det_obj

    def close(self):
        self.CLI_port.write("sensorStop\n".encode())
        self.CLI_port.close()
        self.Data_port.close()
        log.info("Ports closed successfully")
        

def assign_with_overflow_check(array, index, value, label="Unknown"):
    array[index] = value
    if array[index] != value:
        print(f"Overflow detected for [{label}] at index {index} with value {value}:")
        print(f"       -Value: {value} [type: {type(value)}], Assigned: {array[index]}, [type: {type(array[index])}]")