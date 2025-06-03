import time
import logging
from typing import Literal, Optional, Tuple, Union
import numpy as np
import serial

log = logging.getLogger(__name__)



UART_MAGIC_WORD = [2, 1, 4, 3, 6, 5, 8, 7]  # bytearray(b'\x02\x01\x04\x03\x06\x05\x08\x07')
MAGIC_LEN = 8
MIN_PACKET_SIZE = MAGIC_LEN + 8

CONVERTER_4BYTES_32BIT = np.array([1, 256, 65536, 16777216], dtype=np.int32)
CONVERTER_2BYTES_16BIT = np.array([1, 256], dtype=np.int16)

def bytes_to_int32(b: np.ndarray) -> int:
    if b.size != 4:
        raise ValueError("Input array must have exactly 4 bytes for int32 conversion.")
    return np.matmul(b.astype(np.int32), CONVERTER_4BYTES_32BIT)

def bytes_to_int16(b: np.ndarray) -> int:
    if b.size != 2:
        raise ValueError("Input array must have exactly 2 bytes for int16 conversion.")
    return np.matmul(b.astype(np.int16), CONVERTER_2BYTES_16BIT)

class BaseTIBufferedReader:
    CLI_BAUDRATE = 115200
    DATA_BAUDRATE = 921600

    def __init__(self, 
                 CLI_port: Union[str, serial.Serial], 
                 Data_port: Union[str, serial.Serial],
                 config_file_path: str,
                 max_buffer_size: int):
        assert isinstance(CLI_port, str) or isinstance(CLI_port, serial.Serial), "CLI_port must be either a string or a serial.Serial object"
        assert isinstance(Data_port, str) or isinstance(Data_port, serial.Serial), "Data_port must be either a string or a serial.Serial object"
        self.CLI_port = serial.Serial(CLI_port, self.CLI_BAUDRATE) if isinstance(CLI_port, str) else CLI_port
        self.Data_port = serial.Serial(Data_port, self.DATA_BAUDRATE) if isinstance(Data_port, str) else Data_port
        self.config_file_path = config_file_path

        self.max_buffer_size = max_buffer_size

        self._byte_buffer = np.zeros(self.max_buffer_size, dtype=np.uint8)
        self.byte_buffer_volume = 0
        self._read_ptr = 0

    @property
    def byte_buffer(self) -> np.ndarray:
        return self._byte_buffer
    
    @property
    def read_ptr(self) -> int:
        return self._read_ptr

    def get_from_buffer(self, 
                         start_from_cur_ptr: int = 0, 
                         length: int = 1, 
                         preview: bool = False,
                         to_int: Literal[False, 16, 32] = False) -> np.ndarray:
        
        start = self.read_ptr + start_from_cur_ptr
        end = start + length
        assert end <= self.byte_buffer_volume, "Requested length exceeds available data in buffer"
        data = self.byte_buffer[start:end]
        if not preview:
            self._read_ptr = end # NOTE: Pointer is moved forward
        if not to_int:
            return data
        if to_int == 16:
            return bytes_to_int16(data)
        if to_int == 32:
            return bytes_to_int32(data)
        log.warning("Invalid to_int value, returning bytes")
        return data
    
    def find_and_consume_magic(self) -> Tuple[bool, int]:
        if self.byte_buffer_volume <=MIN_PACKET_SIZE:
            return False, 0
        
        possible_magic_locs = np.where(
            self.byte_buffer[0: (self.byte_buffer_volume - MAGIC_LEN)] == UART_MAGIC_WORD[0]
        )

        start_idx = None
        for loc in possible_magic_locs[0][::-1]:
            check = self.byte_buffer[loc: loc+MAGIC_LEN]
            if np.all(check == UART_MAGIC_WORD):
                start_idx = loc
                break
        
        if start_idx is None:
            return False, 0
        
        self._read_ptr = start_idx
        self._compact_buffer()

        if self.byte_buffer_volume <= MIN_PACKET_SIZE:
            return False, 0
        
        total_packet_len = self.get_from_buffer(
            start_from_cur_ptr=12, 
            length=4, 
            preview=True, 
            to_int=32
        )

        if total_packet_len > self.byte_buffer_volume:
            # Incomplete packet
            return False, total_packet_len
        
        return True, total_packet_len
    
    def _compact_buffer(self):
        unread_bytes = self.byte_buffer_volume - self.read_ptr
        if unread_bytes > 0:
            self.byte_buffer[:unread_bytes] = self.byte_buffer[self.read_ptr:self.read_ptr + unread_bytes]
        
        self.byte_buffer_volume = unread_bytes
        self.byte_buffer[self.byte_buffer_volume:] = 0
        self._read_ptr = 0
    
    def clear(self):
        self._byte_buffer = np.zeros(self.max_buffer_size, dtype=np.uint8)
        self.byte_buffer_volume = 0
        self._read_ptr = 0
    
    def connect(self):
        loaded_config = [line.rstrip('\r\n') for line in open(self.config_file_path)]
        filtered_config = [(line if line.endswith('\n') else line + '\n') for line in loaded_config if line != '\n' and not line.startswith('%')]
        
        for line in filtered_config:
            self.CLI_port.write(line.encode())
            time.sleep(0.03)
        self.CLI_port.reset_input_buffer()
        log.info("Chirp configurations sent to radar.")

    def read(self):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    def close(self, close_command: Optional[str] = "sensorStop\n"):
        try:
            if close_command:
                log.info(f"Sending close command . . .")
                self.CLI_port.write(close_command.encode())
            log.info("Closing CLI port . . .")
            self.CLI_port.close()
            log.info("Closing Data port . . .")
            self.Data_port.close()
            log.info("Ports closed successfully")
        except Exception as e:
            log.error(f"Error while closing reader: \n {e.with_traceback()}")
