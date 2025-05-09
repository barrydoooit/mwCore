import time
from typing import Union
import serial

from radario.base import BaseBufferedReader


class ChirpConfigIWR1443:
    def __init__(self, config_file_path: str, CLI_port: Union[str, serial.Serial]):
        assert isinstance(CLI_port, str) or isinstance(CLI_port, serial.Serial), "CLI_port must be either a string or a serial.Serial object"
        if isinstance(CLI_port, str):
            self.CLI_port = serial.Serial(CLI_port, BaseBufferedReader.CLI_BAUDRATE)
        else:
            self.CLI_port = CLI_port
        self.config_file_path = config_file_path
        self._config_parameters = self._parse_config_file()
    
    @property
    def config_parameters(self):
        assert self._config_parameters is not None, "Config parameters have not been parsed yet"
        return self._config_parameters
    
    def _parse_config_file(self):
        config_parameters = (
            {}
        )
        config = self._read_config()
        for i in config:
            split_words = i.split(' ')
            num_rx_antennas = 4
            num_tx_antennas = 3

            if "profileCfg" in split_words[0]:
                start_freq = int(float(split_words[2]))
                idle_time = int(float(split_words[3]))
                ramp_end_time = int(float(split_words[5]))
                freq_slope_const = int(float(split_words[8]))
                num_adc_samples = int(float(split_words[10]))
                num_adc_samples_round_to2 = 1

                while num_adc_samples > num_adc_samples_round_to2:
                    num_adc_samples_round_to2 *= 2
                
                dig_out_sample_rate = int(float(split_words[11]))
            
            elif "frameCfg" in split_words[0]:
                chirp_start_idx = int(split_words[1])
                chirp_end_idx = int(split_words[2])
                num_loops = int(split_words[3])
                num_frames = int(split_words[4])
                self.frame_periodicity = int(float(split_words[5]))
        
        num_chirps_per_frame = (chirp_end_idx - chirp_start_idx + 1) * num_loops
        config_parameters["numDopplerBins"] = num_chirps_per_frame / num_tx_antennas
        config_parameters["numRangeBins"] = num_adc_samples_round_to2
        config_parameters["rangeResolutionMeters"] = 3e8 * dig_out_sample_rate * 1e3 / (2 * freq_slope_const * 1e12 * num_adc_samples)
        config_parameters["rangeIdxToMeters"] = 3e8 * dig_out_sample_rate * 1e3 / (2 * freq_slope_const * 1e12 * config_parameters["numRangeBins"])
        config_parameters["dopplerResolutionMps"] = 3e8 / (2 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 * config_parameters["numDopplerBins"] * num_tx_antennas)
        config_parameters["maxRange"] = 300 * 0.9 * dig_out_sample_rate / (2 * freq_slope_const * 1e3)
        config_parameters["maxVelocity"] = 3e8 / (4 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 * num_tx_antennas)

        return config_parameters

    def _read_config(self) -> list[str]:
        return [
            line.rstrip('\r\n') for line in open(self.config_file_path)
        ]
    
    def send_config(self, close_port: bool = True):
        config = self._read_config()
        for i in config:
            self.CLI_port.write((i + "\n").encode())
            print(i)
            time.sleep(0.01)
        if close_port:
            self.CLI_port.close()