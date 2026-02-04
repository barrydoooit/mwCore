import logging
logger = logging.getLogger(__name__)




from dataclasses import dataclass, field

@dataclass
class AdcConfig:
    """
    Holds all static and dynamic configuration for the ADC capture.
    Derived values are calculated automatically after initialization.
    """
    
    # --- Independent ADC Parameters ---
    chirps_per_frame: int = 128
    rx_antennas: int = 4
    tx_antennas: int = 3  # Active TX antennas in the frame
    samples_per_chirp: int = 256
    iq_format: int = 2  # 2 for complex (I/Q)
    bytes_per_sample_part: int = 2  # 2 for 16-bit

    # --- Independent Network Parameters (Data payload size) ---
    max_packet_size: int = 4096
    # This is the DATA payload size, excluding the 10-byte DCA1000 header
    data_bytes_per_packet: int = 1456 
    
    # --- Derived Parameters (calculated automatically) ---
    bytes_in_frame: int = field(init=False)
    uint16_in_frame: int = field(init=False)

    def __post_init__(self):
        """Calculate derived properties based on the independent params."""
        self.bytes_in_frame = (self.chirps_per_frame * self.rx_antennas * self.tx_antennas * self.iq_format * self.samples_per_chirp * self.bytes_per_sample_part)
        
        self.uint16_in_frame = self.bytes_in_frame // 2

        logger.debug("--- ADC Config Initialized ---")
        logger.debug(f"  Frame Size: {self.bytes_in_frame} bytes")
        logger.debug(f"  uint16 Cnts: {self.uint16_in_frame} samples")
        logger.debug(f"  Packet Size: {self.data_bytes_per_packet} data bytes")
        logger.debug(f"  Packets/Frame: {self.bytes_in_frame / self.data_bytes_per_packet:.2f}")
        logger.debug("------------------------------")

@dataclass
class ProcessConfig:
    # --- Base ADC Parameters (for data structure) ---
    chirps_per_frame: int = 128  # LOOPS_PER_FRAME
    rx_antennas: int = 4         # NUM_RX
    tx_antennas: int = 3         # NUM_TX
    samples_per_chirp: int = 256 # ADC_SAMPLES
    
    # --- Physics Parameters (for calculation) ---
    start_freq_ghz: float = 77.0     # START_FREQ
    freq_slope_mhz_us: float = 60.012 # FREQ_SLOPE
    sample_rate_ksps: float = 4400.0  # SAMPLE_RATE
    idle_time_us: float = 7.0         # IDLE_TIME
    ramp_end_time_us: float = 65.0    # RAMP_END_TIME
    
    # --- Processing Parameters ---
    mmwave_radar_loc: tuple = (0.146517, -3.030810, 1.0371905)
    num_angle_bins: int = 64 # Hardcoded in pc_generation.py
    
    # --- Derived Parameters (calculated automatically) ---
    num_doppler_bins: int = field(init=False)
    num_range_bins: int = field(init=False)
    range_resolution_m: float = field(init=False)
    max_range_m: float = field(init=False)
    doppler_resolution_mps: float = field(init=False)
    max_doppler_mps: float = field(init=False)
    
    # Internal constant
    _SPEED_OF_LIGHT: float = 3e8

    enable_static_clutter_removal: bool = True
    energy_top_128: bool = True
    range_cut: bool = True

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> 'ProcessConfig':
        if isinstance(cfg_dict, ProcessConfig):
            return cfg_dict
        return cls(**cfg_dict)
    
    def __post_init__(self):
        """Calculate derived properties based on the independent params."""
        
        # Bin calculations
        self.num_doppler_bins = self.chirps_per_frame
        self.num_range_bins = self.samples_per_chirp

        # Physics calculations
        c = self._SPEED_OF_LIGHT
        sample_rate_hz = self.sample_rate_ksps * 1e3
        freq_slope_hz_s = self.freq_slope_mhz_us * 1e12
        
        self.range_resolution_m = (c * sample_rate_hz) / (2 * freq_slope_hz_s * self.samples_per_chirp)
        self.max_range_m = (c * sample_rate_hz) / (2 * freq_slope_hz_s)
        
        chirp_time_s = (self.idle_time_us + self.ramp_end_time_us) * 1e-6
        self.doppler_resolution_mps = c / (2 * self.start_freq_ghz * 1e9 * chirp_time_s * self.num_doppler_bins * self.tx_antennas)
        self.max_doppler_mps = c / (4 * self.start_freq_ghz * 1e9 * chirp_time_s * self.tx_antennas)

        logger.debug(f"--- Process Config Initialized ---")
        logger.debug(f"  Range Res: {self.range_resolution_m:.3f} m")
        logger.debug(f"  Doppler Res: {self.doppler_resolution_mps:.3f} m/s")