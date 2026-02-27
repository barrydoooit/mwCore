"""
Configuration for UDP Raw Data Reader

This reader captures raw ADC data from IWR6843 via UDP (DCA1000EVM data forwarding mode),
processes it through DSP, and outputs point clouds compatible with all mwCore apps.
"""

reader_cfg = dict(
    type='UdpRawDataReader',
    
    # Network configuration
    static_ip='192.168.33.30',      # IP of this computer
    adc_ip='192.168.33.180',        # IP of radar/DCA1000
    data_port=4098,                 # UDP port for data
    config_port=4096,               # UDP port for config
    
    # Buffer configuration
    buffer_size=1500,               # Circular buffer size (frames)
    
    # DSP processing options
    enable_static_clutter_removal=True,  # Remove static objects
    energy_top_128=True,                 # Use top 128 energy peaks
    range_cut=True,                      # Cut near/far range bins
    
    # Optional: save raw data to .bin file for offline analysis
    # Uncomment and set path to enable recording:
    # save_to_file='captured_data.bin',
)
