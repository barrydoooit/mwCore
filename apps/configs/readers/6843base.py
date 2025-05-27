reader_cfg=dict(
    type='BufferedPcdReaderIWR6843',
    # CLI_port='COM10',
    # Data_port='COM11',
    CLI_port='/dev/ttyUSB0',
    Data_port='/dev/ttyUSB1',
    config_file_path='./chirp_configs/6843_tracking.cfg',
    # config_file_path='./chirp_configs/6843_vitalsign.cfg'
    # config_file_path='./chirp_configs/6843_mobile_tracker.cfg',
    # config_file_path='./chirp_configs/test.cfg'
)