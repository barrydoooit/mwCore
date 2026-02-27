

_base_ = [
    '../__base__/default_runtime.py',
    './trackers/gtrack_a.py',  # Using gtrack_a variant
]

type = 'TrackingApp'

# UDP Raw Data Reader configuration
reader_cfg = dict(
    type='UdpRawDataReader',
    static_ip='192.168.33.30',
    adc_ip='192.168.33.180',
    data_port=4098,
    config_port=4096,
    buffer_size=1500,
    enable_static_clutter_removal=True,
    energy_top_128=True,
    range_cut=True,
)

# Visualization configuration
vis_cfg = dict(
    type='OnlineTrackingVisualizer',
    tracking_mode='bbox',
    receive_polar=False,
    is_on=True
)
