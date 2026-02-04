_base_ = [
    './readers/6843_udp_raw.py',
]

type = 'BaseMWOnlineApp'

# Enable point cloud visualization
vis_cfg = dict(
    type='OnlinePointCloudVisualizer',
)

reader_cfg = dict(
    type='UdpRawDataReader',
    enable_static_clutter_removal=False,  # ← Disable (saves ~30% CPU)
    energy_top_128=False,                 # ← Disable (saves ~10% CPU)
    range_cut=False,                      # ← Disable (minimal impact)
)