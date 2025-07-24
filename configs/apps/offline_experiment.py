_base_ = [
    './trackers/rkf.py',
    './readers/AReader.py',
]
type = 'BaseMWOfflineApp'
vis_cfg = dict(
    type='OnlineTrackingVisualizer',
    tracking_mode='bbox',  # or 'dot' for dot mode
    receive_polar=True,  # Set to True if tracker outputs polar coordinates
)
