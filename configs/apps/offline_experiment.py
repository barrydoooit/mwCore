_base_ = [
    './trackers/gtrack_a.py',
    './readers/AReader.py',
]
type = 'BaseMWOfflineApp'
vis_cfg = dict(
    type='OnlineTrackingVisualizer',
    tracking_mode='bbox',  # or 'dot' for dot mode
    receive_polar=False,  # Set to True if tracker outputs polar coordinates
)
