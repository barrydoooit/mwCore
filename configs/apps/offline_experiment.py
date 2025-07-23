_base_ = [
    './trackers/gtrack_k.py',
    './readers/AReader.py',
]
type = 'BaseMWOfflineApp'
vis_cfg = dict(
    type='OnlineTrackingVisualizer',
    tracking_mode='bbox',  # or 'dot' for dot mode
)
