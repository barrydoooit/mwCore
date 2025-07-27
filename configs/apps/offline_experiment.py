_base_ = [
    './trackers/rkf.py',
    './readers/AReader.py',
]
type = 'BaseMWOfflineApp'
polar=True
vis_cfg = dict(
    type='OnlineTrackingVisualizer',
    tracking_mode='bbox',  # or 'dot' for dot mode
    receive_polar=polar
)
error_cfg = dict(
    type='ErrorMeasurementThread',
    stats_dir='error_stats',
    polar=polar,
    save_stats=True,  # Whether to save error statistics
    full_metrics=False,  # Whether to calculate and save full metrics
)