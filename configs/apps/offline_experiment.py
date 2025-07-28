_base_ = [
    './trackers/rkf.py',
    './readers/MRIReader.py',
]
type = 'BaseMWOfflineApp'
isExperimentPolar=True
vis_cfg = dict(
    type='OnlineTrackingVisualizer',
    tracking_mode='bbox',  # or 'dot' for dot mode
    receive_polar=isExperimentPolar
)
error_cfg = dict(
    type='ErrorMeasurementThread',
    stats_dir='error_stats',
    polar=isExperimentPolar,
    save_stats=True,  # Whether to save error statistics
    full_metrics=False,  # Whether to calculate and save full metrics
)