_base_ = [
    '../__base__/default_runtime.py',
    './trackers/rkf.py',
    './datasets/mRI_walking.py',
]
type = 'TrackingApp'
isExperimentPolar=True
vis=True
reader_cfg=dict(
    type='PoseEstim3DDatasetReader',
    playback_speed=2.0 if vis else None,
)
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