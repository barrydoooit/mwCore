_base_ = [
    './trackers/gtrack-A.py',
    './readers/6843base.py',
]
type = 'vis_and_track'
radar_cfg = dict(
    sensor_tilt=0,
    sensor_height=1.6,
)