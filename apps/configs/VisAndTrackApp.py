_base_ = [
    './trackers/gtrack-A.py',
    './readers/6843base.py',
]
type = 'VisAndTrackApp'
radar_cfg = dict(
    sensor_tilt=0,
    sensor_height=1.6,
)

vis_cfg = dict(
    tracking_mode="dot",  # Options: "dot", "bbox"
)