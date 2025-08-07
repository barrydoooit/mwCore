_base_ = [
    '../__base__/default_runtime.py',
    './trackers/rkf.py',
    './datasets/mRI_walking.py',
]
type = 'TrackingApp'
vis=True # Set to True to enable visualization
reader_cfg=dict(
    type='PoseEstim3DDatasetReader',
    playback_speed=2.0 if vis else None,
)
vis_cfg = dict(
    type='OnlineTrackingVisualizer',
    tracking_mode='bbox',  # or 'dot' for dot mode
    receive_polar=False,
    is_on=vis
)
evaluators = [
    dict(
        type='SkelPositionalErrorEvaluator',
        keypoints_involved=[11, 12, 5, 6], # HipLeft, HipRight, Left shoulder, Right shoulder
        model_name='RKF',
        dataset_name='mRI-walking',
        out_path='exp_data/tracking/positional_error/',
        mode='all',  # 'absolute', 'bias_corrected', 'displacement', or 'all'
    ),
    dict(
        type='SkelJitterEvaluator',
        keypoints_involved=[11, 12, 5, 6],
        model_name='RKF',
        dataset_name='mRI-walking',
        out_path='exp_data/tracking/jitter_error/',
    ),
    dict(
        type='LatencyEvaluator',
        model_name='RKF',
        dataset_name='mRI-walking',
        out_path='exp_data/tracking/latency/',
    )
]