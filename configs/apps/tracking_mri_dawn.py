_base_ = [
    '../__base__/default_runtime.py',
    './trackers/dawnlh.py',
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
        keypoints_involved=[0], # Spine Base, Left shoulder, Right shoulder
        model_name='Dawn-LH',
        dataset_name='mri',
        out_path='exp_data/tracking/positional_error/',
        mode='all',  # 'absolute', 'bias_corrected', 'displacement', or 'all'
    ),
    dict(
        type='LatencyEvaluator',
        model_name='Dawn-LH',
        dataset_name='mri',
        out_path='exp_data/tracking/latency/',
    ),
    dict(
        type='SkelJitterEvaluator',
        keypoints_involved=[0],
        model_name='Dawn-LH',
        dataset_name='mri',
        out_path='exp_data/tracking/jitter_error/',
    )
]