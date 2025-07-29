pipeline = [
    dict(
        type='LoadMultiFrame3DPoseEstimDatasetFromH5',
        load_pcd_dim=5,
        num_frames=1,
        backup_frames=0,
        empty_frame_op='error',
    ),
    dict(
        type='NormalizePointAttr',
        attr_indices=(3, 4,),
        means=(0.0, 28.98583),
        stds=(0.45029, 35.79703)
    ),
]
data_root = './data/mri'
data_prefix = dict(
    pcd='mmwave',
    skel='skeleton',
)
info_file = 'info_all.pkl'
dataloader = dict(
    batch_size=1,
    num_workers=1,
    shuffle=False,
    dataset=dict(
        type='PoseEstim3DDataset',
        data_root=f"{data_root}",
        info_path=f"{data_root}/{info_file}",
        data_prefix=data_prefix,
        pipeline=pipeline,
        sequence_length=1,
        allow_pad_sequence=False
    )
)