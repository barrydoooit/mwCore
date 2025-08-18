# Short for  Real-Time Multiple-Human Tracking and Fall Detection. Honestly did not put much thought into the name.
# https://github.com/DarkSZChao/MMWave_Radar_Human_Tracking_and_Fall_detection

tracker_cfg = dict(
    keep_radial = True,
    type="RT_MTFHTracker",
    do_dev2standard = False,
    tracker_params = {
        'global_xlim': (-5.0, 5.0),
        'global_ylim': (-5.0, 5.0),
        'global_zlim': (-5.0, 5.0),
        'es_threshold': 0.1,  # Increase to filter out more low-energy noise
        'obj_bin_number': 2,  # Reduce from 10 to limit max tracks
        'dbscan_eps': 0.5,    # Increase to merge closer points into fewer clusters
        'dbscan_min_samples': 15,  # Increase to require more points per cluster
        'dbscan_sort_limit': 3,   # Limit to only 3 best clusters
        'poss_clus_deque_length': 3,  # Reduce history for faster response
        'redundant_clus_remove_cp_dis': 0.6,  # Increase to remove more redundant clusters
        'dis_diff_threshold': 2,  # Increase to allow more movement tolerance
        'size_diff_threshold': 0.8,  # Increase size tolerance
        'obj_delete_timeout': 0.5,  # Reduce timeout to remove stale tracks faster
        'obj_deque_length': 5,  # Reduce history length
    }
)
