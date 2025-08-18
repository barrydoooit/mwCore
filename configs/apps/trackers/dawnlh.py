tracker_cfg = dict(
    keep_radial = True,
    type="DawnLHTracker",
    do_dev2standard = False,
    tracker_params = dict(
            TR_MAX_TRACKS=2,
            DBSCAN_MinPts=30,
            minObjPoints=10,
            feature_cost_multiplier=10.0,  
            cost_of_non_assignment=30.0,
            age_threshold=4,
            visibility_threshold=0.4,
            measurement_noise=1,
            motion_noise=[1.0, 1.0],
        ),
)
