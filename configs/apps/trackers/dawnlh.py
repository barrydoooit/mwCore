tracker_cfg = dict(
    keep_radial = True,
    type="DawnLHTracker",
    do_dev2standard = False,
    tracker_params = dict(
            TR_MAX_TRACKS=2,
            DBSCAN_MinPts=30,
            cost_of_non_assignment=10.0,
            # STATE_VECTOR_TYPE=3,
            # GTRACK_STATE_VECTORS_2DV = 0
            # GTRACK_STATE_VECTORS_2DA = 1
            # GTRACK_STATE_VECTORS_3DV = 2
            # GTRACK_STATE_VECTORS_3DA = 3

            # KF_SPREAD_LIM=[3.5, 3.5, 3.0, 3.0],

            # DB_EPS=1.5,
            # DB_MIN_SAMPLES_MIN=25,
            # TR_VEL_THRES=0,
            # DB_POINTS_THRES=100,
        ),
)
