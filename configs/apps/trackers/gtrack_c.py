keep_radial = True
tracker_type="GTrackCTracker"
tracker_config = dict(
    DB_RANGE_WEIGHT=0.03,
    DB_Z_WEIGHT=0.4, 
    DB_EPS=0.3,
    DB_MIN_SAMPLES_MIN=35,
    FB_FRAMES_BATCH=2,
    TR_MAX_TRACKS=4,
    STATE_VECTOR_TYPE=3
    # GTRACK_STATE_VECTORS_2DV = 0
    # GTRACK_STATE_VECTORS_2DA = 1
    # GTRACK_STATE_VECTORS_3DV = 2
    # GTRACK_STATE_VECTORS_3DA = 3
)
