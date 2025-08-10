# Short for  Real-Time Multiple-Human Tracking and Fall Detection. Honestly did not put much thought into the name.
# https://github.com/DarkSZChao/MMWave_Radar_Human_Tracking_and_Fall_detection

tracker_cfg = dict(
    keep_radial = True,
    type="RT_MTFHTracker",
    do_dev2standard = False,
    tracker_params = dict(
            TR_MAX_TRACKS=2,

        ),
)
