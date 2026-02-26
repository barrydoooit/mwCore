mmwave_radar_cfg = dict(
    mode="3D",
    num_tx=3,
    num_rx=4,
    loops_per_frame=128,
    adc_samples=256,
    start_freq_ghz=77,
    freq_slope_mhz_us=60.012,
    sample_rate_ksps=4400,
    idle_time_us=7,
    ramp_end_time_us=65,
    range_bias_m=0.0,
)

dsp_pipeline_cfg = [
    dict(type="FrameReshaper"),
    dict(type="RangeFFT", window="hann", fft_size=256),

    dict(type="VirtualAntennaAssembler", tx_permutation=(0, 2, 1)),

    dict(
        type="DynamicCaponRAHeatmapWM",
        angle_search_step_deg=0.75,
        diag_loading=0.0010,
        fov_az_deg=(-60.0, 60.0),
        virt_ant_idx_az=(0,1,2,3,8,9,10,11),
        ant_geom_m=(-2,-1,0,1,  0,1,2,3,  2,3,4,5),
    ),

    dict(
        type="DynamicRACfar2PassWM",
        cfar_discard_left_range=4,
        cfar_discard_right_range=4,
        cfar_discard_left_angle=2,
        cfar_discard_right_angle=2,
        ref_win_size_range=8,
        guard_win_size_range=2,
        ref_win_size_angle=12,
        guard_win_size_angle=8,
        range_thre=2.5,
        angle_thre=4.0,
        sidelobe_thre=0.20,
        enable_2nd_pass=1,
        dynamic_flag=1,
    ),

    dict(
        type="DynamicCaponElevDopplerWM",
        elev_search_step_deg=1.5,
        diag_loading=0.0300,
        fov_el_deg=(-30.0, 30.0),
        ant_geom_m=(-2,-1,0,1,  0,1,2,3,  2,3,4,5),
        ant_geom_n=(0,0,0,0,  1,1,1,1,  0,0,0,0),
    ),
]
