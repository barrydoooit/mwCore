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
)

dsp_pipeline_cfg = [
    dict(type='FrameReshaper'),
    dict(type='RangeFFT', window="hamming"),
    dict(type='StaticClutterRemoval', active=True),
    dict(type='DopplerFFT', window="hamming", clutter_removal=False),
    dict(
        type='TopKDetector', 
        top_k=128, 
        range_cut_idx=(25, 125)
    ),
    # dict(
    #     type='NaiveAoA', 
    #     fft_size=64
    # ),
    dict(
        type="AoA_TI_DPU",
        num_angle_bins=64,
        points_first=True,
        azimuth_tx_indices=(0, 1),
        elevation_tx_index=2,
        multi_obj_enable=False,
        aoa_fov_az_deg=(-90, 90),
        aoa_fov_el_deg=(-35, 35),
    )
]