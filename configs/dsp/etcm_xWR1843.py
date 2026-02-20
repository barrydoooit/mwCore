# etcm_xWR1843.py

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
    # keep these (RadarConfig supports them in your current codebase)
    cfar_guard_len=4,
    cfar_noise_len=8,
    cfar_threshold_scale=15.0,
)

dsp_pipeline_cfg = [
    dict(type="FrameReshaper"),

    # 1) Range FFT
    dict(type="RangeFFT", window="hann"),

    # 2) Optional TI DC range signature removal (keep OFF first)
    dict(
        type="CalibDcRangeSig",
        enabled=True,
        negative_bin_idx=-5,
        positive_bin_idx=8,
        num_avg_chirps=256,
        subtract_during_estimation=False,
    ),

    # 3) Optional static clutter removal (mean across loops per range bin)
    dict(type="StaticClutterRemoval", active=True),

    # 4) Doppler FFT (fftshifted doppler axis)
    dict(type="DopplerFFT", window="hamming", clutter_removal=False),

    # 5) ETCM-CFAR (temporal clutter-map CFAR)
    # Fast-start settings so you see points immediately:
    dict(
        type="ETCMCFAR",
        omega=0.001,
        p_fa=1e-6,

        init_mode="mean_first_n",  # boot CM from multiple frames
        init_frames=30,            # ~2.5s at 20 FPS (tune to your frame rate)
        freeze_until_initialized=True,

        # optional post filters (start loose, tighten later)
        min_snr_db=None,            # start here; try 8~12 if still too dense
        peak_grouping=False,
        peak_grouping_size=(3, 3),

        fov_range_m=(0.5, 8.0),
    ),

    # 6) OPTIONAL separate Doppler compensation processor
    # If you enable this, set AoA_TI_DPU.apply_doppler_comp=False below.
    # dict(
    #     type="DopplerCompensation",
    #     enabled=False,
    #     tx_spacing_chirps=1.0,
    #     doppler_fftshifted=True,
    #     reference_tx=0,
    # ),

    # 7) AoA / point cloud
    dict(
        type="AoA_TI_DPU",
        num_angle_bins=64,
        points_first=True,

        # If DopplerCompensation.enabled=True above -> set this False.
        apply_doppler_comp=True,

        # multi-object 2nd peak (TI-style)
        multi_obj_enable=False,
        multi_obj_thresh=0.5,
        multi_obj_exclusion=2,

        # AoA FOV gating
        aoa_fov_az_deg=(-60.0, 60.0),
        aoa_fov_el_deg=(-60.0, 60.0),
    ),
]
