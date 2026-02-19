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

    # 1) Range FFT: ADC samples -> range bins
    dict(type="RangeFFT", window="hann"),

    # 2) Optional TI DC range signature removal (calibDcRangeSig)
    dict(
        type="CalibDcRangeSig",
        enabled=False,
        negative_bin_idx=-5,
        positive_bin_idx=8,
        num_avg_chirps=256,
        subtract_during_estimation=False,
    ),

    # 3) Doppler FFT (2D FFT second dimension): loops -> doppler bins (fftshifted)
    dict(type="DopplerFFT", window="hann", clutter_removal=False),

    # 4) CFAR stage 1: Range direction (procDirection=0)
    #    Note: reads doppler_fft and produces energy_map + range_cfar_mask
    dict(
        type="RangeCFAR",
        averaging_mode="CASO",
        noise_win=8,
        guard_len=4,
        threshold_db=15,
        cyclic=False,
        div_shift=None,   # auto
    ),

    # 5) CFAR stage 2: Doppler direction (procDirection=1)
    #    Note: reads energy_map + range_cfar_mask, produces detected_points
    dict(
        type="DopplerCFAR",
        averaging_mode="CA",
        noise_win=4,
        guard_len=2,
        threshold_db=20.0,
        cyclic=True,
        div_shift=None,  # auto
        peak_grouping=False,
        # optional cfarFovCfg-like filters:
        fov_range_m=(0.0, 11.11),
        fov_doppler_mps=(-2.04, 2.04),
        min_snr_db=None,
    ),

    # 6) AoA (TI-like behaviors: doppler compensation + multi-object + FOV)
    dict(
        type="AoA_TI_DPU",
        num_angle_bins=64,
        points_first=True,
        multi_obj_enable=True,
        multi_obj_thresh=0.5,
        aoa_fov_az_deg=(-90, 90),
        aoa_fov_el_deg=(-90, 90),
    ),
]
