type = 'BaseMWApp'

# --------------------------------
thread_cfg = dict(
    type='OfflineAdcDataReaderWorker',  # This calls the factory in offline_adcbin_reader.py
    playback_speed=1.0,                 # Real-time speed
    reader=dict(
        type='OfflineAdcDataReader',    # The class in adcbin_reader.py
        data_dir='tests',
        file_pattern=r".*\.bin",
        frame_rate=2.0,                # Important for timing calculation
        pipeline=dict(                  # Inject the DSP pipeline
            type='DspPipeline', 
            radar_config= dict(
                mode="3D",
                num_tx=3, num_rx=4, loops_per_frame=128, adc_samples=256,
                start_freq_ghz=77, freq_slope_mhz_us=60.012, sample_rate_ksps=4400,
                idle_time_us=7, ramp_end_time_us=65, cfar_threshold_scale=15.0
            ), 
            pipeline_cfg=[
                dict(type='FrameReshaper'),
                dict(type='RangeFFT'),
                dict(type='StaticClutterRemoval', active=True),
                dict(type='DopplerFFT', clutter_removal=False),
                dict(type='TopKDetector', top_k=128, range_cut_idx=(25, 125)),
                dict(type='NaiveAoA', fft_size=64)
            ]
        )
    )
)

# 3. Visualizer Configuration
# ---------------------------
vis_cfg = dict(
    type='PointCloudOfflineVisualizerV2',
    window_size=(1200, 800),
    max_buffer_frames=2000,             # How many frames to keep in memory
)

# 4. Signal/Slot Connections
# --------------------------
connections = [
    # Worker sends data to Visualizer
    dict(signal='reader_worker.frame_signal', slot='visualizer.receive_frame'),
    
    # Worker sends file status to Visualizer
    dict(signal='reader_worker.progress_signal', slot='visualizer.update_status'),
    
    # Visualizer controls Worker speed (Fast forward when buffering)
    dict(signal='visualizer.request_fast_forward', slot='reader_worker.set_fast_forward'),
]



experiment_name="adc_replay_session",
