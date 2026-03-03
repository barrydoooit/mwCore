_base_ = ['../dsp/ti-mobile-tracker_xWR1843.py']
type = 'BaseMWApp'

# --------------------------------
thread_cfg = dict(
    type='RawAdcUdpFlowReaderWorker',
    reader=dict(
        type='RawAdcUdpFlowReader',
        radar_cfg=_base_.mmwave_radar_cfg,
        pipeline=dict(
            type='DspPipeline', 
            radar_cfg=_base_.mmwave_radar_cfg,
            pipeline_cfg=_base_.dsp_pipeline_cfg
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



experiment_name="adc_read_session",
