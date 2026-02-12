# Offline mmWave DSP (ADC BIN → Point Cloud)

This repo contains an **offline DSP pipeline** to read TI mmWave **raw ADC `.bin`** recordings and turn them into a **3D point cloud** (optionally with velocity/SNR/range).

The pipeline is modular: the reader loads frames from disk and runs each frame through a configurable list of DSP processing modules as defined under `mwcore.signal_processing.processors`.

> **Note / TODO:**  
> The offline ADC `.bin` collector is **planned to be updated soon**. 

---

## Quick start

Run the offline app with the provided config:

```bash
python tools/run_app.py configs/apps/read_adcbin_mmmesh_xWR1843.py