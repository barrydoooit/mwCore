# Raw Radar Signal Processing Integration

This document details the integration of the reading project's raw radar signal processing logic into `mwCore`.

## Objective
To allow `mwCore` to process raw `.bin` radar files (ADC samples) directly, acting as a "Virtual Sensor" that outputs Point Clouds equivalent to hardware processing.

## Changes Implemented

### 1. New Modules
*   **`mwcore.signal_processing`**:
    *   Ported the mathematical core from the previous reading project.
    *   Contains `RadarConfig`, `RadarBinFileReader`, and `StandardRadarProcessor`.
    *   Location: `mwcore/signal_processing/radar_processor.py`
    
*   **`mwcore.radario.readers.offlineReaders.RawBinReader`**:
    *   New adapter class implementing `BaseReader`.
    *   Wraps the signal processing logic to expose it as a standard `mwCore` reader.
    *   Registered via `@READERS.register_module()`.

### 2. Dependency Updates
To support both the new logic and existing `mwCore` features, the following dependencies were added or updated in the environment:
*   **NumPy**: Relaxed constraint to `numpy>=1.26` (to support Python 3.13 where `numpy==1.26` was unavailable).
*   **Added**: `mmengine`, `torch`, `h5py`, `tqdm`, `PyOpenGL`.

## Patches & Difficulties Resolved

### NumPy 2.0 Compatibility
*   **Issue**: The original legacy code used `np.complex` (deprecated/removed in NumPy 1.24+ and 2.0).
*   **Fix**: Patched legacy verification scripts to use Python's built-in `complex` or `np.complex64`.

### Dependency Management
*   **Issue**: `mwCore`'s `pyproject.toml` had strict version pinning (`numpy==1.26`) that conflicted with the available wheels for the user's Python 3.13 environment on macOS ARM64.
*   **Fix**: Relaxed to `numpy>=1.26`.
*   **Issue**: Several imports in `mwCore` (e.g., `mmengine`, `torch`) were missing from the installed environment despite being used in `registry.py` and other files.
*   **Fix**: Manually installed missing dependencies to ensure the integration environment is stable.

## Usage Guide

### 1. Setup
Make sure you are in the `feature/raw-signal-processing` branch.

```bash
# Verify branch
git branch 
# Should show * feature/raw-signal-processing

# Create/Activate Virtual Env
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
pip install mmengine torch h5py tqdm PyOpenGL
```

### 2. Running Verification
We provided a verification script that compares the output of the new `mwCore` integration against the original legacy code.

```bash
# Run math verification (Bit-exact check)
python3 tests/verify_mwcore_math.py
```

*Expected Output:*
```text
Verifying mwCore vs Legacy...
Frame 0 PASS: Max diff 0.0
...
Total: 5 PASS, 0 FAIL
```

### 3. Using the Reader
You can use the new reader in your configs just like any other reader:

```python
from mwcore.registry import READERS

reader = READERS.build(dict(
    type='RawBinReader',
    file_path='/path/to/data.bin'
))

point_cloud = reader.read()
```
