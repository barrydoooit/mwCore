# Installation Guide

## Prerequisites

- Ubuntu / Windows supported
- Python **3.10–3.12** (recommended)
- git
- uv

> Note: PyTorch wheels are published per Python version and platform. You must use a PyTorch version that provides **cp3xx** wheels for your selected CUDA build.

## Install uv

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.cargo/env"
uv --version
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

## Clone repository

```bash
git clone https://github.com/barrydoooit/mwCore.git
cd mwCore
```

## Install

`uv sync` will create/update `.venv` in the repo using `uv.lock`.

### Step 1: Create the environment

Pick **one** PyTorch variant (extras are mutually exclusive):

- `cpu`   → CPU-only wheels
- `cu{xx}{y}` → CUDA xx.y  wheels

For example, when installing for CUDA 12.4:
#### Linux and Windows (PowerShell)

```bash
uv sync --extra cu124
```

### Step 2: Activate the environment

#### Linux

```bash
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install mwcore (editable)

From inside the `mwCore` repo:

```bash
uv pip install -e .
```

### Step 4: Quick import test

```bash
python -c "import mwcore; print('OK')"
```

### Step 5: Verify PyTorch + CUDA (if using a CUDA extra)

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
```

## Python version notes (important)

### Recommended: Python 3.12

If you want the smoothest experience on Windows and Linux, use Python 3.12:

```bash
uv python install 3.12
uv venv --python 3.12
uv sync --extra cu124
```

## Jetson note: PyTorch index

On Jetson, `torch` / `torchvision` usually must come from a Jetson-compatible wheel index. For example, for JetPack 6.x with CUDA 12.6:

```bash
uv pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

Then rerun:

```bash
uv sync
```

Verification:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"
```
