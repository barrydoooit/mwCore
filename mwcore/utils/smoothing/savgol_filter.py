from __future__ import annotations 

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


SavGolayPadding = Literal['reflect', 'symmetric', 'edge', 'wrap', 'constant', 'nearest', 'mirror']

@dataclass
class SavGolayConfig:
    window_length: int # Must be odd and > polyorder
    polyorder: int # Polynomial order (0..window_length-1)
    deriv: int = 0 # Derivative order (0 for smoothing)
    delta: float = 1.0 # Sample spacing
    mode: SavGolayPadding = 'reflect' # Padding mode
    cval: float = 0.0 # Constant value for 'constant' mode

def _normalize_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise np.AxisError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim

def _validate_params(n: int, cfg: SavGolayConfig) -> None:
    if cfg.window_length < 1 or cfg.window_length % 2 == 0:
        raise ValueError(f"window_length must be a positive odd integer, got {cfg.window_length}")
    if cfg.polyorder < 0:
        raise ValueError("polyorder must be >= 0")
    if cfg.window_length <= cfg.polyorder:
        raise ValueError("window_length must be > polyorder")
    if cfg.deriv < 0:
        raise ValueError("deriv must be >= 0")
    if cfg.deriv > cfg.polyorder:
        raise ValueError("deriv must be <= polyorder")
    if cfg.delta <= 0:
        raise ValueError("delta must be > 0")
    if n <= 0:
        raise ValueError("Input length along chosen axis must be > 0")

    # Modes compatible with numpy.pad (allow friendly synonyms)
    pad_modes = {"reflect", "symmetric", "edge", "wrap", "constant"}
    friendly = {"nearest": "edge", "mirror": "reflect"}
    actual_mode = friendly.get(cfg.mode, cfg.mode)
    if actual_mode not in pad_modes:
        raise ValueError(f"Unsupported mode '{cfg.mode}'. Choose from "
                         f"{sorted(list(pad_modes | set(friendly.keys())))}")

    # Extra guard: reflect/symmetric require len(axis) > 1 when pad>0
    if actual_mode in {"reflect", "symmetric"} and n == 1:
        raise ValueError(f"mode='{cfg.mode}' requires axis length > 1; use 'edge' or 'constant' instead.")

def savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    dtype: Any = np.float64,
    eval_index: Optional[int] = None,   # <- NEW (0..W-1), None = center
) -> np.ndarray:
    if window_length % 2 == 0 or window_length <= polyorder or polyorder < 0 or deriv < 0:
        raise ValueError("Require odd window_length > polyorder >= deriv >= 0.")

    W = window_length
    if eval_index is None:
        eval_index = W // 2
    if not (0 <= eval_index < W):
        raise ValueError(f"eval_index must be in [0, {W-1}], got {eval_index}")

    # positions relative to the evaluation point
    k = np.arange(W, dtype=np.float64) - float(eval_index)
    A = np.vander(k, N=polyorder + 1, increasing=True)  # (W, p+1)
    pinvA = np.linalg.pinv(A)                           # (p+1, W)
    coeffs = pinvA[deriv] * math.factorial(deriv) / (delta ** deriv)
    return np.asarray(coeffs, dtype=dtype)


def savgol_filter(
    x: np.ndarray,
    axis: int = -1,
    config: Optional[Union[SavGolayConfig, Dict[str, Any]]] = None,
    *,
    eval_index: Optional[int] = None,          # <- NEW (0..W-1), None=center
    pad: Optional[Tuple[int, int]] = None,     # <- NEW (left, right); default=(m, m)
    **kwargs: Any,
) -> np.ndarray:
    # build cfg as before...
    if config is None:
        config = {}
    if isinstance(config, dict):
        cfg = SavGolayConfig(**{**config, **kwargs})
    elif isinstance(config, SavGolayConfig):
        cfg = SavGolayConfig(**{**config.__dict__, **kwargs})
    else:
        raise TypeError("config must be None, dict, or SavGolayConfig")

    axis = _normalize_axis(axis, x.ndim)
    n = x.shape[axis]
    _validate_params(n, cfg)

    # default symmetric pad to keep output length == input length
    m = cfg.window_length // 2
    if pad is None:
        pad_left, pad_right = m, m
    else:
        pad_left, pad_right = map(int, pad)
        if pad_left < 0 or pad_right < 0:
            raise ValueError("pad must be non-negative")
    # Ensure we produce one output per input sample
    if (pad_left + pad_right) != (cfg.window_length - 1):
        raise ValueError(
            f"pad_left + pad_right must equal window_length-1 ({cfg.window_length-1}) "
            f"to keep the output length unchanged; got {pad_left}+{pad_right}"
        )

    # map mode synonyms
    mode_map = {"nearest": "edge", "mirror": "reflect"}
    pad_mode = mode_map.get(cfg.mode, cfg.mode)

    # dtype
    work_dtype = np.result_type(x.dtype, np.float64)
    x = np.asarray(x, dtype=work_dtype)

    # asymmetric pad on the chosen axis
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (pad_left, pad_right)
    if pad_mode == "constant":
        x_pad = np.pad(x, pad_width, mode=pad_mode, constant_values=cfg.cval)
    else:
        x_pad = np.pad(x, pad_width, mode=pad_mode)

    # coefficients for this evaluation position
    coeffs = savgol_coeffs(
        cfg.window_length, cfg.polyorder, cfg.deriv, cfg.delta,
        dtype=work_dtype, eval_index=eval_index,
    )

    windows = sliding_window_view(x_pad, cfg.window_length, axis=axis)
    y = np.tensordot(windows, coeffs, axes=([-1], [0]))
    return y.astype(x.dtype, copy=False)