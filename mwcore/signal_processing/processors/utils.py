import numpy as np
from typing import Optional, Union

def apply_doppler_compensation(
    data: np.ndarray,
    d_idxs: np.ndarray,
    num_doppler_bins: int,
    num_tx: int,
    tx_offsets: Union[list[int], np.ndarray, None] = None
) -> np.ndarray:
    """
    Compensates for the phase shift introduced by moving targets during the 
    sequential transmission of TDM MIMO radar chirps.
    
    Args:
        data: Complex radar data array of shape (num_tx, num_rx, num_det).
              Modified in-place.
        d_idxs: Array of Doppler bin indices of shape (num_det,). 
                Assumes the underlying FFT data is already fftshift-ed.
        num_doppler_bins: Total size of the Doppler dimension (e.g., loops_per_frame).
        num_tx: Total number of transmitting antennas.
        tx_offsets: Temporal transmission offset for each TX antenna. 
                    If None, defaults to sequential order [0, 1, ..., num_tx-1].
                    
    Returns:
        The phase-compensated data array.
    """
    if tx_offsets is None:
        tx_offsets = np.arange(num_tx, dtype=np.int32)
    else:
        tx_offsets = np.asarray(tx_offsets, dtype=np.int32)
        
    if len(tx_offsets) != num_tx:
        raise ValueError(f"Length of tx_offsets ({len(tx_offsets)}) must match num_tx ({num_tx}).")

    # Recover signed Doppler indices (assuming the underlying FFT is fftshifted)
    # Center bin represents zero velocity
    d_signed = d_idxs.astype(np.int32) - (num_doppler_bins // 2)

    # Calculate the base phase shift per Doppler bin per TX delay
    # Formula: exp(-j * 2pi * doppler_bin / (N_doppler * num_tx))
    base_phase = np.exp(-1j * 2.0 * np.pi * (d_signed.astype(np.float32) / (num_doppler_bins * num_tx)))

    # Apply phase rotation for each TX antenna based on its temporal offset
    for tx in range(num_tx):
        if tx_offsets[tx] == 0:
            continue
        # base_phase[None, :] broadcasts to (1, num_det) to match data[tx, :, :] which is (num_rx, num_det)
        data[tx, :, :] *= base_phase[None, :] ** tx_offsets[tx]

    return data