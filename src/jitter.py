import numpy as np
from scipy.fft import fft

# Some jitter metrics. I'm still not sure what the best metrics are. Most of the pose estimation literature use
# the acceleration error between each predicted and ground truth joint. 

def sliding_window_std(errors, window_size=10, mode='last'):
    """
    Compute the standard deviation of errors over a sliding window.
    """
    if len(errors) < window_size:
        return None  # Not enough data
    if mode == 'all':
        return [np.std(errors[i:i+window_size]) for i in range(len(errors) - window_size + 1)]
    elif mode == 'last':
        return np.std(errors[-window_size:])
    else:
        raise ValueError(f"Invalid mode: {mode}")

def temporal_derivative(errors, timestamps, mode='last'):
    """
    Compute the first derivative (velocity) of errors over time.
    """
    if len(errors) < 2:
        return None  # Not enough data
    if mode == 'all':
        dt = np.diff(timestamps)
        return np.diff(errors) / dt
    elif mode == 'last':
        dt = timestamps[-1] - timestamps[-2]
        return (errors[-1] - errors[-2]) / dt
    else:
        raise ValueError(f"Invalid mode: {mode}")

def temporal_acceleration(errors, timestamps, mode='last'):
    """
    Compute the second derivative (acceleration) of errors over time.
    """
    if len(errors) < 3:
        return None  # Not enough data
    if mode == 'all':
        dt = np.diff(timestamps)
        velocity = np.diff(errors) / dt
        # Compute second derivative (acceleration)
        acceleration = np.diff(velocity) / dt[:-1]
        return acceleration
    
    elif mode == 'last':
        dt1 = timestamps[-1] - timestamps[-2]
        dt2 = timestamps[-2] - timestamps[-3]
        v1 = (errors[-1] - errors[-2]) / dt1
        v2 = (errors[-2] - errors[-3]) / dt2
        return (v1 - v2) / dt1
    else:
        raise ValueError(f"Invalid mode: {mode}")

def peak_to_peak_amplitude(errors, window_size=10, mode='last'):
    """
    Compute the peak-to-peak amplitude over a sliding window.
    """
    if len(errors) < window_size:
        return None  # Not enough data
    if mode == 'all':
        return [np.max(errors[i:i+window_size]) - np.min(errors[i:i+window_size]) for i in range(len(errors) - window_size + 1)]
    elif mode == 'last':
        return np.max(errors[-window_size:]) - np.min(errors[-window_size:])
    else:
        raise ValueError(f"Invalid mode: {mode}")

# I haven't tested this function yet. It's an AI placeholder for now.
def spectral_analysis(errors, window_size=50):
    if len(errors) < window_size:
        return None
    # Apply a window function
    error_window = np.array(errors[-window_size:])
    window = np.hanning(window_size)
    error_window = error_window * window
    # Compute FFT and normalize
    spectrum = np.abs(fft(error_window)) / window_size
    # Use only the first half of the spectrum (real-valued signals)
    spectrum = spectrum[:window_size // 2]
    # Sum high-frequency components (e.g., upper half of the spectrum)
    high_freq_power = np.sum(spectrum[window_size // 4:])
    return high_freq_power
