import numpy as np



def dev2standard(device_height: float,
                  device_tilt: float) -> np.ndarray:
    ang_rad = np.radians(device_tilt)
    R = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(ang_rad), -np.sin(ang_rad), 0],
            [0, np.sin(ang_rad), np.cos(ang_rad), 0],
            [0, 0, 0, 1],
        ], dtype=float
    )

    T = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, device_height],
        [0, 0, 0, 1],
    ], dtype=float)

    M_dev2standard = T @ R
    return M_dev2standard

def standard2dev(device_height: float,
                  device_tilt: float) -> np.ndarray:
    ang_rad = np.radians(device_tilt)
    R_inv = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(ang_rad), np.sin(ang_rad), 0],
            [0, -np.sin(ang_rad), np.cos(ang_rad), 0],
            [0, 0, 0, 1],
        ], dtype=float
    )
    T_inv = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -device_height],
        [0, 0, 0, 1],
    ], dtype=float)
    M_standard2dev = R_inv @ T_inv
    return M_standard2dev