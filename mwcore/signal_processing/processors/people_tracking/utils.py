from __future__ import annotations
import numpy as np



def _nextpow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _steering_vec_nu(m_idx: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """
    Azimuth-only steering vectors at mu=0:
      a_k(nu) = exp(j*pi*m_k*nu)
    Returns: A (Naz, Nant)
    """
    # nu: (Naz,), m_idx: (Nant,)
    return np.exp(1j * np.pi * nu[:, None] * m_idx[None, :]).astype(np.complex64)


def _steering_vec_mu(n_idx: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Elevation-only steering vectors at nu=0:
      b_k(mu) = exp(j*pi*n_k*mu)
    Returns: B (Nel, Nant)
    """
    return np.exp(1j * np.pi * mu[:, None] * n_idx[None, :]).astype(np.complex64)


def _mvdr_spectrum(invR: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    MVDR/Capon spectrum:
      P(i) = 1 / (a_i^H invR a_i)
    Args:
      invR: (Nant,Nant)
      A: (Nbins,Nant) steering vectors
    Returns:
      P: (Nbins,) float32
    """
    # denom_i = a_i^H invR a_i
    # Use einsum: (A* ) (invR) (A^T)
    tmp = A.conj() @ invR  # (Nbins,Nant)
    denom = np.einsum("bi,bi->b", tmp, A).real  # (Nbins,)
    denom = np.maximum(denom, 1e-12)
    return (1.0 / denom).astype(np.float32)


def _cov_inv(Y: np.ndarray, diag_load: float) -> np.ndarray:
    """
    Covariance and inverse with diagonal loading.
    Y: (Nant, Nsnap) complex64
    R = (1/Nsnap) * Y Y^H
    R <- R + diag_load * (trace(R)/Nant) * I
    """
    Nant, Nsnap = Y.shape
    if Nsnap <= 0:
        raise ValueError("Need at least 1 snapshot for covariance.")
    R = (Y @ Y.conj().T) / float(Nsnap)  # (Nant,Nant)
    tr = np.trace(R).real
    alpha = float(diag_load) * (tr / max(1, Nant))
    R = R + alpha * np.eye(Nant, dtype=R.dtype)
    # Use solve for stability
    return np.linalg.inv(R).astype(np.complex64)


def _caso_noise_1d(x: np.ndarray, k: int, ref: int, guard: int, cyclic: bool) -> float:
    """
    CFAR-CASO noise estimate at index k for 1D signal x (linear power).
    noise = min(mean(left_ref), mean(right_ref))
    """
    n = x.shape[0]
    if ref <= 0:
        return 0.0

    def _idx(i: int) -> int:
        return i % n

    if cyclic:
        left = [x[_idx(k - guard - i)] for i in range(1, ref + 1)]
        right = [x[_idx(k + guard + i)] for i in range(1, ref + 1)]
    else:
        l0 = k - guard - ref
        l1 = k - guard
        r0 = k + guard + 1
        r1 = k + guard + 1 + ref
        if l0 < 0 or r1 > n:
            return float("inf")  # outside valid window -> suppress
        left = x[l0:l1]
        right = x[r0:r1]

    nl = float(np.mean(left)) if len(left) else float("inf")
    nr = float(np.mean(right)) if len(right) else float("inf")
    return min(nl, nr)


def _local_max_2d(H: np.ndarray, a: int, r: int) -> bool:
    """
    Simple local maximum check in 2D on heatmap H (Az,Range).
    Checks 8-neighborhood (clamped).
    """
    azN, rN = H.shape
    v = H[a, r]
    for da in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if da == 0 and dr == 0:
                continue
            aa = a + da
            rr = r + dr
            if 0 <= aa < azN and 0 <= rr < rN:
                if H[aa, rr] > v:
                    return False
    return True