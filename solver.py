"""Pseudo-spectral solver for the 1D viscous Burgers equation (the "simulation").

    u_t + u u_x = nu u_xx,   x in [0, 1) periodic,   u(x, 0) = u0(x)

Spatial derivatives are exact in Fourier space (2/3-rule dealiasing); time stepping is
an integrating-factor RK4 scheme, which treats the stiff viscous term exactly. This is
the "slow, expensive simulation" that the neural surrogates in this repo learn to replace.
"""
from __future__ import annotations

import numpy as np

NU = 0.01
T_FINAL = 1.0
DT = 1e-3
N_MODES = 32  # random initial-condition modes, independent of grid resolution


def sample_coeffs(n_samples: int, seed: int, n_modes: int = N_MODES, scale: float = 2.0) -> np.ndarray:
    """Random Fourier coefficients (a_k, b_k) for u0 = sum_k a_k cos(2 pi k x) + b_k sin(2 pi k x).

    Spectrum follows a Gaussian random field with std_k = 25 / (4 pi^2 k^2 + 25), the usual
    benchmark prior for Burgers. Coefficients do not depend on grid size, so the same u0 can be
    evaluated at any resolution (needed for the zero-shot resolution test).
    """
    rng = np.random.default_rng(seed)
    k = np.arange(1, n_modes + 1)
    std = scale * 25.0 / (4 * np.pi**2 * k**2 + 25.0)
    return rng.standard_normal((n_samples, 2, n_modes)) * std


def initial_conditions(coeffs: np.ndarray, n: int) -> np.ndarray:
    """Evaluate u0 on an n-point periodic grid. Returns (n_samples, n)."""
    x = np.arange(n) / n
    k = np.arange(1, coeffs.shape[-1] + 1)
    ang = 2 * np.pi * np.outer(k, x)  # (modes, n)
    return coeffs[:, 0] @ np.cos(ang) + coeffs[:, 1] @ np.sin(ang)


def solve(u0: np.ndarray, nu: float = NU, t_final: float = T_FINAL, dt: float = DT) -> np.ndarray:
    """Advance a batch of initial conditions (n_samples, n) to t_final. Returns u(t_final)."""
    n = u0.shape[-1]
    n_steps = int(round(t_final / dt))
    dt = t_final / n_steps
    k = 2 * np.pi * np.fft.rfftfreq(n, d=1.0 / n)  # wavenumbers for a unit-length domain
    mask = (np.arange(k.size) <= n // 3).astype(float)  # 2/3 dealiasing
    lin = nu * k**2
    E, E2 = np.exp(-lin * dt), np.exp(-lin * dt / 2)

    def nonlinear(vh: np.ndarray) -> np.ndarray:
        u = np.fft.irfft(vh * mask, n=n, axis=-1)
        return -0.5j * k * np.fft.rfft(u * u, axis=-1) * mask

    v = np.fft.rfft(u0, axis=-1) * mask
    for _ in range(n_steps):
        k1 = nonlinear(v)
        k2 = nonlinear(E2 * (v + 0.5 * dt * k1))
        k3 = nonlinear(E2 * v + 0.5 * dt * k2)
        k4 = nonlinear(E * v + dt * E2 * k3)
        v = E * v + dt / 6.0 * (E * k1 + 2.0 * E2 * (k2 + k3) + k4)
    return np.fft.irfft(v, n=n, axis=-1)


def make_dataset(n_samples: int, n: int, seed: int):
    """Returns (u0, u_T) arrays of shape (n_samples, n)."""
    u0 = initial_conditions(sample_coeffs(n_samples, seed), n)
    return u0, solve(u0)
