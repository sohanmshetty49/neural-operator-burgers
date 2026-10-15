import numpy as np

import solver


def test_mean_is_conserved():
    u0 = solver.initial_conditions(solver.sample_coeffs(8, 0), 128)
    uT = solver.solve(u0)
    assert np.allclose(uT.mean(axis=1), u0.mean(axis=1), atol=1e-12)


def test_matches_analytic_heat_decay_for_small_amplitude():
    # For tiny amplitude the nonlinear term is O(eps^2), so u ~ eps * exp(-nu k^2 T) * sin(2 pi x).
    n, eps = 128, 1e-4
    x = np.arange(n) / n
    u0 = eps * np.sin(2 * np.pi * x)[None]
    exact = eps * np.exp(-solver.NU * (2 * np.pi) ** 2 * solver.T_FINAL) * np.sin(2 * np.pi * x)[None]
    rel = np.linalg.norm(solver.solve(u0) - exact) / np.linalg.norm(exact)
    assert rel < 1e-3


def test_spatial_convergence():
    c = solver.sample_coeffs(10, 1)
    coarse = solver.solve(solver.initial_conditions(c, 128))
    fine = solver.solve(solver.initial_conditions(c, 512))[:, ::4]
    rel = np.linalg.norm(coarse - fine, axis=1) / np.linalg.norm(fine, axis=1)
    assert rel.max() < 1e-6


def test_time_step_convergence():
    u0 = solver.initial_conditions(solver.sample_coeffs(10, 2), 128)
    a = solver.solve(u0, dt=1e-3)
    b = solver.solve(u0, dt=5e-4)
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    assert rel.max() < 1e-6


def test_initial_conditions_independent_of_resolution():
    c = solver.sample_coeffs(3, 5)
    lo, hi = solver.initial_conditions(c, 64), solver.initial_conditions(c, 256)
    assert np.allclose(lo, hi[:, ::4], atol=1e-12)
