"""
test_tracking.py — regression tests for the FL + qLPV-MPC cascade.

Motivation
----------
A sign error in the body-frame gravity projection (u_dot: +g·sinθ instead
of −g·sinθ, v_dot mirrored) once shipped in dynamics.py.  It was invisible
in hover — sinθ = sinφ = 0 hides the flipped terms — but produced
positive-feedback divergence the moment the reference moved, ending in
float overflow at ~96 km altitude.  A hover check passes; only a
*maneuvering* closed-loop test catches this class of bug.

These tests run the real simulation (no mocks) and assert closed-loop
performance bounds loose enough to be seed- and machine-independent, but
tight enough that any physics, allocation, or controller regression fails
immediately.

Run:  pytest test_tracking.py -v        (~5 s total)
"""

import numpy as np
import pytest

from simulate import run


def _metrics(data: dict, settle_t: float = 8.0):
    """Position error RMSE [m] over t > settle_t, plus saturation count."""
    mask = data["t"] > settle_t
    # State layout: [u,v,w, p,q,r, x,y,z, phi,theta,psi] — position = cols 6:9
    pos_err = data["refs"][mask, 0:3] - data["states"][mask, 6:9]
    epos    = np.linalg.norm(pos_err, axis=1)
    rmse    = float(np.sqrt(np.mean(epos**2)))
    sats    = int(data["sat"][mask].sum())
    return rmse, sats


def test_figure8_tracks():
    """Nominal figure-8: mm-level tracking, zero saturation."""
    data = run(scenario="figure8", enable_drag=True,
               noise_std=0.0, duration=15.0, verbose=False)
    rmse, sats = _metrics(data)
    assert np.all(np.isfinite(data["states"])), "NaN/Inf in states — divergence"
    assert rmse < 0.05, f"figure-8 RMSE {rmse:.4f} m exceeds 0.05 m bound"
    assert sats == 0,   f"{sats} rotor saturation steps (expected 0)"


def test_figure8_yaw_tracks():
    """Yaw sweep exercises the time-varying A(sigma) entries dormant at psi=0."""
    data = run(scenario="figure8_yaw", enable_drag=True,
               noise_std=0.0, duration=15.0, verbose=False)
    rmse, sats = _metrics(data)
    assert np.all(np.isfinite(data["states"]))
    assert rmse < 0.05, f"yaw-sweep RMSE {rmse:.4f} m exceeds 0.05 m bound"
    assert sats == 0


def test_hover_is_trivial():
    """Hover must be numerically exact — catches equilibrium/mixer breakage."""
    data = run(scenario="hover", enable_drag=True,
               noise_std=0.0, duration=8.0, verbose=False)
    rmse, sats = _metrics(data, settle_t=1.0)
    assert rmse < 1e-6
    assert sats == 0


def test_noise_bounded():
    """Process-noise stress: degraded but bounded, still saturation-free."""
    data = run(scenario="figure8", enable_drag=True,
               noise_std=5e-2, duration=15.0, verbose=False)
    rmse, sats = _metrics(data)
    assert np.all(np.isfinite(data["states"]))
    assert rmse < 0.15, f"noise-run RMSE {rmse:.4f} m exceeds 0.15 m bound"
    assert sats == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
