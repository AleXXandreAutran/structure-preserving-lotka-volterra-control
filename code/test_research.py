"""Check time integration and control transfers."""
import numpy as np
from research_experiments import Parameters, problem, interpolate_control, continuous_time_reference


def test_nested_time_transfer_uses_correct_cells():
    source, fine = (problem('flux', 8, 120), problem('flux', 16, 1920))
    u = np.broadcast_to(np.arange(120)[:, None] / 120, source.shape).copy()
    transferred = interpolate_control(source, fine, u)
    np.testing.assert_allclose(
        transferred[:, 0],
        np.repeat(np.arange(120) / 120, 16),
        rtol=0,
        atol=0,
    )


def test_spatial_transfer_is_exact_for_affine_controls():
    source, fine = (problem('flux', 8, 12), problem('flux', 32, 12))
    u = np.broadcast_to(0.2 + 0.8 * source.x, source.shape).copy()
    np.testing.assert_allclose(
        interpolate_control(source, fine, u),
        np.broadcast_to(0.2 + 0.8 * fine.x, fine.shape),
        rtol=0,
        atol=2e-16,
    )


def test_adaptive_reference_matches_decoupled_exact_harvest():
    par = Parameters(T=2.0, gamma1=0.0, gamma2=0.0, r=1.2, lam=0.05, eta=0.8)
    P = problem('flux', 4, 7, par)
    y0, c = (1.3, 0.7)
    P.initial = np.array([np.zeros(5), np.full(5, y0)])
    rate = par.r - par.q * c
    exact = 0.5 * par.lam * c * c * par.T - par.eta * par.q * c * y0 * np.expm1(rate * par.T) / rate
    result = continuous_time_reference(
        P,
        np.full(P.shape, c),
        M=8,
        rtol=1e-11,
        atol=1e-13,
        initial_data=P.initial,
    )
    assert abs(result['objective'] - exact) < 1e-10


def test_adaptive_reference_respects_control_discontinuities():
    par = Parameters(T=2.0, gamma1=0.0, gamma2=0.0, r=1.2, lam=0.05, eta=0.8)
    P = problem('flux', 4, 4, par)
    P.initial = np.array([np.zeros(5), np.full(5, 1.3)])
    controls = np.array([0.0, 2.0, 0.2, 1.5])
    y = 1.3
    exact = 0.0
    for c in controls:
        rate = par.r - par.q * c
        exact += 0.5 * par.lam * c * c * P.dt - par.eta * par.q * c * y * np.expm1(rate * P.dt) / rate
        y *= np.exp(rate * P.dt)
    u = np.broadcast_to(controls[:, None], P.shape).copy()
    result = continuous_time_reference(
        P,
        u,
        M=8,
        rtol=1e-11,
        atol=1e-13,
        initial_data=P.initial,
    )
    assert abs(result['objective'] - exact) < 1e-10
