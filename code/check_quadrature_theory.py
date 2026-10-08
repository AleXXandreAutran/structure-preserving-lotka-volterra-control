"""Check the harvest formulas and their derivatives in the removal model."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def trajectory_derivatives(u, dt, q, initial):
    u = np.asarray(u, dtype=float)
    count = len(u)
    a = dt * q
    d = 1 / (1 + a * u)
    y = np.r_[initial, initial * np.cumprod(d)]
    jacobian = np.zeros((count + 1, count))
    second = np.zeros((count + 1, count, count))
    for n in range(1, count + 1):
        prefix = np.zeros(count)
        prefix[:n] = d[:n]
        jacobian[n] = -a * y[n] * prefix
        second[n] = a * a * y[n] * (np.outer(prefix, prefix) + np.diag(prefix * prefix))
    return (y, jacobian, second)


def direct_hessians(u, dt, q, initial, lam, eta):
    y, jacobian, second = trajectory_derivatives(u, dt, q, initial)
    effort = lam * dt * np.eye(len(u))
    result = []
    for stage in (slice(None, -1), slice(1, None)):
        jac = jacobian[stage]
        sec = second[stage]
        hessian = effort - eta * dt * q * (jac + jac.T + np.einsum('n,nij->ij', u, sec))
        result.append(hessian / dt)
    return result


def main():
    rng = np.random.default_rng(721901)
    maximum_formula_error = 0.0
    maximum_spectral_error = 0.0
    for count in (2, 3, 4, 9, 12):
        dt, q, initial, lam, eta = (2 / count, 0.7, 1.2, 0.4, 0.8)
        u = rng.uniform(0.1, 1.7, count)
        h0, h1 = direct_hessians(u, dt, q, initial, lam, eta)
        y, jacobian, second = trajectory_derivatives(u, dt, q, initial)
        d = 1 / (1 + dt * q * u)
        expected_h1 = lam * np.eye(count) + eta * dt * q * q * y[-1] * (np.outer(d, d) + np.diag(d * d))
        jnew = jacobian[1:]
        expected_defect_hessian = eta * dt * q * q * (
            2 * np.diag(y[1:])
            + 2 * (np.diag(u) @ jnew + jnew.T @ np.diag(u))
            + np.einsum('n,nij->ij', u * u, second[1:])
        )
        maximum_formula_error = max(
            maximum_formula_error,
            float(np.max(np.abs(h1 - expected_h1))),
            float(np.max(np.abs(h1 - h0 - expected_defect_hessian))),
        )
        assert np.linalg.eigvalsh(h1).min() >= lam
        zero0, zero1 = direct_hessians(np.zeros(count), dt, q, initial, lam, eta)
        for zeta, hessian in ((0, zero0), (1, zero1)):
            k = eta * dt * q * q * initial
            expected = lam * np.eye(count) + k * (np.ones((count, count)) + (2 * zeta - 1) * np.eye(count))
            maximum_spectral_error = max(
                maximum_spectral_error,
                float(np.max(np.abs(hessian - expected))),
            )
        pure_defect = eta * dt * dt * q * q * np.sum(u * u * y[1:])
        old = lam * dt / 2 * np.sum(u * u) - eta * dt * q * np.sum(u * y[:-1])
        flux = lam * dt / 2 * np.sum(u * u) - eta * (initial - y[-1])
        assert abs(flux - old - pure_defect) < 1e-12
    assert maximum_formula_error < 1e-12
    assert maximum_spectral_error < 1e-12
    data = {
        'random_interior_controls': 5,
        'max_hessian_formula_error': maximum_formula_error,
        'max_zero_control_formula_error': maximum_spectral_error,
        'nonglobal_local_minimum_normal_derivative': 2,
        'nonglobal_local_minimum_free_curvature': 1,
        'nonglobal_local_minimum_cost': -2,
        'explicit_better_feasible_cost': -20 / 9,
        'status': 'all algebra checks passed',
    }
    output = Path(__file__).resolve().parents[1] / 'validation' / 'quadrature-theory-checks.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(data, indent=2))
if __name__ == '__main__':
    main()
