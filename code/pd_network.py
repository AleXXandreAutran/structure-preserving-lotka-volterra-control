"""A weighted production-destruction network with diffusion and harvesting."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def assemble_generator(rates, species_weights, spatial_weights, laplacian, diffusivities):
    """Build the diffusion-transfer generator; rates[i,j,a] move species j to i."""
    rates = np.asarray(rates)
    species_weights = np.asarray(species_weights)
    spatial_weights = np.asarray(spatial_weights)
    laplacian = np.asarray(laplacian)
    diffusivities = np.asarray(diffusivities)
    s, _, d = rates.shape
    if rates.shape != (s, s, d):
        raise ValueError('rates must have shape (species, species, nodes)')
    if np.any(rates < 0) or np.any(~np.isfinite(rates)):
        raise ValueError('transfer rates must be finite and nonnegative')
    if np.any(species_weights <= 0) or np.any(spatial_weights <= 0):
        raise ValueError('mass weights must be strictly positive')
    if laplacian.shape != (d, d) or diffusivities.shape != (s,):
        raise ValueError('incompatible diffusion dimensions')
    if np.any(diffusivities < 0):
        raise ValueError('diffusivities must be nonnegative')
    off_diagonal = laplacian.copy()
    np.fill_diagonal(off_diagonal, 0)
    if np.any(off_diagonal < 0):
        raise ValueError('laplacian off-diagonal entries must be nonnegative')
    if not np.allclose(spatial_weights @ laplacian, 0, atol=1e-12):
        raise ValueError('diffusion must conserve the chosen spatial mass')
    generator = np.zeros((s * d, s * d))
    for i in range(s):
        block = slice(i * d, (i + 1) * d)
        generator[block, block] = diffusivities[i] * laplacian
    for i in range(s):
        for j in range(s):
            if i == j:
                continue
            for a in range(d):
                generator[i * d + a, j * d + a] += species_weights[j] / species_weights[i] * rates[i, j, a]
                generator[j * d + a, j * d + a] -= rates[i, j, a]
    mass_weights = np.repeat(species_weights, d) * np.tile(spatial_weights, s)
    return (generator, mass_weights)


def step(state, generator, mortality, harvest, production, tau):
    """Return the new state, componentwise removal and step matrix."""
    state = np.asarray(state)
    mortality = np.asarray(mortality)
    harvest = np.asarray(harvest)
    production = np.asarray(production)
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError('tau must be finite and positive')
    if any((np.any(v < 0) for v in (state, mortality, harvest, production))):
        raise ValueError('state, production and loss rates must be nonnegative')
    matrix = np.eye(len(state)) - tau * generator
    matrix += tau * np.diag(mortality + harvest)
    new_state = np.linalg.solve(matrix, state + tau * production)
    return (new_state, tau * harvest * new_state, matrix)


def run_checks():
    """Check a three-species cycle with diffusion and an empty component."""
    rng = np.random.default_rng(406)
    species_weights = np.array([2.0, 3.0, 5.0])
    spatial_weights = np.array([0.5, 1.0, 1.0, 1.0, 0.5]) / 4.0
    conductance = np.diag(np.full(4, 1.0), 1)
    conductance += conductance.T
    symmetric_flux = conductance - np.diag(conductance.sum(axis=1))
    laplacian = symmetric_flux / spatial_weights[:, None]
    rates = np.zeros((3, 3, 5))
    rates[1, 0] = np.linspace(0.4, 0.8, 5)
    rates[2, 1] = np.linspace(0.2, 0.7, 5)
    rates[0, 2] = np.linspace(0.1, 0.6, 5)
    generator, rho = assemble_generator(
        rates,
        species_weights,
        spatial_weights,
        laplacian,
        np.array([0.01, 0.03, 0.02]),
    )
    state = rng.uniform(0.1, 1.0, size=15)
    state[5:10] = 0.0
    mortality = rng.uniform(0.0, 0.2, size=15)
    harvest = np.tile(np.array([0.0, 0.2, 0.8])[:, None], (1, 5)).ravel()
    production = rng.uniform(0.0, 0.1, size=15)
    rows = []
    assert np.max(np.abs(rho @ generator)) < 1e-13
    for tau in (0.001, 0.1, 2.0, 20.0):
        new, removed, matrix = step(state, generator, mortality, harvest, production, tau)
        mass_change = rho @ (new - state)
        ledger = tau * rho @ production - tau * rho @ (mortality * new)
        ledger -= rho @ removed
        inv = np.linalg.inv(matrix)
        weighted_column_sums = rho @ inv / rho
        assert np.min(new) >= 0
        assert np.min(inv) >= -1e-14
        assert np.max(weighted_column_sums) <= 1 + 1e-13
        assert abs(mass_change - ledger) < 1e-12
        current = state.copy()
        total_removed = 0.0
        total_mortality = 0.0
        old_reward = 0.0
        for _ in range(8):
            old_reward += tau * rho @ (harvest * current)
            current, removal, _ = step(
                current,
                generator,
                mortality,
                harvest,
                np.zeros(15),
                tau,
            )
            total_removed += rho @ removal
            total_mortality += tau * rho @ (mortality * current)
        assert total_removed <= rho @ state + 1e-12
        assert abs(rho @ current + total_removed + total_mortality - rho @ state) < 1e-12
        rows.append({
            'tau': tau,
            'minimum_state': float(np.min(new)),
            'mass_identity_error': float(abs(mass_change - ledger)),
            'resolvent_mass_norm': float(np.max(weighted_column_sums)),
            'initial_mass': float(rho @ state),
            'eight_step_flux_harvest': float(total_removed),
            'eight_step_old_state_reward': float(old_reward),
        })
    return {'network': 'three-species directed cycle with diffusion', 'rows': rows}
if __name__ == '__main__':
    report = run_checks()
    destination = Path(__file__).resolve().parents[1] / 'validation' / 'pd_network_checks.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
