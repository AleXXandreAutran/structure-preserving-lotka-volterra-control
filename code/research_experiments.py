"""Explore harvest rewards, parameters and grid refinement."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import LinearOperator, eigsh
import scipy
import numba
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--phase',
    choices=[
        'all',
        'derivatives',
        'perturbation',
        'parameters',
        'branches',
        'meshes',
        'reference',
    ],
    default='all',
)
parser.add_argument('--code-dir', type=Path, default=None)
parser.add_argument('--output-dir', type=Path, default=None)
ARGS = parser.parse_args() if __name__ == '__main__' else argparse.Namespace(
    phase='all',
    code_dir=None,
    output_dir=None,
)
code_dir = ARGS.code_dir or HERE
sys.path.insert(0, str(code_dir.resolve()))
from lotka import Parameters, Problem, optimize, free_spectrum
OUT = ARGS.output_dir or HERE.parent / 'research_results'
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ['imex', 'left', 'flux']
RNG_SEED = 20261006


def save(name, data):
    (OUT / f'{name}.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load(name, default=None):
    path = OUT / f'{name}.json'
    return json.loads(path.read_text()) if path.exists() else {} if default is None else default


def problem(label, M=20, N=60, parameters=None):
    return Problem(
        M,
        N,
        'imex' if label == 'imex' else 'positive',
        parameters=parameters,
        harvest_rule='flux' if label == 'flux' else 'left',
    )


def interpolate_control(source, destination, u):
    ticks = destination.t / source.dt
    # Keep shared time nodes in the same interval despite roundoff.
    align = 32 * np.finfo(float).eps * max(source.N, destination.N)
    index = np.minimum(np.floor(ticks + align).astype(int), source.N - 1)
    return np.array([np.interp(destination.x, source.x, u[k]) for k in index])


def statistics(P, u):
    _, g, v, p = P.evaluate(u)
    tv = float(np.sum(P.w * np.abs(np.diff(u, axis=0))))
    variation_energy = float(np.sum(P.w * np.diff(u, axis=0) ** 2))
    switching = float(np.sum(P.w * np.sum(
        (u[:-1] < 0.1 * P.par.umax) & (u[1:] > 0.9 * P.par.umax) | (u[:-1] > 0.9 * P.par.umax) & (u[1:] < 0.1 * P.par.umax),
        axis=0,
    )))
    lower = u <= 1e-08
    upper = u >= P.par.umax - 1e-08
    free = ~(lower | upper)
    return dict(
        temporal_variation=tv,
        variation_energy=variation_energy,
        full_switches=switching,
        mean_control=float(np.sum(P.W * u) / P.par.T),
        free=int(np.count_nonzero((u > 1e-08) & (u < P.par.umax - 1e-08))),
        lower_gradient_margin=float(g[lower].min()) if lower.any() else None,
        upper_gradient_margin=float((-g[upper]).min()) if upper.any() else None,
        max_free_gradient=float(np.max(np.abs(g[free]))) if free.any() else None,
        balance_error=float(P.diagnostics(u, v, p, g)['balance_error']),
    )


def attempt(P, start=None, name='run', maxiter=1800, spectrum=False):
    u, report = optimize(P, start=start, tol=2e-07, maxiter=maxiter, maxcor=38)
    hist = report.pop('history')
    report['lbfgsb_memory'] = 38
    if not report['accepted']:
        first = dict(report)
        np.savez_compressed(OUT / f'{name}_initial_rejected.npz', u=u, history=hist)
        u2, retry = optimize(P, start=u, tol=2e-07, maxiter=maxiter, maxcor=8)
        history2 = retry.pop('history')
        retry['lbfgsb_memory'] = 8
        restart_record = dict(retry)
        if retry['residual'] < report['residual']:
            u, report, hist = (u2, retry, history2)
        report['initial_rejected_attempt'] = first
        report['restart_attempt'] = restart_record
        report['restart_reason'] = 'Stationarity threshold missed; restart at returned control with shorter L-BFGS memory.'
    report.update(statistics(P, u))
    report.update(M=P.M, N=P.N, dt=P.dt, h=P.h, parameters=asdict(P.par))
    if spectrum:
        try:
            report['spectrum'], _ = free_spectrum(P, u, tol=1e-06)
        except Exception as error:
            report['spectrum_error'] = repr(error)
    np.savez_compressed(OUT / f'{name}.npz', u=u, history=hist)
    print(
        name,
        {k: report[k] for k in [
            'objective',
            'residual',
            'accepted',
            'iterations',
            'temporal_variation',
        ]},
        flush=True,
    )
    return (u, report)


def continuous_time_reference(P, u, M=80, rtol=1e-09, atol=1e-11, initial_data=None):
    """Integrate between control switches on a fixed spatial grid."""
    R = problem('flux', M=M, N=P.N, parameters=P.par)
    if initial_data is not None:
        R.initial = np.array([np.interp(R.x, P.x, row) for row in initial_data])
    effort = np.array([np.interp(R.x, P.x, row) for row in u])
    par = P.par
    J = M + 1
    y = np.concatenate((R.initial.ravel(), np.zeros(4)))
    nfev = 0
    min_state = float(R.initial.min())
    target = R.targets[:, 0]

    def lap(v):
        z = np.empty_like(v)
        z[1:-1] = v[:-2] - 2 * v[1:-1] + v[2:]
        z[0] = 2 * (v[1] - v[0])
        z[-1] = 2 * (v[-2] - v[-1])
        return z / R.h ** 2
    for n, a in enumerate(effort):

        def rhs(t, z):
            pred, prey = (z[:J], z[J:2 * J])
            pred_rhs = par.mu1 * lap(pred) + pred * (-par.m + par.alpha * prey)
            prey_rhs = par.mu2 * lap(prey) + prey * (par.r - par.beta * pred - par.q * a)
            components = np.array([
                0.5 * par.gamma1 * np.sum(R.w * (pred - target[0]) ** 2),
                0.5 * par.gamma2 * np.sum(R.w * (prey - target[1]) ** 2),
                0.5 * par.lam * np.sum(R.w * a ** 2),
                -par.eta * par.q * np.sum(R.w * a * prey),
            ])
            return np.concatenate((pred_rhs, prey_rhs, components))
        sol = solve_ivp(
            rhs,
            (n * P.dt, (n + 1) * P.dt),
            y,
            method='DOP853',
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        y = sol.y[:, -1]
        min_state = min(min_state, float(sol.y[:2 * J].min()))
        nfev += sol.nfev
    return dict(
        objective=float(y[-4:].sum()),
        components=y[-4:].tolist(),
        M=M,
        rtol=rtol,
        atol=atol,
        nfev=nfev,
        min_state=min_state,
        temporal_method='DOP853, intervals split at every control discontinuity',
    )


def derivatives():
    rng = np.random.default_rng(RNG_SEED)
    result = {}
    base = problem('flux', 10, 24)
    xx, tt = np.meshgrid(base.x, base.t)
    d = 0.35 * np.cos(2 * np.pi * xx) * np.cos(np.pi * tt / base.par.T) + 0.1 * rng.normal(size=base.shape)
    d /= np.max(np.abs(d))
    e = rng.normal(size=base.shape)
    e /= np.max(np.abs(e))
    for label in LABELS:
        P = problem(label, 10, 24)
        u = np.full(P.shape, 0.7)
        f, g, v, p = P.evaluate(u)
        Hd = P.hessian(u, d, v, p)
        He = P.hessian(u, e, v, p)
        rows = []
        for eps in [0.01, 0.005, 0.0025, 0.00125]:
            ff, gg, _, _ = P.evaluate(u + eps * d)
            rows.append(dict(
                epsilon=eps,
                first=abs(ff - f - eps * P.inner(g, d)),
                second=abs(ff - f - eps * P.inner(g, d) - 0.5 * eps ** 2 * P.inner(Hd, d)),
                gradient=P.norm(gg - g - eps * Hd),
            ))
        fit = lambda key: float(np.polyfit(
            np.log([r['epsilon'] for r in rows]),
            np.log([r[key] for r in rows]),
            1,
        )[0])
        symmetry = abs(P.inner(Hd, e) - P.inner(d, He))
        result[label] = dict(
            rows=rows,
            first_slope=fit('first'),
            second_slope=fit('second'),
            gradient_slope=fit('gradient'),
            symmetry=symmetry,
        )
        print(
            'DERIVATIVE',
            label,
            {k: v for k, v in result[label].items() if k != 'rows'},
            flush=True,
        )
        save('derivatives', result)
        assert fit('first') > 1.9 and fit('second') > 2.8 and (fit('gradient') > 1.9) and (symmetry < 1e-10)
    save('derivatives', result)


def perturbation():
    rows = []
    for M in [8, 16, 32]:
        for N in [16, 32, 64, 128, 256, 512, 1024]:
            P0, P1 = (problem('left', M, N), problem('flux', M, N))
            x, t = np.meshgrid(P0.x, P0.t)
            for kind in ['smooth', 'alternating']:
                u = 0.7 + 0.2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * t / P0.par.T) if kind == 'smooth' else np.broadcast_to(
                    (0.7 + 0.2 * (-1.0) ** np.arange(N))[:, None],
                    P0.shape,
                ).copy()
                f0, g0, v, p0 = P0.evaluate(u)
                f1, g1, _, p1 = P1.evaluate(u)
                root = np.sqrt(P0.W.ravel())

                def mv(z):
                    d = (z / root).reshape(P0.shape)
                    return root * (P0.hessian(u, d, v, p0) - P1.hessian(u, d, v, p1)).ravel()
                op = LinearOperator((u.size, u.size), matvec=mv, dtype=float)
                eig, evec = eigsh(
                    op,
                    k=1,
                    which='LM',
                    tol=1e-07,
                    maxiter=1600,
                    v0=np.random.default_rng(RNG_SEED).normal(size=u.size),
                )
                eres = float(np.linalg.norm(mv(evec[:, 0]) - eig[0] * evec[:, 0]))
                ds = {}
                for direction in ['constant', 'alternating', 'spatial']:
                    d = np.ones(P0.shape) if direction == 'constant' else np.broadcast_to(
                        (-1.0) ** np.arange(N)[:, None],
                        P0.shape,
                    ).copy() if direction == 'alternating' else np.broadcast_to(np.cos(2 * np.pi * P0.x), P0.shape).copy()
                    d /= P0.norm(d)
                    ds[direction] = dict(
                        left=P0.inner(d, P0.hessian(u, d, v, p0)),
                        flux=P1.inner(d, P1.hessian(u, d, v, p1)),
                    )
                rows.append(dict(
                    M=M,
                    N=N,
                    dt=P0.dt,
                    control=kind,
                    cost_difference=f0 - f1,
                    gradient_difference=P0.norm(g0 - g1),
                    hessian_difference_norm=abs(float(eig[0])),
                    dominant_signed_eigenvalue=float(eig[0]),
                    eigen_residual=eres,
                    rayleigh=ds,
                ))
                save('perturbation', rows)
                print(
                    'PERTURBATION',
                    M,
                    N,
                    kind,
                    rows[-1]['gradient_difference'],
                    rows[-1]['hessian_difference_norm'],
                    flush=True,
                )


def parameters():
    rows = []
    specs = [('eta', v, Parameters(eta=v)) for v in [0.0, 0.2, 0.4, 0.8, 1.2]]
    specs += [('lambda', v, Parameters(lam=v)) for v in [0.02, 0.05, 0.1, 0.2]]
    specs += [('T', v, Parameters(T=v)) for v in [1.0, 2.0, 4.0, 6.0]]
    for pname, value, par in specs:
        N = int(15 * par.T) if pname == 'T' else 60
        for label in ['left', 'flux']:
            P = problem(label, 20, N, par)
            name = f"parameter_{pname}_{str(value).replace('.', 'p')}_{label}"
            u, r = attempt(P, name=name, maxiter=2200, spectrum=True)
            r.update(parameter_name=pname, parameter_value=value, label=label)
            r['common_reference'] = continuous_time_reference(P, u, M=80)
            rows.append(r)
            save('parameters', rows)


def branches():
    rows = []
    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    for label in ['left', 'flux']:
        controls = {}
        for sense, vv in [('up', values), ('down', values[::-1])]:
            start = None
            for value in vv:
                P = problem(label, 20, 60, Parameters(eta=value))
                name = f"branch_{label}_{sense}_{str(value).replace('.', 'p')}"
                u, r = attempt(P, start=start, name=name, maxiter=2400)
                r.update(label=label, sense=sense, eta=value)
                r['common_reference'] = continuous_time_reference(P, u, M=80)
                controls[sense, value] = u
                if sense == 'down':
                    r['distance_to_up_branch'] = P.norm(u - controls['up', value])
                    r['cost_difference_to_up_branch'] = r['objective'] - next(
                        z['objective']
                        for z in rows
                        if z['label'] == label and z['sense'] == 'up' and z['eta'] == value
                    )
                rows.append(r)
                save('branches', rows)
                start = u if r['accepted'] else None
        for kind in ['zero', 'one', 'upper', 'random', 'frontloaded']:
            P = problem(label, 20, 60)
            start = {
                'zero': np.zeros(P.shape),
                'one': np.ones(P.shape),
                'upper': np.full(P.shape, 2.0),
                'random': np.random.default_rng(RNG_SEED).uniform(0, 2, P.shape),
                'frontloaded': np.broadcast_to(np.where(P.t < P.par.T * 0.45, 2.0, 0.0)[:, None], P.shape).copy(),
            }[kind]
            name = f'multistart_{label}_{kind}'
            u, r = attempt(P, start, name, maxiter=2400, spectrum=True)
            r.update(label=label, sense='multistart', start=kind, eta=0.8)
            r['common_reference'] = continuous_time_reference(P, u, M=80)
            rows.append(r)
            save('branches', rows)


def meshes():
    rows = []
    for label in LABELS:
        previous = None
        last = None
        for N in [30, 60, 120, 240, 480, 960]:
            P = problem(label, 20, N)
            start = None if previous is None else interpolate_control(previous, P, last)
            u, r = attempt(
                P,
                start,
                name=f'temporal_{label}_{N}',
                maxiter=2600,
                spectrum=N == 60,
            )
            r.update(label=label, refinement='temporal')
            if previous is not None:
                r['distance_from_previous'] = P.norm(u - interpolate_control(previous, P, last))
            r['common_reference'] = continuous_time_reference(P, u, M=80)
            rows.append(r)
            save('meshes', rows)
            previous, last = (P, u) if r['accepted'] else (None, None)
        previous = None
        last = None
        for M in [10, 20, 40, 80]:
            P = problem(label, M, 120)
            start = None if previous is None else interpolate_control(previous, P, last)
            u, r = attempt(P, start, name=f'spatial_{label}_{M}', maxiter=2600, spectrum=False)
            r.update(label=label, refinement='spatial')
            if previous is not None:
                r['distance_from_previous'] = P.norm(u - interpolate_control(previous, P, last))
            r['common_reference'] = continuous_time_reference(P, u, M=80)
            rows.append(r)
            save('meshes', rows)
            previous, last = (P, u) if r['accepted'] else (None, None)
    cross = []
    for N in [30, 60, 120, 240, 480, 960]:
        P = problem('flux', 20, N)
        u0 = np.load(OUT / f'temporal_left_{N}.npz')['u']
        u1 = np.load(OUT / f'temporal_flux_{N}.npz')['u']
        ui = np.load(OUT / f'temporal_imex_{N}.npz')['u']
        cross.append(dict(
            M=20,
            N=N,
            left_flux_distance=P.norm(u0 - u1),
            imex_flux_distance=P.norm(ui - u1),
        ))
    save('cross_meshes', cross)


def reference():
    rows = []
    mesh = load('meshes', [])
    for label in LABELS:
        name = f'spatial_{label}_80'
        entry = next((r for r in mesh if r['label'] == label and r['refinement'] == 'spatial' and (r['M'] == 80)))
        P = problem(label, 80, 120)
        u = np.load(OUT / f'{name}.npz')['u']
        for M in [80, 160]:
            for rtol in [1e-08, 1e-10] if M == 80 else [1e-10]:
                r = continuous_time_reference(P, u, M=M, rtol=rtol, atol=rtol * 0.01)
                r.update(
                    label=label,
                    stationary_candidate_accepted=entry['accepted'],
                    discrete_cost=entry['objective'],
                )
                rows.append(r)
                save('reference_accuracy', rows)
        for N in [1920, 3840]:
            V = problem('imex', 80, N)
            ur = interpolate_control(P, V, u)
            r = dict(
                label=label,
                M=80,
                N=N,
                objective=V.objective(ur),
                temporal_method='IMEX and old-state reward',
            )
            rows.append(r)
            save('reference_accuracy', rows)
if __name__ == '__main__':
    save(
        'environment',
        dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            numba=numba.__version__,
            platform=platform.platform(),
            seed=RNG_SEED,
            stationarity_acceptance=1e-06,
            lbfgsb_requested_tolerance=2e-07,
            default_parameters=asdict(Parameters()),
        ),
    )
    for phase in [
        'derivatives',
        'perturbation',
        'parameters',
        'branches',
        'meshes',
        'reference',
    ]:
        if ARGS.phase in ['all', phase]:
            start = perf_counter()
            globals()[phase]()
            print('PHASE COMPLETE', phase, perf_counter() - start, flush=True)
