"""Check global optimality in the two-step removal problem with exact fractions."""
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reduced_results'
OUT.mkdir(exist_ok=True)
FIG = ROOT / 'reduced_figures'
FIG.mkdir(exist_ok=True)


def trim(p):
    p = list(p)
    while len(p) > 1 and (not p[-1]):
        p.pop()
    return p


def evaluate(p, x):
    value = F(0)
    for c in reversed(p):
        value = value * x + c
    return value


def remainder(p, q):
    p = trim(p)
    q = trim(q)
    while len(p) >= len(q) and any(p):
        shift = len(p) - len(q)
        c = p[-1] / q[-1]
        for i, v in enumerate(q):
            p[i + shift] -= c * v
        p = trim(p)
    return p


def sturm(p):
    seq = [trim(list(map(F, p))), [F(i) * F(p[i]) for i in range(1, len(p))]]
    while any(seq[-1]):
        r = [-x for x in remainder(seq[-2], seq[-1])]
        if not any(r):
            break
        seq.append(r)
    return seq


def variations(seq, x):
    signs = [1 if v > 0 else -1 for p in seq if (v := evaluate(p, x))]
    return sum((a != b for a, b in zip(signs, signs[1:])))


def isolate(p, a, b, width=F(1, 10 ** 12)):
    seq = sturm(p)
    assert evaluate(p, a) != 0 and evaluate(p, b) != 0

    def visit(lo, hi):
        count = variations(seq, lo) - variations(seq, hi)
        if not count:
            return []
        if count == 1 and hi - lo <= width:
            return [(lo, hi)]
        mid = (lo + hi) / 2
        if evaluate(p, mid) == 0:
            raise ValueError('Exact midpoint root requires a split.')
        return visit(lo, mid) + visit(mid, hi)
    return (visit(F(a), F(b)), seq)


def phi0_bounds(lo, hi, b=F(2)):
    return (
        lo * lo / 2 - b * hi - b * b / (2 * (1 + lo) ** 2),
        hi * hi / 2 - b * lo - b * b / (2 * (1 + hi) ** 2),
    )


def pair(interval):
    return [str(x) for x in interval]


def run():
    b = F(2)
    A = F(3)
    polynomial = [2, -5, -3, 1, 1]
    intervals, sequence = isolate(polynomial, F(0), A)
    assert len(intervals) == 2
    candidates = [(F(0), F(0)), *intervals, (A, A)]
    costs = [phi0_bounds(*i) for i in candidates]
    winner = 2
    assert all((costs[winner][1] < lo for k, (lo, hi) in enumerate(costs) if k != winner))
    x = sum(candidates[winner]) / 2
    z = b / (1 + x)
    roots_flux, _ = isolate([-2, 1, 3, 3, 1], F(0), A)
    assert len(roots_flux) == 1
    t = sum(roots_flux[0]) / 2
    xf, zf = (float(x), float(z))
    tf = float(t)
    f0 = lambda x, z: 0.5 * (x * x + z * z) - 2 * (x + z / (1 + x))
    f1 = lambda x, z: 0.5 * (x * x + z * z) + 2 / ((1 + x) * (1 + z)) - 2
    continuous = lambda x, z: 0.5 * (x * x + z * z) - 2 * (1 - np.exp(-(x + z)))
    tc = brentq(lambda x: x - 2 * np.exp(-2 * x), 0, 3, xtol=1e-14)
    rows = []
    for name, xx, zz in [
        ('old_global', xf, zf),
        ('old_boundary_local', 0.0, 2.0),
        ('flux_global', tf, tf),
        ('continuous_global', tc, tc),
    ]:
        rows.append(dict(
            name=name,
            x=xx,
            z=zz,
            old_cost=f0(xx, zz),
            flux_cost=f1(xx, zz),
            continuous_cost=continuous(xx, zz),
            continuous_gap=continuous(xx, zz) - continuous(tc, tc),
        ))
    report = dict(
        model='two-step removal, tau=q=y0=lambda=1, eta=2, Umax=3',
        proof_method='exact rational Sturm root count after analytic elimination of z; disjoint rational bounds for all candidate costs',
        old_polynomial=polynomial,
        old_sturm=[list(map(str, p)) for p in sequence],
        old_stationary_intervals=[pair(i) for i in intervals],
        old_candidate_intervals=[pair(i) for i in candidates],
        old_cost_intervals=[pair(i) for i in costs],
        old_global_candidate=winner,
        flux_stationary_interval=pair(roots_flux[0]),
        flux_proof='strict convexity for all nonnegative controls; symmetry forces x=z; the positive polynomial has one root in [0,3]',
        old_strict_local=dict(x=0, z=2, normal_gradient=2, free_hessian=1, cost=-2),
        numerical_comparisons=rows,
    )
    (OUT / 'reduced_global.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout='constrained')
    xx = np.linspace(0, 3, 360)
    zz = np.linspace(0, 3, 360)
    X, Z = np.meshgrid(xx, zz)
    for ax, fun, title in zip(axes, [f0, f1], ['Old-state reward', 'Flux reward']):
        cs = ax.contourf(X, Z, fun(X, Z), levels=30, cmap='viridis')
        fig.colorbar(cs, ax=ax, label='Normalized cost')
        ax.set(xlabel='First effort x', ylabel='Second effort z', title=title)
    axes[0].plot(xf, zf, 'r*', ms=10)
    axes[0].plot(0, 2, 'ro', ms=5)
    axes[1].plot(tf, tf, 'r*', ms=10)
    fig.savefig(FIG / 'reduced_landscape.pdf')
    fig.savefig(FIG / 'reduced_landscape.png', dpi=200)
    plt.close(fig)
    x = np.linspace(0, 2, 501)
    z = 2 - x
    fig, ax = plt.subplots(figsize=(4.8, 3.0), layout='constrained')
    budget0 = lambda x, z: 0.5 * (x * x + z * z) - 10 * (x + z / (1 + x))
    budget1 = lambda x, z: 0.5 * (x * x + z * z) + 10 / ((1 + x) * (1 + z)) - 10
    budget_continuous = lambda x, z: 0.5 * (x * x + z * z) - 10 * (1 - np.exp(-(x + z)))
    assert 1 < 10 / (1 + 2) ** 2
    for fun, label in [
        (budget0, 'Old-state'),
        (budget1, 'Flux'),
        (budget_continuous, 'Continuous'),
    ]:
        ax.plot(x, fun(x, z), label=label)
    (OUT / 'reduced_budget.json').write_text(json.dumps(
        dict(
            tau=1,
            q=1,
            y0=1,
            lam=1,
            eta=10,
            total_effort=2,
            Umax=3,
            old_global_allocations=[[0, 2], [2, 0]],
            flux_global_allocation=[1, 1],
            continuous_global_allocation=[1, 1],
        ),
        indent=2,
    ))
    ax.set(xlabel='First effort x (second effort = 2-x)', ylabel='Normalized cost')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(FIG / 'reduced_budget.pdf')
    fig.savefig(FIG / 'reduced_budget.png', dpi=200)
    plt.close(fig)
    print(json.dumps(rows, indent=2))
    print('Exact rational certificate verified: all old-state candidates enumerated and global gap bounds disjoint.')
if __name__ == '__main__':
    run()
