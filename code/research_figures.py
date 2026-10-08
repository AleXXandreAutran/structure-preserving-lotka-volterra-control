"""Draw the figures for the parameter and grid studies."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir', type=Path, default=None)
parser.add_argument('--figure-dir', type=Path, default=None)
args = parser.parse_args()
root = HERE.parent if (HERE / 'lotka.py').exists() else HERE
OUT = args.results_dir or root / ('research_results' if (HERE / 'lotka.py').exists() else 'results')
FIG = args.figure_dir or root / ('research_figures' if (HERE / 'lotka.py').exists() else 'figures')
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.bbox': 'tight',
    'figure.constrained_layout.use': True,
})
COLORS = {'imex': '#555555', 'left': '#b24c2f', 'flux': '#246a98'}
LABEL = {'imex': 'IMEX', 'left': 'old-state reward', 'flux': 'flux reward'}


def load(name):
    return json.loads((OUT / f'{name}.json').read_text())


def finish(fig, name):
    fig.savefig(FIG / f'{name}.pdf')
    fig.savefig(FIG / f'{name}.png', dpi=180)
    plt.close(fig)
a = load('perturbation')
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
for kind, style in [('smooth', '-'), ('alternating', '--')]:
    rr = [r for r in a if r['M'] == 32 and r['control'] == kind]
    for ax, key in zip(axes[:2], ['gradient_difference', 'hessian_difference_norm']):
        ax.loglog(
            [r['dt'] for r in rr],
            [r[key] for r in rr],
            style + 'o',
            label=kind + ' control',
            markersize=3,
        )
        ax.set_xlabel('time step τ')
        ax.grid(alpha=0.2)
    if kind == 'smooth':
        for label in ['left', 'flux']:
            axes[2].semilogx(
                [r['dt'] for r in rr],
                [r['rayleigh']['alternating'][label] for r in rr],
                'o-',
                color=COLORS[label],
                label=LABEL[label],
                markersize=3,
            )
axes[0].set_ylabel('‖g₀ − g₁‖₂')
axes[1].set_ylabel('‖H₀ − H₁‖₂→₂ (Ritz estimate)')
axes[2].set_ylabel('alternating-direction curvature')
axes[2].set_xlabel('time step τ')
axes[2].axhline(0, color='black', lw=0.7)
axes[2].axhline(0.05, color='#888888', lw=0.7, ls=':', label='effort weight λ')
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
finish(fig, 'quadrature_curvature')
b = load('branches')
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
for label in ['left', 'flux']:
    for sense, style in [('up', '-'), ('down', '--')]:
        rr = sorted([r for r in b if r['label'] == label and r['sense'] == sense], key=lambda r: r['eta'])
        axes[0].plot(
            [r['eta'] for r in rr],
            [r['temporal_variation'] for r in rr],
            style + 'o',
            color=COLORS[label],
            label=LABEL[label] + ', ' + sense,
            markersize=3,
        )
    rr = sorted([r for r in b if r['label'] == label and r['sense'] == 'down'], key=lambda r: r['eta'])
    axes[1].plot(
        [r['eta'] for r in rr],
        [r['distance_to_up_branch'] for r in rr],
        'o-',
        color=COLORS[label],
        label=LABEL[label],
        markersize=3,
    )
for sense, style in [('up', '-'), ('down', '--')]:
    p0 = sorted([r for r in b if r['label'] == 'left' and r['sense'] == sense], key=lambda r: r['eta'])
    p1 = sorted([r for r in b if r['label'] == 'flux' and r['sense'] == sense], key=lambda r: r['eta'])
    axes[2].plot(
        [r['eta'] for r in p0],
        [r0['common_reference']['objective'] - r1['common_reference']['objective'] for r0, r1 in zip(p0, p1)],
        style + 'o',
        label=sense,
        markersize=3,
    )
axes[0].set_ylabel('temporal variation')
axes[1].set_ylabel('L² distance between directional paths')
axes[2].set_ylabel('common-reference cost, old minus flux')
for ax in axes:
    ax.set_xlabel('harvest value η')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
finish(fig, 'directional_continuation')
m = load('meshes')
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
for label in ['imex', 'left', 'flux']:
    rr = sorted([r for r in m if r['refinement'] == 'temporal' and r['label'] == label], key=lambda r: r['N'])
    axes[0].semilogx(
        [r['N'] for r in rr],
        [r['temporal_variation'] for r in rr],
        'o-',
        color=COLORS[label],
        label=LABEL[label],
        markersize=3,
    )
    axes[1].semilogx(
        [r['N'] for r in rr],
        [r['common_reference']['objective'] for r in rr],
        'o-',
        color=COLORS[label],
        label=LABEL[label],
        markersize=3,
    )
    axes[2].loglog(
        [r['N'] for r in rr[1:]],
        [r['distance_from_previous'] for r in rr[1:]],
        'o-',
        color=COLORS[label],
        label=LABEL[label],
        markersize=3,
    )
axes[0].set_ylabel('temporal variation')
axes[1].set_ylabel('common-reference objective')
axes[2].set_ylabel('L² change after temporal refinement')
for ax in axes:
    ax.set_xlabel('time intervals N (M = 20)')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
finish(fig, 'temporal_refinement')
p = load('parameters')
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
for ax, pname, caption in zip(
    axes,
    ['eta', 'lambda', 'T'],
    ['harvest value η', 'effort weight λ', 'horizon T (τ ≈ 1/15)'],
):
    for label in ['left', 'flux']:
        rr = sorted([r for r in p if r['parameter_name'] == pname and r['label'] == label], key=lambda r: r['parameter_value'])
        ax.plot(
            [r['parameter_value'] for r in rr],
            [r['temporal_variation'] for r in rr],
            'o-',
            color=COLORS[label],
            label=LABEL[label],
            markersize=3,
        )
    ax.set_xlabel(caption)
    ax.set_ylabel('temporal variation')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
finish(fig, 'parameter_switching')
fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.5))
for ax, name, title in zip(
    axes.ravel(),
    [
        'branch_left_up_0p8',
        'branch_left_down_0p8',
        'multistart_left_random',
        'multistart_flux_zero',
    ],
    [
        'Old-state reward: ascending path',
        'Old-state reward: descending path',
        'Old-state reward: random start',
        'Flux reward: zero start',
    ],
):
    u = np.load(OUT / f'{name}.npz')['u']
    im = ax.imshow(
        u,
        origin='lower',
        aspect='auto',
        extent=[0, 1, 0, 4],
        vmin=0,
        vmax=2,
        cmap='viridis',
        rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('space x')
    ax.set_ylabel('time t')
fig.colorbar(im, ax=axes.ravel().tolist(), label='control', shrink=0.8)
finish(fig, 'path_controls')
fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.1))
for label in ['imex', 'left', 'flux']:
    rr = sorted([r for r in m if r['refinement'] == 'spatial' and r['label'] == label], key=lambda r: r['M'])
    for ax, key in zip(axes, ['objective', 'temporal_variation']):
        vv = [r['common_reference']['objective'] if key == 'objective' else r[key] for r in rr]
        ax.semilogx(
            [r['M'] for r in rr],
            vv,
            'o-',
            color=COLORS[label],
            label=LABEL[label],
            markersize=3,
        )
        for r, value in zip(rr, vv):
            if not r['accepted']:
                ax.plot(r['M'], value, 'x', markersize=8, color='black')
axes[0].set_ylabel('common-reference objective')
axes[1].set_ylabel('temporal variation')
for ax in axes:
    ax.set_xlabel('space intervals M (N = 120)')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
finish(fig, 'spatial_refinement')
