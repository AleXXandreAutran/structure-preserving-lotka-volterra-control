# Lotka–Volterra optimal control

Minimal reproducibility repository for the numerical methods used in the article **Structure-Preserving Discrete-Adjoint Optimization for Diffusive Lotka–Volterra Harvesting**.

The code implements:

- the IMEX state scheme;
- the positive transfer-implicit state scheme;
- left-endpoint and flux-aligned harvest objectives;
- exact discrete adjoints;
- tangent equations and Hessian-vector products;
- L-BFGS-B, relaxed forward–backward sweep, Anderson acceleration, and spectral projected gradient;
- matrix-free free-Hessian spectral estimates;
- the numerical experiments and derivative tests used in the article.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Tests

```bash
python -m pytest -q code/test_v2.py
```

## Run the experiments

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/run_experiments.py
```

Specific phases can be run separately:

```bash
python code/run_experiments.py --phase derivatives
python code/run_experiments.py --phase reference
python code/run_experiments.py --phase grids
python code/run_experiments.py --phase coarse
python code/run_experiments.py --phase fixed
python code/run_experiments.py --phase multistart
```

The numerical outputs are written to `results/`.

## Generate the figures

```bash
python code/make_figures.py
```

The figures are written to `figures/`.

## Minimal example

```python
import sys
sys.path.insert(0, "code")

from lotka_v2 import Problem, optimize

problem = Problem(M=80, N=120, scheme="positive", harvest_rule="flux")
u, report = optimize(problem)

print(report["objective"])
print(report["residual"])
```

The default parameters, initial data, and target profiles are those used in the reference experiment of the article.
