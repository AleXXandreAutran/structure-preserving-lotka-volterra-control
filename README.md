# Lotka–Volterra optimal control

Numerical optimal control for a diffusive Lotka–Volterra predator–prey system.

The repository includes IMEX and positivity-preserving schemes, exact discrete adjoints, Hessian-vector products, and several constrained optimization methods.

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

## Run

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/run_experiments.py
```

Individual experiment groups can be run with:

```bash
python code/run_experiments.py --phase derivatives
python code/run_experiments.py --phase reference
python code/run_experiments.py --phase grids
python code/run_experiments.py --phase coarse
python code/run_experiments.py --phase fixed
python code/run_experiments.py --phase multistart
```

Results are saved in `results/`.

## Figures

```bash
python code/make_figures.py
```

Figures are saved in `figures/`.

## Example

```python
import sys
sys.path.insert(0, "code")

from lotka_v2 import Problem, optimize

problem = Problem(M=80, N=120, scheme="positive", harvest_rule="flux")
u, report = optimize(problem)

print(report["objective"])
print(report["residual"])
```
