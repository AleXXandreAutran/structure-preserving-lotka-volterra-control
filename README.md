# Lotka–Volterra optimal control

This code solves a harvesting problem for a predator–prey model with diffusion. It compares IMEX with a positive triangular scheme. For the positive scheme, the reward uses either the prey population at the start of each step or the harvest actually removed.

The cost penalizes tracking error and effort, and rewards the catch.

The model is

$$
\begin{aligned}
\partial_t v_1 &= \mu_1\partial_{xx}v_1+v_1(-m+\alpha v_2),\\
\partial_t v_2 &= \mu_2\partial_{xx}v_2+v_2(r-\beta v_1-qu),
\end{aligned}
$$

on $x\in(0,1)$ and $t\in(0,T)$, with zero-flux boundaries and $0\leq u\leq u_{\max}$. Here $v_1$ denotes predators, $v_2$ prey, and $u$ harvest effort. The rates, initial populations and target profiles are set in `code/lotka.py`.

## Run the code

Use Python 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q code
```

Generate the main comparisons and their figures:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/run_experiments.py
python code/make_figures.py
```

Run the parameter, branch and grid studies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/research_experiments.py
python code/research_figures.py
```

Run the smaller examples and algebraic checks:

```bash
python code/reduced_global.py
python code/pd_network.py
python code/check_quadrature_theory.py
```

The full optimization runs take longer than the tests. Use `--help` to see the available options. If you use `--phase`, run `reference` before `grids` or `multistart` in `run_experiments.py`, and `meshes` before `reference` in `research_experiments.py`.

## Files and results

`code/lotka.py` contains the solver, discrete adjoint, Hessian actions and optimization methods. Separate examples check two-step removal and a production–destruction network.

The scripts create their output folders automatically:

| Study | Data | Figures |
| --- | --- | --- |
| Scheme and solver comparisons | `results/` | `figures/` |
| Parameters, branches and grid refinement | `research_results/` | `research_figures/` |
| Two-step removal | `reduced_results/` | `reduced_figures/` |

Network and algebraic checks are saved in `validation/`. Parameters and run details are stored with the generated data. Rerunning experiments replaces that data; run the plotting scripts afterwards to update the figures.

A run is marked accepted when its projected stationarity residual is at most $10^{-6}$. Failed attempts are kept. The two-step example has a global optimality certificate using exact fractions. The coupled reference uses adaptive time integration on a spatial grid.
