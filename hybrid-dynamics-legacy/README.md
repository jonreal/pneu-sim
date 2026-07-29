# hybrid-dynamics-legacy

**Reference only — not part of the active pipeline.**

An early MATLAB hybrid-dynamics model of pneumatic artificial muscle (PAM)
joints. It predates the hybrid physics-structured Neural ODE in this
repository and is kept for historical reference: nothing in the current Python
pipeline (`main.py`, `hnode/`, `comparisions/`) imports or depends on it.

## Contents

| File | Purpose |
| --- | --- |
| `sim_pneus.m` | Closed-loop simulation of the hybrid model. Self-contained entry point. |
| `make_model.m` | Builds the model struct; parameters as name-value options. |
| `odeHybrid.m` | Hybrid ODE integration with guard/reset handling. |
| `fit_model.m` | Fits model parameters against the empirical dataset. |
| `plot_sim.m` | Plotting helpers for simulation output. |
| `make_emperical_data.m` | Preprocessing that produced the fitting dataset. |

## Data not included

The empirical data this model was fit against (~56 MB) is not tracked here:

- `data/e2_5050` — raw benchtop log
- `data/data.mat` — preprocessed `F`, `x`, `dx`, `p1`, `p2` derived from it

Two scripts therefore will not run as-is:

- `fit_model.m` expects `load data/data.mat`
- `make_emperical_data.m` reads `e2_5050` and also calls
  `embedded_process_data()`, which is not in this repo (a Python port exists at
  `comparisions/Kang/Kang/single_pam.py`)

`sim_pneus.m`, `make_model.m`, and `odeHybrid.m` carry their parameters inline
and still run standalone.
