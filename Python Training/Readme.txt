# Hybrid Neural ODE for Antagonistic PAM Joint

This repository contains the implementation of a hybrid physics-structured Neural ODE model for learning the dynamics of an antagonistic pneumatic artificial muscle (PAM) joint. The model integrates physical structure with a neural network to capture complex nonlinear behaviors.

---

## Repository Structure

.
├── TRAIN DATA/
│   ├── nn_3060
│   ├── nn_5050
│   └── nn_8010
│
├── checkpoint/
│   ├── hybrid_model_puresine_16.eqx
│   ├── hybrid_opt_state_puresine_16.eqx
│   ├── best_info_puresine_16.txt
│   └── Read Me
│
├── hnode/
│   ├── core/
│   │   └── models.py
│   ├── data/
│   │   └── loaders.py
│   ├── train/
│   │   └── loop.py
│   └── plot/
│       └── plots.py
│
├── main.py
└── poly44_all_fits.mat

---

## Dataset

The `TRAIN DATA` folder contains three example datasets:

- nn_3060  
- nn_5050  
- nn_8010  

These datasets are provided for demonstration and testing purposes. Additional datasets can be added following the same naming and format.

---

## Checkpoint

The `checkpoint` folder contains:

- Final trained model (`.eqx`)
- Optimizer state (`.eqx`)
- Training summary (`.txt`)

Only the final trained result is included. Intermediate training stages are not provided.

The `Read Me` file inside the checkpoint folder describes the staged training process used to obtain the final model.

---

## Requirements

- Python 3.9+
- JAX
- Equinox
- Diffrax
- Optax
- NumPy
- SciPy
- Matplotlib

---

## Usage

### Train Model

Run training using:

python main.py --data-dir "TRAIN DATA"

Optional arguments include:

--codes 3060 5050 8010     # specify datasets  
--epochs 10000             # number of training epochs  
--lr 1e-2                  # learning rate  
--save-plots               # save plots to checkpoint folder  

---

### Resume Training

python main.py --resume \
    --load-model-name hybrid_model_puresine_16.eqx \
    --load-opt-name hybrid_opt_state_puresine_16.eqx

---

### Evaluate R²

python main.py --eval-r2

Results will be saved to:

Verification/R2_results_225.csv

---

### Plot Results

python main.py --save-plots

Plots will be saved to the checkpoint directory.

---

## Model Overview

The system is modeled as a hybrid neural ODE with state:

y = [x, dx, Pf, Pe]

where:
- x: displacement (mm)
- dx: velocity (mm/s)
- Pf, Pe: chamber pressures (kPa)

A neural network is used to model the net force:

F = NN(mf, me, x, dx)

while the pressure dynamics follow a physics-based formulation using chamber geometry and thermodynamics.

---

## Notes

- Data loading and preprocessing are implemented in `hnode/data/loaders.py`
- Model definition is in `hnode/core/models.py`
- Training loop is in `hnode/train/loop.py`
- Plotting utilities are in `hnode/plot/plots.py`
- Main entry point is `main.py`

The implementation follows a direct simulation-based training approach using Diffrax ODE solvers.

---

## License

For research and academic use.