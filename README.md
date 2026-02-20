# Hybrid Neural ODE for Antagonistic PAM Joint

This repository contains a hybrid physics-structured Neural ODE model for learning the dynamics of an antagonistic pneumatic artificial muscle (PAM) joint.

---

## Repository Structure

```
hnode_example/
├── README.md
├── main.py
├── poly44_all_fits.mat
├── hnode/
│   ├── core/
│   │   └── models.py
│   ├── data/
│   │   └── loaders.py
│   ├── train/
│   │   └── loop.py
│   └── plot/
│       └── plots.py
├── train_data/
│   ├── nn_3060
│   ├── nn_5050
│   └── nn_8010
└── checkpoint/
    ├── hybrid_model_puresine_16.eqx
    ├── hybrid_opt_state_puresine_16.eqx
    └── best_info_puresine_16.txt
```

---

## Dataset

The `train_data` folder contains three example datasets:

- `nn_3060`
- `nn_5050`
- `nn_8010`

These datasets are provided for demonstration and testing purposes.

---

## Checkpoint

The `checkpoint` folder contains:

- Final trained model (`.eqx`)
- Optimizer state (`.eqx`)
- Training summary (`.txt`)

Only the final trained result is included.

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

```bash
python main.py --data-dir train_data
```

Optional arguments:

```bash
--codes 3060 5050 8010
--epochs 10000
--lr 1e-2
--save-plots
```

---

### Resume Training

```bash
python main.py --resume \
    --load-model-name hybrid_model_puresine_16.eqx \
    --load-opt-name hybrid_opt_state_puresine_16.eqx
```

---

### Evaluate R²

```bash
python main.py --eval-r2
```

---

### Plot Results

```bash
python main.py --save-plots
```

---

## Model Overview

The system is modeled as:

```
y = [x, dx, Pf, Pe]
```

- `x`: displacement (mm)  
- `dx`: velocity (mm/s)  
- `Pf`, `Pe`: chamber pressures (kPa)

The neural network models the net force:

```
F = NN(mf, me, x, dx)
```

---

## Notes

- Data loading: `hnode/data/loaders.py`
- Model: `hnode/core/models.py`
- Training: `hnode/train/loop.py`
- Plotting: `hnode/plot/plots.py`
- Entry point: `main.py`

---

## License

For research and academic use.