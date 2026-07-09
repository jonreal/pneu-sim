# pneu-sim

This repository contains simulation and learning-based models for an antagonistic pneumatic artificial muscle (PAM) joint.
The main model is a hybrid physics-structured Neural ODE. The repository also includes GRU/RNN, Koopman, and Kang analytical comparison baselines.

## Repository Structure

```text
pneu-sim/
├── checkpoint/
├── hnode/
├── main.py
├── README.md
├── poly44_all_fits.mat
└── comparisions/
    ├── RNN/
    │   ├── main_RNN.py
    │   ├── RNN/
    │   ├── checkpoint_RNN/
    │   └── readme.md
    ├── Koopman/
    │   ├── main_Koopman.py
    │   ├── Koopman/
    │   ├── checkpoint_Koopman/
    │   └── readme.md
    └── Kang (analytical)/
        ├── main_Kang.py
        ├── Kang/
        ├── checkpoint_Kang/
        └── readme.md
```

## Main Model: HNODE

The top-level `main.py` runs the hybrid Neural ODE model.

```bash
python main.py --data-dir "TRAIN DATA"
```

Use selected datasets:

```bash
python main.py --data-dir "TRAIN DATA" --codes 1010 4545 8080
```

Resume from the saved HNODE checkpoint using the default checkpoint names:

```bash
python main.py --data-dir "TRAIN DATA" --resume
```

Save R2 results and plots:

```bash
python main.py --data-dir "TRAIN DATA" --eval-r2 --save-plots
```

## GRU / RNN Baseline

```bash
python comparisions/RNN/main_RNN.py
```

By default, this script looks for the saved GRU checkpoint in `checkpoint_RNN/`. Use `--train-new` to ignore the saved checkpoint and train a new model.

```bash
python comparisions/RNN/main_RNN.py --train-new --codes 1010 4545 8080
```

## Koopman Baseline

```bash
python comparisions/Koopman/main_Koopman.py
```

By default, this script looks for the saved Koopman model in `checkpoint_Koopman/`. Use `--train-new` to ignore the saved model and train a new one.

```bash
python comparisions/Koopman/main_Koopman.py --train-new --codes 1010 4545 8080
```

## Kang Analytical Baseline

```bash
python "comparisions/Kang (analytical)/main_Kang.py"
```

The Kang baseline does not use a reusable saved model. Each run fits the analytical coefficients from the single-PAM data and then evaluates the joint datasets.

## Data Folders

The code expects joint datasets with names such as `nn_1010`, `nn_4545`, and `nn_8080`.
The default joint-data folder name is:

```text
TRAIN DATA/
```

The Kang baseline also needs single-PAM experiment data. The default single-PAM folder name is:

```text
Single PAM data/
```

Custom data folders can be passed with:

```bash
--data-dir <joint_data_folder>
--single-pam-dir <single_pam_data_folder>
```

## Checkpoint Folders

- `checkpoint/`: saved HNODE model and HNODE outputs.
- `comparisions/RNN/checkpoint_RNN/`: saved GRU/RNN model and RNN outputs.
- `comparisions/Koopman/checkpoint_Koopman/`: saved Koopman model and Koopman outputs.
- `comparisions/Kang (analytical)/checkpoint_Kang/`: Kang run outputs only.

For HNODE, GRU/RNN, and Koopman, the checkpoint folders can contain reusable saved models. For Kang, the folder only stores outputs from the current run.

## Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- JAX
- Equinox
- Diffrax
- Optax
- PyTorch

## Notes

Run commands from the top-level `pneu-sim/` folder.
The folder name `comparisions` is kept to match the current project structure.
