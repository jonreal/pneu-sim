# Kang Analytical Baseline

This folder contains the Kang-type analytical baseline for pneumatic artificial muscle joint-response prediction.

This baseline is different from HNODE, Koopman, and GRU. It does not train or save a reusable neural-network or operator checkpoint. Each run starts from the raw data, fits the Kang force-model coefficients from the single-PAM data, and then evaluates the antagonistic joint datasets.

## Folder Structure

```text
Kang (analytical)/
├── main_Kang.py
├── Kang/
│   ├── __init__.py
│   ├── dynamics.py
│   ├── evaluate.py
│   ├── force_model.py
│   ├── single_pam.py
│   └── utils.py
├── checkpoint_Kang/
└── readme.md
```

`checkpoint_Kang/` is an output folder only. It is not a saved-model folder.

## File Roles

- `main_Kang.py`: command-line entry point.
- `Kang/single_pam.py`: reads single-PAM data and fits Kang force coefficients.
- `Kang/force_model.py`: contains the Kang force equation.
- `Kang/dynamics.py`: simulates the antagonistic joint dynamics.
- `Kang/evaluate.py`: runs the joint datasets and saves evaluation outputs.
- `Kang/utils.py`: helper functions.

## Required Data

The script needs two data locations:

```text
pneu-sim/
├── Single PAM data/
└── TRAIN DATA/
```

The joint datasets are loaded with the shared top-level loader:

```python
from hnode.data.loaders import generate_file_paths, load_training_data_from_file
```

## Basic Run

Run from the top-level `pneu-sim/` folder:

```bash
python "comparisions/Kang (analytical)/main_Kang.py"
```

This run will:

1. load the selected joint datasets,
2. fit the Kang coefficients from the single-PAM data,
3. simulate the antagonistic joint response,
4. save the run outputs in `checkpoint_Kang/`.

## Select Joint Datasets

Use dataset codes:

```bash
python "comparisions/Kang (analytical)/main_Kang.py" --codes <code1> <code2>
```

or use dataset ranges:

```bash
python "comparisions/Kang (analytical)/main_Kang.py" --ranges <start-end>
```

## Custom Data Folders

```bash
python "comparisions/Kang (analytical)/main_Kang.py" \
  --single-pam-dir "Single PAM data" \
  --data-dir "TRAIN DATA"
```

## Useful Options

```text
--single-pam-ids       choose which single-PAM data IDs are used for fitting
--adam-steps           number of Adam fitting iterations
--lbfgs-steps          number of LBFGS fitting iterations
--D0-mm                initial PAM diameter used in the Kang model
--L0-mm                initial PAM length used in the Kang model
--alpha0-deg           initial braid angle used in the Kang model
--m-eff                effective mass used in the joint simulation
--b-eff                effective damping used in the joint simulation
--output-dir           output folder for this run
--save-response-data   save full response data for each evaluated joint dataset
```

## Important Notes

There is no `--resume` option and no `--load-model` option for this baseline.

The outputs are generated fresh every time the script runs. If you change the single-PAM data, joint data, fitted parameter settings, or joint simulation parameters, rerun the script from the beginning.
