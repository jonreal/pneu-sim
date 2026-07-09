# Koopman Baseline

This folder contains the Koopman/EDMDc baseline for PAM joint-response prediction.

## Folder Structure

```text
Koopman/
├── main_Koopman.py
├── Koopman/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   ├── lifting.py
│   ├── model.py
│   ├── plot.py
│   └── utils.py
├── checkpoint_Koopman/
└── readme.md
```

## File Roles

- `main_Koopman.py`: command-line entry point.
- `Koopman/lifting.py`: lifting functions.
- `Koopman/data.py`: data preparation and snapshot matrices.
- `Koopman/model.py`: training, rollout, saving, and loading.
- `Koopman/evaluate.py`: R2/RMSE and rollout-loss evaluation.
- `Koopman/plot.py`: plotting utilities.
- `Koopman/utils.py`: helper functions.

The script uses the shared top-level data loader:

```python
from hnode.data.loaders import generate_file_paths, load_training_data_from_file
```

## Basic Run

Run from the top-level `pneu-sim/` folder:

```bash
python comparisions/Koopman/main_Koopman.py
```

By default, the script first looks for the saved Koopman model in `checkpoint_Koopman/`. If the model is present, it loads the saved model and skips training. If it is missing, the script trains a new model and saves it to `checkpoint_Koopman/`.

## Train a New Model

```bash
python comparisions/Koopman/main_Koopman.py --train-new
```

Use selected training datasets:

```bash
python comparisions/Koopman/main_Koopman.py --train-new --codes 1010 4545 8080
```

Use separate evaluation datasets:

```bash
python comparisions/Koopman/main_Koopman.py --codes 1010 4545 8080 --eval-codes 3030 5050
```

## Evaluate and Plot

```bash
python comparisions/Koopman/main_Koopman.py --eval-r2 --save-plots
```

Disable plotting:

```bash
python comparisions/Koopman/main_Koopman.py --no-plot
```

## Useful Options

```text
--data-dir              folder containing nn_xxyy data files
--codes                 selected training dataset codes
--ranges                selected training dataset ranges
--eval-codes            selected evaluation dataset codes
--eval-ranges           selected evaluation dataset ranges
--lift-type             lifting dictionary type
--ridge                 ridge regularization value
--force-scale           force scaling divisor
--warmup-steps          measured warm-up samples before rollout
--clip-x-norm           normalized-state clipping during rollout
--load-model-path       load an existing Koopman model and skip training
--train-new             ignore the saved model and train from scratch
--save-checkpoint-dir   folder for saved model and run outputs
--eval-r2               save R2/RMSE results
--save-plots            save plots
--no-plot               disable plotting
```

## Notes

The saved `.npz` file is the reusable Koopman model. R2/RMSE files and response-data files are outputs for checking results; they are not the trained model.

The package folder is named `Koopman` with a capital `K`; keep the import name and folder name consistent.
