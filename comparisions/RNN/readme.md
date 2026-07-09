# GRU / RNN Baseline

This folder contains the GRU recurrent-network baseline for PAM joint-response prediction.
The folder is named `RNN` because it is the recurrent-neural-network comparison baseline, while the implemented model is a GRU.

## Folder Structure

```text
RNN/
├── main_RNN.py
├── RNN/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── plot.py
│   ├── train.py
│   └── utils.py
├── checkpoint_RNN/
└── readme.md
```

## File Roles

- `main_RNN.py`: command-line entry point.
- `RNN/dataset.py`: sequence dataset construction.
- `RNN/model.py`: GRU model and checkpoint loading.
- `RNN/train.py`: training and checkpoint saving.
- `RNN/evaluate.py`: rollout and R2/RMSE evaluation.
- `RNN/plot.py`: plotting utilities.
- `RNN/utils.py`: helper functions.

The script uses the shared top-level data loader:

```python
from hnode.data.loaders import generate_file_paths, load_training_data_from_file
```

## Basic Run

Run from the top-level `pneu-sim/` folder:

```bash
python comparisions/RNN/main_RNN.py
```

By default, the script first looks for the saved GRU checkpoint in `checkpoint_RNN/`. If the checkpoint is present, it loads the saved model and skips training. If it is missing, the script trains a new model and saves it to `checkpoint_RNN/`.

## Train a New Model

```bash
python comparisions/RNN/main_RNN.py --train-new
```

Use selected datasets:

```bash
python comparisions/RNN/main_RNN.py --train-new --codes 1010 4545 8080
```

or:

```bash
python comparisions/RNN/main_RNN.py --train-new --ranges 10-80
```

## Evaluate and Plot

```bash
python comparisions/RNN/main_RNN.py --eval-r2 --save-plots
```

Disable plotting:

```bash
python comparisions/RNN/main_RNN.py --no-plot
```

## Useful Options

```text
--data-dir              folder containing nn_xxyy data files
--codes                 selected dataset codes
--ranges                selected dataset ranges
--seq-len               input history length
--pred-horizon          recursive training horizon
--hidden-dim            GRU hidden dimension
--num-layers            number of GRU layers
--batch-size            batch size
--lr                    learning rate
--epochs                maximum training epochs
--patience              early-stopping patience
--use-mass              include mf and me as constant inputs
--load-model-path       load an existing checkpoint and skip training
--train-new             ignore the saved checkpoint and train from scratch
--save-checkpoint-dir   folder for saved model and run outputs
--eval-r2               save R2/RMSE results
--save-plots            save plots
--no-plot               disable plotting
```

## Notes

The saved checkpoint is the reusable GRU model. R2/RMSE files and response-data files are outputs for checking results; they are not the trained model.
