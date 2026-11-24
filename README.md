# pneu-sim

Code, trained models, and sample data for the paper

> X. Wang and J. Realmuto, "Physics-Embedded Neural ODEs for Learning Antagonistic Pneumatic
> Artificial Muscle Dynamics," arXiv:2602.23670, 2026.
> https://doi.org/10.48550/arXiv.2602.23670

The main model is a hybrid Neural ODE (HNODE) of an antagonistic pneumatic artificial muscle (PAM)
joint: the joint and gas equations are written explicitly, and a neural network supplies the net
force of the two muscles as a function of position, velocity, and the two chamber air masses. The
repository also contains the three baselines compared in the paper (GRU, Koopman/EDMDc, and a
Kang-type analytical model).

Release `v0.1.0` is the version that accompanies the paper.

## Contents

| Path | What it is |
|---|---|
| `main.py` | Command-line entry point for the hybrid Neural ODE: training, evaluation, plots |
| `hnode/core/models.py` | The model: joint dynamics, pressure dynamics, learned force network |
| `hnode/data/loaders.py` | Reads a recording, filters it, and estimates the chamber air masses |
| `hnode/train/loop.py` | Training loop, loss functions, and R² evaluation |
| `hnode/plot/plots.py` | Time-series and hysteresis-loop plots |
| `checkpoint/` | The trained model used for the results in the paper (see `checkpoint/README.md` for the training curriculum) |
| `poly44_all_fits.mat` | Single-muscle characterization (polynomial fits) used for the air-mass estimate |
| `train_data/` | Three sample recordings |
| `comparisions/` | The baselines, each with its own `readme.md` |
| `pyproject.toml`, `uv.lock` | Dependencies and their locked versions, for `uv` |
| `CITATION.cff` | Citation information |
| `hybrid-dynamics-legacy/` | An earlier MATLAB model, kept for reference only |

## Installation

The dependencies are declared in `pyproject.toml` and locked in `uv.lock`. With
[uv](https://docs.astral.sh/uv/) installed, run this once in the top-level `pneu-sim/` folder:

```bash
uv sync --extra baselines
```

This creates a virtual environment in `.venv/` with the locked versions. `--extra baselines` adds
PyTorch, which only the GRU and Kang baselines need; plain `uv sync` is enough for the hybrid
Neural ODE and the Koopman baseline.

Run every command from the top-level folder and prefix it with `uv run`, which uses that
environment, as in the examples below. (Activating the environment with
`source .venv/bin/activate` and calling `python` directly works as well.)

The code needs Python 3.11 or newer. The JAX stack is pinned to the versions the code was run with
for the paper: JAX 0.8.2, Equinox 0.13.2, Diffrax 0.7.0, and Optax 0.2.6. The commands in this
file were last run from this environment with Python 3.14 on CPU. Without uv, install the packages
listed in `pyproject.toml` with any other tool.

## Data

Each recording is one co-contraction condition. A file named `nn_XXYY` was recorded with the
flexor commanded to `XX` psi and the extensor to `YY` psi (gauge) before the valves were closed;
`nn_3060` is 30 psi flexor, 60 psi extensor. A recording is a tab-separated text file sampled at
1 kHz. The loader uses the frame counter (ms), the joint angle `x` (deg), the motor current
`current` (A), and the measured chamber pressures `pm1` (flexor) and `pm2` (extensor), in psi
gauge. It converts them to tendon displacement (mm), external force (mN), and absolute pressure
(kPa), and estimates the two chamber air masses from the initial pressures and the equilibrium
positions.

`train_data/` holds three sample recordings (`nn_3060`, `nn_5050`, `nn_8010`), enough to run every
command below. The full set of 225 recordings used in the paper is not included in this
repository.

Select recordings with `--codes 3060 5050` (a list) or `--ranges 10-80` (every 5-psi pair in the
range that exists in the folder), and point to a folder with `--data-dir`.

## Quick start: evaluate the trained model

```bash
uv run python main.py --data-dir train_data --codes 3060 5050 8010 --resume --epochs 0 --eval-r2
```

`--resume` loads the trained model from `checkpoint/`, `--epochs 0` skips training, and
`--eval-r2` simulates each recording from its first sample and writes the coefficient of
determination of each state to `checkpoint/HNODE_R2_results.csv`. For the three sample recordings
the displacement R² is 0.965, 0.984, and 0.877. The script then plots the measured and predicted
responses; add `--save-plots` to save the figure in the checkpoint folder.

## Training

One call of `main.py` trains on the listed recordings until the stopping rule is met and keeps the
best model. The paper's model was trained as a curriculum of 16 such calls, each started from the
previous one's best model with more recordings added (`checkpoint/README.md`).

Continue from the trained model:

```bash
uv run python main.py --data-dir train_data --codes 3060 5050 8010 --resume
```

Start from a new, randomly initialized model:

```bash
uv run python main.py --data-dir train_data --codes 5050
```

Notes:

- New models are saved as `checkpoint/hybrid_model_puresine_new.eqx` (with the optimizer state and
  a text file giving the best epoch and loss), so the trained model is not overwritten. Change the
  names and folders with `--load-*` and `--save-*` (`uv run python main.py --help`).
- Each "epoch" is one optimizer step over all listed recordings, each integrated in full.
- After `--patience` epochs without improvement the model returns to the best one and the
  learning rate (`--lr`, default 1e-2) is multiplied by `--reduce-factor` (0.95). Training stops
  after `--max-patience` epochs without improvement, after `--epochs` epochs (5000), or once the
  loss is below `--threshold-per-dataset` times the number of recordings and has stalled for
  `--patience` epochs. The defaults are `--patience 100` and `--max-patience 1000`, so the
  learning rate is reduced at every stall, as described in the paper.
- Training from a random initialization is unstable without the temporary damping term used in
  the first seven curriculum stages (Section III-B of the paper). That term is fixed at zero in
  this code (`b` in `hnode/core/models.py`), which is its value in the final model.

### Faster training

By default the gradient is computed as in the published training: the recordings are integrated
one after another with a fifth-order solver (Tsit5) at fixed 1-ms steps. Two options make this
faster. Neither changes the model or the loss.

| Option | What changes | Loss | Compile | Per step |
|---|---|---|---|---|
| *(default)* | Published configuration | 129.144 | 185 s | 16.8 s |
| `--batched` | All recordings stacked into one solve, same solver | 129.151 | 69 s | 11.6 s |
| `--batched --solver heun --dt 2e-3` | Stacked solve with Heun's method at 2-ms steps | 128.929 | 60 s | 2.8 s |

The numbers are for the three sample recordings and the trained model on an Apple M3 Ultra (CPU).

- `--batched` is the same computation as the default. In this test the loss differed by 5e-5 and
  the gradient by 5e-4 (relative, single precision). Its compilation time and memory do not grow
  with the number of recordings, which matters most for the later curriculum stages (29
  recordings).
- `--solver heun --dt 2e-3` is a coarser discretization, so its loss differs slightly (0.2% here).
  `--solver` and `--dt` (in seconds) are accepted only together with `--batched`.

```bash
uv run python main.py --data-dir train_data --codes 3060 5050 8010 --resume --batched
uv run python main.py --data-dir train_data --codes 3060 5050 8010 --resume --batched --solver heun --dt 2e-3
```

The model in `checkpoint/` was trained with the default configuration. Evaluation (`--eval-r2`)
and plots always use Tsit5 at 1 ms.

## Baselines

Each baseline has its own entry script and `readme.md`. The GRU and Koopman baselines ship the
models used in the paper and load them by default; their default data folder is `TRAIN DATA/`, so
pass `--data-dir train_data` to use the sample recordings.

GRU (`comparisions/RNN/`), trained on the 29 training conditions with the chamber air masses as
inputs and a recursive training horizon of 500 samples:

```bash
uv run python comparisions/RNN/main_RNN.py --data-dir train_data --codes 3060 5050 8010 --eval-r2 --no-plot
```

Koopman/EDMDc (`comparisions/Koopman/`):

```bash
uv run python comparisions/Koopman/main_Koopman.py --data-dir train_data --codes 3060 5050 8010 --eval-r2 --no-plot
```

Add `--train-new` to either command to train a new model instead of loading the saved one.

Kang-type analytical model (`comparisions/Kang/`): it has no saved model. Each run fits the force
coefficients from single-muscle characterization data (`--single-pam-dir`, default
`Single PAM data/`) and then simulates the joint. That data is not included in this repository,
so this baseline cannot be run from the repository alone.

```bash
uv run python comparisions/Kang/main_Kang.py --single-pam-dir "Single PAM data" --data-dir train_data --codes 3060 5050 8010
```

## Saved models and outputs

- `checkpoint/`: the trained hybrid Neural ODE (`hybrid_model_puresine_16.eqx`), its optimizer
  state, and the outputs of `main.py`.
- `comparisions/RNN/checkpoint_RNN/`: the GRU model (`model_all_checkpoint.pt`) and its outputs.
- `comparisions/Koopman/checkpoint_Koopman/`: the Koopman model (`koopman_model.npz`) and its
  outputs.
- `checkpoint_Kang/` (created in the folder the Kang script is run from; change with
  `--output-dir`): outputs of the current run only.

## Citation

If you use this code or data, please cite:

```bibtex
@misc{wang2026physics,
  title         = {Physics-Embedded Neural {ODEs} for Learning Antagonistic Pneumatic Artificial Muscle Dynamics},
  author        = {Wang, Xinyao and Realmuto, Jonathan},
  year          = {2026},
  eprint        = {2602.23670},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2602.23670}
}
```

## Notes

The folder name `comparisions` is kept as is to match the import paths.
