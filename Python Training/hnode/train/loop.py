# hnode/train/loop.py
import os, time, copy
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import diffrax
from jax import debug
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from hnode.core.models import HybridSystem

@dataclass
class TrainCfg:
    # NN arch
    width: int = 96
    depth: int = 2

    # optimization
    lr: float = 1e-2
    epochs: int = 1000
    patience: int = 100
    max_patience: int = 1000
    reduce_factor: float = 0.95
    threshold_per_dataset: float = 3.0   # raw code: threshold = 3.0 * num_datasets

    # resume control
    resume_training: bool = False

    # --- separate LOAD vs SAVE paths/names ---
    load_checkpoint_dir: str = "checkpoint"
    load_model_name: str = "hybrid_model_5datasets.eqx"
    load_opt_state_name: str = "hybrid_opt_state_5datasets.eqx"
    load_info_name: str = "best_info_5datasets.txt"

    save_checkpoint_dir: str = "checkpoint"
    save_model_name: str = "hybrid_model_5datasets.eqx"
    save_opt_state_name: str = "hybrid_opt_state_5datasets.eqx"
    save_info_name: str = "best_info_5datasets.txt"

    # solver
    dt0: float = 1e-3
    max_steps: int = 100000

def evaluate_r2(model, data_list, save_path="Verification/R2_results_225.csv"):
    #Evaluate R2 for each dataset in data_list and save as CSV.
    #Columns: [dataset_id, mf, me, R2_x, R2_dx, R2_Pf, R2_Pe]

    results = []

    for idx, (x0, ts, y_true, u_fn, mf, me, xeq, Force_array) in enumerate(data_list):
        # Solve model forward like in training
        term = diffrax.ODETerm(lambda t, y, args: model(t, y, u_fn, mf, me))
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0=0.001,
            y0=x0,
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=100000
        )

        y_pred = np.array(sol.ys)
        y_true = np.array(y_true)

        def compute_r2(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Compute R2 for each signal
        R2_x  = compute_r2(y_true[:, 0], y_pred[:, 0])
        R2_dx = compute_r2(y_true[:, 1], y_pred[:, 1])
        R2_Pf = compute_r2(y_true[:, 2], y_pred[:, 2])
        R2_Pe = compute_r2(y_true[:, 3], y_pred[:, 3])

        results.append([idx+1, mf, me, R2_x, R2_dx, R2_Pf, R2_Pe])

        print(f"Dataset {idx+1}: mf={mf:.4f}, me={me:.4f}, R2_x={R2_x:.4f}, R2_dx={R2_dx:.4f}, R2_Pf={R2_Pf:.4f}, R2_Pe={R2_Pe:.4f}")

    # Save as CSV
    header = "dataset_id,mf,me,R2_x,R2_dx,R2_Pf,R2_Pe"
    np.savetxt(save_path, results, delimiter=",", header=header, comments="")
    print(f"✅ R² results saved to {save_path}")

    return np.array(results)

def make_batched_loss_fn(data_list):
    n_datasets = len(data_list)
    @eqx.filter_value_and_grad
    def loss_fn(model: HybridSystem):
        total_loss = 0.0
        total_pen = 0.0
        for x0, ts, y_true, u_fn, mf, me, xeq, Force_array in data_list:
            wrapped_model = lambda t, y, args: model(t, y, u_fn, mf, me)
            term = diffrax.ODETerm(wrapped_model)
            sol = diffrax.diffeqsolve(
                term, diffrax.Tsit5(), ts[0], ts[-1],
                dt0=0.001, y0=x0,
                saveat=diffrax.SaveAt(ts=ts),
                adjoint=diffrax.RecursiveCheckpointAdjoint(),
                max_steps=100000
            )
            # weights: 100 on (x,dx) and 1 on (Pf,Pe)
            mse_pos = np.mean((sol.ys[:, 0:2] - y_true[:, 0:2])**2)
            mse_pressure = np.mean((sol.ys[:, 2:4] - y_true[:, 2:4])**2)

            total_loss += 100 * mse_pos + mse_pressure
        return total_loss / n_datasets
    return loss_fn

def make_train_step(optimizer, loss_fn):
    @eqx.filter_jit
    def train_step(model, opt_state):
        loss, grads = loss_fn(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    return train_step

def _load_paths(cfg: TrainCfg):
    os.makedirs(cfg.load_checkpoint_dir, exist_ok=True)
    return (
        os.path.join(cfg.load_checkpoint_dir, cfg.load_model_name),
        os.path.join(cfg.load_checkpoint_dir, cfg.load_opt_state_name),
        os.path.join(cfg.load_checkpoint_dir, cfg.load_info_name),
    )

def _save_paths(cfg: TrainCfg):
    os.makedirs(cfg.save_checkpoint_dir, exist_ok=True)
    return (
        os.path.join(cfg.save_checkpoint_dir, cfg.save_model_name),
        os.path.join(cfg.save_checkpoint_dir, cfg.save_opt_state_name),
        os.path.join(cfg.save_checkpoint_dir, cfg.save_info_name),
    )

def _save_checkpoint(model, opt_state, loss, epoch, save_model_path, save_opt_path, save_info_path):
    eqx.tree_serialise_leaves(save_model_path, model)
    eqx.tree_serialise_leaves(save_opt_path, opt_state)
    with open(save_info_path, "w") as f:
        f.write(f"best_epoch: {epoch}\n")
        f.write(f"best_loss:  {float(loss):.6e}\n")
    print(f"💾 Saved new best model at epoch {epoch}, loss={loss:.6f}")

def train_model(data_list, *, cfg: TrainCfg, seed: int = 42):
    # use separate paths
    load_model_path, load_opt_path, load_info_path = _load_paths(cfg)
    save_model_path, save_opt_path, save_info_path = _save_paths(cfg)

    lr = cfg.lr
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    key = jax.random.PRNGKey(seed)

    if cfg.resume_training:
        print("✅ Resuming from checkpoint...")
        model = HybridSystem(key=key, width=cfg.width, depth=cfg.depth)

        try:
            model = eqx.tree_deserialise_leaves(load_model_path, model)
        except RuntimeError as e:
            if "has changed dtype" in str(e):
                # load using an int32 'like' for the params leaf
                model_i32 = eqx.tree_at(
                    lambda m: m.params,
                    model,
                    jnp.asarray(model.params, dtype=jnp.int32),
                )
                model = eqx.tree_deserialise_leaves(load_model_path, model_i32)
                # cast params back to float32 so they’re trainable
                model = eqx.tree_at(
                    lambda m: m.params,
                    model,
                    jnp.asarray(model.params, dtype=jnp.float32),
                )
            else:
                raise

        # opt state
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        best_loss = float("inf")
        best_model = copy.deepcopy(model)
        best_opt_state = opt_state
        best_epoch = 0
    else:
        print("🚀 Starting new training...")
        model = HybridSystem(key=key, width=cfg.width, depth=cfg.depth)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        best_loss = float("inf")
        best_model = None
        best_opt_state = None
        best_epoch = 0

    loss_fn = make_batched_loss_fn(data_list)
    train_step = make_train_step(optimizer, loss_fn)

    num_datasets   = len(data_list)
    threshold      = cfg.threshold_per_dataset * num_datasets
    patience       = cfg.patience
    max_patience   = cfg.max_patience
    reduce_factor  = cfg.reduce_factor
    no_improve_eps = 0

    print("    Epoch    TimeElapsed    LearnRate    TrainingLoss    BestLoss    BestEpoch")
    print("    _____    ___________    _________    ____________    ________    _________")
    start_time = time.time()

    for epoch in range(cfg.epochs):
        # one training step
        model, opt_state, loss = train_step(model, opt_state)
        loss_val = float(loss)

        if np.isnan(loss_val) or np.isinf(loss_val):
            print(f"⚠️  Epoch {epoch}: loss={loss_val}, rolling back & cutting LR next")
            model, opt_state = best_model, best_opt_state
            no_improve_eps = patience

        # check improvement
        if loss < best_loss:
            best_loss       = loss
            best_model      = copy.deepcopy(model)
            best_epoch      = epoch
            best_opt_state  = opt_state
            no_improve_eps  = 0

            # save immediately on improvement
            _save_checkpoint(best_model, best_opt_state, best_loss, best_epoch,
                            save_model_path, save_opt_path, save_info_path)
        else:
            no_improve_eps += 1

        # logging
        elapsed = time.time() - start_time
        print(f"{epoch:9d}{elapsed:15.2f}{lr:13.2e}"
              f"{float(loss):15.6f}{float(best_loss):15.6f}{best_epoch:9d}")

        # early stop conditions (identical logic)
        if best_loss < threshold and no_improve_eps >= patience:
            print(f"Early stopping at epoch {epoch}: best_loss={best_loss:.6f} unchanged for {no_improve_eps} epochs.")
            break

        if no_improve_eps >= max_patience:
            print(f"Early stopping at epoch {epoch}: best_loss={best_loss:.6f} unchanged for {no_improve_eps} epochs.")
            break

        if no_improve_eps >= patience:
            print(f"— plateau at epoch {epoch}, cutting LR → {lr*reduce_factor:.1e}")
            model, opt_state = best_model, best_opt_state
            lr *= reduce_factor
            optimizer  = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
            opt_state  = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
            train_step = make_train_step(optimizer, loss_fn)
            no_improve_eps = 0

    # restore best model
    model = best_model
    print(f"Restored model with best loss = {best_loss:.6f} at epoch {best_epoch}")

    return best_model, dict(epoch=best_epoch, loss=float(best_loss))
