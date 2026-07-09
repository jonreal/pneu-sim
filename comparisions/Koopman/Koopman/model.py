import os
import numpy as np

from .data import dataset_to_arrays, fit_scalers, build_snapshot_matrices
from .lifting import lift_state
from .utils import StandardScaler


def solve_ridge_regression(G, Y, ridge=1e-3):
    """
    Solve:
        Y = G @ K.T

    Returns:
        K: [output_dim, feature_dim]

    If ridge > 0:
        K = Y.T G (G.T G + ridge I)^(-1)

    This is more stable than np.linalg.pinv for this data.
    """
    G = np.asarray(G, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    n_features = G.shape[1]

    if ridge > 0:
        lhs = G.T @ G + ridge * np.eye(n_features)
        rhs = G.T @ Y
        K_T = np.linalg.solve(lhs, rhs)
        K = K_T.T
    else:
        K = Y.T @ np.linalg.pinv(G.T)

    return K


def train_koopman(
    data_list,
    lift_type="poly2",
    force_scale=1.0,
    ridge=1e-3,
):
    """
    Main predictor:

        x_{k+1} = K [psi(x_k); u_k]

    This is Koopman/EDMD-style lifting, but the rollout is performed in
    physical state space by relifting the predicted x at every step.

    This is intentionally more stable than pure lifted rollout:
        z_{k+1} = A z_k + B u_k
    """
    x_scaler, u_scaler = fit_scalers(data_list, force_scale=force_scale)

    G, Y, Xk_all = build_snapshot_matrices(
        data_list=data_list,
        x_scaler=x_scaler,
        u_scaler=u_scaler,
        lift_type=lift_type,
        force_scale=force_scale,
    )

    K = solve_ridge_regression(G, Y, ridge=ridge)

    # One-step prediction check
    Y_one = G @ K.T
    one_step_rmse = np.sqrt(np.mean((Y_one - Y) ** 2, axis=0))

    n_lift = lift_state(np.zeros((1, 4)), lift_type=lift_type).shape[1]
    n_u = 1

    print("\n=========== KOOPMAN TRAINING CHECK ===========")
    print(f"Number of datasets       = {len(data_list)}")
    print(f"Number of snapshots      = {G.shape[0]}")
    print(f"Physical state dim       = 4")
    print(f"Input dim                = {n_u}")
    print(f"Lifted state dim         = {n_lift}")
    print(f"Feature dim [z;u]        = {G.shape[1]}")
    print(f"Lift type                = {lift_type}")
    print(f"Force scale divisor      = {force_scale}")
    print(f"Ridge                    = {ridge}")
    print(f"K shape                  = {K.shape}")

    print("\n=========== ONE-STEP NORMALIZED RMSE ===========")
    print(f"x   = {one_step_rmse[0]:.6e}")
    print(f"dx  = {one_step_rmse[1]:.6e}")
    print(f"Pf  = {one_step_rmse[2]:.6e}")
    print(f"Pe  = {one_step_rmse[3]:.6e}")

    return {
        "K": K,
        "x_scaler": x_scaler,
        "u_scaler": u_scaler,
        "lift_type": lift_type,
        "force_scale": force_scale,
        "ridge": ridge,
    }


def rollout_koopman(
    model_dict,
    dataset_tuple,
    warmup_steps=1,
    clip_x_norm=8.0,
    stop_if_nonfinite=True,
):
    """
    Stable relifted rollout:

        z_k = psi(x_k)
        x_{k+1} = K [z_k; u_k]

    At each time step, the predicted physical state is relifted.
    This avoids unconstrained lifted-state explosion.

    warmup_steps:
        Number of initial measured samples copied into prediction.
        After warmup, model predicts recursively.

    clip_x_norm:
        Clip normalized predicted states to avoid impossible numerical blow-up.
        This does not make the model physically perfect; it prevents overflow.
    """
    K = model_dict["K"]
    x_scaler = model_dict["x_scaler"]
    u_scaler = model_dict["u_scaler"]
    lift_type = model_dict["lift_type"]
    force_scale = model_dict["force_scale"]

    ts, X_raw, U_raw = dataset_to_arrays(dataset_tuple, force_scale=force_scale)

    T = len(X_raw)

    Xn_true = x_scaler.transform(X_raw)
    Un = u_scaler.transform(U_raw)

    Xn_pred = np.zeros_like(Xn_true, dtype=np.float64)

    warmup_steps = int(max(1, warmup_steps))
    warmup_steps = min(warmup_steps, T)

    Xn_pred[:warmup_steps, :] = Xn_true[:warmup_steps, :]

    x_curr = Xn_true[warmup_steps - 1:warmup_steps, :]

    for k in range(warmup_steps - 1, T - 1):
        z_curr = lift_state(x_curr, lift_type=lift_type)
        u_curr = Un[k:k + 1, :]

        g_curr = np.hstack([z_curr, u_curr])

        x_next = g_curr @ K.T

        if clip_x_norm is not None:
            x_next = np.clip(x_next, -clip_x_norm, clip_x_norm)

        if not np.all(np.isfinite(x_next)):
            print(f"Warning: non-finite prediction at step k={k}.")
            if stop_if_nonfinite:
                Xn_pred[k + 1:, :] = Xn_pred[k:k + 1, :]
                break

        Xn_pred[k + 1, :] = x_next.reshape(-1)
        x_curr = x_next

    X_pred_raw = x_scaler.inverse_transform(Xn_pred)

    # Return original Force_array in loader units
    _, _, Force_original = dataset_to_arrays(dataset_tuple, force_scale=1.0)

    return ts, X_raw, X_pred_raw, Force_original.reshape(-1)


def save_koopman_model(model_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez(
        save_path,
        K=model_dict["K"],
        X_mean=model_dict["x_scaler"].mean,
        X_std=model_dict["x_scaler"].std,
        U_mean=model_dict["u_scaler"].mean,
        U_std=model_dict["u_scaler"].std,
        lift_type=np.array(model_dict["lift_type"]),
        force_scale=np.array(model_dict["force_scale"]),
        ridge=np.array(model_dict["ridge"]),
    )

    print(f"\nSaved Koopman model to: {save_path}")


def load_koopman_model(load_path):
    data = np.load(load_path, allow_pickle=True)

    x_scaler = StandardScaler()
    u_scaler = StandardScaler()

    x_scaler.mean = data["X_mean"]
    x_scaler.std = data["X_std"]

    u_scaler.mean = data["U_mean"]
    u_scaler.std = data["U_std"]

    model_dict = {
        "K": data["K"],
        "x_scaler": x_scaler,
        "u_scaler": u_scaler,
        "lift_type": str(data["lift_type"].item()),
        "force_scale": float(data["force_scale"]),
        "ridge": float(data["ridge"]),
    }

    print(f"Loaded Koopman model from: {load_path}")
    print(f"Lift type: {model_dict['lift_type']}")
    print(f"Force scale: {model_dict['force_scale']}")
    print(f"Ridge: {model_dict['ridge']}")

    return model_dict
